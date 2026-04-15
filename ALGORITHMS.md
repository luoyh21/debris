# 轨迹仿真与碰撞风险算法说明

本文档说明本项目中“轨迹仿真”和“碰撞风险评估”的**算法原理**与**工程实现**，并给出关键代码入口，便于开发、评审与二次扩展。

---

## 1. 总体流程

系统核心链路如下：

1. 使用 6-DOF 模型生成火箭轨迹（`trajectory/`）。
2. 使用 SGP4 传播目录碎片轨道，并离散成时空轨迹段（`propagator/` + `ingestion/`）。
3. 在数据库中做时空预筛（PostGIS）。
4. 对候选目标做 TCA（最近接近时刻）求解与遭遇平面投影（`lcola/encounter.py`）。
5. 计算碰撞概率 Pc（Foster 数值积分 / Chan 快速近似）。
6. 输出逐阶段风险与飞越窗口（LCOLA）。

---

## 2. 轨迹仿真算法（火箭）

对应模块：
- `trajectory/six_dof.py`
- `trajectory/rocketpy_sim.py`
- `trajectory/launch_phases.py`

### 2.1 动力学状态与坐标系

在 `six_dof.py` 中，积分状态向量为：

\[
\mathbf{x}=[x,y,z,v_x,v_y,v_z,m]
\]

- 位置/速度在 ECEF（km, km/s）
- 质量 `m`（kg）
- 同时输出 ECI 表达（用于后续会遇计算）

主要坐标与转换：
- `geodetic_to_ecef()`
- `ecef_to_geodetic()`
- `ecef_to_eci()`

### 2.2 受力模型

在 ODE 右端（`_build_ode()`）中，总加速度由下列项叠加：

\[
\mathbf{a}=\mathbf{a}_{grav,J2}+\mathbf{a}_{cor}+\mathbf{a}_{cent}+\mathbf{a}_{drag}+\mathbf{a}_{thrust}
\]

1. **重力项（含 J2）**  
   使用地球非球形二阶项修正，函数：`_gravity_j2()`

2. **旋转坐标系惯性项**  
   - Coriolis：\(-2\boldsymbol{\omega}\times\mathbf{v}\)
   - Centrifugal：\(-\boldsymbol{\omega}\times(\boldsymbol{\omega}\times\mathbf{r})\)

3. **气动阻力**  
   动压 \(q=\frac12 \rho v^2\)，其中密度 \(\rho\) 由分层指数大气模型 `atmo_density()` 给出。  
   阻力方向与速度反向，系数由各级 `cd, area` 给定。

4. **推力项**  
   各级火箭按 `RocketStage` 的点火/关机时刻切换。  
   推力方向采用“程序俯仰 + 重力转弯”混合策略：
   - 一级主用时间程序角
   - 上面级可切换为速度分数引导

### 2.3 数值积分与级间分离

`integrate_trajectory()` 使用 `scipy.integrate.solve_ivp(RK45)` 分段积分：

- 每级关机后约 2 秒执行干质量抛弃（stage separation）
- 以“分段重启积分”的方式保证质量状态一致
- 可选终止事件：
  - 落地事件 `ground_impact`
  - 入轨事件 `_make_orbit_insertion_event`（可配置）

### 2.4 Monte Carlo 协方差

`monte_carlo_covariance()` 通过多次扰动仿真估计轨迹不确定性：

- 推力扰动：默认 ±2%（1σ）
- 质量扰动：默认 ±0.5%（1σ）
- 输出每个时刻 \(6\times6\) 协方差（ECI）
- 写回 `TrajectoryPoint.cov_6x6`

该协方差后续进入遭遇平面投影并用于 Pc 计算。

### 2.5 发射阶段划分

`launch_phases.py` 的 `detect_phases()` 将轨迹分为：

- `ASCENT`
- `PARKING_ORBIT`
- `TRANSFER_BURN`
- `POST_SEPARATION`

优先使用关键事件时间（MECO、分离等）；缺失时回退到启发式判定（高度阈值、近圆轨道判定、速度突变等）。

---

## 3. 轨道传播算法（目录碎片）

对应模块：
- `propagator/sgp4_propagator.py`
- `ingestion/ingest_gp.py`

### 3.1 SGP4 传播

`SGP4Propagator` 支持两类输入：

1. 直接 TLE（`TLE_LINE1/2`）
2. GP 平根数字段重建 `Satrec`

通过 `propagate(epoch)` 返回某时刻状态向量 `StateVector`（ECI）。

### 3.2 轨迹段离散

`generate_segments()` 将传播结果按固定时间窗离散为 `OrbitSegment`：

- 每段默认 10 分钟
- 每段默认 7 个采样点
- 同时保留 ECI 点与近似地理点（lat/lon/alt）

这些轨迹段入库后由 PostGIS 做快速时空筛选。

---

## 4. 碰撞风险评估算法

对应模块：
- `lcola/encounter.py`
- `lcola/foster_pc.py`
- `lcola/fly_through.py`
- `ingestion/collision_risk.py`（历史/简化流程）

### 4.1 时空预筛（PostGIS）

在 `fly_through.py` 的 `_spatial_prefilter()` 中，先对每个发射阶段进行粗筛：

- 时间条件：轨迹段时间窗重叠
- 空间条件：轨迹包围框 / 扩展阈值（默认 200 km 粗筛）

目的：把海量目录对象缩小为可精算的候选集。

### 4.2 TCA 求解（最近接近时刻）

`encounter.py::find_tca()` 的方法：

1. 对两条轨迹分别做三次样条插值（`CubicSpline`）
2. 在重叠时间窗粗采样寻找最小距离邻域
3. 在局部区间上用 `minimize_scalar(method="bounded")` 做一维极小化
4. 用有限差分估计相对速度向量

输出：
- `tca_s`
- `miss_distance_km`
- `r_rel_km`
- `v_rel_kms`

### 4.3 遭遇平面构造与协方差投影

`build_encounter_frame(v_rel)` 构造遭遇平面基：

- \(\hat e_{\xi}\)：沿相对速度方向（法向）
- \(\hat e_{\eta}\)：与 \(\hat e_{\xi}\) 正交
- \(\hat e_{\zeta}=\hat e_{\xi}\times \hat e_{\eta}\)

使用投影矩阵 \(T=[\hat e_{\zeta},\hat e_{\eta}]\)：

- 失配向量投影：\(\mathbf{m}_{2D}=T^T\mathbf{r}_{rel}\)
- 协方差投影：\(\Sigma_{2D}=T^T(C_1+C_2)T\)

得到 2D 会遇几何，即 Foster/Chan 的直接输入。

### 4.4 Pc 计算：Foster 与 Chan

#### Foster（高精度）

`foster_pc.py::foster_pc()` 采用双重积分 `scipy.integrate.dblquad`：

\[
P_c=\iint_{x^2+y^2\le HBR^2}\mathcal{N}(\mathbf{r};\mathbf{m},\Sigma)\,dx\,dy
\]

特点：
- 物理含义直接、结果稳定
- 计算开销大，适合精算或结果确认

#### Chan（快速）

`foster_pc.py::chan_pc()` 主要用于批量筛查：

- 近各向同性情况走非中心卡方 CDF 快速路径
- 一般情形当前实现会回退到 Foster 数值积分（保证可用性与正确性）

> 在 `batch_pc()` 中，先快筛，再对高风险前 10% 用 Foster 精化。

### 4.5 LCOLA 飞越扫描

`fly_through.py::FlyThroughScreener.screen()` 对发射窗口逐时刻扫描（如每 60s）：

1. 对每个候选发射时刻、每个发射阶段：
   - 空间预筛
   - 传播候选碎片
   - 求 TCA 与遭遇几何
   - 计算 Pc
2. 汇总每个发射时刻 `max_pc`
3. 按阈值判定 blackout（禁射）
4. 合并连续 blackout/safe 时段

默认阈值（代码常量）：
- `PC_UNCREWED = 1e-5`
- `PC_CREWED = 1e-6`
- `MISS_ABSOLUTE_KM = 25`

判定逻辑通常为：
- `Pc >= threshold` 或 `miss_distance < 25km` 即触发高风险标记。

---

## 5. 工程实现要点（与理论对应）

1. **两级筛选架构**：先 PostGIS 粗筛、再精算，可把复杂度控制在可交互范围。  
2. **轨迹与风险解耦**：火箭轨迹（6-DOF）与目录碎片（SGP4）独立建模，接口在 TCA 会遇层汇合。  
3. **数值稳健性**：协方差矩阵在投影后会做对称化与正定修正，避免积分异常。  
4. **精度/性能折中**：Chan 用于快筛，Foster 用于最终可信结果。  
5. **支持任务级评估**：既支持单次逐阶段风险（`assess_launch_phases`），也支持窗口扫描（`FlyThroughScreener.screen`）。

---

## 6. 当前假设与局限

1. SGP4 误差随预报时长增长，长时间窗需谨慎。  
2. 默认协方差在部分场景下仍是各向同性近似（无外部 CDM 时）。  
3. 火箭气动与控制律为工程简化模型，不是飞控级高保真闭环。  
4. Chan 一般情形尚未完整实现解析级数，当前可用但依赖回退路径。  
5. ECEF↔ECI 采用工程近似（未引入完整章动/岁差链路）。

---

## 7. 关键函数索引

- 轨迹积分：`trajectory/six_dof.py::integrate_trajectory`
- MC 协方差：`trajectory/six_dof.py::monte_carlo_covariance`
- 阶段划分：`trajectory/launch_phases.py::detect_phases`
- SGP4 传播：`propagator/sgp4_propagator.py::SGP4Propagator.propagate`
- TCA 求解：`lcola/encounter.py::find_tca`
- 遭遇几何：`lcola/encounter.py::compute_encounter`
- Foster Pc：`lcola/foster_pc.py::foster_pc`
- LCOLA 扫描：`lcola/fly_through.py::FlyThroughScreener.screen`
- 逐阶段评估：`lcola/fly_through.py::assess_launch_phases`

---

## 8. 建议的后续增强

1. 完整实现 Chan 各向异性解析项，减少回退积分次数。  
2. 引入更真实的外部协方差（CDM/OD），替代统一默认 \(\sigma\)。  
3. 对 TCA 搜索加入多峰保护（长时间窗、机动场景）。  
4. 引入并行化候选评估（按 NORAD 分片）提升吞吐。  
5. 增加算法回归测试集（典型会遇几何 + 基准 Pc）。

本项目里有两条“轨迹”：

火箭轨迹（你发射器本身）
碎片轨迹（目录目标）
1) 火箭轨迹（6-DOF 数值积分）
在 trajectory/six_dof.py 里，状态是
[x, y, z, vx, vy, vz, m]（位置、速度、质量），用 solve_ivp 做时间积分。

每一步都会算这些力/项：

地球引力（含 J2 扁率修正）
地球自转带来的科里奥利/离心项（ECEF 下）
大气阻力（随高度的分层指数大气）
推力项（按级间点火/关机时刻 + 俯仰程序）
所以它不是“画一条预设曲线”，而是物理方程一步步积分出来的。
并且支持 monte_carlo_covariance 做多次扰动，得到轨迹协方差。

2) 碎片轨迹（SGP4 传播）
在 propagator/sgp4_propagator.py 里，碎片用 TLE/GP 元素通过 SGP4 传播到目标时刻，得到 ECI 位置速度。
LCOLA 主流程 lcola/fly_through.py 里会对候选碎片再次传播、求 TCA、再算 Pc。

碎片是固定的吗？
结论：不固定，但有“时间快照”特性。

物理上不固定：系统把碎片当作“按轨道运动”的对象，随时间变化。
单次评估时：通常假设碎片“无机动、按 SGP4 自然飞行”，所以在该次计算里其轨道模型是确定的。
数据层面：如果底层 TLE/GP 不更新，看起来会“像固定”；更新数据库后，传播结果会变。
工程上两种用法：
trajectory_segments 是预生成轨迹段（用于快速预筛，像缓存快照）
精算阶段（LCOLA）会再次做 SGP4 传播，不只依赖静态段