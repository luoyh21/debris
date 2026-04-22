# Space Debris Monitor

空间碎片监测与火箭发射碰撞风险评估系统。

## 系统架构

```
space_debris/
├── .env                              # 配置文件
├── run.py                            # CLI 统一入口
├── requirements.txt
│
├── config/
│   └── settings.py                   # 从 .env 加载所有配置
│
├── fetcher/
│   └── spacetrack_client.py          # Space-Track.org REST API 客户端
│                                     # 支持：GP均根数、CDM合取事件、衰落预报
│
├── propagator/
│   └── sgp4_propagator.py            # SGP4 轨道传播器
│                                     # 基于 python-sgp4 v2.25
│                                     # generate_segments() → 3D LineStringZ 片段
│
├── database/
│   ├── models.py                     # SQLAlchemy ORM 模型
│   ├── db.py                         # 连接池 + session_scope()
│   └── migrations/
│       └── 001_init_postgis.sql      # PostGIS 建表 + GIST 空间索引 + 视图
│
├── ingestion/
│   ├── ingest_gp.py                  # 完整流水线：拉取 → catalog → 传播 → 轨迹段
│   └── collision_risk.py             # 历史遗留：简单 Pc 估算（已被 lcola/ 取代）
│
├── trajectory/                       # 6-DOF 轨迹仿真模块
│   ├── __init__.py
│   ├── six_dof.py                    # 核心 6-DOF ECEF 积分器
│   │                                 # J2 引力 + USSA-76 大气 + 推力（重力转弯）
│   │                                 # Coriolis + 离心加速度（旋转框架）
│   ├── rocketpy_sim.py               # 预设运载火箭 + 仿真入口
│   │                                 # 预设：CZ-5B / Falcon 9 / Ariane 6
│   ├── oem_io.py                     # CCSDS OEM 2.0 ASCII/KVN 读写
│   │                                 # 含 21 元素协方差块
│   └── launch_phases.py              # 发射阶段检测
│                                     # ASCENT / PARKING_ORBIT / TRANSFER_BURN / POST_SEPARATION
│
├── lcola/                            # 发射合取分析（LCOLA）模块
│   ├── __init__.py
│   ├── encounter.py                  # TCA 求解 + 遭遇平面几何
│   │                                 # 三次样条插值 + minimize_scalar 精化
│   ├── foster_pc.py                  # Foster (1992) 2-D 碰撞概率
│   │                                 # scipy.integrate.dblquad 极坐标数值积分
│   └── fly_through.py                # 完整 LCOLA 飞越筛选流水线
│                                     # PostGIS 预筛 → SGP4 传播 → TCA → Foster Pc
│                                     # assess_launch_phases()：逐阶段碰撞风险评估
│
├── agent/
│   └── debris_agent.py               # LLM 工具调用 Agent（4个内置工具）
│
└── streamlit_app/
    └── app.py                        # 8 页面 Streamlit 仪表盘
```

## 数据库表

| 表名 | 说明 |
|------|------|
| `catalog_objects` | NORAD 目标对象（碎片/载荷/火箭级），含近地点/远地点高度 |
| `gp_elements` | GP 均根数（TLE 数据），附历元 |
| `trajectory_segments` | SGP4 传播的 3D LineStringZ 轨迹段（时空索引） |
| `collision_risks` | 碰撞概率评估结果（历史遗留，新流程使用 lcola/ 模块） |

## 配置 .env

```env
SPACETRACK_USERNAME=your@email.com
SPACETRACK_PASSWORD=your_password

OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini

DB_HOST=localhost
DB_PORT=5432
DB_NAME=space_debris
DB_USER=postgres
DB_PASSWORD=postgres

SEGMENT_MINUTES=10          # 轨迹段时间窗口（分钟）
PROPAGATION_HORIZON_DAYS=3  # 向未来传播天数
```

## 快速开始

### 1. 准备 PostgreSQL + PostGIS

```bash
# 确认 PostgreSQL 运行
pg_lsclusters

# 如未运行，启动
pg_ctlcluster 14 main start

# 创建数据库
su -c "psql -c 'CREATE DATABASE space_debris;'" postgres
su -c "psql -c \"ALTER USER postgres PASSWORD 'postgres';\"" postgres
```

### 2. 初始化数据库 Schema

```bash
cd /mnt/space_debris
python3 run.py init-db
```

### 3. 从 Space-Track 拉取数据并传播轨道

```bash
# 全量下载（默认，约 25 000–27 000 个在轨目标）
# 首次运行耗时约 30–90 分钟（含 SGP4 传播 + 数据库写入）
python3 run.py ingest

# 后台运行（推荐）
nohup python3 run.py ingest > /tmp/ingest.log 2>&1 &
tail -f /tmp/ingest.log   # 实时查看进度

# 仅下载最新 N 个对象（快速验证连通性）
python3 run.py ingest --limit 1000

# 仅拉取不传播（仅写入目录和 GP 数据）
python3 run.py ingest --no-propagate

# 自定义传播窗口（默认 3 天 / 10 分钟段）
python3 run.py ingest --horizon-days 3 --seg-minutes 10
```

> **数据规模参考（全量）**
> - `catalog_objects`：~25 000–27 000 行
> - `gp_elements`：~26 000 行（部分目标无新鲜 TLE 被过滤）
> - `trajectory_segments`：~10–12 百万行（3 天 × 144 段/对象）

### 4. 运行轨迹仿真（6-DOF）

```bash
# 使用预设运载火箭进行仿真，输出 CCSDS OEM 文件
python3 run.py simulate --vehicle CZ-5B --launch-utc 2026-04-15T06:00:00Z \
    --lat 19.61 --lon 110.95 --azimuth 90 \
    --t-max 6000 --dt 10 --output /tmp/czb5_nominal.oem

# 可用预设：CZ-5B / Falcon9 / Ariane6
# 不加 --output 则仅打印轨迹摘要
```

### 5. 运行 LCOLA 飞越筛选

```bash
# 对给定 OEM 文件在 ±30 分钟窗口内逐 5 分钟步长评估碰撞风险
python3 run.py lcola --oem /tmp/czb5_nominal.oem \
    --launch-utc 2026-04-15T06:00:00Z \
    --window-minus 30 --window-plus 30 --step 5 \
    --hbr 0.02 --crewed \
    --output /tmp/lcola_result.json
```

### 6. 启动 Streamlit 仪表盘

```bash
python3 run.py app --port 8501
# 访问 http://localhost:8501
```

## Streamlit 页面说明

| 页面 | 功能 |
|------|------|
| **🌍 系统概览** | 总览：对象数、碎片数、轨迹段数、高危事件数；类型分布图；高度分布图 |
| **📚 目标目录** | 可筛选的目标对象数据库表（类型/国家/名称/近地点高度） |
| **🛤️ 轨迹片段** | 按时间窗口和 NORAD ID 查询 SGP4 轨迹段 |
| **🚀 轨迹仿真** | 配置运载火箭与发射参数，运行 6-DOF 仿真；高度/速度/质量曲线；Monte Carlo 协方差 |
| **📄 OEM 管理** | 从仿真结果生成 CCSDS OEM 2.0 文件；或解析已有 OEM 文件 |
| **⚠️ LCOLA 飞越筛选** | 多发射时刻窗口扫描；Pc 曲线；禁射窗口；合取事件表 |
| **☄️ 碰撞风险** | 基于轨迹仿真结果，逐发射阶段评估碰撞风险；阶段汇总 + Foster Pc 事件表（对数坐标柱状图） |
| **💬 AI 助手** | LLM 对话窗口，可用自然语言询问碎片数据库，支持 MCP 工具调用 |

> **典型工作流**：
> 1. **目标目录** — 确认在轨碎片数据已就绪
> 2. **🚀 轨迹仿真** — 配置并运行 6-DOF 仿真，结果存入 session
> 3. **📄 OEM 管理** — 可选：导出 CCSDS OEM 文件留档
> 4. **碰撞风险** — 加载 session 中的仿真结果，逐阶段评估 Foster Pc
> 5. **⚠️ LCOLA 飞越筛选** — 多时刻窗口扫描，确定最优发射时刻

## AI Agent 工具

Agent 具备 4 个内置工具 + 2 个 MCP 工具，可在对话中自动调用：

| 工具 | 类型 | 说明 |
|------|------|------|
| `query_debris_count` | 内置 | 统计指定区域/高度的碎片数量 |
| `query_high_risk_conjunctions` | 内置 | 返回最高碰撞概率事件 |
| `query_catalog_stats` | 内置 | 按类型/国家统计目标对象 |
| `run_sql` | 内置 | 执行任意只读 SQL 查询 |
| `query_debris_in_region` | MCP | 在指定地理区域和高度范围内检索在轨目标（PostGIS 空间索引） |
| `predict_launch_collision_risk` | MCP | 给定发射参数，运行 6-DOF 仿真 + Foster Pc 逐阶段碰撞风险预测 |

示例问题：
- "LEO 轨道（200–2000 km）有多少碎片？"
- "搜索文昌上空 500 km 范围内、高度 200–2000 km 的碎片，列出前 20 个"
- "用长征五号B从文昌向正东方向发射，预测明天 06:00 UTC 发射的各阶段碰撞风险"
- "俄罗斯有多少在轨载荷？"

MCP 服务端单独启动：
```bash
# stdio 模式（供 Claude Desktop / claude-code 接入）
python -m mcp.server

# HTTP/SSE 模式（供其他客户端接入，端口 8888）
python -m mcp.server --http
```

## 碰撞概率计算方法

采用 **Foster (1992) 2-D 碰撞概率**数值积分算法：

$$P_c = \iint_{x^2+y^2 \le HBR^2} \frac{1}{2\pi\sqrt{|\Sigma|}} \exp\!\left[-\tfrac{1}{2}(\mathbf{r}-\mathbf{m})^\top \Sigma^{-1} (\mathbf{r}-\mathbf{m})\right] dx\, dy$$

- 遭遇平面：以相对速度方向 $\hat{e}_\xi$ 和轨道面法向 $\hat{e}_\eta$ 为基
- 硬体半径（HBR）：20 m（碎片 + 载具联合）
- 批量预筛：Chan (2003) 级数近似快速计算，前 10% 高风险事件用 Foster 精化
- 风险等级：🔴 RED ($P_c \ge 10^{-5}$，任务无论有无人均需关注）· 🟠 AMBER ($P_c \ge 10^{-6}$，载人任务须关注）· 🟡 YELLOW ($P_c \ge 10^{-7}$，持续监视）· 🟢 GREEN

### 6-DOF 积分器

| 项目 | 说明 |
|------|------|
| 积分框架 | ECEF 旋转坐标系（含 Coriolis + 离心加速度） |
| 引力模型 | J2 摄动（$J_2 = 1.08263\times10^{-3}$） |
| 大气模型 | USSA-1976 指数分层（0–700 km，9 层） |
| 推力模式 | 重力转弯（俯仰踢动后沿速度矢量推力） |
| 积分方法 | `scipy.integrate.solve_ivp` RK45，rtol=1e-8，atol=1e-10 |
| Monte Carlo | 50 次推力（±2%）+ 质量（±0.5%）扰动，输出 (N, 6, 6) 协方差 |

PostGIS 空间预筛：
```sql
-- 粗筛：200 km 包围盒
SELECT * FROM trajectory_segments
WHERE geom_eci && ST_Expand(launch_bbox_eci, 200)
  AND time_start < t_end AND time_end > t_start;
```

## 已完成功能总览

> 以下为系统已全面实现并部署的功能模块，涵盖数据采集、轨道力学仿真、碰撞风险评估、长期任务规划、AI 智能分析与多维度可视化。

### 一、数据采集与多源融合

| 数据源 | 类型 | 规模 | 说明 |
|--------|------|------|------|
| **Space-Track.org** | GP 均根数 / TLE | ~27,000 在轨目标 | 美国战略司令部（18 SDS）官方目录，含碎片、载荷、火箭箭体 |
| **Jonathan McDowell GCAT** | 卫星目录 + 发射日志 | **68,690 条**（1957—2026） | 哈佛-史密松天体物理中心维护，全球最权威的发射历史日志，含部分不公开军事载荷修正数据 |
| **UNOOSA / Our World in Data** | 年度发射统计 | **1,274 条**（116 国，1957—2025） | 联合国外层空间事务厅（UNOOSA）登记的各国年度发射数量，通过 OWID API 自动获取 |
| **UCS Satellite Database** | 在轨卫星详情 | **7,560 条**（105 国） | 忧思科学家联盟维护，含卫星用途（军/民/商）、运营商、设计寿命、发射质量、轨道参数 |
| **ESA DISCOS** | 欧空局空间物体数据库 | **10,000 条**（API 批量获取） | 欧洲空间局 DISCOSweb API，含物体质量、截面积、碎片数量、预测再入日期 |
| **Asterank** | 小行星 / 近地天体（NEO）专题库 | **数千条**（API + 本地缓存） | http://www.asterank.com 维护，含小行星开普勒轨道根数、光谱类型、Δv、经济开采估值（price/profit） |

系统自动完成数据去重与清洗：
- Space-Track 数据用于实时轨道传播（SGP4）和碰撞风险计算
- GCAT 数据用于历史发射趋势统计、在轨目标演化分析，国家/地区按标准代码映射（US→美国、CIS/SU/RU→俄罗斯/苏联、PRC/CN→中国等 19 个国家）
- 入库脚本 `scripts/ingest_external.py` 支持增量更新

**数据库新增表**：

| 表名 | 说明 |
|------|------|
| `external_yearly_launches` | GCAT 按年/国家/目标类型统计的发射量（1,392 行） |
| `external_country_yearly_payload` | GCAT 按年/国家的有效载荷发射量（672 行） |
| `external_cumulative_onorbit` | GCAT 逐年累计在轨数量（含发射-衰减差分，210 行） |
| `external_onorbit_snapshot` | GCAT 当前在轨快照（44 行） |
| `external_ucs_satellites`   | UCS 在轨卫星详细信息（7,560 行） |
| `external_esa_discos`       | ESA DISCOS 物体物理参数（10,000 行） |
| `external_unoosa_launches`  | UNOOSA 年度发射统计（1,274 行） |
| `external_asterank`         | Asterank 小行星 / NEO 专题库（数千行） |

### 二、轨道力学仿真

| 模块 | 方法 | 精度 |
|------|------|------|
| **6-DOF 弹道仿真** | ECEF 坐标系数值积分（RK45），J2 引力 + USSA-76 大气阻力 + 推力矢量（重力转弯） | rtol=1e-8, atol=1e-10 |
| **SGP4 轨道传播** | 基于 GP/TLE 数据，python-sgp4 v2.25，批量传播 27,000+ 在轨目标 | 短期（<3天）误差 ~1 km |
| **轨道预报可视化** | 用户选取目标 NORAD ID → 实时 SGP4 传播 → 3D 椭圆轨道渲染（颜色渐变标注时序） | 支持 OEM 导入/导出 |
| **运载火箭预设** | CZ-5B（长征五号B）/ Falcon 9（猎鹰九号）/ Ariane 6（阿丽亚娜6号） | 各级推力、流量、质量参数 |

### 三、碰撞风险评估体系

#### 3.1 短期碰撞概率（LCOLA 飞越筛选）

完整的 **Launch Collision Avoidance** 流水线：

1. **PostGIS 空间预筛** — 200 km 包围盒 + GIST 索引快速候选筛选
2. **SGP4 碎片传播** — 对候选碎片在发射窗口内逐秒传播
3. **TCA 求解** — 线性插值 + 均匀网格扫描求最近接近时刻（`find_tca_fast`，8–15× 加速）
4. **Foster (1992) 2-D Pc** — 遭遇平面双高斯数值积分，Chan (2003) 级数近似快速筛选
5. **多时刻窗口扫描** — 滑动发射时刻，输出 Pc 曲线与禁射窗口

风险等级：🔴 RED（Pc ≥ 10⁻⁵）· 🟠 AMBER（Pc ≥ 10⁻⁶，载人须关注）· 🟡 YELLOW（Pc ≥ 10⁻⁷）· 🟢 GREEN

#### 3.2 长期任务碰撞风险评估

基于 **NASA ORDEM 3.1 碎片通量模型** + **泊松蒙特卡洛**的工程级评估：

$$P_c = 1 - \exp(-F \cdot A \cdot \Delta t)$$

- **ORDEM 3.1 通量表**：覆盖 200–2000 km 高度，>10 cm 和 >1 cm 两挡碎片通量，倾角修正因子
- **泊松蒙特卡洛**：n=2000 次试验，输出聚合碰撞概率 Pc_agg、交会次数分布（均值/P95）、最近逼近距离分布
- **碎片年增长率**：简化 LEGEND 模型，按高度段差异化增长率预测
- **可配置参数**：轨道高度、倾角、任务寿命（1–30年）、卫星面积、交会警戒距离、HBR、位置不确定度

**输出指标**：
- Pc（>10 cm 可追踪碎片）/ Pc（>1 cm 含不可追踪）
- 年碰撞率
- 交会次数（均值 / P95）
- 最近逼近距离（中位 / P95）
- 聚合碰撞概率 Pc_agg（蒙特卡洛聚合）
- 数据库中最接近目标轨道的 30 颗碎片详情

### 四、AI 智能分析助手

Agent 具备 **6 个 MCP 工具**，可在自然语言对话中自动调用：

| 工具 | 说明 |
|------|------|
| `query_debris_in_region` | 在指定地理区域和高度范围内检索在轨目标（PostGIS 空间索引） |
| `predict_launch_collision_risk` | 6-DOF 仿真 + Foster Pc 逐阶段碰撞风险预测 |
| `get_debris_reentry_forecast` | 预报即将再入大气层的空间目标 |
| `get_object_tle` | 获取指定 NORAD 编号目标的最新 TLE 轨道根数 |
| `query_debris_by_rcs` | 按雷达截面积（RCS）大小类别筛选空间目标 |
| `forecast_conjunction_risk` | **长期任务碰撞风险预测**（ORDEM 3.1 + 泊松 MC），返回 Pc_agg、交会次数、建议规避燃料量 |

**新增能力**：
- 支持长周期问题——如"5年内会有多少次小于2km的接近？建议配备多少规避燃料？"
- 强制工具调用规则：涉及长期风险、交会次数、Pc_agg 等问题时，Agent 自动调用 `forecast_conjunction_risk`，禁止以"缺少参数"拒绝
- 示例问题点击后**填入编辑框**（而非直接发送），用户可修改后再提交
- 聊天消息中的 Markdown 表格支持横向滚动，不再溢出重叠

### 五、可视化探索平台

8 个主页面 + 5 个可视化子标签：

#### 主页面

| 页面 | 功能亮点 |
|------|----------|
| **系统概览** | 在轨目标总数 / 碎片数量 / 轨迹片段总数（自定义 HTML 卡片防截断）；目标类型分布；半长轴高度分布（log10）；轨道层分布（LEO/MEO/GEO/HEO）；倾角分布；**历年航天发射趋势**（年代汇总 + 近年逐年 + 在轨演化）；主要国家 Top 15 |
| **可视化探索** | 三维沉浸式 5 子标签平台（详见下方） |
| **目标目录** | 可筛选数据库表 |
| **轨迹仿真** | 6-DOF 运载火箭仿真，高度/速度/质量曲线 |
| **LCOLA 飞越筛选** | 多发射时刻窗口扫描，Pc 曲线，禁射窗口，实时进度条（含预处理阶段详细状态） |
| **碰撞风险** | 逐发射阶段 Foster Pc 评估 |
| **长期风险评估** | ORDEM 3.1 + 泊松 MC，6 项指标卡片（3×2 布局），4 张分析图表，碎片环境详情 |
| **AI 助手** | 6 个 MCP 工具，长期预测能力，可编辑示例问题 |

#### 可视化探索 5 个子标签

| 子标签 | 内容 |
|--------|------|
| **全球碎片态势** | pydeck 全球投影 + Plotly 高度分布（LEO / 全空间 log10） |
| **高度分层下钻** | VLEO/LEO/MEO/GEO/HEO 五层 3D 地球+碎片点云，层内目标统计 |
| **目标轨道预报** | 选取 NORAD ID → SGP4 传播 → 3D 轨道渲染（颜色渐变标注时序），OEM 导入/导出 |
| **火箭发射碎片预警** | 6-DOF 仿真 → 时间轴拖动 → 实时查询附近碎片 → 3D 近场态势 |
| **发射历史趋势分析** | 年代汇总堆叠柱状图、近年逐年发射量、分国别折线趋势、2020 后地区对比横向条形图、在轨目标历史演化曲线、关键里程碑时间轴 |

### 六、数据安全

- 数据库对 LLM 触发的 SQL 实施**只读保护**：关键字黑名单（CREATE/DROP/ALTER/INSERT/UPDATE/DELETE）+ `SET TRANSACTION READ ONLY`
- 禁止 SQL 注入：多语句拦截 + 参数化查询

### 七、部署方式

- **Docker Compose 一键部署**：`docker compose up -d`
- 包含三个服务：`db`（PostgreSQL 15 + PostGIS）、`app`（Streamlit 前端，端口 8501）、`api`（REST API + 文档，端口 8000）
- PostgreSQL 数据持久化于本地 `./pgdata` 目录
- 健康检查自动等待数据库就绪后再启动前后端服务
- Streamlit 监听 `0.0.0.0:8501`，API 监听 `0.0.0.0:8000`，支持局域网访问
- 系统说明文档：`http://<host>:8000/docs`
- REST API Swagger：`http://<host>:8000/api/docs`

---

## 数据源接入状态

| 数据源 | 状态 | 数据库表 | 说明 |
|--------|------|----------|------|
| **Space-Track.org** | ✅ 已接入 | `catalog_objects`, `gp_elements`, `trajectory_segments` | 自动拉取 + SGP4 传播 |
| **Jonathan McDowell GCAT** | ✅ 已接入 | `external_yearly_launches`, `external_country_yearly_payload`, `external_cumulative_onorbit`, `external_onorbit_snapshot` | `scripts/ingest_external.py` |
| **UNOOSA / Our World in Data** | ✅ 已接入 | `external_unoosa_launches` | 1,274 条，116 国，1957—2025 |
| **UCS Satellite Database** | ✅ 已接入 | `external_ucs_satellites` | 7,560 条，105 国，含用途/运营商/寿命/质量 |
| **ESA DISCOS** | ✅ 已接入 | `external_esa_discos` | 10,000 条，含质量/截面积/碎片数/预测再入 |
| **Asterank** | ✅ 已接入 | `external_asterank` | 小行星 / NEO 专题库，含开普勒根数、光谱、Δv、估值（`scripts/ingest_asterank.py`） |

## Windows 本地部署（无 Docker）

> 以下步骤适用于 Windows 10/11，在 **PowerShell** 中执行。  
> 全程不需要 Docker，数据库和虚拟环境都位于项目目录内。

> **提示**：自带的 **Windows PowerShell 5.x** 不支持 `&&` 串连命令，请用分号 `;` 分行执行，或安装 [PowerShell 7+](https://github.com/PowerShell/PowerShell)（支持 `&&`）。

### 前置条件

| 软件 | 版本 | 下载地址 |
|------|------|----------|
| **Python** | 3.10 – 3.12 | https://www.python.org/downloads/ （安装时勾选 **Add to PATH**） |
| **PostgreSQL + PostGIS** | PG 15/16 + PostGIS 3.4 | https://www.enterprisedb.com/downloads/postgres-postgresql-downloads （安装时用 Stack Builder 勾选 PostGIS） |
| **Git** | 任意 | https://git-scm.com/download/win |

> PostgreSQL 安装时设置的超级用户密码需要记住（下文假设为 `postgres`）。

### Step 1：克隆仓库

```powershell
git clone https://github.com/<your-org>/debris.git
cd debris
```

> 若你使用自己 fork 的地址，克隆后进入的文件夹名以仓库名为准（本仓库为 `debris`）。

### Step 2：创建 Python 虚拟环境（位于项目目录内）

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> 如果 `pip install psycopg2-binary` 失败，可尝试 `pip install psycopg2` 或从 https://www.lfd.uci.edu/~gohlke/pythonlibs/ 下载对应 wheel。

### Step 3：配置 `.env`

在项目根目录创建 `.env`：将 **`.env.example` 复制为 `.env`** 后再编辑填入密钥（`.env` 已被 Git 忽略，不会提交）。

```powershell
copy .env.example .env
```

`.env` 示例内容：

```env
# Space-Track 账号（免费注册 https://www.space-track.org/auth/createAccount）
SPACETRACK_USERNAME=your@email.com
SPACETRACK_PASSWORD=your_password

# OpenAI / LLM（AI 助手功能需要，不需要可留空）
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini

# 数据库（与 PostgreSQL 安装时设置一致）
DB_HOST=localhost
DB_PORT=5432
DB_NAME=space_debris
DB_USER=postgres
DB_PASSWORD=postgres

# 传播参数
SEGMENT_MINUTES=10
PROPAGATION_HORIZON_DAYS=3

# ESA DISCOS API（可选，用于获取物体质量/截面积数据）
# 在 https://discosweb.esoc.esa.int/ 注册后获取
ESA_DISCOS_TOKEN=
```

### Step 4：创建数据库

打开 **pgAdmin** 或 **psql**（位于 PostgreSQL 安装目录的 `bin` 下；若未加入 PATH，可使用完整路径，如 `& "C:\Program Files\PostgreSQL\16\bin\psql.exe"`，版本号因安装而异）：

```powershell
# 方法一：用 psql（需已登录同一 Windows 用户为 postgres 配置了认证，或设置环境变量 PGPASSWORD）
$env:PGPASSWORD = "postgres"   # 与你在安装 Postgres 时设置的超级用户密码一致
psql -U postgres -h localhost -c "CREATE DATABASE space_debris;"
psql -U postgres -h localhost -d space_debris -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# 方法二：在 pgAdmin 图形界面中
# 右键 Databases → Create → Database → 名称填 space_debris
# 然后在 space_debris 上右键 → Query Tool → 执行：CREATE EXTENSION IF NOT EXISTS postgis;
```

### Step 5：初始化数据库表结构

```powershell
python run.py init-db
```

### Step 6：拉取全量数据（6 个数据源）

数据摄入分两步，可在不同终端窗口中并行执行：

**终端 1 — Space-Track（核心数据，约 30–90 分钟）：**

```powershell
.\venv\Scripts\Activate.ps1
python run.py ingest
```

> 首次运行需要从 Space-Track.org 下载约 29,000 条在轨目标数据，含 SGP4 轨道传播和轨迹段写入。  
> 如想快速验证，可先 `python run.py ingest --limit 1000` 只拉 1000 条。

**终端 2 — 外部数据源（GCAT + UNOOSA + UCS + ESA DISCOS）：**

```powershell
.\venv\Scripts\Activate.ps1
python scripts/ingest_external.py
```

> 此命令依次拉取 4 个外部数据源：
> - **GCAT**（Jonathan McDowell 发射日志）— 从本地 `data/external/jm_satcat.tsv` 导入
> - **UNOOSA**（联合国发射统计）— 从 API 自动下载
> - **UCS**（在轨卫星数据库）— 从本地 `data/external/ucs_satellites.xlsx` 导入
> - **ESA DISCOS**（欧空局物体数据库）— 从 API 获取（需要 `ESA_DISCOS_TOKEN`），约 20 分钟

也可单独运行某个数据源：

```powershell
python scripts/ingest_external.py --ucs       # 仅 UCS
python scripts/ingest_external.py --esa       # 仅 ESA DISCOS
python scripts/ingest_external.py --unoosa    # 仅 UNOOSA
python scripts/ingest_external.py --gcat      # 仅 GCAT
python scripts/ingest_external.py --limit 500 # 每源各限制 500 条（冒烟测试）
```

**终端 3 — Asterank 小行星 / NEO 专题库（几秒内完成）：**

```powershell
.\venv\Scripts\Activate.ps1
python scripts/ingest_asterank.py              # 默认拉取 5000 条
python scripts/ingest_asterank.py --limit 500  # 限制 500 条
```

> Asterank 数据独立于地球在轨目标（不进入 `v_unified_objects`），
> 入库到 `external_asterank` 表，可在目标目录页的「小行星 / NEO (Asterank)」标签查看。

### Step 7：创建统一视图

数据摄入完成后，创建多源数据统一视图：

```powershell
python scripts/create_unified_view.py
```

> 此步骤将 Space-Track、UCS、ESA DISCOS 三库数据去重合并为 `v_unified_objects` 物化视图，  
> 同时通过 COSPAR 编号交叉比对为 ESA 记录推断国家代码，通过名称模式和 ESA mission 推断卫星用途。

### Step 8：启动系统

需要同时启动 **前端**（Streamlit）和 **API/文档服务**（FastAPI），建议使用两个终端：

**终端 A — Streamlit 前端：**

```powershell
.\venv\Scripts\Activate.ps1
python run.py app --port 8501
```

**终端 B — API & 文档服务：**

```powershell
.\venv\Scripts\Activate.ps1
python run.py api --port 8000
```

浏览器访问：
- **http://localhost:8501** — 系统主界面（Streamlit 仪表盘）
- **http://localhost:8000/docs** — 系统说明文档
- **http://localhost:8000/api/docs** — REST API Swagger 交互文档

### 需要补充的文件

以下文件包含敏感信息或大文件，不在 Git 仓库中，需手动准备：

| 文件 | 说明 | 获取方式 |
|------|------|----------|
| `.env` | 配置文件（API 密钥等） | 按 Step 3 模板创建 |
| `data/external/ucs_satellites.xlsx` | UCS 卫星数据库 | 从 https://www.ucsusa.org/nuclear-weapons/space-weapons/satellite-database 下载 |
| `data/external/jm_satcat.tsv` | GCAT 卫星目录 | 从 https://planet4589.org/space/gcat/tsv/cat/satcat.tsv 下载 |
| `data/external/jm_launch.tsv` | GCAT 发射日志 | 从 https://planet4589.org/space/gcat/tsv/launch/launch.tsv 下载 |

> `data/external/` 目录如不存在需手动创建：`mkdir data\external`

### 完整命令汇总（一键复制）

以下按顺序逐段执行即可；若在 **PowerShell 5.x** 下请避免使用 `&&`（可改用分号或分行）。

```powershell
# 1. 克隆 & 进入目录
git clone https://github.com/luoyh21/debris.git
cd debris

# 2. 创建虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3. 配置环境变量
copy .env.example .env
# 用记事本或编辑器编辑 .env，填入 Space-Track / OpenAI 等

# 4. 创建数据库（密码与安装 PostgreSQL 时设置的超级用户一致）
$env:PGPASSWORD = "postgres"
psql -U postgres -h localhost -c "CREATE DATABASE space_debris;"
psql -U postgres -h localhost -d space_debris -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# 5. 初始化表结构
python run.py init-db

# 6. 拉取数据（两个终端并行）
python run.py ingest                        # 终端 1：Space-Track
python scripts/ingest_external.py           # 终端 2：GCAT+UNOOSA+UCS+ESA
python scripts/ingest_asterank.py           # 终端 3：Asterank 小行星/NEO

# 7. 创建统一视图
python scripts/create_unified_view.py

# 8. 启动前端 + API 服务（两个终端）
python run.py app --port 8501               # 终端 A：Streamlit 前端 → http://localhost:8501
python run.py api --port 8000               # 终端 B：API + 文档 → http://localhost:8000/docs
```

### 常见问题

| 问题 | 解决方法 |
|------|----------|
| PowerShell 报 `&&` 不是有效语句分隔符 | 使用 PowerShell 7+，或把命令拆成多行 / 用分号 `;` 分隔 |
| `psycopg2` 安装失败 | 改用 `pip install psycopg2-binary`，或安装 Visual C++ Build Tools |
| `CREATE EXTENSION postgis` 失败 | PostgreSQL 安装时需通过 Stack Builder 额外安装 PostGIS |
| Space-Track 登录失败 | 确认 `.env` 中账号密码正确，需在 space-track.org 注册 |
| `rocketpy` 安装慢或失败 | `pip install rocketpy --no-build-isolation`，需要已安装 numpy |
| 端口 5432 被占用 | 检查是否已有 PostgreSQL 服务在运行，或改 `.env` 中的 `DB_PORT` |
| ESA DISCOS 返回 401 | 在 discosweb.esoc.esa.int 注册获取 Token 填入 `.env` |
| 页面显示"暂无数据" | 检查 Step 6 数据摄入是否完成，Step 7 统一视图是否已创建 |
| 侧边栏"系统说明文档"打不开 | 确认 API 服务已启动（Step 8 终端 B），检查端口 8000 是否可访问 |

---

## 后续扩展方向

- [ ] 接入 SGP4-XP 扩展摄动模型（更高精度，适合 >12h 预报）
- [x] 3D 轨道可视化（Plotly 3D scatter + 颜色渐变时序标注）
- [ ] 实时 CDM 推送告警
- [x] MCP 工具封装（6 个工具：区域搜索 / 发射风险 / 再入预报 / TLE 查询 / RCS 筛选 / 长期风险预测）
- [x] ORDEM 3.1 长期任务碰撞风险评估
- [x] 历年航天发射趋势（GCAT 68,690 条多源数据融合）
- [x] AI Agent 长期预测能力（泊松统计工具 + 规避燃料估算）
- [ ] 协方差矩阵接入（从 Space-Track CDM 获取真实误差椭球）
- [ ] Alfano (2005) 短期遭遇精化方法
- [ ] 定时自动重新摄入（cron / Airflow），保持轨迹段滚动更新
- [x] 接入 UCS / UNOOSA / ESA 补充数据源（UNOOSA 1,274 条 + UCS 7,560 条 + ESA DISCOS 10,000 条）
