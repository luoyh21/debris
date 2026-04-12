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

## 后续扩展方向

- [ ] 接入 SGP4-XP 扩展摄动模型（更高精度，适合 >12h 预报）
- [ ] 3D 轨道可视化（Cesium / deck.gl）
- [ ] 实时 CDM 推送告警
- [x] MCP 工具封装（`query_debris_in_region` + `predict_launch_collision_risk`）
- [ ] 协方差矩阵接入（从 Space-Track CDM 获取真实误差椭球）
- [ ] Alfano (2005) 短期遭遇精化方法
- [ ] 定时自动重新摄入（cron / Airflow），保持轨迹段滚动更新
