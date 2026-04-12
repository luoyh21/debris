"""空间碎片监测仪表盘（中文界面）

启动: streamlit run streamlit_app/app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import threading
import time as _time_mod
from datetime import datetime, timezone, timedelta
from sqlalchemy import text


class _LcolaProgress:
    """Thread-safe progress container for LCOLA background screening.

    Background thread ONLY mutates attributes of this object.
    It never calls st.session_state[key] = ... (which raises
    StreamlitAPIException outside a ScriptRunContext).
    The main thread (and @st.fragment) reads these attributes via
    st.session_state['_lcola_ps'].
    """
    __slots__ = ('step', 'total', 'stop_req', 'done', 'error', 'report', 'start_time')

    def __init__(self, total: int):
        self.step       = 0
        self.total      = total
        self.stop_req   = False
        self.done       = False
        self.error      = None
        self.report     = None
        self.start_time = _time_mod.time()

st.set_page_config(
    page_title="空间碎片监测系统",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 数据库连接（懒加载）
# ------------------------------------------------------------------
@st.cache_resource
def get_db_session_factory():
    try:
        from database.db import get_session_factory
        return get_session_factory()
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Read-only SQL query with a 60-second result cache.

    Results are cached by (sql, params) key for 60 s so that repeated
    page reruns (e.g. sidebar navigation, 30-s fragment fire) do not hit
    the database unnecessarily.  Pass params=None (default) for static
    queries; pass a plain dict for parametrised ones.
    """
    factory = get_db_session_factory()
    if factory is None:
        return pd.DataFrame()
    sess = factory()
    try:
        result = sess.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as exc:
        st.error(f"数据库错误：{exc}")
        return pd.DataFrame()
    finally:
        sess.close()



def _bar_chart_zero(data: "pd.Series | pd.DataFrame") -> None:
    """st.bar_chart replacement that forces the y-axis to start at zero."""
    import altair as alt
    df = data.reset_index()
    cols = df.columns.tolist()
    x_col, y_col = cols[0], cols[1]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x_col, sort=None, title=x_col),
            y=alt.Y(y_col, scale=alt.Scale(zero=True), title=y_col),
        )
        .properties(height=300, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _line_chart_zero(data: "pd.DataFrame") -> None:
    """st.line_chart replacement that forces the y-axis to start at zero."""
    import altair as alt
    df = data.reset_index()
    cols = df.columns.tolist()
    x_col, y_col = cols[0], cols[1]
    chart = (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, scale=alt.Scale(zero=True), title=y_col),
        )
        .properties(height=300, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------------
# 侧边栏导航
# ------------------------------------------------------------------
st.sidebar.title("🛸 空间碎片监测系统")
page = st.sidebar.radio(
    "导航菜单",
    ["🌍 系统概览", "🌐 可视化探索", "📚 目标目录", "🛤️ 轨迹片段", "🚀 轨迹仿真", "📄 OEM 管理", "⚠️ LCOLA 飞越筛选", "☄️ 碰撞风险", "💬 AI 助手"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"UTC 时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.caption("数据来源：Space-Track.org")

# ── 全局后台任务轮询（每 30 秒自动检测完成，触发 toast + 全页刷新）───────────────
@st.fragment(run_every=30)
def _bg_poll_fragment():
    """Detect LCOLA background-thread completion and fire a toast notification.

    This fragment ONLY handles completion detection — it writes no sidebar
    elements (Streamlit forbids st.sidebar inside a fragment).  Sidebar
    progress is rendered in the regular main-thread code below, which runs
    on every full-page rerun that this fragment triggers.
    """
    ps: _LcolaProgress | None = st.session_state.get("_lcola_ps")
    if ps is None or not st.session_state.get("_lcola_running"):
        return
    if not ps.done:
        return

    # Completion path — write to session_state (main-thread safe here)
    st.session_state["_lcola_running"] = False
    if ps.error:
        st.session_state["_lcola_error"] = ps.error
        st.toast(f"❌ LCOLA 计算失败：{ps.error}", icon="❌")
    elif ps.report is not None:
        _rpt = ps.report
        st.session_state["lcola_report"]       = _rpt
        st.session_state["_lcola_n_blackouts"] = len(_rpt.blackout_windows)
        st.session_state["_lcola_n_events"]    = len(_rpt.top_events)
        st.toast(
            f"🛸 LCOLA 飞越筛选完成！  {len(_rpt.blackout_windows)} 个禁发窗口 · "
            f"{len(_rpt.top_events)} 条合取事件",
            icon="✅",
        )
    st.rerun()


_bg_poll_fragment()

# ── 侧边栏后台进度状态（每 30 秒自动刷新，无论用户在哪个页面）─────────────────────
# Calling the fragment *inside* `with st.sidebar:` is the Streamlit-supported
# way to let a fragment write to the sidebar without raising an exception.
with st.sidebar:
    @st.fragment(run_every=30)
    def _sidebar_lcola_fragment():
        if not st.session_state.get("_lcola_running"):
            return
        ps: _LcolaProgress | None = st.session_state.get("_lcola_ps")
        if ps is None:
            return
        if ps.total > 0:
            step    = ps.step
            total   = ps.total
            elapsed = _time_mod.time() - ps.start_time
            eta_str = f" · 剩余≈{elapsed / step * (total - step):.0f}s" if step > 0 else ""
            sb_text = f"⏳ LCOLA 计算中 {step}/{total}{eta_str}"
        else:
            sb_text = "⏳ LCOLA 正在后台计算…"
        st.markdown(
            f'<span style="color:#ffcc00;font-size:12px">{sb_text}</span>',
            unsafe_allow_html=True,
        )

    _sidebar_lcola_fragment()

# ------------------------------------------------------------------
# 页面：系统概览
# ------------------------------------------------------------------
if page == "🌍 系统概览":
    st.title("🌍 空间环境概览")

    col1, col2, col3, col4 = st.columns(4)

    _stats_df = run_query("""
        SELECT
            (SELECT COUNT(*)                                           FROM catalog_objects)                       AS total,
            (SELECT COUNT(*) FROM catalog_objects WHERE object_type = 'DEBRIS')                                   AS debris,
            (SELECT COUNT(*)                                           FROM trajectory_segments)                  AS segs,
            (SELECT COUNT(*) FROM collision_risks  WHERE probability > 1e-6)                                     AS risks
    """)
    if not _stats_df.empty:
        _s = _stats_df.iloc[0]
        col1.metric("在轨目标总数",        int(_s["total"]))
        col2.metric("空间碎片数量",        int(_s["debris"]))
        col3.metric("轨迹片段总数",        int(_s["segs"]))
        col4.metric("高风险事件(Pc>1e-6)", int(_s["risks"]))
    else:
        for c in (col1, col2, col3, col4):
            c.metric("–", "–")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 目标类型分布")
        type_df = run_query("""
            SELECT
                CASE object_type
                    WHEN 'DEBRIS'      THEN '空间碎片'
                    WHEN 'PAYLOAD'     THEN '有效载荷'
                    WHEN 'ROCKET BODY' THEN '火箭箭体'
                    ELSE '未知'
                END AS 类型,
                COUNT(*) AS 数量
            FROM catalog_objects
            GROUP BY object_type
            ORDER BY 数量 DESC
        """)
        if not type_df.empty:
            _bar_chart_zero(type_df.set_index("类型")["数量"])
        else:
            st.info("暂无数据，请先运行：`python3 run.py ingest`")

    with col_right:
        st.subheader("📈 轨道高度分布（近地点，km）")
        alt_df = run_query("""
            SELECT
                FLOOR(perigee_km / 200) * 200 AS 高度区间_km,
                COUNT(*) AS 数量
            FROM catalog_objects
            WHERE perigee_km IS NOT NULL AND perigee_km > 0 AND perigee_km < 40000
            GROUP BY 1
            ORDER BY 1
        """)
        if not alt_df.empty:
            _bar_chart_zero(alt_df.set_index("高度区间_km")["数量"])
        else:
            st.info("暂无高度数据")

    st.subheader("🌐 主要国家在轨目标数量（Top 15）")
    country_df = run_query("""
        SELECT
            country_code AS 国家代码,
            COUNT(*) AS 目标数量,
            SUM(CASE WHEN object_type='DEBRIS' THEN 1 ELSE 0 END) AS 碎片数,
            SUM(CASE WHEN object_type='PAYLOAD' THEN 1 ELSE 0 END) AS 载荷数
        FROM catalog_objects
        WHERE country_code IS NOT NULL
        GROUP BY country_code
        ORDER BY 目标数量 DESC
        LIMIT 15
    """)
    if not country_df.empty:
        st.dataframe(country_df, use_container_width=True)

    st.subheader("📥 数据摄入状态")
    status_df = run_query("""
        SELECT
            source AS 来源,
            DATE(ingested_at) AS 日期,
            COUNT(*) AS 记录数
        FROM gp_elements
        GROUP BY 1, 2
        ORDER BY 2 DESC
        LIMIT 10
    """)
    if not status_df.empty:
        st.dataframe(status_df, use_container_width=True)
    else:
        st.info("暂无 GP 均根数记录")

# ------------------------------------------------------------------
# 页面：可视化探索
# ------------------------------------------------------------------
elif page == "🌐 可视化探索":
    from streamlit_app.viz_explorer import render_viz_explorer
    render_viz_explorer()

# ------------------------------------------------------------------
# 页面：目标目录
# ------------------------------------------------------------------
elif page == "📚 目标目录":
    st.title("📋 NORAD 目标目录")

    with st.expander("🔍 筛选条件", expanded=True):
        col1, col2, col3 = st.columns(3)
        obj_type = col1.selectbox(
            "目标类型",
            ["全部", "DEBRIS（碎片）", "PAYLOAD（载荷）", "ROCKET BODY（箭体）", "UNKNOWN（未知）"]
        )
        country = col2.text_input("国家代码（如 US、CN、RU）", "")
        name_search = col3.text_input("名称关键词", "")

        col4, col5, col6 = st.columns(3)
        perigee_min = col4.number_input("近地点下限 (km)", value=0, step=100)
        perigee_max = col5.number_input("近地点上限 (km)", value=50000, step=500)
        rcs_size = col6.selectbox("RCS 尺寸", ["全部", "SMALL", "MEDIUM", "LARGE"])

    type_map = {
        "全部": None,
        "DEBRIS（碎片）": "DEBRIS",
        "PAYLOAD（载荷）": "PAYLOAD",
        "ROCKET BODY（箭体）": "ROCKET BODY",
        "UNKNOWN（未知）": "UNKNOWN",
    }

    where_clauses = ["(perigee_km IS NULL OR perigee_km BETWEEN :pmin AND :pmax)"]
    params: dict = {"pmin": perigee_min, "pmax": perigee_max}

    otype = type_map[obj_type]
    if otype:
        where_clauses.append("object_type = :otype")
        params["otype"] = otype
    if country.strip():
        where_clauses.append("country_code = :country")
        params["country"] = country.strip().upper()
    if name_search.strip():
        where_clauses.append("name ILIKE :ns")
        params["ns"] = f"%{name_search.strip()}%"
    if rcs_size != "全部":
        where_clauses.append("rcs_size = :rcs")
        params["rcs"] = rcs_size

    where_sql = " AND ".join(where_clauses)
    df = run_query(f"""
        SELECT
            norad_cat_id   AS "NORAD ID",
            name           AS "名称",
            CASE object_type
                WHEN 'DEBRIS'      THEN '碎片'
                WHEN 'PAYLOAD'     THEN '载荷'
                WHEN 'ROCKET BODY' THEN '箭体'
                ELSE object_type
            END            AS "类型",
            country_code   AS "国家",
            launch_date    AS "发射日期",
            ROUND(perigee_km::numeric, 1) AS "近地点(km)",
            ROUND(apogee_km::numeric, 1)  AS "远地点(km)",
            ROUND(inclination::numeric, 2) AS "倾角(°)",
            rcs_size       AS "RCS",
            object_id      AS "国际编号"
        FROM catalog_objects
        WHERE {where_sql}
        ORDER BY norad_cat_id
        LIMIT 2000
    """, params)

    st.write(f"**共 {len(df)} 条记录**（最多显示 2000 条）")
    if not df.empty:
        st.dataframe(df, use_container_width=True, height=550)
    else:
        st.warning("当前筛选条件下无数据，请调整筛选范围或检查数据摄入状态。")

# ------------------------------------------------------------------
# 页面：轨迹片段
# ------------------------------------------------------------------
elif page == "🛤️ 轨迹片段":
    st.title("🛰️ 轨迹片段查询")
    st.caption("由 SGP4 传播器生成，每段为 3D LineStringZ（ECI/大地坐标双存储）")

    col1, col2, col3 = st.columns(3)
    norad_filter  = col1.text_input("NORAD ID（空=全部）", "")
    time_window   = col2.selectbox("时间窗口", ["未来 1 小时", "未来 6 小时", "未来 24 小时", "全部"])
    obj_type_filter = col3.selectbox("目标类型", ["全部", "碎片", "载荷", "箭体"])

    t_now = datetime.now(timezone.utc)
    t_map = {"未来 1 小时": 1, "未来 6 小时": 6, "未来 24 小时": 24, "全部": 24 * 365}
    t_end = t_now + timedelta(hours=t_map[time_window])

    extra_clauses = []
    params2: dict = {"t_now": t_now, "t_end": t_end}

    if norad_filter.strip():
        extra_clauses.append("ts.norad_cat_id = :nid")
        params2["nid"] = int(norad_filter.strip())

    type_sql_map = {"碎片": "DEBRIS", "载荷": "PAYLOAD", "箭体": "ROCKET BODY"}
    if obj_type_filter != "全部":
        extra_clauses.append("co.object_type = :otype")
        params2["otype"] = type_sql_map[obj_type_filter]

    extra_where = ("AND " + " AND ".join(extra_clauses)) if extra_clauses else ""

    seg_df = run_query(f"""
        SELECT
            ts.norad_cat_id            AS "NORAD ID",
            co.name                    AS "名称",
            CASE co.object_type
                WHEN 'DEBRIS'      THEN '碎片'
                WHEN 'PAYLOAD'     THEN '载荷'
                WHEN 'ROCKET BODY' THEN '箭体'
                ELSE co.object_type END AS "类型",
            to_char(ts.t_start AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI') AS "开始时间(UTC)",
            to_char(ts.t_end   AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI') AS "结束时间(UTC)",
            ROUND(ST_Length(ts.geom_eci)::numeric, 1)                      AS "ECI弧长(km)"
        FROM trajectory_segments ts
        JOIN catalog_objects co ON co.norad_cat_id = ts.norad_cat_id
        WHERE ts.t_start <= :t_end AND ts.t_end >= :t_now
        {extra_where}
        ORDER BY ts.t_start
        LIMIT 1000
    """, params2)

    st.write(f"**找到 {len(seg_df)} 条片段**")
    if not seg_df.empty:
        st.dataframe(seg_df, use_container_width=True, height=550)
    else:
        st.info("当前时间窗口内无匹配片段")

# ------------------------------------------------------------------
# 页面：碰撞风险（统一使用 6-DOF 仿真轨迹 + Foster Pc）
# ------------------------------------------------------------------
elif page == "☄️ 碰撞风险":
    st.title("🎯 发射各阶段碰撞风险评估")
    st.caption(
        "轨迹来源：6-DOF 数值积分（trajectory/six_dof.py）｜"
        "算法：Foster (1992) 2-D Pc 数值积分｜"
        "碎片来源：Space-Track GP 目录 + SGP4 传播"
    )

    # ── 功能说明卡片 ────────────────────────────────────────────────────────
    st.info(
        "**本页面解决的问题：对于一个已确定的发射时刻，各飞行阶段各有多高的碰撞风险？**\n\n"
        "固定发射时刻后，将完整飞行轨迹划分为上升段、停泊轨道段等阶段，"
        "逐阶段与在轨碎片进行 TCA 求解 + Foster Pc 计算，输出每个阶段的风险等级（RED/AMBER/YELLOW/GREEN）"
        "及详细合取事件表。\n\n"
        "📌 **与「⚠️ LCOLA 飞越筛选」的区别**：LCOLA 页面扫描一段时间窗口内的多个候选发射时刻，"
        "输出禁射窗口（回答『什么时候发射安全』）；"
        "本页面针对单一固定发射时刻，给出各阶段的具体 Pc 数值和风险等级"
        "（回答『选定时刻的碰撞风险有多高，哪个阶段最危险』）。"
    )

    # ── 依赖检查 ─────────────────────────────────────────────────────────────
    if "sim_result" not in st.session_state or "sim_phases" not in st.session_state:
        st.info(
            "请先在 **🚀 轨迹仿真** 页面运行仿真，生成轨迹数据后再来此页面评估碰撞风险。"
        )
        st.stop()

    result = st.session_state["sim_result"]
    phases = st.session_state["sim_phases"]

    # ── 配置区 ────────────────────────────────────────────────────────────────
    with st.expander("⚙️ 评估参数", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        hbr_m      = c1.number_input("联合硬体半径 HBR (m)", value=20, step=5,
                                     help="火箭 + 碎片外接球半径之和，默认 20 m")
        hbr_km     = hbr_m / 1000.0
        crewed     = c2.checkbox("载人任务（Pc 门限 1e-6）", value=False)
        fine_km    = c3.number_input("精细筛选距离 (km)", value=50, step=10)
        coarse_km  = c4.number_input("粗筛距离 (km)", value=200, step=50)
        inject_demo = st.checkbox(
            "🧪 注入演示威胁（Demo Threats）",
            value=True,
            help="注入三个标注为 🧪 DEMO 的合成合取事件，用于展示系统功能。"
                 "真实碎片数据库较小时推荐开启。",
        )

    t0_utc = result.config.launch_utc
    pc_thresh = 1e-6 if crewed else 1e-5

    # ── 任务摘要 ─────────────────────────────────────────────────────────────
    st.markdown(
        f"**运载火箭：** {result.config.vehicle_name}　"
        f"**发射场：** {result.config.launch_lat_deg:.2f}°N "
        f"{result.config.launch_lon_deg:.2f}°E　"
        f"**T0：** {t0_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC　"
        f"**Pc 门限：** {pc_thresh:.0e}"
    )

    phase_names_cn = {
        "ASCENT":          "上升段",
        "PARKING_ORBIT":   "停泊轨道段",
        "TRANSFER_BURN":   "变轨推进段",
        "POST_SEPARATION": "分离后脱轨段",
    }

    if st.button("▶ 开始碰撞风险评估", type="primary"):
        from lcola.fly_through import assess_launch_phases

        progress_bar = st.progress(0, text="准备评估…")
        status_text  = st.empty()

        def _cb(phase_name, i, n):
            frac = (phases.index(next(p for p in phases if p.name == phase_name))
                    / len(phases)) + (i / max(n, 1)) / len(phases)
            frac = min(frac, 0.99)
            cn = phase_names_cn.get(phase_name, phase_name)
            progress_bar.progress(frac, text=f"评估 {cn}… ({i}/{n})")
            status_text.caption(f"空间预筛选候选目标：{n} 个　已处理：{i} 个")

        try:
            summaries = assess_launch_phases(
                phases=phases,
                launch_time=t0_utc,
                hbr_km=hbr_km,
                pc_threshold=0.0,
                coarse_km=float(coarse_km),
                fine_km=float(fine_km),
                inject_demo=inject_demo,
                progress_cb=_cb,
            )
            st.session_state["risk_summaries"] = summaries
            progress_bar.progress(1.0, text="评估完成")
            status_text.empty()

            total_events = sum(len(s.events) for s in summaries)
            st.success(
                f"评估完成！{len(summaries)} 个阶段，"
                f"发现 {total_events} 条合取事件"
            )
        except Exception as exc:
            progress_bar.empty()
            st.error(f"评估失败：{exc}")
            st.exception(exc)

    # ── 结果展示 ─────────────────────────────────────────────────────────────
    if "risk_summaries" in st.session_state:
        import pandas as pd
        summaries = st.session_state["risk_summaries"]

        risk_icon = {"RED": "🔴", "AMBER": "🟠", "YELLOW": "🟡", "GREEN": "🟢"}

        # ── 阶段摘要总览 ──────────────────────────────────────────────────────
        st.subheader("📊 各飞行阶段风险摘要")
        summary_rows = []
        for s in summaries:
            summary_rows.append({
                "阶段":             phase_names_cn.get(s.phase_name, s.phase_name),
                "MET 时段 (s)":    f"{s.t_start_met:.0f} – {s.t_end_met:.0f}",
                "UTC 开始":        s.t_start_utc.strftime("%H:%M:%S"),
                "UTC 结束":        s.t_end_utc.strftime("%H:%M:%S"),
                "候选碎片数":       s.n_candidates,
                "有效评估数":       s.n_evaluated,
                "合取事件数":       len(s.events),
                "最大 Pc":          f"{s.max_pc:.3e}" if s.max_pc > 0 else "–",
                "风险等级":         risk_icon.get(s.risk_level, "") + " " + s.risk_level,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # ── Pc 柱状图（各阶段最大 Pc，对数坐标） ────────────────────────────────
        st.subheader("各阶段最大碰撞概率 (Pc)")
        try:
            import plotly.graph_objects as go

            PC_FLOOR = 1e-12   # bars with Pc=0 land here on log scale

            bar_colors = {
                "RED":    "#e74c3c",
                "AMBER":  "#e67e22",
                "YELLOW": "#f1c40f",
                "GREEN":  "#2ecc71",
            }
            phases_labels = [phase_names_cn.get(s.phase_name, s.phase_name)
                             for s in summaries]
            pc_vals   = [max(s.max_pc, PC_FLOOR) for s in summaries]
            colors    = [bar_colors.get(s.risk_level, "#95a5a6") for s in summaries]
            hover_txt = [
                f"{phase_names_cn.get(s.phase_name, s.phase_name)}<br>"
                f"max Pc = {s.max_pc:.3e}<br>风险等级 = {s.risk_level}"
                for s in summaries
            ]

            fig_pc = go.Figure(go.Bar(
                x=phases_labels,
                y=pc_vals,
                marker_color=colors,
                hovertext=hover_txt,
                hoverinfo="text",
            ))
            fig_pc.add_hline(y=1e-5, line_dash="dash", line_color="#e74c3c",
                             annotation_text="Pc_uncrewed 1e-5", annotation_position="top right")
            fig_pc.add_hline(y=1e-6, line_dash="dot",  line_color="#e67e22",
                             annotation_text="Pc_crewed 1e-6",   annotation_position="top right")
            fig_pc.update_layout(
                yaxis_type="log",
                yaxis_range=[-12, 0],
                yaxis_title="最大碰撞概率 Pc（对数坐标）",
                xaxis_title="飞行阶段",
                height=360,
                margin=dict(t=30, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_pc, use_container_width=True)
            st.caption("虚线：1e-5（无人载荷门限）；点线：1e-6（载人任务门限）；零值显示在底部（1e-12）")

        except ImportError:
            # fallback to native chart if plotly missing
            import pandas as pd
            pc_chart = pd.DataFrame({
                "阶段": [phase_names_cn.get(s.phase_name, s.phase_name) for s in summaries],
                "最大 Pc": [s.max_pc if s.max_pc > 0 else 1e-15 for s in summaries],
            }).set_index("阶段")
            st.bar_chart(pc_chart)  # Pc values always ≥ 0; native chart OK for fallback
            st.caption(f"Pc 门限参考：载人 1e-6，普通载荷 1e-5")

        # ── 分阶段详细事件 ────────────────────────────────────────────────────
        st.subheader("🔍 各阶段高风险碎片详情")

        all_events_rows = []   # collect all for global table

        for s in summaries:
            phase_cn = phase_names_cn.get(s.phase_name, s.phase_name)
            icon      = risk_icon.get(s.risk_level, "")
            with st.expander(
                f"{icon} {phase_cn}  │  合取事件 {len(s.events)} 条  │  "
                f"max Pc = {s.max_pc:.3e}  │  {s.risk_level}",
                expanded=(s.risk_level in ("RED", "AMBER") and len(s.events) > 0),
            ):
                if not s.events:
                    st.info("该阶段无合取事件（距离 > 精细筛选阈值）")
                    continue

                # Phase risk profile
                if s.risk_text:
                    st.caption(f"📝 {s.risk_text}")

                rows = []
                for ev in s.events:
                    rows.append({
                        "风险":             risk_icon.get(ev.risk_level, ""),
                        "NORAD ID":         ev.norad_cat_id,
                        "目标名称":         ev.object_name,
                        "TCA (UTC)":        ev.tca.strftime("%m-%d %H:%M:%S"),
                        "最近距离 (km)":    round(ev.miss_distance_km, 3),
                        "Foster Pc":        f"{ev.probability:.4e}",
                        "Pc 误差":          f"{ev.pc_error:.1e}" if not (ev.pc_error != ev.pc_error) else "–",
                        "相对速度 (km/s)":  round(ev.v_rel_kms, 3),
                        "禁发标记":         "⛔" if ev.is_blackout else "✓",
                    })
                    all_events_rows.append({
                        "阶段":             phase_cn,
                        "风险":             risk_icon.get(ev.risk_level, "") + ev.risk_level,
                        "NORAD ID":         ev.norad_cat_id,
                        "目标名称":         ev.object_name,
                        "TCA (UTC)":        ev.tca.strftime("%m-%d %H:%M:%S"),
                        "最近距离 (km)":    round(ev.miss_distance_km, 3),
                        "Foster Pc":        f"{ev.probability:.4e}",
                        "相对速度 (km/s)":  round(ev.v_rel_kms, 3),
                    })

                df_ev = pd.DataFrame(rows)
                st.dataframe(df_ev, use_container_width=True, height=min(400, 45 + 35*len(rows)))

        # ── 全局汇总表（所有阶段合并，按 Pc 降序） ────────────────────────────
        if all_events_rows:
            st.subheader("📋 全部合取事件汇总（按 Pc 降序）")
            df_all = pd.DataFrame(all_events_rows)
            st.dataframe(df_all, use_container_width=True, height=500)

            # 禁发事件单独列出
            blackout_events = [
                ev for s in summaries for ev in s.events if ev.is_blackout
            ]
            if blackout_events:
                st.error(
                    f"⛔ 发现 {len(blackout_events)} 个**禁发**合取事件"
                    f"（Pc ≥ {pc_thresh:.0e} 或 Miss Distance < 25 km）"
                )
                bo_rows = [{
                    "阶段":            phase_names_cn.get(ev.phase, ev.phase),
                    "NORAD ID":        ev.norad_cat_id,
                    "目标名称":        ev.object_name,
                    "TCA (UTC)":       ev.tca.strftime("%m-%d %H:%M:%S"),
                    "最近距离 (km)":   round(ev.miss_distance_km, 3),
                    "Foster Pc":       f"{ev.probability:.4e}",
                } for ev in sorted(blackout_events,
                                   key=lambda e: e.probability, reverse=True)]
                st.dataframe(pd.DataFrame(bo_rows), use_container_width=True)
            else:
                st.success("✅ 当前发射方案无禁发合取事件")

# ------------------------------------------------------------------
# 页面：AI 助手
# ------------------------------------------------------------------
elif page == "💬 AI 助手":
    st.title("💬 AI 碎片分析助手")
    st.caption(
        "可用自然语言询问碎片数据库或调用 MCP 工具，例如：\n"
        "- 「低地球轨道（200~2000km）有多少碎片？按目标类型和国家统计前5名」\n"
        "- 「搜索文昌上空 500km 内、高度 200~2000km 的碎片，列出近地点最低的前10个」\n"
        "- 「用长征五号B从文昌发射，方位角90°，预测明天发射的各阶段碰撞风险」"
    )

    # ── MCP 工具文档面板 ──────────────────────────────────────────────
    with st.expander("🔧 可用 MCP 工具（5 个）", expanded=False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
**🛰️ query_debris_in_region**

在指定地理区域和高度范围内检索在轨空间目标。

| 参数 | 说明 | 默认 |
|------|------|------|
| `lat_deg` | 中心纬度（°） | — |
| `lon_deg` | 中心经度（°） | — |
| `radius_km` | 搜索半径（km） | 500 |
| `alt_min_km` | 最低轨道高度（km） | 0 |
| `alt_max_km` | 最高轨道高度（km） | 2000 |
| `object_type` | DEBRIS/PAYLOAD/ROCKET BODY/ALL | ALL |
| `t_start_utc` | 时间窗口起点（ISO-8601） | 当前时刻 |
| `hours` | 时间窗口长度（小时） | 6 |
| `limit` | 最多返回目标数 | 50 |

**示例：**
> 「搜索文昌（19.61°N, 110.95°E）上空500km内、高度200~2000km的所有碎片」

---

**🌍 get_debris_reentry_forecast**

预报即将再入大气层的空间目标。查询 `catalog_objects` 中已有 `decay_date` 或近地点过低的目标。

| 参数 | 说明 | 默认 |
|------|------|------|
| `days_ahead` | 预报窗口（天） | 30 |
| `alt_max_km` | 无确认再入日期时近地点阈值（km） | 300 |
| `object_type` | DEBRIS/PAYLOAD/ROCKET BODY/ALL | ALL |
| `limit` | 最多返回目标数 | 50 |

**返回：** NORAD ID、名称、类型、国家、确认再入日期、距今天数、轨道高度

**示例：**
> 「未来30天有哪些碎片预计再入大气层？」
> 「近地点低于200km的待衰减碎片有哪些？」

---

**📡 get_object_tle**

获取指定 NORAD 编号目标的最新 TLE 轨道根数，可用于外部 SGP4 传播计算。

| 参数 | 说明 | 默认 |
|------|------|------|
| `norad_cat_id` | NORAD 目标编号（必填） | — |

**返回：** TLE Line1/Line2、轨道历元、六根数（倾角/偏心率/平均运动/升交点赤经/近地点幅角/平近点角/B*阻力系数）

**示例：**
> 「给我 ISS（NORAD 25544）的 TLE 轨道根数」
> 「获取 NORAD 12345 的最新轨道数据」
""")
        with col_t2:
            st.markdown("""
**🚀 predict_launch_collision_risk**

对指定发射任务进行 6-DOF 仿真 + Foster Pc 碰撞风险评估。

| 参数 | 说明 | 默认 |
|------|------|------|
| `vehicle` | 运载火箭（CZ-5B/Falcon9/Ariane6） | CZ-5B |
| `launch_lat_deg` | 发射场纬度（°） | 19.61（文昌） |
| `launch_lon_deg` | 发射场经度（°） | 110.95（文昌） |
| `launch_az_deg` | 发射方位角（°，0=北，90=东） | 90（正东） |
| `launch_utc` | 发射时刻（ISO-8601 UTC） | 明日06:00 |
| `t_max_s` | 仿真时长（秒，600~7200） | 3600 |
| `include_demo_threats` | 是否注入演示威胁 | true |

**返回：** 各阶段风险等级（🔴🟠🟡🟢）、最高 Pc 合取事件列表、中文建议

**示例：**
> 「用长征五号B从文昌发射，方位角90°，预测明天06:00 UTC的碰撞风险」

---

**🔍 query_debris_by_rcs**

按雷达截面积（RCS）大小类别筛选空间目标，用于威胁辨别与目录完整性分析。

| 参数 | 说明 | 默认 |
|------|------|------|
| `rcs_sizes` | 类别列表：SMALL/MEDIUM/LARGE | 全部 |
| `alt_min_km` | 最低轨道高度（km） | 0 |
| `alt_max_km` | 最高轨道高度（km） | 2000 |
| `object_type` | DEBRIS/PAYLOAD/ROCKET BODY/ALL | ALL |
| `limit` | 最多返回目标数 | 50 |

RCS 分级：
- **SMALL** — < 0.1 m²，难以跟踪，位置不确定性大
- **MEDIUM** — 0.1–1 m²
- **LARGE** — > 1 m²，最易跟踪，碎裂后产碎片量最大

**示例：**
> 「LEO（200~2000km）中有多少 LARGE 级碎片？」
> 「筛选 SMALL 级目标评估载人任务避撞窗口」
""")

    # ── 示例问题快捷按钮 ─────────────────────────────────────────────
    st.markdown("##### 💡 示例问题")
    example_cols = st.columns(3)
    _examples = [
        ("🛰️ 区域碎片查询",
         "搜索文昌发射场（19.61°N, 110.95°E）上空500km范围内、高度200~2000km的碎片，列出前20个"),
        ("🚀 发射风险预测",
         "用长征五号B从文昌（纬度19.61°，经度110.95°）向正东方向发射，"
         "预测明天06:00 UTC发射的各阶段碰撞风险，给出风险等级和建议"),
        ("📊 轨道带分布",
         "统计低地球轨道（perigee_km 200~2000km）内各类型目标数量，"
         "按 object_type 分组并列出代表性目标名称各3个"),
        ("🌍 再入预报",
         "预测未来30天内有哪些空间目标即将再入大气层，列出确认再入日期和近地点高度"),
        ("📡 获取 TLE",
         "获取国际空间站（ISS，NORAD 25544）的最新TLE轨道根数，列出六根数"),
        ("🔍 大型碎片统计",
         "统计LEO（200~2000km）中LARGE级别的碎片数量，列出近地点最低的前10个"),
    ]
    for i, (label, question) in enumerate(_examples):
        col = example_cols[i % 3]
        with col:
            if st.button(label, use_container_width=True, key=f"example_{i}"):
                st.session_state["_prefill_question"] = question
                st.rerun()

    # 处理示例按钮预填充
    prefill = st.session_state.pop("_prefill_question", None)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史对话
    for msg in st.session_state.chat_history:
        role_label = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role_label):
            st.markdown(msg["content"])

    # 清除按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清除对话历史"):
            st.session_state.chat_history = []
            st.rerun()

    # 输入框
    user_input = st.chat_input("请输入您的问题…") or prefill
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                try:
                    from agent.debris_agent import chat
                    history = st.session_state.chat_history[:-1][-10:]
                    reply = chat(history, user_input)
                except Exception as exc:
                    reply = f"调用出错：{exc}"
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ------------------------------------------------------------------
# 页面：轨迹仿真（6-DOF）
# ------------------------------------------------------------------
elif page == "🚀 轨迹仿真":
    st.title("🚀 火箭轨迹仿真（6-DOF 数值积分）")
    st.caption(
        "基于 ECEF 坐标系 6 自由度 ODE（J2 引力 + 大气阻力 + 推力矢量），"
        "重力转弯俯仰程序，蒙特卡洛协方差估计。"
    )

    with st.form("sim_form"):
        col1, col2, col3 = st.columns(3)
        vehicle_name = col1.selectbox("运载火箭", ["CZ-5B", "Falcon9", "Ariane6"],
                                      help="CZ-5B=长征五号B  Falcon9=猎鹰九  Ariane6=阿丽亚娜6")
        launch_utc_str = col2.text_input("发射时间（UTC）", "2026-04-15T06:00:00")
        azimuth = col3.number_input("发射方位角（°）", value=90.0, step=1.0,
                                    help="0=正北 90=正东")
        col4, col5, col6 = st.columns(3)
        lat = col4.number_input("发射场纬度（°）", value=19.61, format="%.4f")
        lon = col5.number_input("发射场经度（°）", value=110.95, format="%.4f")
        t_max = col6.number_input("仿真时长（s）", value=3600, step=300)
        col7, col8, col9 = st.columns(3)
        dt_out  = col7.number_input("输出步长（s）", value=10, step=5)
        run_mc  = col8.checkbox("运行蒙特卡洛协方差（~50次）", value=True)
        mc_runs = col9.number_input("MC 次数", value=30, step=10, disabled=not run_mc)
        submitted = st.form_submit_button("▶ 运行仿真", type="primary")

    if submitted:
        from datetime import timezone as _tz
        try:
            launch_utc = datetime.fromisoformat(launch_utc_str).replace(tzinfo=_tz.utc)
        except ValueError as e:
            st.error(f"发射时间格式错误: {e}")
            st.stop()

        with st.spinner("6-DOF 积分中…（首次约 5–20 秒）"):
            try:
                from trajectory.rocketpy_sim import SimConfig, simulate
                from trajectory.launch_phases import detect_phases
                import numpy as np

                cfg = SimConfig(
                    vehicle_name=vehicle_name,
                    launch_lat_deg=lat, launch_lon_deg=lon,
                    launch_alt_km=0.04, launch_az_deg=azimuth,
                    launch_utc=launch_utc,
                    t_max_s=float(t_max), dt_out_s=float(dt_out),
                    run_mc=run_mc, mc_n_runs=int(mc_runs),
                )
                result = simulate(cfg)
                phases = detect_phases(
                    result.nominal,
                    t_meco1=result.t_meco1, t_stage_sep=result.t_stage_sep,
                    t_meco2=result.t_meco2, t_payload_sep=result.t_payload_sep,
                )
                st.session_state["sim_result"] = result
                st.session_state["sim_phases"] = phases
                st.success(f"仿真完成：{len(result.nominal)} 个输出点，{len(phases)} 个阶段")
            except Exception as exc:
                st.error(f"仿真失败：{exc}")
                st.exception(exc)

    if "sim_result" in st.session_state:
        result = st.session_state["sim_result"]
        phases = st.session_state["sim_phases"]
        nom    = result.nominal
        import numpy as np
        import pandas as pd

        # 飞行阶段摘要
        st.subheader("飞行阶段摘要")
        phase_rows = []
        for ph in phases:
            a0, a1 = ph.alt_range_km
            phase_rows.append({
                "阶段": ph.name, "MET起(s)": ph.t_start_met, "MET止(s)": ph.t_end_met,
                "近地点(km)": round(a0, 0), "远地点(km)": round(a1, 0),
                "平均速度(km/s)": round(ph.mean_speed_kms, 3),
                "风险特征": ph.risk_profile,
            })
        st.dataframe(pd.DataFrame(phase_rows), use_container_width=True)

        # 高度–速度曲线
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("高度随时间变化 (km)")
            df_alt = pd.DataFrame({
                "MET (s)": [p.t_met_s for p in nom],
                "高度 (km)": [p.alt_km for p in nom],
            }).set_index("MET (s)")
            _line_chart_zero(df_alt)

        with col_r:
            st.subheader("速度大小随时间变化 (km/s)")
            df_vel = pd.DataFrame({
                "MET (s)": [p.t_met_s for p in nom],
                "速度 (km/s)": [float(np.linalg.norm(p.vel_eci)) for p in nom],
            }).set_index("MET (s)")
            _line_chart_zero(df_vel)

        # 质量消耗
        st.subheader("质量随时间变化 (kg)")
        df_mass = pd.DataFrame({
            "MET (s)": [p.t_met_s for p in nom],
            "质量 (kg)": [p.mass_kg for p in nom],
        }).set_index("MET (s)")
        _line_chart_zero(df_mass)

        # 位置轨迹表
        with st.expander("状态向量数据表（前100行）"):
            rows = [{"MET(s)": p.t_met_s,
                     "alt(km)": round(p.alt_km,2),
                     "lat(°)": round(p.lat_deg,3),
                     "lon(°)": round(p.lon_deg,3),
                     "vx(km/s)": round(p.vel_eci[0],4),
                     "vy(km/s)": round(p.vel_eci[1],4),
                     "vz(km/s)": round(p.vel_eci[2],4),
                     "mass(kg)": round(p.mass_kg,0),
                     } for p in nom[:100]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # 协方差可视化
        if result.covariances is not None:
            st.subheader("位置不确定性（MC 协方差 1-σ, km）")
            mc_times = np.arange(0, result.config.t_max_s, result.mc_dt_s)
            n = min(len(mc_times), len(result.covariances))
            sigma_pos = [float(np.sqrt(np.trace(result.covariances[i,:3,:3])/3))
                         for i in range(n)]
            df_cov = pd.DataFrame({
                "MET (s)": mc_times[:n],
                "位置1σ (km)": sigma_pos,
            }).set_index("MET (s)")
            _line_chart_zero(df_cov)

# ------------------------------------------------------------------
# 页面：OEM 管理
# ------------------------------------------------------------------
elif page == "📄 OEM 管理":
    st.title("📄 CCSDS OEM 轨道星历管理")
    st.caption(
        "符合 CCSDS 502.0-B-3 的 ASCII/KVN 格式。包含位置+速度状态向量及可选协方差块。"
    )

    tab_gen, tab_view = st.tabs(["生成 OEM 文件", "查看 / 解析 OEM"])

    with tab_gen:
        st.subheader("从仿真结果生成 OEM")
        if "sim_result" not in st.session_state:
            st.info("请先在「轨迹仿真」页面运行仿真。")
        else:
            result = st.session_state["sim_result"]
            phases = st.session_state["sim_phases"]
            mission_id = st.text_input("任务编号", "2026-001")
            oem_path   = st.text_input("输出文件路径", "/tmp/mission.oem")

            if st.button("📥 生成 OEM 文件"):
                try:
                    from trajectory.oem_io import sim_result_to_oem_segments, write_oem
                    segs = sim_result_to_oem_segments(result, phases, mission_id=mission_id)
                    write_oem(oem_path, segs)
                    st.success(f"OEM 文件已写入：{oem_path}，共 {len(segs)} 段")

                    # Show preview
                    with open(oem_path, "r") as fh:
                        content = fh.read()
                    st.code(content[:3000] + ("…（截断）" if len(content) > 3000 else ""), language="text")
                except Exception as exc:
                    st.error(f"生成失败：{exc}")

    with tab_view:
        st.subheader("解析现有 OEM 文件")
        oem_input_path = st.text_input("OEM 文件路径", "/tmp/mission.oem", key="oem_view_path")
        if st.button("📂 解析"):
            try:
                from trajectory.oem_io import read_oem
                import pandas as pd
                segs = read_oem(oem_input_path)
                st.success(f"解析成功：{len(segs)} 个 META 段")
                for i, seg in enumerate(segs):
                    with st.expander(f"段 {i+1}: {seg.object_name}  ({len(seg.states)} 条状态)"):
                        rows = [{
                            "UTC": s.epoch.strftime("%Y-%m-%dT%H:%M:%S"),
                            "X(km)": round(s.pos_km[0],3),
                            "Y(km)": round(s.pos_km[1],3),
                            "Z(km)": round(s.pos_km[2],3),
                            "VX(km/s)": round(s.vel_kms[0],6),
                            "VY(km/s)": round(s.vel_kms[1],6),
                            "VZ(km/s)": round(s.vel_kms[2],6),
                            "有协方差": "✓" if s.cov_6x6 is not None else "–",
                        } for s in seg.states]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as exc:
                st.error(f"解析失败：{exc}")

# ------------------------------------------------------------------
# 页面：LCOLA 飞越筛选
# ------------------------------------------------------------------
elif page == "⚠️ LCOLA 飞越筛选":
    st.title("⚠️ 发射碰撞规避（LCOLA）飞越窗口扫描")

    # ── 功能说明卡片 ────────────────────────────────────────────────────────
    st.info(
        "**本页面解决的问题：在 ±N 分钟的发射窗口内，哪些时刻可以安全发射？**\n\n"
        "通过扫描多个候选发射时刻（每隔 step 秒取一个），对每个时刻运行完整的 "
        "PostGIS 空间预筛 → SGP4 传播 → TCA 求解 → Foster Pc 数值积分流程，"
        "最终输出 **禁射窗口（Blackout）** 和 **安全窗口（Safe）**，帮助任务规划员选择最优发射时刻。\n\n"
        "📌 **与「☄️ 碰撞风险」的区别**：碰撞风险页面针对 *单一固定发射时刻*，"
        "逐飞行阶段给出 Pc 和风险等级（回答『这个时刻安不安全』）；"
        "本页面针对 *一段时间窗口内的多个候选时刻*，找出哪些时刻触发禁射条件"
        "（回答『窗口内什么时候可以发射』）。两者互补，不可替代。"
    )

    # ── 配置面板 ────────────────────────────────────────────────────────────
    with st.expander("⚙️ 筛选配置", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        crewed = col1.checkbox("载人任务（Pc 门限 1e-6）", value=False)
        hbr    = col2.number_input("联合硬体半径 HBR (m)", value=20, step=5) / 1000.0
        win_m  = col3.number_input("窗口宽度（分钟，各方向）", value=30, step=5,
                                   help="以标称发射时刻为中心，向前后各扩展 N 分钟")
        step_s = col4.number_input("筛选步长（秒）", value=60, step=30,
                                   help="相邻两个候选发射时刻之间的间隔。步长越小精度越高但耗时越长")

    # ── 耗时估算 ────────────────────────────────────────────────────────────
    n_steps_est = int(win_m * 2 * 60 / max(step_s, 1)) + 1
    st.caption(
        f"⏱ 本次将评估 **{n_steps_est} 个**候选发射时刻。"
        f"  预计耗时：碎片数据库为空时约 **{max(1, n_steps_est//30)}-{max(2, n_steps_est//10)} 秒**；"
        f"碎片数据库已填充时约 **{max(1, n_steps_est//4)}-{max(2, n_steps_est)} 秒**。"
        f"  计算在后台进行，可切换到其他页面，完成后右上角弹窗通知。",
    )

    if "sim_result" not in st.session_state:
        st.info("请先在「轨迹仿真」页面运行仿真并生成轨迹数据。")
        st.stop()

    result = st.session_state["sim_result"]
    phases = st.session_state.get("sim_phases", [])
    pc_thresh = 1e-6 if crewed else 1e-5

    st.markdown(
        f"**当前轨迹：** {result.config.vehicle_name}  |  "
        f"T0 = {result.config.launch_utc.strftime('%Y-%m-%dT%H:%M:%S')} UTC  |  "
        f"Pc 门限 = {pc_thresh:.0e}"
    )

    # ── 发射按钮 ────────────────────────────────────────────────────────────
    col_btn, col_stop = st.columns([2, 1])
    run_clicked  = col_btn.button(
        "▶ 开始飞越筛选（后台运行）", type="primary",
        disabled=bool(st.session_state.get("_lcola_running")),
    )
    stop_clicked = col_stop.button(
        "⏹ 取消", disabled=not bool(st.session_state.get("_lcola_running"))
    )

    if stop_clicked:
        _ps = st.session_state.get("_lcola_ps")
        if _ps:
            _ps.stop_req = True       # thread reads this; no st.session_state write
        st.session_state["_lcola_running"] = False
        st.warning("已请求取消，当前步骤执行完后停止。")

    if run_clicked and not st.session_state.get("_lcola_running"):
        from lcola.fly_through import FlyThroughScreener
        from trajectory.oem_io import sim_result_to_oem_segments

        t0      = result.config.launch_utc
        w_open  = t0 - timedelta(minutes=float(win_m))
        w_close = t0 + timedelta(minutes=float(win_m))
        segs    = sim_result_to_oem_segments(result, phases)
        screener = FlyThroughScreener(
            oem_segments=segs,
            phases=phases,
            mission_name=result.config.vehicle_name,
            pc_threshold=pc_thresh,
            hbr_km=hbr,
        )

        # ── create progress object (main-thread); thread mutates its attributes ─
        ps = _LcolaProgress(n_steps_est)
        st.session_state["_lcola_ps"]      = ps
        st.session_state["_lcola_running"] = True
        st.session_state.pop("_lcola_just_done", None)
        st.session_state.pop("lcola_report",     None)
        st.session_state.pop("_lcola_error",     None)

        def _bg_screen():
            # NEVER write to st.session_state here — silently fails without ScriptRunContext.
            # Only mutate attributes of `ps` (captured from outer scope).
            def _cb(step, total):
                ps.step  = step
                ps.total = total
                if ps.stop_req:
                    raise InterruptedError("用户取消")
            try:
                report   = screener.screen(
                    w_open, w_close, t0,
                    step_s=float(step_s),
                    progress_cb=_cb,
                )
                ps.report = report
            except InterruptedError:
                ps.error = "用户已取消筛选。"
            except Exception as exc:
                ps.error = str(exc)
            finally:
                ps.done = True   # main thread (fragment or global check) reads this

        threading.Thread(target=_bg_screen, daemon=True).start()
        st.rerun()

    # ── 后台运行中：显示进度（非阻塞 fragment，仅在计算进行时注册 1s 刷新）────────
    # Only register the run_every fragment when computation is actually active.
    # When idle, there is no need for a 1-second timer — doing so would cause
    # 60 partial reruns per minute even while nothing is happening.
    if st.session_state.get("_lcola_running"):
        @st.fragment(run_every="1s")
        def _lcola_progress_fragment():
            ps: _LcolaProgress | None = st.session_state.get("_lcola_ps")
            if ps is None:
                return

            # ── task completed ────────────────────────────────────────────
            if ps.done:
                st.session_state["_lcola_running"] = False
                if ps.error:
                    st.session_state["_lcola_error"] = ps.error
                elif ps.report is not None:
                    _rpt = ps.report
                    st.session_state["lcola_report"]       = _rpt
                    st.session_state["_lcola_n_blackouts"] = len(_rpt.blackout_windows)
                    st.session_state["_lcola_n_events"]    = len(_rpt.top_events)
                    st.toast(
                        f"🛸 LCOLA 飞越筛选完成！  {len(_rpt.blackout_windows)} 个禁发窗口 · "
                        f"{len(_rpt.top_events)} 条合取事件",
                        icon="✅",
                    )
                st.rerun()
                return

            # ── still running: skip if nothing changed ────────────────────
            _last = st.session_state.get("_lcola_last_step", -1)
            step, total = ps.step, (ps.total or n_steps_est)
            if step == _last and step > 0:
                return
            st.session_state["_lcola_last_step"] = step

            pct = min(step / max(total, 1), 0.99)
            elapsed = _time_mod.time() - ps.start_time
            eta_s = f"  预计剩余 {elapsed / step * (total - step):.0f}s" if step > 0 else ""
            st.progress(
                pct,
                text=f"飞越筛选中… {step}/{total} 个发射时刻  ⏱ {elapsed:.0f}s{eta_s}",
            )
            st.info(
                f"⏳ 正在后台计算（{step}/{total}），您可以切换到其他页面，完成后将在右上角弹窗通知。"
            )

        _lcola_progress_fragment()

    # ── 错误提示 ────────────────────────────────────────────────────────────
    if "lcola_error" in st.session_state:
        st.error(f"筛选失败：{st.session_state.pop('_lcola_error', '')}")

    # ── 结果展示 ────────────────────────────────────────────────────────────
    if "lcola_report" in st.session_state:
        report = st.session_state["lcola_report"]
        import numpy as np
        import pandas as pd

        # 成功通知
        st.success(
            f"✅ 筛选完成！评估了 {len(report.results)} 个发射时刻，"
            f"发现 {len(report.blackout_windows)} 个禁发窗口，"
            f"共 {len(report.top_events)} 条合取事件"
        )

        # Pc 时间曲线
        st.subheader("发射时刻偏移量 vs 最大碰撞概率 (Pc)")
        st.caption("横轴为相对于标称发射时刻的偏移秒数；红色阴影区域为禁射窗口（Pc ≥ 门限）")
        if report.results:
            df_pc = pd.DataFrame({
                "偏移量(s)": [r.t_launch_offset_s for r in report.results],
                "最大 Pc":   [r.max_pc            for r in report.results],
            }).set_index("偏移量(s)")
            _line_chart_zero(df_pc)
            st.caption(f"Pc 门限 = {report.pc_threshold:.0e}（{'载人' if crewed else '非载人'}任务）")

        # 禁发/安全窗口
        col_bo, col_sa = st.columns(2)
        with col_bo:
            st.subheader("🔴 禁射窗口（Blackout Windows）")
            bows = report.blackout_windows
            if bows:
                df_bo = pd.DataFrame([
                    {"开始(UTC)": t0.strftime("%H:%M:%S"),
                     "结束(UTC)": t1.strftime("%H:%M:%S"),
                     "持续(min)": round((t1-t0).total_seconds()/60, 1)}
                    for t0, t1 in bows
                ])
                st.dataframe(df_bo, use_container_width=True)
            else:
                st.success("窗口内无禁射时段，全时段可发射")

        with col_sa:
            st.subheader("🟢 安全发射窗口（Safe Windows）")
            safe = report.safe_windows
            if safe:
                df_sa = pd.DataFrame([
                    {"开始(UTC)": t0.strftime("%H:%M:%S"),
                     "结束(UTC)": t1.strftime("%H:%M:%S"),
                     "持续(min)": round((t1-t0).total_seconds()/60, 1)}
                    for t0, t1 in safe
                ])
                st.dataframe(df_sa, use_container_width=True)
            else:
                st.warning("窗口内无完整安全时段")

        # 高风险合取事件
        st.subheader("🔍 高风险合取事件（Top 50，按 Pc 降序）")
        if report.top_events:
            risk_color = {"RED": "🔴", "AMBER": "🟠", "YELLOW": "🟡", "GREEN": "🟢"}
            rows = [{
                "风险": risk_color.get(ev.risk_level, ""),
                "NORAD ID":       ev.norad_cat_id,
                "目标名称":       ev.object_name,
                "飞行阶段":       ev.phase,
                "TCA (UTC)":      ev.tca.strftime("%m-%d %H:%M:%S"),
                "最近距离(km)":   round(ev.miss_distance_km, 3),
                "Foster Pc":      f"{ev.probability:.3e}",
                "相对速度(km/s)": round(ev.v_rel_kms, 3),
                "偏移量(s)":      round(ev.t_launch_offset_s, 0),
            } for ev in report.top_events]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500)
        else:
            st.info("未发现合取事件（筛选阈值内）")

        # 按阶段统计
        st.subheader("按飞行阶段统计")
        if report.top_events:
            from collections import Counter
            phase_cnt = Counter(ev.phase for ev in report.top_events)
            df_ph = pd.DataFrame(list(phase_cnt.items()), columns=["阶段", "合取事件数"])
            _bar_chart_zero(df_ph.set_index("阶段"))
