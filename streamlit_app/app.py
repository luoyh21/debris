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

from streamlit_app.nav_icons import (
    SIDEBAR_NAV_BLUE_CSS,
    icon_inline,
    risk_dot_html,
    section_title,
    sidebar_brand_row,
    title_row,
)

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_FAVICON_PATH = os.path.join(_ASSETS_DIR, "favicon.svg")

NAV_PAGES = [
    ("overview",  "系统概览",       "overview"),
    ("viz",       "可视化探索",     "viz"),
    ("catalog",   "目标目录",       "catalog"),
    ("sim",       "轨迹仿真",       "sim"),
    ("lcola",     "LCOLA 飞越筛选", "lcola"),
    ("collision", "碰撞风险",       "collision"),
    ("longterm",  "长期风险评估",   "longterm"),
    ("ai",        "AI 助手",        "ai"),
]


def _norad_display(nid) -> str:
    """Catalog NORAD IDs are positive; synthetic demo threats use negative placeholders."""
    if nid is None:
        return "—"
    try:
        n = int(nid)
    except (TypeError, ValueError):
        return str(nid)
    if n < 0:
        return f"合成 #{abs(n)}"
    return str(n)



class _LcolaProgress:
    """Thread-safe progress container for LCOLA background screening.

    Background thread ONLY mutates attributes of this object.
    It never calls st.session_state[key] = ... (which raises
    StreamlitAPIException outside a ScriptRunContext).
    The main thread (and @st.fragment) reads these attributes via
    st.session_state['_lcola_ps'].

    Progress convention:
      step < 0  → preparation phase (status_msg describes what's happening)
      step ≥ 0  → scan phase (step / total launch-time slots evaluated)
    """
    __slots__ = ('step', 'total', 'stop_req', 'done', 'error', 'report',
                 'start_time', 'status_msg')

    def __init__(self, total: int):
        self.step       = 0
        self.total      = total
        self.stop_req   = False
        self.done       = False
        self.error      = None
        self.report     = None
        self.start_time = _time_mod.time()
        self.status_msg = "初始化中…"

st.set_page_config(
    page_title="空间碎片监测系统",
    page_icon=_FAVICON_PATH if os.path.isfile(_FAVICON_PATH) else "🛸",
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

    Uses session_scope() from database.db which has automatic pool_pre_ping,
    pool_recycle, and up-to-3-retry logic on OperationalError — this prevents
    long-running Streamlit sessions from dying on stale TCP connections.
    Falls back to a direct session (legacy path) if the import fails.
    """
    try:
        from database.db import session_scope
        with session_scope() as sess:
            result = sess.execute(text(sql), params or {})
            rows = result.fetchall()
            cols = list(result.keys())
        return pd.DataFrame(rows, columns=cols)
    except ImportError:
        # Fallback: direct session (no retry)
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
    except Exception as exc:
        st.error(f"数据库错误：{exc}")
        return pd.DataFrame()



def _bar_chart_zero(data: "pd.Series | pd.DataFrame",
                    x_label_angle: int | None = None) -> None:
    """st.bar_chart replacement that forces the y-axis to start at zero.

    ``x_label_angle`` overrides the X-axis tick label rotation in degrees
    (e.g. ``0`` for horizontal labels).  Default ``None`` keeps Altair's
    automatic behaviour (long labels rotate to vertical).
    """
    import altair as alt
    df = data.reset_index()
    cols = df.columns.tolist()
    x_col, y_col = cols[0], cols[1]
    x_axis = (alt.Axis(labelAngle=int(x_label_angle))
              if x_label_angle is not None else alt.Undefined)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x_col, sort=None, title=x_col, axis=x_axis),
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


def _overview_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    color: str = "#1677ff",
    x_tick_angle: int = 0,
):
    """Styled Plotly bar chart for overview dashboard."""
    import plotly.express as px

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        text=y_col,
        color_discrete_sequence=[color],
    )
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
        marker_line_color="#0e4fbf",
        marker_line_width=0.7,
        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y:,.0f}}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=12, r=12, t=24, b=12),
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", size=14, color="#0f172a"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.28,
        hoverlabel=dict(bgcolor="white"),
        showlegend=False,
    )
    fig.update_xaxes(
        title_text=x_col,
        tickangle=x_tick_angle,
        showgrid=False,
        linecolor="#cbd5e1",
    )
    fig.update_yaxes(
        title_text=y_col,
        rangemode="tozero",
        tickformat=",.0f",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.24)",
        zeroline=False,
    )
    return fig


def _overview_histogram(
    df: pd.DataFrame,
    x_col: str,
    *,
    x_title: str,
    y_title: str = "数量",
    nbins: int = 60,
    color: str = "#1677ff",
):
    """Styled Plotly histogram for overview dashboard."""
    import plotly.express as px

    _d = df.copy()
    _d[x_col] = pd.to_numeric(_d[x_col], errors="coerce")
    _d = _d.dropna(subset=[x_col])
    if _d.empty:
        return None

    fig = px.histogram(
        _d,
        x=x_col,
        nbins=nbins,
        color_discrete_sequence=[color],
    )
    fig.update_traces(
        marker_line_color="#0e4fbf",
        marker_line_width=0.5,
        opacity=0.95,
        hovertemplate=f"{x_title}: %{{x:,.1f}}<br>{y_title}: %{{y:,.0f}}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=12, r=12, t=24, b=12),
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", size=14, color="#0f172a"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.03,
        hoverlabel=dict(bgcolor="white"),
        showlegend=False,
    )
    fig.update_xaxes(
        title_text=x_title,
        tickformat=",.0f",
        showgrid=False,
        linecolor="#cbd5e1",
    )
    fig.update_yaxes(
        title_text=y_title,
        rangemode="tozero",
        tickformat=",.0f",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.24)",
        zeroline=False,
    )
    return fig


def _overview_orbit_layer_chart(df: pd.DataFrame, alt_col: str):
    """Layered orbit distribution chart by semi-major-axis altitude."""
    import plotly.graph_objects as go

    _d = df.copy()
    _d[alt_col] = pd.to_numeric(_d[alt_col], errors="coerce")
    _d = _d.dropna(subset=[alt_col])
    _d = _d[_d[alt_col] >= 0]
    if _d.empty:
        return None

    bins = [0, 450, 2000, 35786 - 2000, 35786 + 2000, 120000, float("inf")]
    labels = ["VLEO(0-450)", "LEO(450-2000)", "MEO(2000-33786)", "GEO带(±2000)", "HEO(37786-120000)", "深空/高椭圆(>120000)"]
    layer = pd.cut(_d[alt_col], bins=bins, labels=labels, right=False, include_lowest=True)
    cnt = layer.value_counts().reindex(labels, fill_value=0)
    total = int(cnt.sum()) or 1
    pct = (cnt / total * 100).round(1)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=cnt.values,
            text=[f"{c:,.0f}<br>{p:.1f}%" for c, p in zip(cnt.values, pct.values)],
            textposition="outside",
            marker=dict(
                color=["#3b82f6", "#1677ff", "#06b6d4", "#8b5cf6", "#f59e0b", "#ef4444"],
                line=dict(color="#0f172a", width=0.5),
            ),
            hovertemplate="轨道层: %{x}<br>数量: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=12, r=12, t=24, b=12),
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", size=13, color="#0f172a"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.18,
        showlegend=False,
    )
    fig.update_xaxes(title_text="轨道层", tickangle=-8, showgrid=False, linecolor="#cbd5e1")
    fig.update_yaxes(
        title_text="数量",
        tickformat=",.0f",
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.24)",
        zeroline=False,
    )
    return fig


def _overview_log10_histogram(df: pd.DataFrame, x_col: str):
    """Histogram on log10(x+1) axis to reduce long-tail empty space."""
    import plotly.express as px
    import numpy as np

    _d = df.copy()
    _d[x_col] = pd.to_numeric(_d[x_col], errors="coerce")
    _d = _d.dropna(subset=[x_col])
    _d = _d[_d[x_col] >= 0]
    if _d.empty:
        return None

    _d["log10_x"] = (_d[x_col] + 1.0).map(lambda v: float(np.log10(v)))

    fig = px.histogram(
        _d,
        x="log10_x",
        nbins=60,
        color_discrete_sequence=["#1677ff"],
    )
    fig.update_traces(
        marker_line_color="#0e4fbf",
        marker_line_width=0.5,
        opacity=0.95,
        hovertemplate="log10(半长轴高度+1): %{x:.2f}<br>数量: %{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=12, r=12, t=24, b=12),
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", size=14, color="#0f172a"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.03,
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="半长轴高度 (km, log10刻度)",
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=["0", "10", "100", "1k", "10k", "100k", "1M"],
        showgrid=False,
        linecolor="#cbd5e1",
    )
    fig.update_yaxes(
        title_text="数量",
        rangemode="tozero",
        tickformat=",.0f",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.24)",
        zeroline=False,
    )
    return fig


# ------------------------------------------------------------------
# 侧边栏导航（SVG 图标 + 文本按钮）
# ------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "overview"

st.sidebar.markdown(sidebar_brand_row(), unsafe_allow_html=True)
st.sidebar.markdown(SIDEBAR_NAV_BLUE_CSS, unsafe_allow_html=True)  # 全局 primary 蓝色主题  # 全局 primary 蓝色主题
st.sidebar.caption("功能导航")
for _nav_key, _nav_label, _nav_icon in NAV_PAGES:
    _c_ic, _c_bt = st.sidebar.columns([0.14, 0.86])
    with _c_ic:
        st.markdown(
            f'<div style="padding-top:10px">{icon_inline(_nav_icon, 20)}</div>',
            unsafe_allow_html=True,
        )
    with _c_bt:
        _active = st.session_state["nav_page"] == _nav_key
        if st.button(
            _nav_label,
            key=f"nav_{_nav_key}",
            use_container_width=True,
            type="primary" if _active else "secondary",
        ):
            if st.session_state["nav_page"] != _nav_key:
                st.session_state["nav_page"] = _nav_key
                st.rerun()

page = st.session_state["nav_page"]

st.sidebar.markdown("---")
try:
    _host_header = st.context.headers.get("Host", "")
    _api_host = _host_header.split(":")[0] if _host_header else "localhost"
except Exception:
    _api_host = "localhost"
_docs_url = f"http://{_api_host}:8502/docs"
_docs_ic, _docs_bt = st.sidebar.columns([0.14, 0.86])
with _docs_ic:
    _docs_svg = icon_inline("docs", 20).replace('stroke="#1e3a5f"', 'stroke="#2C84BC"')
    st.markdown(
        f'<div style="padding-top:10px">{_docs_svg}</div>',
        unsafe_allow_html=True,
    )
with _docs_bt:
    st.markdown(
        f'<a href="{_docs_url}" target="_blank" style="'
        'display:flex;align-items:center;justify-content:center;'
        'padding:0.3rem 0.75rem;margin:4px 0;'
        'border:1px solid #2C84BC;border-radius:0.5rem;'
        'font-size:0.875rem;font-weight:400;color:#ffffff;'
        'text-decoration:none;background:#2C84BC;cursor:pointer;'
        'min-height:2.5rem;line-height:1.6;width:100%;'
        'font-family:Source Sans Pro,sans-serif;'
        'transition:background .15s,border-color .15s"'
        ' onmouseover="this.style.background=\'#24709e\';this.style.borderColor=\'#24709e\'"'
        ' onmouseout="this.style.background=\'#2C84BC\';this.style.borderColor=\'#2C84BC\'"'
        '>系统说明文档</a>',
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")
st.sidebar.caption(f"UTC 时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.caption("数据来源：Space-Track · UCS · ESA DISCOS · GCAT · UNOOSA · Asterank · NASA TechPort")

# Track current page in session_state so sidebar fragments can read it.
# Also clear LCOLA done notification when user visits the LCOLA page.
st.session_state["_current_page"] = page
if page == "lcola":
    st.session_state.pop("_lcola_done_notify", None)

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
        st.toast(f"LCOLA 计算失败：{ps.error}")
    elif ps.report is not None:
        _rpt = ps.report
        st.session_state["lcola_report"]       = _rpt
        st.session_state["_lcola_n_blackouts"] = len(_rpt.blackout_windows)
        st.session_state["_lcola_n_events"]    = len(_rpt.top_events)
        st.session_state["_lcola_done_notify"] = True
        st.toast(
            f"LCOLA 飞越筛选完成：{len(_rpt.blackout_windows)} 个禁发窗口 · "
            f"{len(_rpt.top_events)} 条合取事件",
        )
    st.rerun()


_bg_poll_fragment()

# ── 侧边栏后台进度状态（每 30 秒自动刷新，无论用户在哪个页面）─────────────────────
# Calling the fragment *inside* `with st.sidebar:` is the Streamlit-supported
# way to let a fragment write to the sidebar without raising an exception.
with st.sidebar:
    @st.fragment(run_every=30)
    def _sidebar_lcola_fragment():
        running = st.session_state.get("_lcola_running")
        done_notify = st.session_state.get("_lcola_done_notify")
        if not running and not done_notify:
            return
        if running:
            ps: _LcolaProgress | None = st.session_state.get("_lcola_ps")
            if ps is None:
                return
            step    = ps.step
            total   = ps.total
            elapsed = _time_mod.time() - ps.start_time
            status_msg = getattr(ps, "status_msg", "")
            if step < 0:
                # preparation phase — show descriptive text, not a raw negative index
                sb_text = f"LCOLA 预处理中… {status_msg}"
            elif total > 0:
                eta_str = (f" · 剩余≈{elapsed / step * (total - step):.0f}s"
                           if step > 0 else "")
                sb_text = f"LCOLA 计算中 {step}/{total}{eta_str}"
            else:
                sb_text = "LCOLA 正在后台计算…"
            st.markdown(
                f'<span style="color:#ffcc00;font-size:12px">{sb_text}</span>',
                unsafe_allow_html=True,
            )
        elif done_notify:
            n_bo = st.session_state.get("_lcola_n_blackouts", "?")
            n_ev = st.session_state.get("_lcola_n_events", "?")
            st.markdown(
                f'<span style="color:#00cc66;font-size:12px">'
                f'LCOLA 完成：{n_bo} 个禁发窗口 · {n_ev} 条合取事件</span>',
                unsafe_allow_html=True,
            )

    _sidebar_lcola_fragment()

# ------------------------------------------------------------------
# 页面：系统概览
# ------------------------------------------------------------------
if page == "overview":
    st.markdown(title_row("overview", "空间环境概览"), unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:18px 22px;margin-bottom:18px;color:#1e293b">
          <div style="font-size:1.05em;font-weight:600;margin-bottom:8px">
            关于本系统
          </div>
          <p style="margin:0 0 10px 0;line-height:1.75">
            本系统是一套面向<strong>航天发射任务规划人员</strong>和<strong>空间碎片研究人员</strong>的
            <strong>全链路空间态势感知平台</strong>。从数据采集、轨道预报，到碰撞风险评估和发射窗口优化，
            提供端到端的决策支持。
          </p>
          <p style="margin:0;line-height:1.75">
            系统整合了 <strong>Space-Track、UCS、ESA DISCOS、GCAT、UNOOSA</strong>
            五大权威数据源，统一去重后形成 <strong>60,000+</strong> 在轨目标的综合目录，
            并独立接入 <strong>Asterank</strong> 小行星 / 近地天体（NEO）专题库
            与 <strong>NASA TechPort</strong> 航天技术项目组合（约 20,000 项目）；
            同时提供 <strong>8 个交互式功能页面</strong>和 <strong>6 个 REST API 接口</strong>。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _overview_card(col, label: str, value, sub: str = "") -> None:
        """Render a metric card via HTML so large numbers never get truncated."""
        sub_h = (f"<div style='font-size:0.68em;color:#94A3B8;margin-top:3px'>{sub}</div>"
                 if sub else "")
        col.markdown(
            f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;"
            f"padding:14px 10px;text-align:center'>"
            f"<div style='font-size:0.78em;color:#475569;margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:1.45em;font-weight:700;color:#1E293B;"
            f"word-break:break-all'>{value}</div>{sub_h}</div>",
            unsafe_allow_html=True,
        )

    # Top-level metrics from unified view (ST + UCS + ESA)
    _stats_df = run_query("""
        SELECT
            COUNT(*)                                                     AS total,
            SUM(CASE WHEN object_type = 'PAYLOAD'     THEN 1 ELSE 0 END) AS payloads,
            SUM(CASE WHEN object_type = 'DEBRIS'      THEN 1 ELSE 0 END) AS debris,
            SUM(CASE WHEN object_type = 'ROCKET BODY' THEN 1 ELSE 0 END) AS rockets,
            COUNT(DISTINCT primary_source)                                AS n_sources
        FROM v_unified_objects
    """)

    c1, c2, c3, c4 = st.columns(4)
    if not _stats_df.empty:
        _s = _stats_df.iloc[0]
        _overview_card(c1, "在轨目标总数", f"{int(_s['total']):,}",
                       f"融合 {int(_s['n_sources'])} 个数据源")
        _overview_card(c2, "有效载荷", f"{int(_s['payloads']):,}", "PAYLOAD")
        _overview_card(c3, "空间碎片", f"{int(_s['debris']):,}", "DEBRIS")
        _overview_card(c4, "火箭箭体", f"{int(_s['rockets']):,}", "ROCKET BODY")
    else:
        for c in (c1, c2, c3, c4):
            _overview_card(c, "–", "–")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(section_title("chart_bar", "目标类型分布（多源融合）"), unsafe_allow_html=True)
        type_df = run_query("""
            SELECT
                CASE object_type
                    WHEN 'DEBRIS'      THEN '空间碎片'
                    WHEN 'PAYLOAD'     THEN '有效载荷'
                    WHEN 'ROCKET BODY' THEN '火箭箭体'
                    ELSE '未知'
                END AS 类型,
                COUNT(*) AS 数量
            FROM v_unified_objects
            GROUP BY object_type
            ORDER BY 数量 DESC
        """)
        if not type_df.empty:
            fig_type = _overview_bar_chart(type_df, "类型", "数量", x_tick_angle=0)
            st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("暂无数据，请先运行：`python3 run.py ingest`")

    with col_right:
        st.markdown(section_title("chart_line", "轨道分布（半长轴）"), unsafe_allow_html=True)
        # From raw GP elements: a = (mu / n^2)^(1/3), altitude = a - R_earth
        sma_df = run_query("""
            WITH latest_gp AS (
                SELECT DISTINCT ON (norad_cat_id)
                    norad_cat_id,
                    mean_motion
                FROM gp_elements
                WHERE mean_motion IS NOT NULL AND mean_motion > 0
                ORDER BY norad_cat_id, epoch DESC
            )
            SELECT
                POWER(398600.4418 / POWER((2 * PI() * mean_motion / 86400.0), 2), 1.0 / 3.0) - 6378.137
                    AS 半长轴高度_km
            FROM latest_gp
            WHERE mean_motion > 0
        """)
        fig_sma = _overview_log10_histogram(sma_df, "半长轴高度_km") if not sma_df.empty else None
        if fig_sma is not None:
            st.plotly_chart(fig_sma, use_container_width=True)
        else:
            st.info("暂无半长轴数据")

    st.markdown(section_title("layers", "轨道层分布（按半长轴高度分层）"), unsafe_allow_html=True)
    if not sma_df.empty:
        fig_layer = _overview_orbit_layer_chart(sma_df, "半长轴高度_km")
        if fig_layer is not None:
            st.plotly_chart(fig_layer, use_container_width=True)
        else:
            st.info("暂无可分层的轨道数据")
    else:
        st.info("暂无可分层的轨道数据")

    st.markdown(section_title("chart_line", "轨道倾角分布（来自 GP 原始数据）"), unsafe_allow_html=True)
    inc_df = run_query("""
        WITH latest_gp AS (
            SELECT DISTINCT ON (norad_cat_id)
                norad_cat_id,
                inclination
            FROM gp_elements
            WHERE inclination IS NOT NULL
            ORDER BY norad_cat_id, epoch DESC
        )
        SELECT inclination AS 倾角_deg
        FROM latest_gp
        WHERE inclination >= 0 AND inclination <= 180
    """)
    fig_inc = _overview_histogram(
        inc_df,
        "倾角_deg",
        x_title="倾角 (°)",
        y_title="数量",
        nbins=72,  # ~2.5° per bin
        color="#1890ff",
    ) if not inc_df.empty else None
    if fig_inc is not None:
        st.plotly_chart(fig_inc, use_container_width=True)
    else:
        st.info("暂无倾角数据")

    # ── 历年航天发射趋势 ───────────────────────────────────────────────────────
    st.markdown(section_title("rocket", "历年航天发射趋势（1957—今）"), unsafe_allow_html=True)
    st.caption(
        "数据来源：GCAT (McDowell) + UNOOSA + Space-Track 多源融合，"
        "仅统计有效载荷（PAYLOAD）发射。Starlink 等巨型星座驱动 2020 年后跃升明显。"
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _load_launch_hist():
        from streamlit_app.launch_trend import load_launch_history
        return load_launch_history()

    _lt = _load_launch_hist()
    if _lt:
        from streamlit_app.launch_trend import (
            make_annual_launch_fig,
            make_cumulative_fig,
            make_decade_summary_fig,
        )
        _lt_col1, _lt_col2 = st.columns(2)
        with _lt_col1:
            st.markdown("##### 年代发射量汇总（1957—今）")
            st.plotly_chart(
                make_decade_summary_fig(_lt["annual_by_region"]),
                use_container_width=True,
            )
        with _lt_col2:
            st.markdown("##### 近年逐年发射量（2010—今）")
            st.plotly_chart(
                make_annual_launch_fig(_lt["annual_by_region"]),
                use_container_width=True,
            )

        st.markdown("##### 在轨目标数量历史演化（按类型）")
        st.plotly_chart(
            make_cumulative_fig(_lt.get("cumulative", pd.DataFrame())),
            use_container_width=True,
        )
    else:
        st.info("暂无历史发射数据，请确认数据库已完成摄入。")

    st.markdown(section_title("globe_meridians", "主要国家/地区在轨目标数量（Top 15）"), unsafe_allow_html=True)
    country_df = run_query("""
        SELECT
            country_code AS 国家代码,
            COUNT(*) AS 目标数量,
            SUM(CASE WHEN object_type='DEBRIS' THEN 1 ELSE 0 END) AS 碎片数,
            SUM(CASE WHEN object_type='PAYLOAD' THEN 1 ELSE 0 END) AS 载荷数,
            SUM(CASE WHEN object_type='ROCKET BODY' THEN 1 ELSE 0 END) AS 箭体数
        FROM v_unified_objects
        WHERE country_code IS NOT NULL AND country_code != ''
        GROUP BY country_code
        ORDER BY 目标数量 DESC
        LIMIT 15
    """)
    if not country_df.empty:
        st.dataframe(country_df, use_container_width=True)

    st.markdown(section_title("download", "多源数据摄入状态"), unsafe_allow_html=True)

    # Main breakdown: per *source-table* row counts × inferred object type.
    # We deliberately do NOT pull this from v_unified_objects because that
    # view dedupes by NORAD ID — an ESA / UCS row that is also present in
    # Space-Track gets *folded* into the Space-Track row via LEFT JOIN
    # (with has_esa / has_ucs flags), so the unified view shows only the
    # *exclusive* portion of each external source.  The摄入状态 table is
    # supposed to reflect "how many rows did each source contribute", so
    # we count the underlying source tables directly.
    _src_type_df = run_query("""
        SELECT 'Space-Track' AS 数据源, COALESCE(object_type,'UNKNOWN') AS 目标类型,
               COUNT(*) AS 数量
        FROM catalog_objects GROUP BY 1, 2
        UNION ALL
        SELECT 'UCS', 'PAYLOAD', COUNT(*)
        FROM external_ucs_satellites
        UNION ALL
        SELECT 'ESA-DISCOS',
            CASE
                WHEN "objectClass" IN ('Payload','Payload Mission Related Object')
                     THEN 'PAYLOAD'
                WHEN "objectClass" = 'Rocket Body'
                     THEN 'ROCKET BODY'
                WHEN "objectClass" IN ('Payload Fragmentation Debris',
                     'Rocket Fragmentation Debris','Payload Debris','Rocket Debris',
                     'Rocket Mission Related Object','Other Debris')
                     THEN 'DEBRIS'
                ELSE 'UNKNOWN'
            END,
            COUNT(*)
        FROM external_esa_discos GROUP BY 2
        ORDER BY 1, 3 DESC
    """)
    _src_total_df = run_query("""
        SELECT '数据源' AS 数据源, 0 AS 合计 WHERE FALSE
        UNION ALL SELECT 'Space-Track', COUNT(*) FROM catalog_objects
        UNION ALL SELECT 'UCS', COUNT(*) FROM external_ucs_satellites
        UNION ALL SELECT 'ESA-DISCOS', COUNT(*) FROM external_esa_discos
        ORDER BY 2 DESC
    """)

    if not _src_type_df.empty:
        # Pivot: source as rows, object type as columns
        _pivot = _src_type_df.pivot_table(
            index="数据源", columns="目标类型", values="数量",
            aggfunc="sum", fill_value=0,
        ).reset_index()
        _pivot.columns.name = None
        # Add total column
        type_cols = [c for c in _pivot.columns if c != "数据源"]
        _pivot["合计"] = _pivot[type_cols].sum(axis=1)
        # Reorder columns
        desired_order = ["数据源", "PAYLOAD", "DEBRIS", "ROCKET BODY", "UNKNOWN", "合计"]
        cols_final = [c for c in desired_order if c in _pivot.columns]
        _pivot = _pivot[cols_final]
        _rename = {"PAYLOAD": "有效载荷", "DEBRIS": "空间碎片",
                   "ROCKET BODY": "火箭箭体", "UNKNOWN": "未知"}
        _pivot = _pivot.rename(columns=_rename)
        # Format numbers
        for c in _pivot.columns:
            if c != "数据源":
                _pivot[c] = _pivot[c].apply(lambda x: f"{int(x):,}")
        st.dataframe(_pivot, use_container_width=True, hide_index=True)

        # Summary line
        _total_obj = run_query("SELECT COUNT(*) AS n FROM v_unified_objects")
        _n = int(_total_obj.iloc[0]["n"]) if not _total_obj.empty else 0
        _n_src = _src_total_df["数据源"].nunique() if not _src_total_df.empty else 0
        st.caption(f"融合 **{_n_src}** 个数据源 · 去重后共 **{_n:,}** 个空间目标")

    # Supplementary: GCAT launch history + UNOOSA aggregate stats + Asterank
    # (asteroids / NEO) + NASA TechPort (technology project portfolio). These
    # are NOT Earth-orbiting satellites/debris, so they live in their own
    # tables rather than v_unified_objects.
    with st.expander("补充与专题数据源（发射统计 · 小行星 · NASA 技术项目）", expanded=False):
        _aux_rows = []
        _aux_tables = [
            ("external_yearly_launches",        "GCAT 年度发射统计"),
            ("external_cumulative_onorbit",     "GCAT 累计在轨统计"),
            ("external_country_yearly_payload", "GCAT 国别载荷统计"),
            ("external_unoosa_launches",        "UNOOSA 年度发射统计"),
            ("external_asterank",               "Asterank 小行星 / 近地天体目录"),
            ("external_techport",               "NASA TechPort 航天技术项目组合"),
        ]
        for tbl, label in _aux_tables:
            try:
                eq = run_query(f"SELECT COUNT(*) AS cnt FROM {tbl}")
                if not eq.empty and int(eq.iloc[0]["cnt"]) > 0:
                    _aux_rows.append((label, int(eq.iloc[0]["cnt"])))
            except Exception:
                pass
        if _aux_rows:
            _aux_df = pd.DataFrame(_aux_rows, columns=["数据源", "记录数"])
            _aux_df["记录数"] = _aux_df["记录数"].apply(lambda x: f"{x:,}")
            st.dataframe(_aux_df, use_container_width=True, hide_index=True)
            st.caption(
                "GCAT / UNOOSA 为历史发射趋势的聚合统计数据，用于可视化探索页面的发射趋势分析；"
                "Asterank 为独立的小行星/近地天体专题库（http://www.asterank.com），"
                "TechPort 为 NASA 航天技术项目组合（https://techport.nasa.gov），"
                "三者均与地球在轨目标目录相互独立，分别可在目标目录页的"
                "「小行星 / NEO (Asterank)」与「NASA 技术项目 (TechPort)」标签查看。"
            )

# ------------------------------------------------------------------
# 页面：可视化探索
# ------------------------------------------------------------------
elif page == "viz":
    from streamlit_app.viz_explorer import render_viz_explorer
    render_viz_explorer()

# ------------------------------------------------------------------
# 页面：目标目录
# ------------------------------------------------------------------
elif page == "catalog":
    st.markdown(title_row("catalog", "空间目标统一目录"), unsafe_allow_html=True)
    st.caption("融合 Space-Track、UCS Satellite Database、ESA DISCOS 三大数据源，"
               "按 NORAD ID 去重后统一展示；此外提供独立的 Asterank 小行星 / 近地天体目录、"
               "以及 NASA TechPort 航天技术项目组合（不同物理实体，分别浏览）。")

    _cat_tab_earth, _cat_tab_aster, _cat_tab_tp = st.tabs(
        ["地球在轨目标（多源融合）", "小行星 / NEO（Asterank）", "NASA 技术项目 (TechPort)"]
    )

    with _cat_tab_earth:
        with st.expander("筛选条件", expanded=True):
            col1, col2, col3 = st.columns(3)
            obj_type = col1.selectbox(
                "目标类型",
                ["全部", "DEBRIS（碎片）", "PAYLOAD（载荷）", "ROCKET BODY（箭体）", "UNKNOWN（未知）"]
            )
            country = col2.text_input("国家代码（如 US、PRC、RU、CIS）", "",
                                      help="CN/China → PRC，USA → US，USSR → CIS")
            name_search = col3.text_input("名称关键词", "")

            col4, col5, col6 = st.columns(3)
            perigee_min = col4.number_input("近地点下限 (km)", value=0, step=100)
            perigee_max = col5.number_input("近地点上限 (km)", value=50000, step=500)
            rcs_size = col6.selectbox("RCS 尺寸", ["全部", "SMALL", "MEDIUM", "LARGE"])

            col7, col8 = st.columns(2)
            primary_src = col7.selectbox("主数据源",
                                         ["全部", "Space-Track", "UCS", "ESA-DISCOS"])
            enrich_filter = col8.selectbox("数据enrichment",
                                           ["全部", "含 UCS 数据", "含 ESA 数据",
                                            "有用途信息", "有用户类型信息"])

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
            _country_aliases = {
                "CN": "PRC", "CHINA": "PRC", "中国": "PRC",
                "USA": "US", "美国": "US",
                "USSR": "CIS", "RUSSIA": "CIS", "俄罗斯": "CIS",
                "JAPAN": "JPN", "日本": "JPN",
                "INDIA": "IND", "印度": "IND",
            }
            _cc = country.strip().upper()
            _cc = _country_aliases.get(_cc, _cc)
            where_clauses.append("country_code = :country")
            params["country"] = _cc
        if name_search.strip():
            where_clauses.append("name ILIKE :ns")
            params["ns"] = f"%{name_search.strip()}%"
        if rcs_size != "全部":
            where_clauses.append("rcs_size = :rcs")
            params["rcs"] = rcs_size
        if primary_src != "全部":
            where_clauses.append("primary_source = :psrc")
            params["psrc"] = primary_src
        if enrich_filter == "含 UCS 数据":
            where_clauses.append("has_ucs = true")
        elif enrich_filter == "含 ESA 数据":
            where_clauses.append("has_esa = true")
        elif enrich_filter == "有用途信息":
            where_clauses.append("inferred_purpose IS NOT NULL")
        elif enrich_filter == "有用户类型信息":
            where_clauses.append("inferred_users IS NOT NULL")

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
                primary_source AS "主数据源",
                CASE WHEN has_ucs THEN '✓' ELSE '' END AS "UCS",
                CASE WHEN has_esa THEN '✓' ELSE '' END AS "ESA",
                inferred_purpose AS "用途",
                inferred_users   AS "用户类型",
                ucs_purpose    AS "用途(UCS原始)",
                ucs_users      AS "用户(UCS原始)",
                ROUND(esa_mass_kg::numeric, 2) AS "质量kg(ESA)",
                ROUND(esa_cross_section_m2::numeric, 4) AS "截面m²(ESA)",
                esa_mission    AS "任务(ESA)",
                CASE esa_active WHEN true THEN '在轨' WHEN false THEN '已衰减' ELSE '' END AS "状态(ESA)"
            FROM v_unified_objects
            WHERE {where_sql}
            ORDER BY (name IS NOT NULL) DESC, launch_date DESC NULLS LAST, norad_cat_id DESC
            LIMIT 2000
        """, params)

        st.write(f"**共 {len(df)} 条记录**（最多显示 2000 条）")
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=550)
        else:
            st.warning("当前筛选条件下无数据，请调整筛选范围或检查数据摄入状态。")

    # ── Tab 2：Asterank 小行星 / NEO 独立目录 ────────────────────────────
    with _cat_tab_aster:
        st.caption(
            "Asterank 数据源（http://www.asterank.com） 维护的小行星 / 近地天体（NEO）"
            "开放目录。与地球在轨目标不同，这里是围绕太阳运行的天体，包含开普勒轨道"
            "根数、经济开采估值（price/profit）、Δv、光谱类型等字段。"
        )

        try:
            _ast_total = run_query("SELECT COUNT(*) AS n FROM external_asterank")
            _ast_n = int(_ast_total.iloc[0]["n"]) if not _ast_total.empty else 0
        except Exception:
            _ast_n = 0
            st.warning(
                "未检测到 `external_asterank` 表。请先运行："
                "`python scripts/ingest_asterank.py`（容器内："
                "`docker compose run --rm app python scripts/ingest_asterank.py`）"
                "后刷新页面。"
            )

        if _ast_n > 0:
            with st.expander("筛选条件", expanded=True):
                ac1, ac2, ac3 = st.columns(3)
                ast_name = ac1.text_input("名称关键词", "", key="_ast_name")
                ast_class = ac2.text_input("轨道族 class（如 APO、AMO、ATE）", "", key="_ast_class")
                ast_spec = ac3.text_input("光谱类型 spec", "", key="_ast_spec")

                ac4, ac5, ac6 = st.columns(3)
                ast_dv_max = ac4.number_input("Δv 上限 (km/s) — 越小越易开采", value=0.0, step=1.0,
                                              help="0 表示不限制", key="_ast_dv")
                ast_diam_min = ac5.number_input("最小直径 (km)", value=0.0, step=0.1, key="_ast_diam")
                ast_profit_min = ac6.number_input("最小估值 profit ($)", value=0.0, step=1.0e8,
                                                  format="%.2e", key="_ast_profit")

            a_where: list[str] = []
            a_params: dict = {}
            if ast_name.strip():
                a_where.append("(full_name ILIKE :an OR prov_des ILIKE :an)")
                a_params["an"] = f"%{ast_name.strip()}%"
            if ast_class.strip():
                a_where.append('"class" ILIKE :ac')
                a_params["ac"] = f"%{ast_class.strip()}%"
            if ast_spec.strip():
                a_where.append("spec ILIKE :asp")
                a_params["asp"] = f"%{ast_spec.strip()}%"
            if ast_dv_max and ast_dv_max > 0:
                a_where.append("(dv IS NULL OR dv <= :advm)")
                a_params["advm"] = float(ast_dv_max)
            if ast_diam_min and ast_diam_min > 0:
                a_where.append("diameter >= :adm")
                a_params["adm"] = float(ast_diam_min)
            if ast_profit_min and ast_profit_min > 0:
                a_where.append("profit >= :apm")
                a_params["apm"] = float(ast_profit_min)
            a_where_sql = " AND ".join(a_where) if a_where else "TRUE"

            a_df = run_query(f"""
                SELECT
                    full_name                          AS "名称",
                    prov_des                           AS "临时编号",
                    "class"                            AS "轨道族",
                    spec                               AS "光谱",
                    ROUND(a::numeric, 3)               AS "半长轴 a (AU)",
                    ROUND(e::numeric, 4)               AS "偏心率 e",
                    ROUND(i::numeric, 3)               AS "倾角 i (°)",
                    ROUND(per::numeric, 1)             AS "周期 (d)",
                    ROUND(diameter::numeric, 3)        AS "直径 (km)",
                    ROUND(albedo::numeric, 3)          AS "反照率",
                    ROUND(moid::numeric, 4)            AS "MOID (AU)",
                    ROUND(dv::numeric, 2)              AS "Δv (km/s)",
                    price                              AS "估值 price ($)",
                    profit                             AS "利润 profit ($)"
                FROM external_asterank
                WHERE {a_where_sql}
                ORDER BY profit DESC NULLS LAST, diameter DESC NULLS LAST
                LIMIT 2000
            """, a_params)

            st.write(f"**共 {len(a_df)} 条记录**（库中共 {_ast_n:,} 条；最多显示 2000 条）")
            if not a_df.empty:
                st.dataframe(a_df, use_container_width=True, height=550)
            else:
                st.info("当前筛选条件下无数据。")

    # ── Tab 3：NASA TechPort 航天技术项目组合 ───────────────────────────
    with _cat_tab_tp:
        st.caption(
            "NASA TechPort 数据源（https://techport.nasa.gov） 维护的 NASA 资助 / 跟踪的"
            "技术项目组合。包含项目标题、描述、TRL（Technology Readiness Level）、"
            "起止日期、责任组织、技术分类（NASA Taxonomy）、目标方向（Earth/Moon/Mars …）等字段，"
            "与地球在轨目标 / 小行星目录相互独立，存储于 `external_techport` 表。"
        )

        try:
            _tp_total = run_query("SELECT COUNT(*) AS n FROM external_techport")
            _tp_n = int(_tp_total.iloc[0]["n"]) if not _tp_total.empty else 0
        except Exception:
            _tp_n = 0
            st.warning(
                "未检测到 `external_techport` 表。请先运行："
                "`python scripts/ingest_techport.py`（容器内："
                "`docker compose run --rm app python scripts/ingest_techport.py`）"
                "后刷新页面。"
            )

        if _tp_n > 0:
            with st.expander("筛选条件", expanded=True):
                tc1, tc2, tc3 = st.columns(3)
                tp_kw       = tc1.text_input("名称 / 描述关键词", "", key="_tp_kw")
                tp_status   = tc2.selectbox(
                    "项目状态",
                    ["全部", "Active", "Completed", "Pending", "Cancelled"],
                    key="_tp_status",
                )
                tp_org_type = tc3.selectbox(
                    "责任组织类型",
                    ["全部", "Academia", "Industry", "NASA_Center",
                     "NASA_Research_Center", "Other_Government_Agency"],
                    key="_tp_org_type",
                )

                tc4, tc5, tc6 = st.columns(3)
                tp_country = tc4.text_input("国家代码（如 US、CA、JPN）", "",
                                            key="_tp_country")
                tp_tx_code = tc5.text_input("技术分类 code（如 TX06、TX17）", "",
                                            key="_tp_tx_code",
                                            help="TX01–TX17，可只填前缀做模糊匹配")
                tp_trl_min = tc6.number_input("TRL 当前下限（1–9）", min_value=0,
                                              max_value=9, value=0, step=1,
                                              key="_tp_trl_min",
                                              help="0 表示不限制")

                tc7, tc8 = st.columns(2)
                tp_year_min = tc7.number_input("起始年份 ≥", min_value=0,
                                               max_value=2100, value=0, step=1,
                                               key="_tp_year_min",
                                               help="0 表示不限制")
                tp_year_max = tc8.number_input("结束年份 ≤", min_value=0,
                                               max_value=2100, value=0, step=1,
                                               key="_tp_year_max",
                                               help="0 表示不限制")

            t_where: list[str] = []
            t_params: dict = {}
            if tp_kw.strip():
                t_where.append("(title ILIKE :tk OR description ILIKE :tk)")
                t_params["tk"] = f"%{tp_kw.strip()}%"
            if tp_status != "全部":
                t_where.append("status = :ts")
                t_params["ts"] = tp_status
            if tp_org_type != "全部":
                t_where.append("lead_org_type = :tot")
                t_params["tot"] = tp_org_type
            if tp_country.strip():
                t_where.append("lead_org_country = :tc")
                t_params["tc"] = tp_country.strip().upper()
            if tp_tx_code.strip():
                t_where.append("primary_tx_code ILIKE :ttc")
                t_params["ttc"] = f"{tp_tx_code.strip()}%"
            if tp_trl_min and tp_trl_min > 0:
                t_where.append("trl_current >= :ttrl")
                t_params["ttrl"] = int(tp_trl_min)
            if tp_year_min and tp_year_min > 0:
                t_where.append("start_year >= :tymin")
                t_params["tymin"] = int(tp_year_min)
            if tp_year_max and tp_year_max > 0:
                t_where.append("end_year <= :tymax")
                t_params["tymax"] = int(tp_year_max)
            t_where_sql = " AND ".join(t_where) if t_where else "TRUE"

            t_df = run_query(f"""
                SELECT
                    project_id          AS "项目 ID",
                    title               AS "项目名称",
                    status              AS "状态",
                    release_status      AS "发布状态",
                    start_date          AS "开始",
                    end_date            AS "结束",
                    trl_begin           AS "TRL 起",
                    trl_current         AS "TRL 当前",
                    trl_end             AS "TRL 终",
                    program_acronym     AS "项目所属计划",
                    lead_org_acronym    AS "责任机构",
                    lead_org_type       AS "机构类型",
                    lead_org_country    AS "国家",
                    lead_org_state      AS "州 / 省",
                    primary_tx_code     AS "技术分类 code",
                    primary_tx_title    AS "技术分类",
                    destination_types   AS "目标方向",
                    view_count          AS "浏览数"
                FROM external_techport
                WHERE {t_where_sql}
                ORDER BY (status='Active') DESC,
                         trl_current DESC NULLS LAST,
                         start_date DESC NULLS LAST
                LIMIT 2000
            """, t_params)

            st.write(f"**共 {len(t_df)} 条记录**（库中共 {_tp_n:,} 条；最多显示 2000 条）")
            if not t_df.empty:
                st.dataframe(t_df, use_container_width=True, height=550)

                # 详情面板：选一个项目展开看 description / benefits 全文
                _ids = t_df["项目 ID"].dropna().astype(int).tolist()
                if _ids:
                    sel_pid = st.selectbox(
                        "查看项目详情（description / benefits）",
                        options=[None] + _ids,
                        format_func=lambda v: ("（不查看）" if v is None
                                               else f"{v} — "
                                               + str(t_df.set_index('项目 ID').loc[v, '项目名称'])[:80]),
                        key="_tp_detail_pid",
                    )
                    if sel_pid:
                        det = run_query(
                            "SELECT title, description, benefits, "
                            "primary_tx_title, lead_org_name, status, start_date, end_date "
                            "FROM external_techport WHERE project_id = :pid",
                            {"pid": int(sel_pid)},
                        )
                        if not det.empty:
                            row = det.iloc[0]
                            st.markdown(f"### {row['title']}")
                            st.caption(
                                f"状态：{row['status']} ｜ 起止：{row['start_date']} → {row['end_date']} ｜ "
                                f"责任机构：{row['lead_org_name']} ｜ 技术分类：{row['primary_tx_title']}"
                            )
                            if row.get("description"):
                                st.markdown("**项目描述：**")
                                st.write(row["description"])
                            if row.get("benefits"):
                                st.markdown("**预期效益：**")
                                st.write(row["benefits"])
            else:
                st.info("当前筛选条件下无数据。")

# ------------------------------------------------------------------
# 页面：轨迹片段（已迁移到可视化探索 → 轨道预报）
# ------------------------------------------------------------------
elif page == "segments":
    st.info(
        "**轨迹片段** 功能已整合到「可视化探索 → 轨道预报」标签页，"
        "在那里可选定任意目标并在三维地球上查看真实传播轨迹（含椭圆轨道可视化）。"
    )
    if st.button("前往可视化探索", type="primary"):
        st.session_state["page"] = "viz"
        st.rerun()

# ------------------------------------------------------------------
# 页面：碰撞风险（统一使用 6-DOF 仿真轨迹 + Foster Pc）
# ------------------------------------------------------------------
elif page == "collision":
    st.markdown(title_row("collision", "发射各阶段碰撞风险评估"), unsafe_allow_html=True)
    st.caption(
        "轨迹来源：6-DOF 数值积分（trajectory/six_dof.py）｜"
        "算法：Foster (1992) 2-D Pc 数值积分｜"
        "碎片来源：Space-Track + UCS + ESA DISCOS 多源统一目录 · SGP4 传播"
    )

    # ── 功能说明卡片 ────────────────────────────────────────────────────────
    st.info(
        "**本页面解决的问题：对于一个已确定的发射时刻，各飞行阶段各有多高的碰撞风险？**\n\n"
        "固定发射时刻后，将完整飞行轨迹划分为上升段、停泊轨道段等阶段，"
        "逐阶段与在轨碎片进行 TCA 求解 + Foster Pc 计算，输出每个阶段的风险等级（RED/AMBER/YELLOW/GREEN）"
        "及详细合取事件表。\n\n"
        "**与「LCOLA 飞越筛选」的区别：** LCOLA 页面扫描一段时间窗口内的多个候选发射时刻，"
        "输出禁射窗口（回答『什么时候发射安全』）；"
        "本页面针对单一固定发射时刻，给出各阶段的具体 Pc 数值和风险等级"
        "（回答『选定时刻的碰撞风险有多高，哪个阶段最危险』）。"
    )

    # ── 依赖检查 ─────────────────────────────────────────────────────────────
    if "sim_result" not in st.session_state or "sim_phases" not in st.session_state:
        st.info(
            "请先在 **轨迹仿真** 页面运行仿真，生成轨迹数据后再来此页面评估碰撞风险。"
        )
        st.stop()

    result = st.session_state["sim_result"]
    phases = st.session_state["sim_phases"]

    # ── 配置区 ────────────────────────────────────────────────────────────────
    with st.expander("评估参数", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        hbr_m      = c1.number_input("联合硬体半径 HBR (m)", value=20, step=5,
                                     help="火箭 + 碎片外接球半径之和，默认 20 m")
        hbr_km     = hbr_m / 1000.0
        crewed     = c2.checkbox("载人任务（Pc 门限 1e-6）", value=False)
        fine_km    = c3.number_input("精细筛选距离 (km)", value=50, step=10)
        coarse_km  = c4.number_input("粗筛距离 (km)", value=200, step=50)
        inject_demo = st.checkbox(
            "注入演示威胁（Demo Threats）",
            value=True,
            help="注入三条标注为 DEMO 的合成合取事件（无真实 NORAD 编号，表格中显示为「合成 #n」）。"
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

    if st.button("开始碰撞风险评估", type="primary"):
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

        # ── 阶段摘要总览 ──────────────────────────────────────────────────────
        st.markdown(section_title("chart_bar", "各飞行阶段风险摘要"), unsafe_allow_html=True)
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
                "风险等级":         s.risk_level,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # ── Pc 柱状图（各阶段最大 Pc，对数坐标） ────────────────────────────────
        st.markdown(section_title("chart_line", "各阶段最大碰撞概率 (Pc)"), unsafe_allow_html=True)
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
        st.markdown(section_title("magnifier", "各阶段高风险碎片详情"), unsafe_allow_html=True)

        all_events_rows = []   # collect all for global table

        for s in summaries:
            phase_cn = phase_names_cn.get(s.phase_name, s.phase_name)
            with st.expander(
                f"{phase_cn}  │  合取事件 {len(s.events)} 条  │  "
                f"max Pc = {s.max_pc:.3e}  │  {s.risk_level}",
                expanded=(s.risk_level in ("RED", "AMBER") and len(s.events) > 0),
            ):
                if not s.events:
                    st.info("该阶段无合取事件（距离 > 精细筛选阈值）")
                    continue

                st.markdown(
                    f'{risk_dot_html(s.risk_level)}<span style="font-weight:600">'
                    f"{s.risk_level}</span> 阶段风险",
                    unsafe_allow_html=True,
                )
                # Phase risk profile
                if s.risk_text:
                    st.caption(s.risk_text)

                rows = []
                for ev in s.events:
                    rows.append({
                        "风险等级":         ev.risk_level,
                        "NORAD ID":         _norad_display(ev.norad_cat_id),
                        "目标名称":         ev.object_name,
                        "TCA (UTC)":        ev.tca.strftime("%m-%d %H:%M:%S"),
                        "最近距离 (km)":    round(ev.miss_distance_km, 3),
                        "Foster Pc":        f"{ev.probability:.4e}",
                        "Pc 误差":          f"{ev.pc_error:.1e}" if not (ev.pc_error != ev.pc_error) else "–",
                        "相对速度 (km/s)":  round(ev.v_rel_kms, 3),
                        "禁发标记":         "是" if ev.is_blackout else "否",
                    })
                    all_events_rows.append({
                        "阶段":             phase_cn,
                        "风险等级":         ev.risk_level,
                        "NORAD ID":         _norad_display(ev.norad_cat_id),
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
            st.markdown(section_title("clipboard", "全部合取事件汇总（按 Pc 降序）"), unsafe_allow_html=True)
            df_all = pd.DataFrame(all_events_rows)
            st.dataframe(df_all, use_container_width=True, height=500)

            # 禁发事件单独列出
            blackout_events = [
                ev for s in summaries for ev in s.events if ev.is_blackout
            ]
            if blackout_events:
                st.error(
                    f"发现 {len(blackout_events)} 个**禁发**合取事件"
                    f"（Pc ≥ {pc_thresh:.0e} 或 Miss Distance < 25 km）"
                )
                bo_rows = [{
                    "阶段":            phase_names_cn.get(ev.phase, ev.phase),
                    "NORAD ID":        _norad_display(ev.norad_cat_id),
                    "目标名称":        ev.object_name,
                    "TCA (UTC)":       ev.tca.strftime("%m-%d %H:%M:%S"),
                    "最近距离 (km)":   round(ev.miss_distance_km, 3),
                    "Foster Pc":       f"{ev.probability:.4e}",
                } for ev in sorted(blackout_events,
                                   key=lambda e: e.probability, reverse=True)]
                st.dataframe(pd.DataFrame(bo_rows), use_container_width=True)
            else:
                st.success("当前发射方案无禁发合取事件")

# ------------------------------------------------------------------
# 页面：长期任务碰撞风险评估
# ------------------------------------------------------------------
elif page == "longterm":
    from streamlit_app.longterm_risk import render_longterm_risk
    render_longterm_risk()

# ------------------------------------------------------------------
# 页面：AI 助手
# ------------------------------------------------------------------
elif page == "ai":
    st.markdown(title_row("ai", "AI 碎片分析助手"), unsafe_allow_html=True)
    st.caption(
        "可用自然语言询问碎片数据库或调用 MCP 工具，例如：\n"
        "- 「低地球轨道（200~2000km）有多少碎片？按目标类型和国家统计前5名」\n"
        "- 「搜索文昌上空 500km 内、高度 200~2000km 的碎片，列出近地点最低的前10个」\n"
        "- 「用长征五号B从文昌发射，方位角90°，预测明天发射的各阶段碰撞风险」"
    )

    # ── MCP 工具文档面板 ──────────────────────────────────────────────
    with st.expander("可用 MCP 工具（6 个）", expanded=False):
        st.markdown("""
**1. query_debris_in_region** — 在指定地理区域和高度范围内检索在轨空间目标

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

> 示例：「搜索文昌（19.61°N, 110.95°E）上空500km内、高度200~2000km的所有碎片」

---

**2. predict_launch_collision_risk** — 对指定发射任务进行 6-DOF 仿真 + Foster Pc 碰撞风险评估

| 参数 | 说明 | 默认 |
|------|------|------|
| `vehicle` | 运载火箭（CZ-5B/Falcon9/Ariane6） | CZ-5B |
| `launch_lat_deg` | 发射场纬度（°） | 19.61（文昌） |
| `launch_lon_deg` | 发射场经度（°） | 110.95（文昌） |
| `launch_az_deg` | 发射方位角（°，0=北，90=东） | 90（正东） |
| `launch_utc` | 发射时刻（ISO-8601 UTC） | 明日06:00 |
| `t_max_s` | 仿真时长（秒，600~7200） | 3600 |
| `include_demo_threats` | 是否注入演示威胁 | true |

> 示例：「用长征五号B从文昌发射，方位角90°，预测明天06:00 UTC的碰撞风险」

---

**3. get_debris_reentry_forecast** — 预报即将再入大气层的空间目标

| 参数 | 说明 | 默认 |
|------|------|------|
| `days_ahead` | 预报窗口（天） | 30 |
| `alt_max_km` | 近地点阈值（km） | 300 |
| `object_type` | DEBRIS/PAYLOAD/ROCKET BODY/ALL | ALL |
| `limit` | 最多返回目标数 | 50 |

> 示例：「未来30天有哪些碎片预计再入大气层？」

---

**4. get_object_tle** — 获取指定 NORAD 编号目标的最新 TLE 轨道根数

| 参数 | 说明 | 默认 |
|------|------|------|
| `norad_cat_id` | NORAD 目标编号（必填） | — |

> 示例：「给我 ISS（NORAD 25544）的 TLE 轨道根数」

---

**5. query_debris_by_rcs** — 按雷达截面积（RCS）大小类别筛选空间目标

| 参数 | 说明 | 默认 |
|------|------|------|
| `rcs_sizes` | 类别列表：SMALL/MEDIUM/LARGE | 全部 |
| `alt_min_km` | 最低轨道高度（km） | 0 |
| `alt_max_km` | 最高轨道高度（km） | 2000 |
| `object_type` | DEBRIS/PAYLOAD/ROCKET BODY/ALL | ALL |
| `limit` | 最多返回目标数 | 50 |

> 示例：「LEO（200~2000km）中有多少 LARGE 级碎片？」

---

**6. forecast_conjunction_risk** — 长期任务寿命期碰撞风险预测（NASA ORDEM 3.1 + 泊松 MC）

| 参数 | 说明 | 默认 |
|------|------|------|
| `alt_km` | 目标轨道高度（km） | 800 |
| `inc_deg` | 轨道倾角（°） | 53.0 |
| `mission_years` | 任务寿命年数 | 5 |
| `conjunction_km` | 交会距离阈值（km） | 2 |
| `sat_area_m2` | 卫星碰撞截面积（m²） | 10 |
| `band_km` | 高度搜索带宽（km） | 200 |

> 示例：「5年内会有多少次小于2km的接近？建议配备多少规避燃料？」
""")

    # ── CSS：修复聊天消息中 Markdown 表格重叠与滚动问题 ───────────────
    st.markdown("""
<style>
/* 聊天消息内的 Markdown 表格：横向可滚动、防止溢出重叠 */
[data-testid="stChatMessage"] table {
    display: block !important;
    overflow-x: auto !important;
    max-width: 100% !important;
    margin-bottom: 0.6em !important;
    border-collapse: collapse !important;
    position: relative !important;
    z-index: 0 !important;
}
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {
    white-space: nowrap !important;
    padding: 4px 10px !important;
    border: 1px solid #E2E8F0 !important;
}
[data-testid="stChatMessage"] {
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

    # ── 示例问题快捷按钮 ─────────────────────────────────────────────
    st.markdown(section_title("idea", "示例问题", level=4, icon_size=18), unsafe_allow_html=True)
    example_cols = st.columns(3)
    _examples = [
        ("区域碎片查询",
         "搜索文昌发射场（19.61°N, 110.95°E）上空500km范围内、高度200~2000km的碎片，列出前20个"),
        ("发射风险预测",
         "用长征五号B从文昌（纬度19.61°，经度110.95°）向正东方向发射，"
         "预测明天06:00 UTC发射的各阶段碰撞风险，给出风险等级和建议"),
        ("轨道带分布",
         "统计低地球轨道（perigee_km 200~2000km）内各类型目标数量，"
         "按 object_type 分组并列出代表性目标名称各3个"),
        ("再入预报",
         "预测未来30天内有哪些空间目标即将再入大气层，列出确认再入日期和近地点高度"),
        ("获取 TLE",
         "获取国际空间站（ISS，NORAD 25544）的最新TLE轨道根数，列出六根数"),
        ("长期接近预测",
         "我计划在800km高度（倾角53°）运营一颗卫星，寿命5年，"
         "预测5年内会有多少次小于2km的接近？聚合碰撞概率有多高？建议配备多少规避燃料？"),
    ]
    for i, (label, question) in enumerate(_examples):
        col = example_cols[i % 3]
        with col:
            if st.button(label, use_container_width=True, key=f"example_{i}"):
                st.session_state["_ai_draft"] = question
                # Streamlit caches the widget's bound value under its key
                # forever; if we don't clear it, picking a different example
                # will keep showing the previous question's text.  Reset both
                # the widget state and a small "version" suffix so the
                # text_area is fully re-instantiated on the next run.
                st.session_state.pop("ai_draft_editor", None)
                st.session_state["_ai_draft_ver"] = (
                    int(st.session_state.get("_ai_draft_ver", 0)) + 1
                )
                st.rerun()

    # 处理待发送消息（来自"发送草稿"按钮）
    pending_send = st.session_state.pop("_ai_send_pending", None)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史对话
    for msg in st.session_state.chat_history:
        role_label = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role_label):
            st.markdown(msg["content"])

    # 清除按钮
    if st.session_state.chat_history:
        if st.button("清除对话历史"):
            st.session_state.chat_history = []
            st.rerun()

    # 输入区：若有示例草稿则显示可编辑区域，否则显示 chat_input
    _ai_draft = st.session_state.get("_ai_draft", "")
    if _ai_draft:
        st.caption("示例问题已填入，可在下方编辑后发送：")
        # Append the version suffix to the key so each example click yields
        # a *new* widget instance whose initial value is correctly applied
        # (Streamlit otherwise re-uses the previous widget's stored value).
        _ed_key = f"ai_draft_editor_v{int(st.session_state.get('_ai_draft_ver', 0))}"
        edited_draft = st.text_area(
            "问题内容", value=_ai_draft, height=90,
            key=_ed_key, label_visibility="collapsed",
        )
        # 使用 HTML 按钮行，避免 st.columns 在窄屏下换行
        st.markdown("""
<style>
#ai-draft-btns { display:flex; gap:8px; margin-top:4px; }
#ai-draft-btns button { white-space: nowrap !important; }
</style>""", unsafe_allow_html=True)
        col_send, col_cancel, col_pad = st.columns([2, 2, 10])
        send_clicked   = col_send.button("发送", type="primary", key="ai_send_draft",
                                         use_container_width=True)
        cancel_clicked = col_cancel.button("清空", key="ai_cancel_draft",
                                           use_container_width=True)
        if send_clicked:
            st.session_state["_ai_send_pending"] = edited_draft
            st.session_state["_ai_draft"] = ""
            st.rerun()
        if cancel_clicked:
            st.session_state["_ai_draft"] = ""
            st.rerun()
        user_input = None
    else:
        user_input = pending_send or st.chat_input("请输入您的问题…")

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

        # 若消息来自草稿按钮（pending_send），本次渲染未生成 chat_input；
        # 强制重新渲染，让 chat_input 出现，保证后续对话可继续。
        if pending_send:
            st.rerun()

# ------------------------------------------------------------------
# 页面：轨迹仿真（6-DOF）
# ------------------------------------------------------------------
elif page == "sim":
    st.markdown(title_row("sim", "火箭轨迹仿真（6-DOF 数值积分）"), unsafe_allow_html=True)
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
        dt_out = st.number_input("输出步长（s）", value=10, step=5)
        submitted = st.form_submit_button("运行仿真", type="primary")

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
                    run_mc=False,
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


# ------------------------------------------------------------------
# 页面：OEM 管理（已整合到可视化探索 → 轨道预报）
# ------------------------------------------------------------------
elif page == "oem":
    st.info("OEM 功能已整合到「可视化探索 → 轨道预报」，在那里可导出/导入 OEM 文件。")
    if st.button("前往可视化探索", type="primary"):
        st.session_state["page"] = "viz"
        st.rerun()

# ------------------------------------------------------------------
# 页面：LCOLA 飞越筛选
# ------------------------------------------------------------------
elif page == "lcola":
    st.markdown(title_row("lcola", "发射碰撞规避（LCOLA）飞越窗口扫描"), unsafe_allow_html=True)

    # ── 功能说明卡片 ────────────────────────────────────────────────────────
    st.info(
        "**本页面解决的问题：在 ±N 分钟的发射窗口内，哪些时刻可以安全发射？**\n\n"
        "通过扫描多个候选发射时刻（每隔 step 秒取一个），对每个时刻运行完整的 "
        "PostGIS 空间预筛 → SGP4 传播 → TCA 求解 → Foster Pc 数值积分流程，"
        "最终输出 **禁射窗口（Blackout）** 和 **安全窗口（Safe）**，帮助任务规划员选择最优发射时刻。\n\n"
        "**与「碰撞风险」的区别：** 碰撞风险页面针对 *单一固定发射时刻*，"
        "逐飞行阶段给出 Pc 和风险等级（回答『这个时刻安不安全』）；"
        "本页面针对 *一段时间窗口内的多个候选时刻*，找出哪些时刻触发禁射条件"
        "（回答『窗口内什么时候可以发射』）。两者互补，不可替代。"
    )

    # ── 配置面板 ────────────────────────────────────────────────────────────
    with st.expander("筛选配置", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        crewed = col1.checkbox("载人任务（Pc 门限 1e-6）", value=False)
        hbr    = col2.number_input("联合硬体半径 HBR (m)", value=20, step=5) / 1000.0
        win_m  = col3.number_input("窗口宽度（分钟，各方向）", value=30, step=5,
                                   help="以标称发射时刻为中心，向前后各扩展 N 分钟")
        step_s = col4.number_input("筛选步长（秒）", value=60, step=30,
                                   help="相邻两个候选发射时刻之间的间隔。步长越小精度越高但耗时越长")
        lcola_inject_demo = st.checkbox(
            "注入演示威胁（Demo Threats）",
            value=True,
            help="在标称发射时刻附近注入合成禁发窗口（~5 分钟），用于演示 LCOLA 工作流。"
                 "目录规模较小时推荐开启。",
        )

    # ── 耗时估算 ────────────────────────────────────────────────────────────
    n_steps_est = int(win_m * 2 * 60 / max(step_s, 1)) + 1
    st.caption(
        f"本次将评估 **{n_steps_est} 个**候选发射时刻。"
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
        "开始飞越筛选（后台运行）", type="primary",
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
            def _cb(step, total, msg=None):
                ps.step  = step
                ps.total = total
                if msg is not None:
                    ps.status_msg = msg
                elif step >= 0:
                    ps.status_msg = f"扫描发射时刻 {step}/{total}"
                if ps.stop_req:
                    raise InterruptedError("用户取消")
            try:
                report   = screener.screen(
                    w_open, w_close, t0,
                    step_s=float(step_s),
                    progress_cb=_cb,
                    inject_demo=lcola_inject_demo,
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
                        f"LCOLA 飞越筛选完成：{len(_rpt.blackout_windows)} 个禁发窗口 · "
                        f"{len(_rpt.top_events)} 条合取事件",
                    )
                st.rerun()
                return

            # 每次进入页面都重绘进度条（离开再回来时 step 可能未变，不能跳过渲染）
            step, total = ps.step, (ps.total or n_steps_est)
            status_msg = getattr(ps, "status_msg", "")
            st.session_state["_lcola_last_step"] = step

            elapsed = _time_mod.time() - ps.start_time
            if step < 0:
                # preparation phase (spatial filter / TLE fetch / pre-propagation)
                pct = 0.02
                eta_s = ""
                prog_text = f"预处理中… {status_msg}  ·  已用 {elapsed:.0f}s"
            else:
                pct = min(step / max(total, 1), 0.99)
                eta_s = (f"  预计剩余 {elapsed / step * (total - step):.0f}s"
                         if step > 0 else "")
                prog_text = f"飞越筛选中… {step}/{total} 个发射时刻 · 已用 {elapsed:.0f}s{eta_s}"
            st.progress(pct, text=prog_text)
            st.info(
                f"正在后台计算（{status_msg}），可切换到其他页面，完成后右上角将弹窗通知。"
            )

        _lcola_progress_fragment()

    # ── 错误提示 ────────────────────────────────────────────────────────────
    if st.session_state.get("_lcola_error"):
        st.error(f"筛选失败：{st.session_state.pop('_lcola_error', '')}")

    # ── 结果展示 ────────────────────────────────────────────────────────────
    if "lcola_report" in st.session_state:
        report = st.session_state["lcola_report"]
        import numpy as np
        import pandas as pd

        # 成功通知
        st.success(
            f"筛选完成：已评估 {len(report.results)} 个发射时刻，"
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
            st.markdown(section_title("ban", "禁射窗口（Blackout Windows）"), unsafe_allow_html=True)
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
            st.markdown(section_title("check", "安全发射窗口（Safe Windows）"), unsafe_allow_html=True)
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
        st.markdown(section_title("magnifier", "高风险合取事件（Top 50，按 Pc 降序）"), unsafe_allow_html=True)
        if report.top_events:
            rows = [{
                "风险等级":       ev.risk_level,
                "NORAD ID":       _norad_display(ev.norad_cat_id),
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
            _bar_chart_zero(df_ph.set_index("阶段"), x_label_angle=0)
