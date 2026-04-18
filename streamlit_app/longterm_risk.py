"""
长期任务碰撞风险评估 —— ORDEM 3.1 通量模型 + Poisson Monte Carlo
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

_C_BLUE   = "#3B82F6"
_C_AMBER  = "#F59E0B"
_C_RED    = "#EF4444"
_C_GREEN  = "#22C55E"
_C_PURPLE = "#A855F7"
_C_GREY   = "#6B7280"
_C_TEAL   = "#14B8A6"


def _risk_label(pc: float) -> tuple[str, str]:
    if pc >= 1e-3:  return "极高风险", _C_RED
    if pc >= 1e-4:  return "高风险",   _C_RED
    if pc >= 1e-5:  return "中等风险", _C_AMBER
    if pc >= 1e-6:  return "低风险",   _C_BLUE
    return "极低风险", _C_GREEN


def _fmt_pc(pc: float) -> str:
    if pc <= 0:  return "< 10⁻¹⁵"
    exp  = math.floor(math.log10(pc))
    mant = pc / 10 ** exp
    return f"{mant:.2f}×10^{exp}" if abs(mant - 1.0) >= 0.05 else f"10^{exp}"


def _card(label: str, value: str, color: str = _C_BLUE, sub: str = "") -> str:
    sub_html = (f"<div style='font-size:0.73em;color:#94A3B8;margin-top:2px;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{sub}</div>"
                if sub else
                "<div style='font-size:0.73em;margin-top:2px'>&nbsp;</div>")
    return (
        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;"
        f"border-radius:10px;padding:12px 10px;text-align:center;"
        f"min-height:100px;display:flex;flex-direction:column;justify-content:center'>"
        f"<div style='font-size:0.76em;color:#475569;margin-bottom:4px'>{label}</div>"
        f"<div style='font-size:1.3em;font-weight:700;color:{color};"
        f"white-space:nowrap'>{value}</div>{sub_html}</div>"
    )


# ─── cached simulation ────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_mc(alt_km, inc_deg, mission_years, hbr_km,
               conjunction_km, sigma_km, n_mc, band_km, sat_area_m2):
    from mission_risk.mc_risk import fetch_debris_environment, run_monte_carlo
    env = fetch_debris_environment(float(alt_km), float(inc_deg), float(band_km))
    return run_monte_carlo(
        env=env, inc_deg=float(inc_deg),
        mission_years=float(mission_years),
        hbr_km=float(hbr_km),
        conjunction_km=float(conjunction_km),
        sigma_km=float(sigma_km),
        n_mc=int(n_mc),
        sat_area_m2=float(sat_area_m2),
    )


# ─── charts ───────────────────────────────────────────────────────────────────

def _pc_curve_fig(report) -> go.Figure:
    yr = report.ts_months / 12
    fig = go.Figure()
    # 1cm curve (upper bound, shaded)
    fig.add_trace(go.Scatter(
        x=yr, y=report.ts_pc_flux_1cm,
        mode="lines", name=">1 cm 全部碎片（含不可追踪）",
        line=dict(color=_C_RED, dash="dot", width=1.5),
    ))
    # 10cm curve (main)
    fig.add_trace(go.Scatter(
        x=yr, y=report.ts_pc_flux,
        mode="lines", name=">10 cm 可追踪碎片",
        line=dict(color=_C_BLUE, width=2.5),
        fill="tonexty", fillcolor="rgba(239,68,68,0.08)",
    ))
    for thresh, lbl, col in [(1e-4, "高风险 10⁻⁴", _C_RED),
                              (1e-5, "触发阈值 10⁻⁵", _C_AMBER)]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=col, line_width=1.2,
                      annotation_text=lbl, annotation_position="right",
                      annotation_font_color=col)
    fig.update_layout(
        xaxis_title="任务年限（年）",
        yaxis_title="累积碰撞概率 P_c",
        yaxis=dict(type="log", exponentformat="e"),
        height=310, margin=dict(t=20, b=40, l=70, r=20),
        legend=dict(orientation="h", y=1.02, x=0),
        template="plotly_white",
    )
    return fig


def _conj_rate_fig(report) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=report.ts_months / 12, y=report.ts_n_conj_mean,
        mode="lines", line=dict(color=_C_TEAL, width=2.5),
        name="期望累积交会次数",
    ))
    fig.update_layout(
        xaxis_title="任务年限（年）",
        yaxis=dict(title=f"期望交会次数（< {report.conjunction_km:.0f} km）",
                   rangemode="tozero"),
        height=240, margin=dict(t=10, b=40, l=60, r=20),
        template="plotly_white",
    )
    return fig


def _conj_hist_fig(report) -> go.Figure:
    counts = report.conj_count_samples
    fig = go.Figure(go.Histogram(
        x=counts, nbinsx=max(int(counts.max()) + 2, 10),
        marker_color=_C_BLUE, opacity=0.8,
    ))
    fig.add_vline(x=report.n_conj_mean, line_dash="solid", line_color=_C_AMBER,
                  annotation_text=f"均值 {report.n_conj_mean:.1f}",
                  annotation_font_color=_C_AMBER)
    fig.add_vline(x=report.n_conj_p95, line_dash="dash", line_color=_C_RED,
                  annotation_text=f"P95 {report.n_conj_p95:.0f}",
                  annotation_font_color=_C_RED)
    fig.update_layout(
        xaxis=dict(title=f"交会次数（<{report.conjunction_km:.0f} km）",
                   rangemode="tozero"),
        yaxis=dict(title="仿真频次", rangemode="tozero"),
        height=280, margin=dict(t=20, b=40, l=50, r=20),
        template="plotly_white",
    )
    return fig


def _miss_dist_fig(report) -> go.Figure:
    fin = report.min_miss_samples[np.isfinite(report.min_miss_samples)]
    fig = go.Figure()
    if len(fin) > 0:
        fig.add_trace(go.Histogram(x=fin, nbinsx=40,
                                    marker_color=_C_PURPLE, opacity=0.8))
        for pct, lbl, col in [(50, "中位数", _C_BLUE), (95, "P95", _C_RED)]:
            val = float(np.percentile(fin, pct))
            fig.add_vline(x=val, line_dash="dash", line_color=col,
                          annotation_text=f"{lbl}: {val:.2f} km",
                          annotation_font_color=col)
    fig.update_layout(
        xaxis=dict(title="最小交会距离（km）", rangemode="tozero"),
        yaxis=dict(title="仿真频次", rangemode="tozero"),
        height=280, margin=dict(t=20, b=40, l=50, r=20),
        template="plotly_white",
    )
    return fig


def _pc_dist_fig(report) -> go.Figure:
    valid = report.pc_agg_samples[report.pc_agg_samples > 1e-20]
    if len(valid) == 0:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="无交会事件（所有试次 Pc_agg ≈ 0）",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))],
            height=280, template="plotly_white", margin=dict(t=20, b=40))
        return fig
    fig = go.Figure(go.Histogram(
        x=np.log10(np.clip(valid, 1e-20, 1.0)),
        nbinsx=40, marker_color=_C_RED, opacity=0.8,
    ))
    fig.update_layout(
        xaxis=dict(title="log₁₀(P_c,agg)",
                   tickvals=[-15,-12,-10,-8,-6,-5,-4,-3,-2],
                   ticktext=["10⁻¹⁵","10⁻¹²","10⁻¹⁰","10⁻⁸",
                             "10⁻⁶","10⁻⁵","10⁻⁴","10⁻³","10⁻²"]),
        yaxis=dict(title="仿真频次", rangemode="tozero"),
        height=280, margin=dict(t=20, b=40, l=50, r=20),
        template="plotly_white",
    )
    for lv, lbl, col in [(-4, "10⁻⁴", _C_RED), (-5, "10⁻⁵", _C_AMBER)]:
        fig.add_vline(x=lv, line_dash="dot", line_color=col,
                      annotation_text=lbl, annotation_font_color=col,
                      annotation_position="top right")
    return fig


def _inc_dist_fig(env) -> go.Figure:
    if len(env.inclinations) == 0:
        fig = go.Figure()
        fig.add_annotation(text="该高度带无数据库碎片（ORDEM通量模型不受影响）",
                            xref="paper", yref="paper", x=0.5, y=0.5,
                            showarrow=False, font=dict(size=12))
        fig.update_layout(height=200, template="plotly_white",
                          margin=dict(t=10, b=30))
        return fig
    fig = go.Figure(go.Histogram(x=env.inclinations, nbinsx=36,
                                  marker_color=_C_GREY, opacity=0.7))
    fig.update_layout(
        xaxis_title="轨道倾角（°）",
        yaxis=dict(title="目标数量", rangemode="tozero"),
        height=200, margin=dict(t=10, b=30, l=50, r=10),
        template="plotly_white",
    )
    return fig


# ─── main renderer ────────────────────────────────────────────────────────────

def render_longterm_risk():
    from streamlit_app.nav_icons import section_title

    st.markdown(section_title("longterm", "长期任务碰撞风险评估"),
                unsafe_allow_html=True)
    st.caption(
        "**算法**：NASA ORDEM 3.1 通量表（Krisko 2016）+ 泊松蒙特卡洛 —— "
        "P_c = 1 − exp(−F·A·Δt)，F 按高度与倾角插值，覆盖 >10 cm 及 >1 cm 两挡碎片。"
    )

    # ════════════════════════════════════════════════════════════════════════
    # 两行输入 Form
    # ════════════════════════════════════════════════════════════════════════
    with st.form("longterm_form"):
        st.markdown("**第一行：轨道与任务参数**")
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        launch_date   = r1c1.date_input("发射日期", value=date(2028, 4, 2),
                                         min_value=date(2025, 1, 1))
        alt_km        = r1c2.number_input("高度 (km)", min_value=200,
                                           max_value=40_000, value=800, step=50,
                                           help="800 km = LEO高密度区；2000 km = LEO/MEO交界")
        inc_deg       = r1c3.number_input("倾角 (°)", min_value=0.0,
                                           max_value=180.0, value=53.0, step=1.0)
        mission_years = r1c4.number_input("寿命 (年)", min_value=1,
                                           max_value=30, value=5, step=1)
        sat_area_m2   = r1c5.number_input("面积 (m²)", min_value=1.0,
                                           max_value=1000.0, value=10.0, step=1.0,
                                           help="卫星有效截面积，典型小卫星 5–20 m²")

        st.markdown("**第二行：接近参数与仿真设置**")
        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
        conjunction_km = r2c1.selectbox("警戒距 (km)", [1, 2, 5, 10, 20], index=2,
                                         help="< 此距离计为一次交会")
        hbr_m         = r2c2.number_input("HBR (m)", min_value=1,
                                           max_value=50, value=10, step=1,
                                           help="联合硬体半径")
        sigma_km      = r2c3.number_input("位置σ (km)", min_value=0.1,
                                           max_value=10.0, value=1.5, step=0.1,
                                           help="TLE传播1-σ不确定度（各轴）")
        n_mc          = r2c4.selectbox("MC 次数", [500, 1000, 2000, 5000], index=1)
        band_km       = r2c5.selectbox("带宽 (km)", [100, 200, 400], index=2,
                                        help="DB查询高度窗口（不影响ORDEM通量）")

        submitted = st.form_submit_button("▶  开始仿真", type="primary")

    # ════════════════════════════════════════════════════════════════════════
    # 结果区
    # ════════════════════════════════════════════════════════════════════════
    if not submitted and "longterm_report" not in st.session_state:
        st.info(
            "**默认示例**：2028年发射，800 km / 53° 轨道，5年寿命，卫星面积 10 m²，"
            "5 km 交会警戒距离。\n\n"
            "点击「开始仿真」后将展示：\n"
            "- ORDEM 3.1 通量法碰撞概率（>10 cm / >1 cm 两挡）\n"
            "- 预期危险交会次数 MC 分布\n"
            "- 最近逼近距离分布\n"
            "- 累积碰撞概率随时间演化曲线"
        )
        return

    if submitted:
        with st.spinner(f"运行中（ORDEM 通量 + MC×{n_mc}）…"):
            try:
                report = _cached_mc(
                    alt_km=float(alt_km), inc_deg=float(inc_deg),
                    mission_years=float(mission_years),
                    hbr_km=hbr_m / 1000.0,
                    conjunction_km=float(conjunction_km),
                    sigma_km=float(sigma_km),
                    n_mc=int(n_mc), band_km=float(band_km),
                    sat_area_m2=float(sat_area_m2),
                )
                st.session_state["longterm_report"] = report
            except Exception as exc:
                st.error(f"仿真失败：{exc}")
                log.exception("longterm_risk simulation error")
                return

    report = st.session_state.get("longterm_report")
    if report is None:
        return

    env = report.env
    label_txt, label_color = _risk_label(report.pc_orbit_10cm)

    # ── 指标卡（3+3 两行） ─────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(_card("Pc（>10 cm）<br>可追踪碎片", _fmt_pc(report.pc_orbit_10cm),
                       label_color, sub="ORDEM 3.1 直接法"),
                 unsafe_allow_html=True)
    mc2.markdown(_card("Pc（>1 cm）<br>含全部碎片", _fmt_pc(report.pc_orbit_1cm),
                       _C_RED, sub="含不可追踪×50"),
                 unsafe_allow_html=True)
    mc3.markdown(_card("年碰撞率<br>ORDEM通量法", f"{report.collision_rate_per_year:.2e}",
                       _C_GREY, sub="次/年"),
                 unsafe_allow_html=True)

    mc4, mc5, mc6 = st.columns(3)
    mc4.markdown(_card(f"交会次数（均值）<br>阈值 &lt;{report.conjunction_km:.0f} km",
                       f"{report.n_conj_mean:.1f} 次", _C_BLUE,
                       sub=f"P95 = {report.n_conj_p95:.0f} 次"),
                 unsafe_allow_html=True)
    mc5.markdown(_card("最近逼近（中位）<br>&nbsp;",
                       f"{report.min_miss_p50_km:.2f} km", _C_PURPLE,
                       sub=f"P95 = {report.min_miss_p95_km:.2f} km"),
                 unsafe_allow_html=True)
    mc6.markdown(_card("Pc_agg<br>Poisson MC 均值", _fmt_pc(report.agg_pc_mean),
                       _C_TEAL, sub="n=MC 聚合"),
                 unsafe_allow_html=True)

    st.markdown(
        f"<div style='padding:8px 14px;border-radius:8px;"
        f"background:{label_color}18;border-left:4px solid {label_color};"
        f"margin:8px 0 14px 0;font-size:0.91em'>"
        f"<b style='color:{label_color}'>{label_txt}</b>"
        f"&ensp;·&ensp;{int(report.mission_years)} 年寿命期"
        f"&ensp;·&ensp;{int(report.altitude_km)} km / {report.inclination_deg:.0f}°"
        f"&ensp;·&ensp;卫星面积 {report.sat_area_m2:.0f} m²"
        f"&ensp;·&ensp;ORDEM F = {env.flux_10cm:.2e} /m²/yr"
        f"&ensp;·&ensp;年交会率 {report.lambda_conj_per_year:.1f} 次/年</div>",
        unsafe_allow_html=True,
    )

    # ── 图表 tabs ─────────────────────────────────────────────────────────────
    tab_curve, tab_conj, tab_miss, tab_pc, tab_env = st.tabs([
        "📈 累积Pc曲线", "💥 交会次数分布", "📏 逼近距离分布",
        "🎲 Pc_agg分布", "🌐 碎片环境",
    ])

    with tab_curve:
        st.markdown("##### ORDEM 3.1 通量法累积碰撞概率")
        st.caption("蓝线 = >10 cm 可追踪碎片（主线），红点线 = >1 cm 全部碎片（上限）")
        st.plotly_chart(_pc_curve_fig(report), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("##### 期望累积交会次数")
        st.plotly_chart(_conj_rate_fig(report), use_container_width=True,
                        config={"displayModeBar": False})
        # 年度汇总
        rows = []
        for yr in sorted(set([1, 2, 3, 5, int(report.mission_years)])):
            m = min(int(yr * 12), len(report.ts_months) - 1)
            pc10 = report.ts_pc_flux[m]
            pc1  = report.ts_pc_flux_1cm[m]
            lbl, _ = _risk_label(pc10)
            rows.append({
                "年份": f"第 {yr} 年",
                "Pc（>10cm）": _fmt_pc(pc10),
                "Pc（>1cm）": _fmt_pc(pc1),
                "期望交会次数": f"{report.ts_n_conj_mean[m]:.1f}",
                "风险等级": lbl,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_conj:
        st.markdown("##### 任务寿命内交会次数 MC 分布")
        st.plotly_chart(_conj_hist_fig(report), use_container_width=True,
                        config={"displayModeBar": False})
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("期望交会（均值）", f"{report.n_conj_mean:.1f} 次")
        cc2.metric("期望交会（P95）",  f"{report.n_conj_p95:.0f} 次")
        cc3.metric("年交会率", f"{report.lambda_conj_per_year:.1f} 次/年")
        st.info(
            f"交会率来自 ORDEM 3.1 通量放大至 5 km 球面：\n"
            f"λ = F_10cm × π × d_thresh² = {env.flux_10cm:.2e} × π × "
            f"({int(report.conjunction_km*1000)} m)² = {report.lambda_conj_per_year:.1f} 次/年"
        )

    with tab_miss:
        st.markdown("##### 最近逼近距离 MC 分布")
        st.plotly_chart(_miss_dist_fig(report), use_container_width=True,
                        config={"displayModeBar": False})
        d1, d2, d3 = st.columns(3)
        d1.metric("最近逼近（中位）",  f"{report.min_miss_p50_km:.3f} km")
        d2.metric("最近逼近（P95）",   f"{report.min_miss_p95_km:.3f} km")
        d3.metric("最近逼近（最坏）",  f"{report.min_miss_absolute_km:.3f} km")

    with tab_pc:
        st.markdown("##### Pc_agg MC 分布（对数尺度）")
        st.plotly_chart(_pc_dist_fig(report), use_container_width=True,
                        config={"displayModeBar": False})
        p1, p2, p3 = st.columns(3)
        p1.metric("Pc_agg 均值",  _fmt_pc(report.agg_pc_mean))
        p2.metric("Pc_agg 中位",  _fmt_pc(report.agg_pc_p50))
        p3.metric("Pc_agg P95",   _fmt_pc(report.agg_pc_p95))
        st.markdown(
            "**NASA NPR 8715.6B / ESA DRAMA 阈值**：\n"
            "- 触发规避机动：Pc ≥ 10⁻⁵（非载人）/ 10⁻⁶（载人）\n"
            "- 任务级聚合风险上限：**P_c,agg ≤ 10⁻⁴**"
        )

    with tab_env:
        e1, e2, e3, e4 = st.columns(4)
        e1.markdown(_card("DB 高度带目标数",
                          f"{env.n_objects:,}",
                          _C_BLUE, sub=f"{int(report.altitude_km)}±{int(env.band_km/2)} km"),
                    unsafe_allow_html=True)
        e2.markdown(_card("ORDEM F >10 cm",
                          f"{env.flux_10cm:.2e}",
                          _C_AMBER, sub="obj/m²/yr"),
                    unsafe_allow_html=True)
        e3.markdown(_card("ORDEM F >1 cm",
                          f"{env.flux_1cm:.2e}",
                          _C_RED, sub="obj/m²/yr"),
                    unsafe_allow_html=True)
        e4.markdown(_card("碎片年增长率",
                          f"{env.growth_rate*100:.1f} %",
                          _C_GREY, sub="简化 LEGEND 模型"),
                    unsafe_allow_html=True)

        # Closest objects (always available, no band filter)
        st.markdown("##### 数据库中最接近目标轨道的碎片（前 30 条，按均值高度偏差排序）")
        if not env.closest_objects.empty:
            st.dataframe(env.closest_objects, use_container_width=True, hide_index=True)
        else:
            st.warning("数据库查询失败，无法获取最近碎片信息。")

        # Inclination distribution (only meaningful when band has data)
        st.markdown("##### 高度带内碎片倾角分布")
        st.plotly_chart(_inc_dist_fig(env), use_container_width=True,
                        config={"displayModeBar": False})

        if not env.top_objects.empty:
            with st.expander(f"高度带内全部目标列表（{env.n_objects} 个）"):
                st.dataframe(env.top_objects, use_container_width=True, hide_index=True)

    with st.expander("📐 算法说明"):
        st.markdown(f"""
**核心公式（ORDEM 3.1 / ESA MASTER-8 标准）**：

$$P_c = 1 - e^{{-F \\cdot A \\cdot \\Delta t}}$$

| 参数 | 值 |
|------|-----|
| 通量 F（>10 cm） | {env.flux_10cm:.2e} obj/m²/yr |
| 通量 F（>1 cm） | {env.flux_1cm:.2e} obj/m²/yr |
| 卫星截面积 A | {report.sat_area_m2:.0f} m² |
| 任务寿命 Δt | {report.mission_years:.0f} 年 |
| 碎片增长率 | {env.growth_rate*100:.1f} %/年 |

**通量来源**：内置 ORDEM 3.1 校准表（Krisko 2016），按高度对数线性插值 + 倾角修正因子。

**主要限制**：① ORDEM 3.1 校准至 2016 年，后续 Starlink 等大型星座碎片未充分建模；  
② 本模型为工程估算，精确任务设计应使用 NASA ORDEM 官方软件或 ESA DRAMA。
""")
