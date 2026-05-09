"""Streamlit page: 规避策略 / Collision-avoidance maneuver design.

Three tabs map onto the three solvers in :mod:`avoidance`:
  1. 高推力脉冲 — B 平面解析 + Lagrange 乘子
  2. 持续小推力 — 单次迭代 SCP / SOCP-lite
  3. 上升段走廊 — 时空走廊 + MPC 一步

The page either pulls the latest collision-risk events from ``st.session_state``
("risk_summaries") or lets the user enter the conjunction parameters by hand.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    from .nav_icons import title_row, section_title
except Exception:                                                                 # pragma: no cover
    from streamlit_app.nav_icons import title_row, section_title

from avoidance import (
    AvoidanceSolution,
    ConjunctionInputs,
    inputs_from_event,
    optimal_impulsive_dv,
    design_low_thrust_burn,
    design_ascent_correction,
)


def _add_earth(fig: go.Figure, R: float = 6378.137) -> None:
    """Reference Earth wireframe sphere."""
    u, v = np.linspace(0, 2*np.pi, 36), np.linspace(0, np.pi, 18)
    uu, vv = np.meshgrid(u, v)
    x = R*np.cos(uu)*np.sin(vv); y = R*np.sin(uu)*np.sin(vv); z = R*np.cos(vv)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        showscale=False, opacity=0.25, hoverinfo="skip",
        colorscale=[[0, "#0e2a47"], [1, "#0e2a47"]],
    ))


def _traj_fig(sol: AvoidanceSolution, title: str) -> go.Figure:
    fig = go.Figure()
    _add_earth(fig)
    if sol.nominal_traj:
        x = [s.r_eci[0] for s in sol.nominal_traj]
        y = [s.r_eci[1] for s in sol.nominal_traj]
        z = [s.r_eci[2] for s in sol.nominal_traj]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            name="原轨迹（无机动）",
            line=dict(color="#94a3b8", width=4, dash="dash"),
        ))
    if sol.modified_traj:
        x = [s.r_eci[0] for s in sol.modified_traj]
        y = [s.r_eci[1] for s in sol.modified_traj]
        z = [s.r_eci[2] for s in sol.modified_traj]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines+markers",
            name="规避后轨迹",
            line=dict(color="#22d3ee", width=5),
            marker=dict(size=3, color="#22d3ee"),
        ))
        # Maneuver point
        r0 = sol.modified_traj[0].r_eci
        fig.add_trace(go.Scatter3d(
            x=[r0[0]], y=[r0[1]], z=[r0[2]],
            mode="markers+text",
            marker=dict(size=8, color="#facc15", symbol="diamond"),
            text=["机动点火"],
            textposition="top center",
            name="点火时刻",
        ))
        # TCA point
        rt = sol.modified_traj[-1].r_eci
        fig.add_trace(go.Scatter3d(
            x=[rt[0]], y=[rt[1]], z=[rt[2]],
            mode="markers+text",
            marker=dict(size=8, color="#ef4444"),
            text=["TCA"],
            textposition="top center",
            name="TCA",
        ))
    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="X ECI (km)", yaxis_title="Y ECI (km)", zaxis_title="Z ECI (km)",
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", y=-0.05),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _metric_row(sol: AvoidanceSolution) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("机动前错失距离", f"{sol.miss_before_km*1000:.1f} m")
    c2.metric("机动后错失距离", f"{sol.miss_after_km*1000:.1f} m",
               delta=f"{(sol.miss_after_km - sol.miss_before_km)*1000:+.1f} m")
    c3.metric("|ΔV|", f"{sol.dv_mag_kms*1000:.2f} m/s")
    if sol.propellant_kg is not None:
        c4.metric("推进剂消耗", f"{sol.propellant_kg:.3f} kg")
    else:
        c4.metric("推进剂消耗", "n/a")

    c5, c6, c7 = st.columns(3)
    c5.metric("Pc 机动前", f"{sol.pc_before:.2e}")
    c6.metric("Pc 机动后", f"{sol.pc_after:.2e}")
    if sol.pc_before > 0 and sol.pc_after > 0:
        c7.metric("风险下降倍数", f"{sol.pc_before / max(sol.pc_after,1e-300):.1e}×")
    else:
        c7.metric("风险下降倍数", "—")


def _list_events_from_session() -> list[tuple[str, object]]:
    """Pull (label, event) tuples from `risk_summaries` in session_state."""
    items: list[tuple[str, object]] = []
    summaries = st.session_state.get("risk_summaries") or []
    for s in summaries:
        for ev in (getattr(s, "events", None) or []):
            label = (
                f"[{getattr(ev,'risk_level','?')}] "
                f"{ev.phase}  T+{int(ev.t_launch_offset_s)}s  "
                f"NORAD {ev.norad_cat_id}  "
                f"miss {ev.miss_distance_km*1000:.0f}m  "
                f"Pc {ev.probability:.1e}"
            )
            items.append((label, ev))
    items.sort(key=lambda kv: getattr(kv[1], "probability", 0.0), reverse=True)
    return items


def render_avoidance_page() -> None:
    """Top-level entry: render the 规避策略 page."""
    st.markdown(title_row("collision", "碎片规避策略生成器"), unsafe_allow_html=True)
    st.caption(
        "针对碰撞风险评估输出的合取事件，分别按"
        "**B 平面解析（脉冲）/ 序列凸规划（持续小推力）/ 时空走廊（上升段）**"
        "三类算法生成规避方案，输出 ΔV、推进剂、修正后错失距离 / Pc 与三维轨迹对比。"
    )

    # ── input source --------------------------------------------------------
    sim_result = st.session_state.get("sim_result")
    events     = _list_events_from_session()

    with st.expander("输入与不确定性参数", expanded=True):
        if sim_result is None:
            st.info("尚未在「轨迹仿真」页面运行任务仿真——请先生成 `SimResult` 后再来。")
            st.stop()
        if not events:
            st.warning("未在 session 中检测到碰撞事件。请先在「碰撞风险」页面运行评估，或在下方手填。")
        labels = ["✋ 手动输入合取参数"] + [lab for lab, _ in events]
        choice = st.selectbox("数据来源", labels, key="av_evt_pick")

        if choice == labels[0]:
            colA, colB, colC = st.columns(3)
            tca_offset_s = colA.number_input(
                "TCA（自发射时刻偏移，秒）", min_value=10, max_value=600,
                value=240, step=10,
            )
            miss_m = colB.number_input("当前最近距离 (m)", min_value=1.0,
                                       max_value=20_000.0, value=120.0, step=10.0)
            v_rel_kms = colC.number_input("相对速度 |v_rel| (km/s)",
                                          min_value=0.5, max_value=18.0,
                                          value=14.5, step=0.5)
            class _SyntheticEvent:
                def __init__(s):
                    s.tca = sim_result.config.launch_utc + timedelta(seconds=tca_offset_s)
                    s.t_launch_offset_s = tca_offset_s
                    s.phase   = "POST_SEPARATION"
                    s.miss_distance_km = miss_m / 1000.0
                    s.v_rel_kms = v_rel_kms
                    s.norad_cat_id = -999
                    s.object_name  = "(手动输入)"
                    s.probability  = 0.0
            event = _SyntheticEvent()
        else:
            event = events[labels.index(choice) - 1][1]

        col1, col2 = st.columns(2)
        sigma_km = col1.number_input("组合 1-σ 位置不确定度 σ (km)", min_value=0.05,
                                     max_value=20.0, value=1.5, step=0.1,
                                     help="主目标 + 威胁目标位置不确定度的二范数和。"
                                          "Space-Track TLE ≈ 1–3 km，DISCOS 高精度产品 ≈ 0.1 km。")
        hbr_km   = col2.number_input("联合硬体半径 HBR (m)", min_value=1.0,
                                     max_value=200.0, value=20.0, step=1.0) / 1000.0
        inputs = inputs_from_event(event, sim_result,
                                    sigma_combined_km=float(sigma_km),
                                    hbr_km=float(hbr_km))

        st.caption(
            f"已选事件：**{getattr(event,'object_name','?')}** "
            f"（NORAD {getattr(event,'norad_cat_id','?')}）"
            f"  ·  TCA 错失 {inputs.miss_distance_km*1000:.1f} m  ·  "
            f"|v_rel| {inputs.v_rel_kms:.2f} km/s  ·  TCA = "
            f"{inputs.tca.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

    # ── three solvers ------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "① 高推力脉冲（B 平面解析）",
        "② 持续小推力（SCP / SOCP）",
        "③ 上升段走廊（时空 + MPC）",
    ])

    # ── Tab 1: impulsive ----------------------------------------------------
    with tab1:
        with st.expander("脉冲规避参数", expanded=True):
            ic1, ic2, ic3, ic4 = st.columns(4)
            dv_budget_ms = ic1.number_input(
                "ΔV 预算 (m/s)", min_value=0.1, max_value=200.0,
                value=5.0, step=0.5, key="av_imp_dvb",
            )
            lead_min = ic2.number_input(
                "提前点火（分钟前）", min_value=1.0, max_value=720.0,
                value=15.0, step=1.0, key="av_imp_lead",
            )
            isp_imp = ic3.number_input("Isp (s) — 化学", min_value=200,
                                        max_value=460, value=320, step=10,
                                        key="av_imp_isp")
            mass_imp = ic4.number_input("航天器干重 (kg)", min_value=100,
                                         max_value=20000, value=4500, step=100,
                                         key="av_imp_m")
        sol1 = optimal_impulsive_dv(
            inputs,
            dv_budget_kms=dv_budget_ms / 1000.0,
            maneuver_lead_s=lead_min * 60.0,
            isp_s=float(isp_imp),
            dry_mass_kg=float(mass_imp),
        )
        _metric_row(sol1)
        st.plotly_chart(_traj_fig(sol1, "B 平面脉冲机动 ‒ 轨迹对比"),
                         use_container_width=True, key="av_imp_traj")
        with st.expander("算法说明 / 解释", expanded=False):
            for n in sol1.notes:
                st.markdown(f"- {n}")
            st.markdown(
                "**算法核心**：将协方差与位置矢量投影到与相对速度垂直的 B 平面，"
                "建立线性敏感度 $M = P\\,\\Phi_{rv}$；在 $\\|\\Delta v\\| \\le \\Delta v_{max}$"
                "约束下用 Lagrange 乘子求解，最优方向 = $M^TM$ 的最大特征向量。"
                "解析解使预警生成后毫秒级返回首选机动方案。"
            )

    # ── Tab 2: low-thrust ---------------------------------------------------
    with tab2:
        with st.expander("持续小推力参数", expanded=True):
            lc1, lc2, lc3, lc4 = st.columns(4)
            thrust_n = lc1.number_input("推力 (N)", min_value=0.05, max_value=20.0,
                                         value=1.0, step=0.05, key="av_lt_T")
            burn_min = lc2.number_input("点火时长（分钟）", min_value=2.0,
                                         max_value=600.0, value=10.0, step=1.0,
                                         key="av_lt_burn")
            isp_lt = lc3.number_input("Isp (s) — 电推", min_value=1500,
                                       max_value=8000, value=3000, step=100,
                                       key="av_lt_isp")
            mass_lt = lc4.number_input("初始湿重 (kg)", min_value=200,
                                        max_value=20000, value=4500, step=100,
                                        key="av_lt_m")
        sol2 = design_low_thrust_burn(
            inputs,
            burn_duration_s=burn_min * 60.0,
            thrust_N=float(thrust_n),
            spacecraft_mass_kg=float(mass_lt),
            isp_s=float(isp_lt),
        )
        _metric_row(sol2)
        st.plotly_chart(_traj_fig(sol2, "持续小推力机动 ‒ 轨迹对比"),
                         use_container_width=True, key="av_lt_traj")
        with st.expander("算法说明 / 解释", expanded=False):
            for n in sol2.notes:
                st.markdown(f"- {n}")
            st.markdown(
                "**算法核心**：把 HCW 输入矩阵 $\\Phi_{rv}$ 在 $[t_{man}, T_{TCA}]$ 上对推力施加时间积分，"
                "得到 *积分敏感度* $B(T)$；将连续推力建模为 LVLH 系常值加速度 $\\alpha$，"
                "在凸松弛下解 $\\max\\|P B \\alpha\\|^2 \\;\\text{s.t.}\\; \\|\\alpha\\|\\le a_{max}$，"
                "等价 SOCP — 一次迭代即收敛于线性化解。多目标 / 多碎片场景可加约束迭代为完整 SCP。"
            )

    # ── Tab 3: ascent corridor ----------------------------------------------
    with tab3:
        with st.expander("上升段时空走廊参数", expanded=True):
            ac1, ac2, ac3 = st.columns(3)
            t_launch_s = ac1.number_input("机动施加时刻 MET (s)", min_value=0.0,
                                           max_value=300.0, value=0.0, step=10.0,
                                           key="av_asc_t")
            max_az = ac2.number_input("方位角走廊 ±Δaz (°)", min_value=0.5,
                                       max_value=10.0, value=3.0, step=0.5,
                                       key="av_asc_az")
            max_pi = ac3.number_input("俯仰走廊 ±Δθ (°)", min_value=0.1,
                                       max_value=5.0, value=1.0, step=0.1,
                                       key="av_asc_pi")
        sol3 = design_ascent_correction(
            inputs,
            t_launch_s=float(t_launch_s),
            max_dazimuth_deg=float(max_az),
            max_dpitch_deg=float(max_pi),
        )
        _metric_row(sol3)
        st.plotly_chart(_traj_fig(sol3, "上升段时空走廊机动 ‒ 轨迹对比"),
                         use_container_width=True, key="av_asc_traj")
        with st.expander("算法说明 / 解释", expanded=False):
            for n in sol3.notes:
                st.markdown(f"- {n}")
            st.markdown(
                "**算法核心**：上升段不能任意施加 3-DOF Δv，仅可在 *可行驶走廊* 内"
                "微调发射方位与俯仰程序。把碎片云膨胀边界投影到时变时空矩形包络，"
                "求 $\\max(\\text{miss})$ s.t.  Δaz∈[-Δaz_max, Δaz_max] 等。"
                "实时引导利用 MPC 在 1 Hz 周期内重解，本页面给出第一迭代解析解。"
            )

    # ── consolidated comparison --------------------------------------------
    st.markdown(section_title("chart_bar", "三种规避方案对比"), unsafe_allow_html=True)
    comp = []
    for label, s in [("脉冲（B 平面）", sol1),
                     ("持续小推力", sol2),
                     ("上升段走廊", sol3)]:
        comp.append({
            "方案":            label,
            "ΔV (m/s)":        round(s.dv_mag_kms * 1000, 3),
            "持续时长 (s)":   round(s.burn_duration_s, 1),
            "推进剂 (kg)":    None if s.propellant_kg is None
                              else round(s.propellant_kg, 4),
            "错失前 (m)":      round(s.miss_before_km * 1000, 1),
            "错失后 (m)":      round(s.miss_after_km * 1000, 1),
            "Pc 前":           f"{s.pc_before:.2e}",
            "Pc 后":           f"{s.pc_after:.2e}",
        })
    import pandas as _pd
    st.dataframe(_pd.DataFrame(comp), use_container_width=True, hide_index=True)
