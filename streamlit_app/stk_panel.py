"""STK 交叉验证的 Streamlit 复用面板。

用法（在轨迹仿真页 / 碰撞页 / 轨道预报页中嵌入）::

    from streamlit_app.stk_panel import render_sgp4_validation_panel
    render_sgp4_validation_panel(line1, line2, norad_id=12345)

    from streamlit_app.stk_panel import render_six_dof_validation_panel
    render_six_dof_validation_panel(sim_result)

设计要点
--------
* 统一通过 :func:`stk_validation.detect_stk_availability` 探测当前主机能力，
  非 Windows / 缺 SDK 时按钮 ``disabled=True``，并给出 ``help`` 提示。
* 长耗时运行放在 ``st.spinner`` 中；写聚合 JSON 由 runner 内部完成。
* 显示 RMS / RIC 表格 + In-track 时间序列折线，便于研判误差模式。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import streamlit as st


def _availability_card(key_prefix: str) -> "Any":
    """渲染顶部"STK 可用性"卡片，并返回 :class:`StkAvailability`。"""
    from stk_validation import detect_stk_availability

    avail = detect_stk_availability()
    if avail.available:
        msg = (
            f"✅ 检测到 STK Python 接口（SDK = `{avail.sdk}`"
            f"{'，版本 ' + avail.sdk_version if avail.sdk_version else ''}）"
            f"\n\n操作系统：{avail.os_name}"
            + (f"，安装目录：`{avail.install_dir}`" if avail.install_dir else "")
        )
        st.success(msg)
    else:
        if avail.reason == "os_unsupported":
            st.warning(
                f"⚠ 当前操作系统 `{avail.os_name}` 不支持 Ansys STK，"
                f"「STK 交叉验证」按钮已禁用。\n\n"
                f"{avail.install_hint}"
            )
        else:
            st.warning(
                "⚠ 未在本机检测到可用的 STK Python SDK（`ansys-stk-core` / "
                "`comtypes` / `pywin32` 均未导入成功）。\n\n"
                f"{avail.install_hint or ''}\n\n"
                "下方运行将自动回退到 **参考实现真值**（`sgp4` 库 + HPOP-lite RK45 积分），"
                "仍可给出可信的自洽性误差，但不再是 Ansys STK 真值。"
            )
    return avail


def _render_report(report: Any, key_prefix: str) -> None:
    """渲染验证报告：Verdict + RMS 表格 + In-track 折线。"""
    import pandas as pd

    if report is None:
        return

    verdict = "✅ 通过" if report.passed else "⚠ 偏差超阈值"
    badge_color = "#15803d" if report.passed else "#b45309"
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:8px;"
        f"background:{badge_color}22;border-left:4px solid {badge_color};"
        f"font-weight:600;color:{badge_color};font-size:1.05em;'>"
        f"{verdict} · {report.candidate.split('（')[0]} ↔ {report.reference}</div>",
        unsafe_allow_html=True,
    )

    # 顶部 4 个数字
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("位置 RMS", f"{report.pos_rms_km*1000:.2f} m")
    c2.metric("位置 Max", f"{report.pos_max_km*1000:.2f} m")
    c3.metric("In-track RMS", f"{report.in_track_rms_km*1000:.2f} m")
    c4.metric("速度 RMS", f"{report.vel_rms_kms*1e3:.3f} mm/s")

    if report.notes:
        for n in report.notes:
            st.caption(f"• {n}")

    # RIC 分解表
    df_rms = pd.DataFrame([{
        "方向": "Radial（径向）",
        "RMS (m)": round(report.radial_rms_km * 1000.0, 3),
    }, {
        "方向": "In-track（沿轨）",
        "RMS (m)": round(report.in_track_rms_km * 1000.0, 3),
    }, {
        "方向": "Cross-track（轨道法向）",
        "RMS (m)": round(report.cross_track_rms_km * 1000.0, 3),
    }, {
        "方向": "总位置",
        "RMS (m)": round(report.pos_rms_km * 1000.0, 3),
    }])
    st.markdown("**RIC 分解（RMS）**")
    st.dataframe(df_rms, use_container_width=True, hide_index=True)

    # In-track / Radial / Cross-track 时间序列折线
    if getattr(report, "samples", None):
        df_ts = pd.DataFrame([{
            "MET (s)": s["t_offset_s"],
            "径向 (m)": s["radial_err_km"] * 1000.0,
            "沿轨 (m)": s["in_track_err_km"] * 1000.0,
            "轨道法向 (m)": s["cross_track_err_km"] * 1000.0,
            "位置 (m)": s["pos_err_km"] * 1000.0,
        } for s in report.samples]).set_index("MET (s)")
        st.markdown("**误差随时间演化**")
        st.line_chart(df_ts, use_container_width=True, height=260)

    with st.expander("原始报告 JSON（已写入 `data/validation/stk_validation.json`）", expanded=False):
        st.json(report.to_dict())


def render_sgp4_validation_panel(
    *,
    line1: Optional[str] = None,
    line2: Optional[str] = None,
    norad_id: int = 0,
    duration_default_h: float = 24.0,
    threshold_default_m: float = 5.0,
    key_prefix: str = "stk_sgp4",
) -> None:
    """嵌入"SGP4 vs STK"对照面板（轨道目录 / 轨道预报页可调用）。"""
    st.markdown("##### STK 交叉验证（SGP4）")
    with st.container(border=True):
        avail = _availability_card(key_prefix)

        col1, col2 = st.columns([3, 2])
        line1_in = col1.text_input(
            "TLE Line 1",
            value=line1 or "",
            key=f"{key_prefix}_l1",
            placeholder="1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990",
        )
        line2_in = col2.text_input(
            "TLE Line 2",
            value=line2 or "",
            key=f"{key_prefix}_l2",
            placeholder="2 25544  51.6400 130.0000 0001000   0.0000   0.0000 15.50000000000010",
        )

        c1, c2, c3 = st.columns(3)
        dur_h = c1.number_input(
            "对比时长（小时）", value=float(duration_default_h),
            step=1.0, min_value=0.1, max_value=240.0,
            key=f"{key_prefix}_durh",
        )
        step_min = c2.number_input(
            "采样步长（分钟）", value=10.0,
            step=1.0, min_value=0.1, max_value=120.0,
            key=f"{key_prefix}_stepmin",
        )
        thr_m = c3.number_input(
            "通过阈值（位置 RMS, m）", value=float(threshold_default_m),
            step=0.1, min_value=0.001,
            key=f"{key_prefix}_thr",
            help="LEO 上 SGP4 vs STK SGP4 理论上一致；阈值越紧越能暴露实现差异。",
        )

        os_disable = bool(not avail.os_supported)
        btn_help = (
            "Ansys STK Engine 仅 Windows 可用；当前 OS 不支持，按钮已禁用。"
            if os_disable else
            ("将运行 STK SGP4（如 STK 不可用则退化为 sgp4 库参考实现），"
             "再与本系统内置 SGP4 对照位置 RMS / In-track / Cross-track。")
        )
        run_clicked = st.button(
            "▶ 运行 STK SGP4 交叉验证",
            type="primary", disabled=os_disable,
            key=f"{key_prefix}_run", help=btn_help,
            use_container_width=True,
        )

        if run_clicked:
            if not (line1_in.strip() and line2_in.strip()):
                st.error("请先填写 TLE Line 1 / Line 2。")
            else:
                with st.spinner("正在调用 STK / 参考传播器并计算 RMS 误差…"):
                    try:
                        from stk_validation import run_sgp4_validation
                        report = run_sgp4_validation(
                            line1_in.strip(), line2_in.strip(),
                            norad_id=int(norad_id),
                            duration_s=float(dur_h * 3600.0),
                            step_s=float(step_min * 60.0),
                            threshold_km=float(thr_m / 1000.0),
                        )
                        st.session_state[f"{key_prefix}_last"] = report
                    except Exception as exc:
                        st.error(f"STK 验证失败：{exc}")

        report = st.session_state.get(f"{key_prefix}_last")
        if report is not None:
            _render_report(report, key_prefix)
        else:
            st.caption(
                "尚未运行验证。点击上方按钮后会把结果追加到 "
                "`data/validation/stk_validation.json`，并在『算法验证文档』里同步显示。"
            )


def render_six_dof_validation_panel(
    sim_result: Any,
    *,
    key_prefix: str = "stk_6dof",
) -> None:
    """嵌入"6-DOF / 数值积分 vs STK HPOP"对照面板（轨迹仿真页可调用）。"""
    st.markdown("##### STK HPOP 交叉验证（6-DOF / 入轨惯性段）")
    with st.container(border=True):
        avail = _availability_card(key_prefix)

        c1, c2, c3 = st.columns(3)
        dur_min = c1.number_input(
            "推演时长（分钟）", value=30.0,
            step=5.0, min_value=1.0, max_value=720.0,
            key=f"{key_prefix}_durmin",
        )
        step_s = c2.number_input(
            "采样步长（秒）", value=60.0,
            step=10.0, min_value=1.0, max_value=600.0,
            key=f"{key_prefix}_steps",
        )
        thr_km = c3.number_input(
            "通过阈值（位置 RMS, km）", value=5.0,
            step=0.5, min_value=0.001,
            key=f"{key_prefix}_thr",
            help="LEO 短期 HPOP-vs-J2 典型偏差 1–10 km；按需收紧阈值。",
        )
        col_a, col_b = st.columns(2)
        mass_kg = col_a.number_input(
            "卫星质量（kg）", value=1000.0, step=50.0, min_value=1.0,
            key=f"{key_prefix}_mass",
        )
        drag_area = col_b.number_input(
            "迎风面积（m²）", value=10.0, step=1.0, min_value=0.01,
            key=f"{key_prefix}_area",
        )

        variant = st.radio(
            "Candidate 算法变体",
            options=["baseline", "optimized", "egm4x4", "egm6", "egm8_msise", "compare-all"],
            index=5, horizontal=True,
            key=f"{key_prefix}_variant",
            help=(
                "baseline    = 当前 trajectory/six_dof.py (J2 + USSA-76)；"
                "optimized   = J2+J3+J4 + 月日扌动 + SRP + USSA-76（zonal 高阶提升）；"
                "egm4x4      = EGM96 球谐 4×4（含 sectorial+tesseral）+ 月日 + SRP + USSA-76；"
                "egm6        = EGM96 球谐 6×6 + 月日 + SRP + USSA-76（**6h RMS≈200m 当前最优**）；"
                "egm8_msise  = EGM96 球谐 8×8 + 月日 + SRP + NRLMSISE-00（短弧 30min 最佳，长弧依赖 F107/Ap 与 STK 是否一致）；"
                "compare-all = 五变体共享同一份 STK HPOP 真值，并行显示改善百分比。"
            ),
        )

        os_disable = bool(not avail.os_supported)
        btn_help = (
            "Ansys STK Engine 仅 Windows 可用；当前 OS 不支持，按钮已禁用。"
            if os_disable else
            ("将以 6-DOF 仿真末段 ECI 状态为初值，分别用 STK HPOP（首选）和"
             "本系统等价积分器向前推 N 分钟，对照 RMS / RIC 误差。")
        )
        run_clicked = st.button(
            "▶ 运行 STK HPOP 交叉验证",
            type="primary", disabled=os_disable,
            key=f"{key_prefix}_run", help=btn_help,
            use_container_width=True,
        )

        if run_clicked:
            with st.spinner("正在调用 STK HPOP / 参考积分并计算 RMS 误差…"):
                try:
                    from stk_validation import run_six_dof_validation
                    all_variants = ("baseline", "optimized", "egm4x4", "egm6", "egm8_msise")
                    if variant in all_variants:
                        report = run_six_dof_validation(
                            sim_result,
                            duration_s=float(dur_min * 60.0),
                            step_s=float(step_s),
                            threshold_km=float(thr_km),
                            mass_kg=float(mass_kg),
                            drag_area_m2=float(drag_area),
                            algorithm_variant=variant,
                        )
                        st.session_state[f"{key_prefix}_last"] = report
                        st.session_state.pop(f"{key_prefix}_compare", None)
                    else:
                        compare = []
                        for v in all_variants:
                            compare.append(run_six_dof_validation(
                                sim_result,
                                duration_s=float(dur_min * 60.0),
                                step_s=float(step_s),
                                threshold_km=float(thr_km),
                                mass_kg=float(mass_kg),
                                drag_area_m2=float(drag_area),
                                algorithm_variant=v,
                            ))
                        st.session_state[f"{key_prefix}_compare"] = tuple(compare)
                        st.session_state[f"{key_prefix}_last"] = compare[-1]
                except Exception as exc:
                    st.error(f"STK HPOP 验证失败：{exc}")

        compare = st.session_state.get(f"{key_prefix}_compare")
        if compare is not None:
            import pandas as pd
            labels = ("baseline", "optimized", "egm4×4", "egm6", "egm8+MSISE")
            reps = compare
            st.markdown("**五变体算法升级对比（共享同一份 STK HPOP 真值）**")
            def _mk_row(label, attr):
                vals = [getattr(r, attr) * 1000 for r in reps]
                base = vals[0]
                row = {"误差项": label}
                for name, v in zip(labels, vals):
                    row[f"{name} (m)"] = round(v, 2)
                    if name != "baseline":
                        row[f"{name} 改善%"] = round(100 * (1 - v / max(base, 1e-9)), 1)
                return row
            comp_df = pd.DataFrame([
                _mk_row("位置 RMS", "pos_rms_km"),
                _mk_row("Radial RMS", "radial_rms_km"),
                _mk_row("In-track RMS", "in_track_rms_km"),
                _mk_row("Cross-track RMS", "cross_track_rms_km"),
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.caption(
                "改善 % = (1 − variant / baseline) × 100。"
                "**EGM6** = EGM96 球谐 6×6（33 sectorial+tesseral 项），LEO 6h 通常 RMS≈200 m，**当前推荐档**。"
                "**EGM8+MSISE** = EGM96 8×8 + NRLMSISE-00 大气；短弧（≤30 min）通常优于 EGM6，"
                "长弧表现取决于 F107/Ap 与 STK SpaceWeather.spw 是否一致。"
            )

        report = st.session_state.get(f"{key_prefix}_last")
        if report is not None:
            st.markdown("**当前显示报告：** " + report.label)
            _render_report(report, key_prefix)
        else:
            st.caption(
                "尚未运行验证。运行后将把结果追加到 "
                "`data/validation/stk_validation.json`，并在『算法验证文档』中同步显示。"
            )
