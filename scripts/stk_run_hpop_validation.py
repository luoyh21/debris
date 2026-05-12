"""跑 6-DOF / 数值积分 vs STK HPOP 实际验证（baseline 与 optimized 对比）。

- baseline   : 与 trajectory/six_dof.py 当前实现一致 (J2 + USSA-76)
- optimized : 按 STK 验证文档"建议修正"提升后 (J2+J3+J4 + 月日扌动 + SRP + USSA-76)

两次都用同一个 STK HPOP（EGM2008 21x21 + NRLMSISE2000 + 月日 + SRP）作真值，
报告 baseline / optimized 各自的 RIC RMS，并把两次结果一起写到验证 JSON，
供文档前端展示"算法升级前后效果"。
"""
from __future__ import annotations

import io, sys, math
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from stk_validation.comparison import compute_rms_errors
from stk_validation.report import save_report
from stk_validation.stk_adapter import _open_stk, shutdown_stk, stk_propagate_hpop
from stk_validation.runner import _coast_propagate_j2

MU_KM3S2 = 398600.4418


def _make_initial_state(alt_km: float = 408.0,
                        inc_deg: float = 51.6) -> tuple[list, datetime]:
    R_E = 6378.137
    r = R_E + alt_km
    inc = math.radians(inc_deg)
    pos = [r, 0.0, 0.0]
    v_circ = math.sqrt(MU_KM3S2 / r)
    vel = [0.0, v_circ * math.cos(inc), v_circ * math.sin(inc)]
    epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return pos + vel, epoch


def _print_report(tag: str, report) -> None:
    print(f"\n── {tag} ──")
    print(f"  Radial      RMS = {report.radial_rms_km*1000:9.2f} m")
    print(f"  In-track    RMS = {report.in_track_rms_km*1000:9.2f} m")
    print(f"  Cross-track RMS = {report.cross_track_rms_km*1000:9.2f} m")
    print(f"  位置        RMS = {report.pos_rms_km*1000:9.2f} m  (max = {report.pos_max_km*1000:.2f} m)")
    print(f"  速度        RMS = {report.vel_rms_kms*1000:9.4f} m/s")
    print(f"  passed = {report.passed}")
    if report.notes:
        print("  diagnostics:")
        for n in report.notes:
            print(f"    · {n}")


def _make_report(
    t_offsets, ref_pos, ref_vel, cand_pos, cand_vel,
    *, label, candidate_label, reference_label, epoch, extra=None,
):
    return compute_rms_errors(
        t_offsets, ref_pos, ref_vel, cand_pos, cand_vel,
        label=label,
        reference=reference_label,
        candidate=candidate_label,
        threshold_km=5.0,
        epoch_utc=epoch,
        keep_samples=200,
        extra=extra or {},
    )


def _run_one_scenario(state6, epoch, duration_s: float, step_s: float, scenario_tag: str,
                      session=None):
    pos0 = np.array(state6[:3]); vel0 = np.array(state6[3:])
    t_offsets = [k * step_s for k in range(int(duration_s // step_s) + 1)]
    print(f"\n=== 场景：{scenario_tag}  duration={duration_s/60:.0f}min  step={step_s:.0f}s  N={len(t_offsets)} ===")

    print("[1/6] candidate-baseline (J2 + USSA-76)…")
    base_pos, base_vel = _coast_propagate_j2(
        pos0, vel0, t_offsets, epoch=epoch,
        mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
    )
    print(f"  完成，末端 |r| = {np.linalg.norm(base_pos[-1]):.2f} km")

    print("[2/6] candidate-optimized (J2+J3+J4 + 月日扌动 + SRP + USSA-76)…")
    opt_pos, opt_vel = _coast_propagate_j2(
        pos0, vel0, t_offsets, epoch=epoch,
        mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
        use_j3=True, use_j4=True,
        use_third_body=True, use_srp=True, srp_cr=1.5,
    )
    print(f"  完成，末端 |r| = {np.linalg.norm(opt_pos[-1]):.2f} km")

    print("[3/6] candidate-egm4x4 (EGM96 4×4 + 月日 + SRP + USSA-76)…")
    egm_pos, egm_vel = _coast_propagate_j2(
        pos0, vel0, t_offsets, epoch=epoch,
        mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
        use_egm_n=4, use_third_body=True, use_srp=True, srp_cr=1.5,
    )
    print(f"  完成，末端 |r| = {np.linalg.norm(egm_pos[-1]):.2f} km")

    print("[4/6] candidate-egm6 (EGM96 6×6 + 月日 + SRP + USSA-76)  ← 最优档")
    egm6_pos, egm6_vel = _coast_propagate_j2(
        pos0, vel0, t_offsets, epoch=epoch,
        mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
        use_egm_n=6, use_third_body=True, use_srp=True, srp_cr=1.5,
    )
    print(f"  完成，末端 |r| = {np.linalg.norm(egm6_pos[-1]):.2f} km")

    print("[5/6] candidate-egm8_msise (EGM96 8×8 + 月日 + SRP + NRLMSISE-00)…")
    egm8_pos, egm8_vel = _coast_propagate_j2(
        pos0, vel0, t_offsets, epoch=epoch,
        mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
        use_egm_n=8, use_third_body=True, use_srp=True, srp_cr=1.5,
        use_nrlmsise=True,
        nrlmsise_f107=165.0, nrlmsise_f107a=165.0, nrlmsise_ap=10.0,
    )
    print(f"  完成，末端 |r| = {np.linalg.norm(egm8_pos[-1]):.2f} km")

    print("[6/6] reference: STK HPOP (EGM2008 21x21 + NRLMSISE-00 + 月日 + SRP)…")
    own_session = False
    if session is None:
        session = _open_stk()
        own_session = True
    if session is None:
        print("  ✗ 无法启动 STK"); return None
    try:
        ref = stk_propagate_hpop(
            initial_state_eci=state6, epoch=epoch, t_offsets_s=t_offsets,
            mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
            gravity_degree=21, session=session,
        )
        if ref is None:
            print("  ✗ STK HPOP 推演失败"); return None
        ref_pos, ref_vel = ref
        print(f"  完成，末端 |r| = {np.linalg.norm(ref_pos[-1]):.2f} km")
    finally:
        if own_session:
            shutdown_stk(session)

    print(f"[ 计算 RIC 误差 ]")

    extra_common = {
        "scenario": scenario_tag,
        "duration_s": duration_s, "step_s": step_s,
        "initial_state_eci_km": pos0.tolist(),
        "initial_state_vel_kms": vel0.tolist(),
        "reference_force_model": "STK HPOP (EGM2008 21x21 + NRLMSISE-00 + Sun/Moon + SRP)",
    }
    base_report = _make_report(
        t_offsets, ref_pos, ref_vel, base_pos, base_vel,
        label=f"six_dof_vs_hpop_baseline_{int(duration_s/60):d}min",
        candidate_label="本系统 6-DOF baseline (J2 + USSA-76，与 trajectory/six_dof.py 一致)",
        reference_label="Ansys STK HPOP (EGM2008 21x21 + NRLMSISE-00 + 月日 + SRP)",
        epoch=epoch,
        extra={**extra_common, "algorithm_variant": "baseline",
               "candidate_force_model": "J2 + USSA-76"},
    )
    opt_report = _make_report(
        t_offsets, ref_pos, ref_vel, opt_pos, opt_vel,
        label=f"six_dof_vs_hpop_optimized_{int(duration_s/60):d}min",
        candidate_label="本系统 6-DOF optimized (J2+J3+J4 + 月日扌动 + SRP + USSA-76)",
        reference_label="Ansys STK HPOP (EGM2008 21x21 + NRLMSISE-00 + 月日 + SRP)",
        epoch=epoch,
        extra={**extra_common, "algorithm_variant": "optimized",
               "candidate_force_model":
                   "J2+J3+J4 + Sun/Moon point mass + SRP (Cr=1.5) + USSA-76"},
    )
    egm_report = _make_report(
        t_offsets, ref_pos, ref_vel, egm_pos, egm_vel,
        label=f"six_dof_vs_hpop_egm4x4_{int(duration_s/60):d}min",
        candidate_label="本系统 6-DOF EGM4×4 (EGM96 球谐 4×4 + 月日扌动 + SRP + USSA-76)",
        reference_label="Ansys STK HPOP (EGM2008 21x21 + NRLMSISE-00 + 月日 + SRP)",
        epoch=epoch,
        extra={**extra_common, "algorithm_variant": "egm4x4",
               "candidate_force_model":
                   "EGM96 球谐 4×4 + Sun/Moon + SRP + USSA-76"},
    )
    egm6_report = _make_report(
        t_offsets, ref_pos, ref_vel, egm6_pos, egm6_vel,
        label=f"six_dof_vs_hpop_egm6_{int(duration_s/60):d}min",
        candidate_label="本系统 6-DOF EGM6 (EGM96 球谐 6×6 + 月日扌动 + SRP + USSA-76)",
        reference_label="Ansys STK HPOP (EGM2008 21x21 + NRLMSISE-00 + 月日 + SRP)",
        epoch=epoch,
        extra={**extra_common, "algorithm_variant": "egm6",
               "candidate_force_model":
                   "EGM96 球谐 6×6 (33 项 sectorial+tesseral) + Sun/Moon + SRP + USSA-76"},
    )
    egm8_report = _make_report(
        t_offsets, ref_pos, ref_vel, egm8_pos, egm8_vel,
        label=f"six_dof_vs_hpop_egm8_msise_{int(duration_s/60):d}min",
        candidate_label="本系统 6-DOF EGM8+MSISE (EGM96 球谐 8×8 + 月日扌动 + SRP + NRLMSISE-00)",
        reference_label="Ansys STK HPOP (EGM2008 21x21 + NRLMSISE-00 + 月日 + SRP)",
        epoch=epoch,
        extra={**extra_common, "algorithm_variant": "egm8_msise",
               "candidate_force_model":
                   "EGM96 球谐 8×8 (60 项 sectorial+tesseral) + Sun/Moon + SRP + NRLMSISE-00 (F107=165, Ap=10)"},
    )

    def _imp(rep):
        return 100.0 * (1.0 - rep.pos_rms_km / max(base_report.pos_rms_km, 1e-9))
    imp_opt  = _imp(opt_report)
    imp_egm  = _imp(egm_report)
    imp_egm6 = _imp(egm6_report)
    imp_egm8 = _imp(egm8_report)

    for rep, imp in [(opt_report, imp_opt), (egm_report, imp_egm),
                     (egm6_report, imp_egm6), (egm8_report, imp_egm8)]:
        rep.extra.update({
            "improvement_vs_baseline_pos_rms_pct": round(imp, 2),
            "baseline_pos_rms_m": round(base_report.pos_rms_km * 1000, 3),
            "self_pos_rms_m":     round(rep.pos_rms_km * 1000, 3),
            "paired_baseline_label": base_report.label,
        })
    base_report.extra.update({
        "paired_optimized_label":  opt_report.label,
        "paired_egm4x4_label":     egm_report.label,
        "paired_egm6_label":       egm6_report.label,
        "paired_egm8_msise_label": egm8_report.label,
    })

    save_report(base_report); save_report(opt_report); save_report(egm_report)
    save_report(egm6_report); save_report(egm8_report)

    _print_report(f"{scenario_tag} BASELINE",   base_report)
    _print_report(f"{scenario_tag} OPTIMIZED",  opt_report)
    _print_report(f"{scenario_tag} EGM4×4",     egm_report)
    _print_report(f"{scenario_tag} EGM6",       egm6_report)
    _print_report(f"{scenario_tag} EGM8+MSISE", egm8_report)
    print(f"\n  {'变体':14s} {'位置 RMS':>12s} {'改善%':>8s}")
    print(f"  {'baseline':14s} {base_report.pos_rms_km*1000:9.2f} m  {'—':>8s}")
    print(f"  {'optimized':14s} {opt_report.pos_rms_km*1000:9.2f} m  {imp_opt:+7.1f}%")
    print(f"  {'egm4x4':14s} {egm_report.pos_rms_km*1000:9.2f} m  {imp_egm:+7.1f}%")
    print(f"  {'egm6':14s} {egm6_report.pos_rms_km*1000:9.2f} m  {imp_egm6:+7.1f}%")
    print(f"  {'egm8_msise':14s} {egm8_report.pos_rms_km*1000:9.2f} m  {imp_egm8:+7.1f}%")
    return base_report, opt_report, egm_report, egm6_report, egm8_report


def main() -> None:
    state6, epoch = _make_initial_state(alt_km=408.0, inc_deg=51.6)
    print(f"初值 ECI: pos={state6[:3]}, vel={state6[3:]}, epoch={epoch.isoformat()}")

    session = _open_stk()
    if session is None:
        print("无法启动 STK，退出"); return
    try:
        _run_one_scenario(state6, epoch, duration_s=1800.0, step_s=30.0,
                          scenario_tag="ISS-like LEO 408km / 30min", session=session)
        _run_one_scenario(state6, epoch, duration_s=21600.0, step_s=60.0,
                          scenario_tag="ISS-like LEO 408km / 6h", session=session)
    finally:
        shutdown_stk(session)


if __name__ == "__main__":
    main()
