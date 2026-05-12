"""HPOP 消融实验：缓存 STK ref，对 candidate 各项力学（J2/J3/J4/月日/SRP）单独开启。

跑一次 STK HPOP（很慢），缓存到 .npz；之后逐项打开 candidate 力学，看哪一项真正改善 RMS。
"""
from __future__ import annotations
import io, sys, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np

from stk_validation.runner import _coast_propagate_j2
from stk_validation.comparison import compute_rms_errors

CACHE = ROOT / "data" / "validation" / "_stk_hpop_cache_408km.npz"

MU = 398600.4418
R_E = 6378.137


def _make_state(alt_km=408.0, inc_deg=51.6):
    r = R_E + alt_km
    inc = math.radians(inc_deg)
    pos = [r, 0.0, 0.0]
    v = math.sqrt(MU / r)
    vel = [0.0, v * math.cos(inc), v * math.sin(inc)]
    return pos + vel, datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def get_or_build_ref(t_offsets, state, epoch, force=False):
    if CACHE.exists() and not force:
        d = np.load(CACHE)
        if (np.array_equal(d["t_offsets"], np.asarray(t_offsets))
            and np.allclose(d["state"], state)
            and d["epoch_iso"].item() == epoch.isoformat()):
            return d["pos"].tolist(), d["vel"].tolist()
    print(f"  cache miss → 生成 STK HPOP 真值（一次性）…")
    from stk_validation.stk_adapter import _open_stk, shutdown_stk, stk_propagate_hpop
    s = _open_stk()
    if s is None:
        raise RuntimeError("STK 不可用")
    try:
        ref = stk_propagate_hpop(state, epoch, t_offsets,
                                 mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
                                 gravity_degree=21, session=s)
    finally:
        shutdown_stk(s)
    if ref is None:
        raise RuntimeError("STK HPOP 失败")
    pos, vel = ref
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE,
             pos=np.array(pos), vel=np.array(vel),
             t_offsets=np.asarray(t_offsets, dtype=float),
             state=np.asarray(state, dtype=float),
             epoch_iso=np.array(epoch.isoformat()))
    print(f"  ref 已缓存到 {CACHE.name}")
    return pos, vel


def measure(name, pos, vel, ref_pos, ref_vel, t_offsets, epoch):
    rep = compute_rms_errors(
        t_offsets, ref_pos, ref_vel, pos, vel,
        label=name, reference="STK HPOP", candidate=name,
        threshold_km=5.0, epoch_utc=epoch, keep_samples=0,
    )
    print(f"  {name:50s}  pos RMS = {rep.pos_rms_km*1000:8.2f} m  "
          f"R={rep.radial_rms_km*1000:7.1f}  I={rep.in_track_rms_km*1000:7.1f}  "
          f"C={rep.cross_track_rms_km*1000:7.1f}")
    return rep


def main():
    state, epoch = _make_state(408.0, 51.6)
    pos0 = np.array(state[:3]); vel0 = np.array(state[3:])
    duration_s = 21600.0
    step_s = 30.0
    t_offsets = [k * step_s for k in range(int(duration_s // step_s) + 1)]
    print(f"场景：ISS-like LEO 408 km，inc 51.6°，6 h 时段，{step_s:.0f}s 步长")
    print(f"采样点 = {len(t_offsets)}")
    print()

    print("[1/N] 准备 STK HPOP ref")
    ref_pos, ref_vel = get_or_build_ref(t_offsets, state, epoch)
    print()

    print("[消融] 各组合 → STK HPOP")
    cases = [
        ("J2 + USSA76 (baseline)",                  dict()),
        ("J2 + J3 + USSA76",                        dict(use_j3=True)),
        ("J2 + J3 + J4 + USSA76",                   dict(use_j3=True, use_j4=True)),
        ("J2 + 月日 + USSA76",                       dict(use_third_body=True)),
        ("J2 + SRP + USSA76",                       dict(use_srp=True)),
        ("J2 + J3 + J4 + 月日 + USSA76",             dict(use_j3=True, use_j4=True,
                                                          use_third_body=True)),
        ("J2 + J3 + J4 + 月日 + SRP + USSA76 (all)", dict(use_j3=True, use_j4=True,
                                                          use_third_body=True, use_srp=True)),
        ("EGM96 4×4 + USSA76",                      dict(use_egm_n=4)),
        ("EGM96 4×4 + 月日 + SRP + USSA76",         dict(use_egm_n=4, use_third_body=True, use_srp=True)),
        ("EGM96 6×6 + 月日 + SRP + USSA76 (NEW)",   dict(use_egm_n=6, use_third_body=True, use_srp=True)),
        ("EGM96 8×8 + 月日 + SRP + USSA76 (NEW)",   dict(use_egm_n=8, use_third_body=True, use_srp=True)),
        ("EGM96 8×8 + 月日 + SRP + NRLMSISE(F107=165)", dict(use_egm_n=8, use_third_body=True, use_srp=True,
                                                              use_nrlmsise=True)),
        ("EGM96 8×8 + 月日 + SRP + NRLMSISE(F107=70)",  dict(use_egm_n=8, use_third_body=True, use_srp=True,
                                                              use_nrlmsise=True,
                                                              nrlmsise_f107a=70.0, nrlmsise_f107=70.0,
                                                              nrlmsise_ap=4.0)),
        ("EGM96 6×6 + 月日 + SRP + NRLMSISE(F107=70)",  dict(use_egm_n=6, use_third_body=True, use_srp=True,
                                                              use_nrlmsise=True,
                                                              nrlmsise_f107a=70.0, nrlmsise_f107=70.0,
                                                              nrlmsise_ap=4.0)),
    ]
    for name, kwargs in cases:
        cand_pos, cand_vel = _coast_propagate_j2(
            pos0, vel0, t_offsets, epoch=epoch,
            mass_kg=1000.0, drag_area_m2=10.0, drag_cd=2.2,
            **kwargs,
        )
        measure(name, cand_pos, cand_vel, ref_pos, ref_vel, t_offsets, epoch)


if __name__ == "__main__":
    main()
