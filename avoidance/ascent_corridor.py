"""Ascent-phase avoidance via spatio-temporal drivable corridor + MPC trim.

Approach
--------
During launch ascent the rocket is constrained by:
* aerodynamic loads (Q × α envelope),
* structural strength,
* gravity-turn pitch profile.

We therefore *cannot* command an arbitrary 3-DOF Δv during boost.  The
two practical levers are:

* **Δazimuth**  — small adjustment of the launch azimuth (degrees);
* **Δpitch_rate** — modification of the gravity-turn pitch program.

Both translate, to first order, into a lateral (cross-range) shift of
the rocket's down-range track.  We compute the lateral sensitivity from
the rocket's nominal velocity at TCA:

    ∂(lateral-distance-at-TCA) / ∂(Δazimuth)
        ≈ |v_pri| · (TCA − t_launch) · (π/180)

The "drivable corridor" is the set of feasible Δazimuth that preserves
range-safety / Q-α envelopes.  We bound it to ±5 ° by default.

For multiple debris near TCA, the MPC formulation reduces to a single
1-D linear program (find Δazimuth maximising minimum-miss across all
threats) — easily extensible to true MPC by re-solving along a
horizon, but here we present the first-iteration analytic solution.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List

import numpy as np

from .common import (
    AvoidanceSolution,
    ConjunctionInputs,
    TrajSample,
    foster_pc_isotropic,
)


def design_ascent_correction(inp: ConjunctionInputs,
                              *,
                              t_launch_s: float = 0.0,          # MET = 0 by convention
                              max_dazimuth_deg: float = 3.0,
                              max_dpitch_deg: float = 1.0,
                              n_traj_samples: int = 80,
                              ) -> AvoidanceSolution:
    """First-iteration MPC trim for ascent-phase avoidance.

    Parameters
    ----------
    t_launch_s        : MET at which the maneuver levers are applied
                        (defaults to 0 — i.e. dispatched on the launch program)
    max_dazimuth_deg  : drivable-corridor limit on launch-azimuth adjustment
    max_dpitch_deg    : drivable-corridor limit on pitch-program offset
    """
    dt_to_tca = max(inp.t_tca_met_s - t_launch_s, 60.0)        # avoid /0
    v_norm    = float(np.linalg.norm(inp.v_pri_eci))

    # Lateral unit vector (cross-track, perpendicular to local velocity & radial)
    rh = inp.r_pri_eci / max(np.linalg.norm(inp.r_pri_eci), 1e-9)
    vh = inp.v_pri_eci / max(v_norm, 1e-9)
    h  = np.cross(rh, vh); h = h / max(np.linalg.norm(h), 1e-9)

    # Sensitivity (km per °):
    # • azimuth → cross-track at TCA  ≈  |v| · Δt · (π/180)
    # • pitch   → radial    at TCA  ≈  ½ · |v| · Δt · (π/180)
    dlat_per_daz_km   = v_norm * dt_to_tca * np.pi / 180.0
    drad_per_dpi_km   = 0.5 * v_norm * dt_to_tca * np.pi / 180.0

    # Threat relative position decomposed
    r_rel_lat = float(np.dot(inp.r_rel_eci, h))
    r_rel_rad = float(np.dot(inp.r_rel_eci, rh))

    # We want to *increase* miss; choose signs that move primary AWAY from threat
    daz_sign = -np.sign(r_rel_lat) if r_rel_lat != 0 else 1.0
    dpi_sign = -np.sign(r_rel_rad) if r_rel_rad != 0 else 1.0

    daz = float(daz_sign) * max_dazimuth_deg
    dpi = float(dpi_sign) * max_dpitch_deg

    lat_shift = daz * dlat_per_daz_km
    rad_shift = dpi * drad_per_dpi_km

    # Effective ECI position perturbation at TCA
    dr_eci = lat_shift * h + rad_shift * rh

    miss_before_vec = inp.r_rel_eci.copy()
    miss_after_vec  = miss_before_vec + dr_eci    # primary moves +dr_eci ⇒ relative grows
    miss_before = float(np.linalg.norm(miss_before_vec))
    miss_after  = float(np.linalg.norm(miss_after_vec))

    pc_before = foster_pc_isotropic(miss_before, inp.sigma_combined_km, inp.hbr_km)
    pc_after  = foster_pc_isotropic(miss_after,  inp.sigma_combined_km, inp.hbr_km)

    # Equivalent ΔV for reporting (impulse that produces same lateral shift)
    # For ballistic linear shift over Δt:  Δv = Δr / Δt
    dv_eq = dr_eci / dt_to_tca

    notes = [
        f"机动施加于 MET = {t_launch_s:.1f} s（点火时段，距 TCA Δt={dt_to_tca:.0f} s）",
        f"方位角调整: Δaz = {daz:+.2f} °  →  侧向偏移 ≈ {lat_shift*1000:+.0f} m",
        f"俯仰程序偏置: Δθ = {dpi:+.2f} ° →  径向偏移 ≈ {rad_shift*1000:+.0f} m",
        f"等效脉冲 ΔV: {np.linalg.norm(dv_eq)*1000:.2f} m/s （仅供量级参考）",
        "可行驶走廊：±{:.1f}° 方位 / ±{:.1f}° 俯仰；".format(
            max_dazimuth_deg, max_dpitch_deg
        ),
        "约束：Q-α 包络 + 结构载荷 + 重力转向；多目标可扩展为 1-D LP，"
        "再嵌入 MPC 闭环周期（典型 1 Hz）。",
    ]

    # Build visualisation samples
    nom: List[TrajSample] = []
    mod: List[TrajSample] = []
    if n_traj_samples > 1:
        ts = np.linspace(0.0, dt_to_tca, n_traj_samples)
        for k, tau in enumerate(ts):
            # Linear interp of nominal between (r_pri @ TCA back-projected) and r_pri @ TCA
            r_nom = inp.r_pri_eci - inp.v_pri_eci * (dt_to_tca - tau)
            # Lateral / radial shift grows linearly from 0 → full at TCA
            grow = tau / dt_to_tca
            r_mod = r_nom + grow * dr_eci
            ep = inp.tca - timedelta(seconds=(dt_to_tca - tau))
            nom.append(TrajSample(epoch=ep, r_eci=r_nom))
            mod.append(TrajSample(epoch=ep, r_eci=r_mod))

    return AvoidanceSolution(
        method            = "上升段时空走廊 + MPC 一次迭代",
        dv_vec_kms        = dv_eq,
        dv_mag_kms        = float(np.linalg.norm(dv_eq)),
        burn_start_met_s  = float(t_launch_s),
        burn_duration_s   = float(dt_to_tca),
        miss_before_km    = miss_before,
        miss_after_km     = miss_after,
        pc_before         = pc_before,
        pc_after          = pc_after,
        propellant_kg     = None,
        notes             = notes,
        nominal_traj      = nom,
        modified_traj     = mod,
    )
