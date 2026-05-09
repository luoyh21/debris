"""Continuous low-thrust avoidance burn — single-iteration SCP.

For electric / continuous-thrust propulsion the burn must be spread over
a window ``[t_start, t_TCA]``.  We use a *first-iteration* sequential
convex programming (SCP) approximation:

1. Linearise the state about the nominal trajectory (HCW STM, same
   sensitivity used in :mod:`avoidance.bplane`).

2. For a uniform thrust acceleration vector ``a`` (constant in LVLH),
   the equivalent total ΔV is ``Δv_total = a · T``  where ``T`` is the
   burn duration.

3. The optimal direction is given by the same B-plane SVD as the
   impulsive case, but the sensitivity is the *integrated* HCW input
   matrix:  ``B(T) = ∫₀ᵀ Φ_rv(τ) dτ`` (mean-lead).  For a uniform burn
   ending at TCA:

       B(T) ≈ (1/T) · ∫_{t_man}^{TCA} Φ_rv(TCA − τ) dτ

4. Magnitude obeys the rocket equation:
       Δv_total = Isp · g₀ · ln(m_init / m_final)
       a_max    = (T_thrust / m_avg)             (km/s²)

This is the "single-iteration SCP / SOCP-lite" from the brief — it
produces a feasible thrust profile in milliseconds and converges on
the convex relaxation in one shot for the linear-in-control limit.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List

import numpy as np

from .bplane import bplane_from_states, _propagate_pre_post
from .common import (
    AvoidanceSolution,
    ConjunctionInputs,
    TrajSample,
    foster_pc_isotropic,
    hcw_stm,
    lvlh_basis,
    mean_motion,
    MU_EARTH,
)


def _integrated_input_matrix_eci(inp: ConjunctionInputs,
                                  burn_duration_s: float,
                                  n_quad: int = 32) -> np.ndarray:
    """Numerically integrate Φ_rv(TCA−τ) for τ ∈ [t_TCA−T, t_TCA]."""
    T = burn_duration_s
    n = mean_motion(inp.r_pri_eci, inp.v_pri_eci)
    R = lvlh_basis(inp.r_pri_eci, inp.v_pri_eci)

    quad_t = np.linspace(0.0, T, n_quad)
    dt_q   = T / max(n_quad - 1, 1)
    accum_lvlh = np.zeros((3, 3))
    for tau in quad_t:
        # τ is time-since-burn-start; lead-time at this slice = T − τ
        Phi = hcw_stm(T - tau, n)
        accum_lvlh += Phi[:3, 3:]
    accum_lvlh *= dt_q                        # km / (km/s) units → km / (km/s²)·s = km / (km/s)
    # Now we apply a constant LVLH acceleration α (km/s²); position shift ≈ accum · α
    return R @ accum_lvlh @ R.T


def design_low_thrust_burn(inp: ConjunctionInputs,
                            *,
                            burn_duration_s: float = 600.0,
                            thrust_N: float = 1.0,
                            spacecraft_mass_kg: float = 4500.0,
                            isp_s: float = 3000.0,
                            n_traj_samples: int = 80,
                            ) -> AvoidanceSolution:
    """Design a continuous-thrust avoidance burn ending at TCA.

    Parameters
    ----------
    burn_duration_s     : duration of the constant-thrust segment
    thrust_N            : magnitude of thrust vector (Newtons)
    spacecraft_mass_kg  : initial wet mass
    isp_s               : specific impulse (default 3000 s ≈ Hall-effect)
    """
    # Integrated sensitivity (3×3 km / (km/s²))
    B_eci = _integrated_input_matrix_eci(inp, burn_duration_s)

    # Maximum LVLH acceleration available from this thruster
    a_max_kms2 = (thrust_N / spacecraft_mass_kg) / 1000.0   # km/s²

    h_pri = np.cross(inp.r_pri_eci, inp.v_pri_eci)
    bf    = bplane_from_states(inp.v_rel_eci, h_pri)
    P     = np.row_stack([bf.xi, bf.zeta])                 # 2×3

    M     = P @ B_eci                                       # 2×3, units km / (km/s²)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    a_dir = Vt[0]                                            # principal direction

    # Sign: align with desired miss-vector growth
    miss_vec_2d = bf.project(inp.r_rel_eci)
    if float(np.dot(M @ a_dir, miss_vec_2d)) < 0:
        a_dir = -a_dir

    a_vec = a_max_kms2 * a_dir
    dv_total = a_vec * burn_duration_s        # km/s, ECI

    # Predicted post-burn miss
    new_miss_2d = miss_vec_2d + (M @ a_vec)   # M is km / (km/s²); a_vec is km/s²; result km
    miss_after  = float(np.linalg.norm(new_miss_2d))
    miss_before = float(np.linalg.norm(miss_vec_2d))

    pc_before = foster_pc_isotropic(miss_before, inp.sigma_combined_km, inp.hbr_km)
    pc_after  = foster_pc_isotropic(miss_after,  inp.sigma_combined_km, inp.hbr_km)

    # Propellant via Tsiolkovsky
    ve = isp_s * 9.80665 / 1000.0
    propellant = float(spacecraft_mass_kg * (1.0 - np.exp(-np.linalg.norm(dv_total) / ve)))

    # Build constant thrust profile (N points, all equal direction)
    n_prof = 32
    profile = np.tile(a_vec, (n_prof, 1))     # (n,3) km/s²

    notes = [
        f"持续推力幅值: {thrust_N:.2f} N  →  a = {a_max_kms2*1e6:.3f} mm/s²",
        f"等效总 ΔV: {np.linalg.norm(dv_total)*1000:.2f} m/s",
        f"推进剂消耗: {propellant:.3f} kg  (Isp = {isp_s:.0f} s)",
        f"主奇异值（积分敏感度）: {S[0]:.4e} km / (km/s²)",
        f"σ₂/σ₁ ≈ {S[1]/S[0]:.3f}",
        "实现：单次迭代 SCP / 线性 SOCP — 已锁定推力方向后求最大允许 |Δv|；"
        "若需多目标 / 多碎片同时规避，可把 ‖Δv‖ ≤ Δv_max 改为 SOCP 约束并迭代。",
    ]

    nom, mod = _propagate_pre_post(inp, dv_total, burn_duration_s, n_traj_samples)
    return AvoidanceSolution(
        method            = "持续小推力 SCP 单次迭代",
        dv_vec_kms        = dv_total,
        dv_mag_kms        = float(np.linalg.norm(dv_total)),
        burn_start_met_s  = inp.t_tca_met_s - burn_duration_s,
        burn_duration_s   = float(burn_duration_s),
        thrust_profile    = profile,
        miss_before_km    = miss_before,
        miss_after_km     = miss_after,
        pc_before         = pc_before,
        pc_after          = pc_after,
        propellant_kg     = propellant,
        notes             = notes,
        nominal_traj      = nom,
        modified_traj     = mod,
    )
