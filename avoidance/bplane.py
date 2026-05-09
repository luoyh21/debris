"""B-plane (encounter-plane) impulsive ΔV solver.

Theory
------
The B-plane is the plane perpendicular to the relative velocity ``v_rel``
at TCA.  Its in-plane axes (Öpik convention):

    η̂  = v_rel / |v_rel|                    (out-of-plane, normal to B-plane)
    ζ̂  = (v_rel × h_pri) / |…|              (B-plane "down-track")
    ξ̂  = ζ̂ × η̂                              (B-plane "cross-track")

The position offset projected onto (ξ̂, ζ̂) is the *miss distance vector*.

Sensitivity to an impulsive ΔV applied at ``Δt = TCA − t_man`` seconds
*before* TCA (relative motion modelled with the HCW state-transition
matrix Φ_rv(Δt) of the rocket's reference orbit):

    Δr_ECI(TCA)  ≈  R_lvlh→eci · Φ_rv(Δt) · R_lvlh→eciᵀ · Δv_ECI

We then project Δr onto the B-plane and seek

    max  ‖B_proj · Δv‖²        s.t.  ‖Δv‖ ≤ Δv_max

The optimum is the **principal eigenvector** of  B_projᵀ B_proj, scaled
to the budget.  This is the closed-form analytic solver referenced in
the user's brief ("Lagrange multipliers + B-plane").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import List

import numpy as np

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


@dataclass
class BPlaneFrame:
    """B-plane orthonormal axes in ECI."""
    eta:  np.ndarray   # out-of-plane (along v_rel)
    xi:   np.ndarray   # B-plane axis 1
    zeta: np.ndarray   # B-plane axis 2

    def project(self, v: np.ndarray) -> np.ndarray:
        """Return (xi, zeta) components of vector ``v``."""
        return np.array([float(np.dot(v, self.xi)),
                         float(np.dot(v, self.zeta))])


def bplane_from_states(v_rel_eci: np.ndarray,
                        h_pri_eci: np.ndarray) -> BPlaneFrame:
    """Build B-plane axes from relative velocity & primary's orbit normal."""
    eta = v_rel_eci / max(np.linalg.norm(v_rel_eci), 1e-12)
    z   = np.cross(eta, h_pri_eci)
    z_n = np.linalg.norm(z)
    if z_n < 1e-9:
        # fallback when v_rel is nearly parallel to h
        any_axis = np.array([0.0, 0.0, 1.0])
        z = np.cross(eta, any_axis)
        z_n = np.linalg.norm(z)
        if z_n < 1e-9:
            z = np.cross(eta, np.array([1.0, 0.0, 0.0]))
            z_n = np.linalg.norm(z)
    zeta = z / z_n
    xi   = np.cross(zeta, eta)
    xi   = xi / max(np.linalg.norm(xi), 1e-12)
    return BPlaneFrame(eta=eta, xi=xi, zeta=zeta)


def _sensitivity_eci(inp: ConjunctionInputs,
                     dt_lead_s: float) -> np.ndarray:
    """3×3 Jacobian d(Δr_ECI@TCA) / d(Δv_ECI@maneuver), HCW approximation."""
    n = mean_motion(inp.r_pri_eci, inp.v_pri_eci)
    R = lvlh_basis(inp.r_pri_eci, inp.v_pri_eci)        # LVLH→ECI
    Phi = hcw_stm(dt_lead_s, n)
    Phi_rv = Phi[:3, 3:]                                # 3×3 r←v block
    return R @ Phi_rv @ R.T


def optimal_impulsive_dv(inp: ConjunctionInputs,
                          *,
                          dv_budget_kms: float = 0.05,
                          maneuver_lead_s: float = 1800.0,
                          isp_s: float = 320.0,
                          dry_mass_kg: float = 4500.0,
                          n_traj_samples: int = 80,
                          ) -> AvoidanceSolution:
    """Solve the B-plane optimal impulsive maneuver.

    Parameters
    ----------
    dv_budget_kms      : maximum |Δv| allowed [km/s]
    maneuver_lead_s    : maneuver fires this many seconds **before** TCA
    isp_s, dry_mass_kg : if both >0, propellant cost is reported
    n_traj_samples     : visualisation samples for nominal vs modified
    """
    # ── frame and projection ───────────────────────────────────────────────
    h_pri = np.cross(inp.r_pri_eci, inp.v_pri_eci)
    bf    = bplane_from_states(inp.v_rel_eci, h_pri)

    J_eci = _sensitivity_eci(inp, dt_lead_s=maneuver_lead_s)   # 3×3
    # Projection onto B-plane (2×3)
    P = np.row_stack([bf.xi, bf.zeta])
    # Combined linear map  Δv_ECI → (Δξ, Δζ)
    M = P @ J_eci                                                # 2×3

    # Optimum direction: principal right-singular vector of M
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    u_opt = Vt[0]                                                # 1st right-sing vector
    dv = dv_budget_kms * u_opt

    # Sign: choose direction that *increases* miss distance (push along r_rel)
    miss_vec_2d = bf.project(inp.r_rel_eci)
    shift_2d    = M @ dv
    if float(np.dot(shift_2d, miss_vec_2d)) < 0:
        dv = -dv

    # Predicted post-maneuver miss
    new_miss_2d = miss_vec_2d + (M @ dv)
    miss_after  = float(np.linalg.norm(new_miss_2d))
    miss_before = float(np.linalg.norm(miss_vec_2d))

    # Pc (isotropic combined)
    pc_before = foster_pc_isotropic(miss_before, inp.sigma_combined_km, inp.hbr_km)
    pc_after  = foster_pc_isotropic(miss_after,  inp.sigma_combined_km, inp.hbr_km)

    # Tsiolkovsky propellant cost
    propellant = None
    if isp_s > 0 and dry_mass_kg > 0:
        ve   = isp_s * 9.80665 / 1000.0    # km/s
        prop = dry_mass_kg * (np.exp(np.linalg.norm(dv) / ve) - 1.0)
        propellant = float(prop)

    notes: List[str] = [
        f"B-plane principal singular value: {S[0]:.4e} km / (km/s)",
        f"σ₂/σ₁ ratio: {S[1] / S[0]:.3f} (≈0 → maneuver direction is highly preferred)",
        f"ΔV applied {maneuver_lead_s/60.0:.1f} min before TCA",
        f"Maneuver direction (ECI): [{dv[0]:+.4f}, {dv[1]:+.4f}, {dv[2]:+.4f}] km/s",
    ]

    # Trajectory samples for visualisation (Keplerian two-body, post-maneuver)
    nom, mod = _propagate_pre_post(inp, dv, maneuver_lead_s, n_traj_samples)

    return AvoidanceSolution(
        method            = "B-plane analytic impulsive",
        dv_vec_kms        = dv,
        dv_mag_kms        = float(np.linalg.norm(dv)),
        burn_start_met_s  = inp.t_tca_met_s - maneuver_lead_s,
        burn_duration_s   = 0.0,
        miss_before_km    = miss_before,
        miss_after_km     = miss_after,
        pc_before         = pc_before,
        pc_after          = pc_after,
        propellant_kg     = propellant,
        notes             = notes,
        nominal_traj      = nom,
        modified_traj     = mod,
    )


# ─── two-body propagation for visualisation ──────────────────────────────────

def _kepler_step(r: np.ndarray, v: np.ndarray, dt_s: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Universal-variable two-body propagation (single step).
    Robust enough for short visualisation horizons (≤ a few hours).
    """
    mu = MU_EARTH
    r0 = float(np.linalg.norm(r))
    v0 = float(np.linalg.norm(v))
    if r0 < 1.0:
        return r.copy(), v.copy()

    alpha = 2.0 / r0 - v0 * v0 / mu                     # 1/a
    sqrt_mu = np.sqrt(mu)

    # initial χ guess
    if alpha > 1e-9:
        chi = sqrt_mu * dt_s * alpha
    elif abs(alpha) < 1e-9:
        h = np.cross(r, v)
        p = float(np.dot(h, h)) / mu
        s = 0.5 * (np.pi / 2.0 - np.arctan(3.0 * np.sqrt(mu / (p ** 3)) * dt_s))
        w = np.arctan(np.tan(s) ** (1.0 / 3.0))
        chi = np.sqrt(p) * 2.0 / np.tan(2.0 * w)
    else:
        a = 1.0 / alpha
        chi = (np.sign(dt_s) * np.sqrt(-a) *
               np.log(-2.0 * mu * alpha * dt_s /
                      (np.dot(r, v) + np.sign(dt_s) * np.sqrt(-mu * a) * (1.0 - r0 * alpha))))

    # Newton iteration
    for _ in range(60):
        psi = chi * chi * alpha
        c2, c3 = _stumpff(psi)
        rdotv = float(np.dot(r, v))
        r_new = (chi * chi * c2 +
                 rdotv / sqrt_mu * chi * (1.0 - psi * c3) +
                 r0 * (1.0 - psi * c2))
        delta = (sqrt_mu * dt_s -
                 chi * chi * chi * c3 -
                 rdotv / sqrt_mu * chi * chi * c2 -
                 r0 * chi * (1.0 - psi * c3))
        chi += delta / max(r_new, 1e-3)
        if abs(delta) < 1e-9:
            break

    psi = chi * chi * alpha
    c2, c3 = _stumpff(psi)
    f    = 1.0 - chi * chi * c2 / r0
    g    = dt_s - chi ** 3 * c3 / sqrt_mu
    rv   = f * r + g * v
    rmag = float(np.linalg.norm(rv))
    fdot = sqrt_mu / (r0 * rmag) * chi * (psi * c3 - 1.0)
    gdot = 1.0 - chi * chi * c2 / rmag
    vv   = fdot * r + gdot * v
    return rv, vv


def _stumpff(psi: float) -> tuple[float, float]:
    if psi > 1e-6:
        s = np.sqrt(psi)
        c2 = (1.0 - np.cos(s)) / psi
        c3 = (s - np.sin(s)) / np.sqrt(psi ** 3)
    elif psi < -1e-6:
        s = np.sqrt(-psi)
        c2 = (1.0 - np.cosh(s)) / psi
        c3 = (np.sinh(s) - s) / np.sqrt((-psi) ** 3)
    else:
        c2, c3 = 0.5, 1.0 / 6.0
    return float(c2), float(c3)


def _propagate_pre_post(inp: ConjunctionInputs,
                        dv: np.ndarray,
                        maneuver_lead_s: float,
                        n_samples: int,
                        post_window_s: float | None = None,
                        ) -> tuple[List[TrajSample], List[TrajSample]]:
    """Two-body propagate the rocket from t_man over ±maneuver_lead window."""
    # Reference state: at maneuver epoch (back-propagate from r_pri/v_pri)
    if post_window_s is None:
        post_window_s = maneuver_lead_s
    t_total = maneuver_lead_s + post_window_s
    dt = t_total / max(n_samples - 1, 1)

    # Back-propagate r_pri/v_pri (which were given at TCA) to maneuver epoch
    r_man, v_man = _kepler_step(inp.r_pri_eci, inp.v_pri_eci, -maneuver_lead_s)
    r_n, v_n = r_man.copy(), v_man.copy()
    r_m, v_m = r_man.copy(), v_man.copy() + dv

    nom: List[TrajSample] = []
    mod: List[TrajSample] = []
    t0 = inp.tca - timedelta(seconds=maneuver_lead_s)
    for k in range(n_samples):
        nom.append(TrajSample(epoch=t0 + timedelta(seconds=k * dt),
                              r_eci=r_n.copy()))
        mod.append(TrajSample(epoch=t0 + timedelta(seconds=k * dt),
                              r_eci=r_m.copy()))
        if k == n_samples - 1:
            break
        r_n, v_n = _kepler_step(r_n, v_n, dt)
        r_m, v_m = _kepler_step(r_m, v_m, dt)
    return nom, mod
