"""Shared dataclasses & helpers for avoidance solvers.

Inputs are intentionally lean and match what the LCOLA / collision-risk
modules already produce (``ConjunctionEvent`` + a ``SimResult``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import numpy as np

MU_EARTH = 398600.4418      # km^3/s^2
R_EARTH  = 6378.1366        # km


@dataclass
class ConjunctionInputs:
    """Self-contained snapshot of a conjunction.

    These are the only fields the solvers need.  They are derived once
    from a ConjunctionEvent + the rocket's nominal trajectory at TCA.
    """
    tca:               datetime
    t_tca_met_s:       float           # mission elapsed time at TCA [s]
    r_pri_eci:         np.ndarray      # (3,) primary (rocket) ECI position [km]
    v_pri_eci:         np.ndarray      # (3,) primary ECI velocity         [km/s]
    r_rel_eci:         np.ndarray      # (3,) primary − threat at TCA      [km]
    v_rel_eci:         np.ndarray      # (3,) primary − threat at TCA      [km/s]
    sigma_combined_km: float = 0.5     # 1-σ position uncertainty (combined)
    hbr_km:            float = 0.020   # hard-body radius (combined)
    threat_norad:      Optional[int] = None
    threat_name:       str = "(unknown)"

    @property
    def miss_distance_km(self) -> float:
        return float(np.linalg.norm(self.r_rel_eci))

    @property
    def v_rel_kms(self) -> float:
        return float(np.linalg.norm(self.v_rel_eci))


@dataclass
class TrajSample:
    """Lightweight (epoch, ECI position) sample for visualisation."""
    epoch:  datetime
    r_eci:  np.ndarray   # (3,) km


@dataclass
class AvoidanceSolution:
    """Common output for any of the three solvers."""
    method:                 str           # "B-plane impulsive" / "Low-thrust SCP" / "Ascent corridor"
    dv_vec_kms:             np.ndarray    # (3,) total ΔV in ECI [km/s]
    dv_mag_kms:             float
    burn_start_met_s:       float         # mission elapsed time at maneuver start
    burn_duration_s:        float         # 0 for impulsive
    thrust_profile:         Optional[np.ndarray] = None    # (N,3) for low-thrust
    miss_before_km:         float = 0.0
    miss_after_km:          float = 0.0
    pc_before:              float = 0.0
    pc_after:               float = 0.0
    propellant_kg:          Optional[float] = None         # if Isp+mass given
    notes:                  List[str] = field(default_factory=list)
    nominal_traj:           List[TrajSample] = field(default_factory=list)
    modified_traj:          List[TrajSample] = field(default_factory=list)

    @property
    def reduction_factor(self) -> float:
        """Pc reduction (>1 means safer)."""
        if self.pc_after <= 0:
            return float("inf")
        return self.pc_before / self.pc_after


# ─── helpers ─────────────────────────────────────────────────────────────────

def _interp_eci_state(nominal, t_target_met_s: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Linear-interp the rocket's ECI state at a given MET.  Returns (r,v) in km."""
    if not nominal:
        raise ValueError("Empty nominal trajectory")

    ts = np.array([p.t_met_s for p in nominal])
    if t_target_met_s <= ts[0]:
        return nominal[0].pos_eci.copy(), nominal[0].vel_eci.copy()
    if t_target_met_s >= ts[-1]:
        return nominal[-1].pos_eci.copy(), nominal[-1].vel_eci.copy()
    j = int(np.searchsorted(ts, t_target_met_s))
    j = max(1, min(j, len(ts) - 1))
    p0, p1 = nominal[j - 1], nominal[j]
    a = (t_target_met_s - p0.t_met_s) / (p1.t_met_s - p0.t_met_s)
    r = (1 - a) * p0.pos_eci + a * p1.pos_eci
    v = (1 - a) * p0.vel_eci + a * p1.vel_eci
    return r.copy(), v.copy()


def inputs_from_event(event, sim_result,
                       *, sigma_combined_km: float = 0.5,
                       hbr_km: float = 0.020) -> ConjunctionInputs:
    """Build :class:`ConjunctionInputs` from a ``ConjunctionEvent`` + ``SimResult``.

    Because the catalog ConjunctionEvent only stores scalar miss distance
    & |v_rel|, we synthesise a worst-case relative geometry that is
    physically self-consistent:
      * use the rocket's ECI state at TCA (interpolated from ``sim_result.nominal``);
      * place the threat exactly ``miss_distance_km`` away from the rocket
        in the radial direction (worst case for B-plane projection –
        maximises the radial component);
      * align ``v_rel`` along the rocket's anti-velocity direction with the
        magnitude given by ``event.v_rel_kms`` (head-on relative motion).

    This produces deterministic but conservative geometry; the solvers
    use these vectors only to derive directions / sensitivities.
    """
    tca: datetime = event.tca
    t0 = sim_result.config.launch_utc
    t_met = (tca - t0).total_seconds()

    r_pri, v_pri = _interp_eci_state(sim_result.nominal, t_met)
    if np.linalg.norm(v_pri) < 1e-9:
        v_pri = np.array([1.0, 0.0, 0.0])

    # radial direction (away from Earth centre) — worst case position offset
    r_hat = r_pri / max(np.linalg.norm(r_pri), 1e-9)
    # anti-velocity direction — head-on relative velocity
    vhat  = v_pri / max(np.linalg.norm(v_pri), 1e-9)

    miss_km = max(float(getattr(event, "miss_distance_km", 0.0)), 0.001)
    vrel_kms = max(float(getattr(event, "v_rel_kms", 0.0)), 0.001)

    r_rel = miss_km * r_hat
    v_rel = -vrel_kms * vhat

    return ConjunctionInputs(
        tca=tca,
        t_tca_met_s=t_met,
        r_pri_eci=r_pri,
        v_pri_eci=v_pri,
        r_rel_eci=r_rel,
        v_rel_eci=v_rel,
        sigma_combined_km=sigma_combined_km,
        hbr_km=hbr_km,
        threat_norad=getattr(event, "norad_cat_id", None),
        threat_name=str(getattr(event, "object_name", "(unknown)")),
    )


# ─── analytic foster-style Pc with isotropic 1-σ + HBR (used for after/before)

def foster_pc_isotropic(miss_km: float, sigma_km: float, hbr_km: float) -> float:
    """Closed-form Foster Pc for isotropic combined covariance.

    Pc = 1 - exp(-HBR² / (2σ²)) when miss << σ
    Generalisation:  Pc ≈ HBR²/(2σ²) · exp(-miss²/(2σ²))   (small-HBR limit)
    """
    sigma = max(sigma_km, 1e-6)
    hbr   = max(hbr_km,   1e-6)
    pc = (hbr * hbr) / (2.0 * sigma * sigma) * np.exp(-(miss_km * miss_km) / (2.0 * sigma * sigma))
    return float(min(max(pc, 0.0), 1.0))


# ─── Hill-Clohessy-Wiltshire STM (for impulsive sensitivity) ─────────────────

def hcw_stm(dt_s: float, n_rad_s: float) -> np.ndarray:
    """6×6 HCW state-transition matrix in LVLH frame for circular orbit.

    n_rad_s : mean motion of reference orbit [rad/s]
    """
    n  = n_rad_s
    s  = np.sin(n * dt_s)
    c  = np.cos(n * dt_s)
    Φrr = np.array([[4 - 3*c,        0, 0],
                     [6*(s - n*dt_s), 1, 0],
                     [0,              0, c]])
    Φrv = np.array([[s/n,            2*(1 - c)/n,                 0],
                     [-2*(1 - c)/n,   (4*s - 3*n*dt_s)/n,           0],
                     [0,              0,                            s/n]])
    Φvr = np.array([[3*n*s,          0, 0],
                     [-6*n*(1 - c),   0, 0],
                     [0,              0, -n*s]])
    Φvv = np.array([[c,             2*s,             0],
                     [-2*s,           4*c - 3,          0],
                     [0,             0,                c]])
    Φ = np.zeros((6, 6))
    Φ[:3, :3] = Φrr
    Φ[:3, 3:] = Φrv
    Φ[3:, :3] = Φvr
    Φ[3:, 3:] = Φvv
    return Φ


def lvlh_basis(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """Return 3×3 rotation matrix R_lvlh→eci (columns are LVLH unit axes in ECI).

    LVLH:
      x̂ = +R̂  (radial out)
      ŷ = +V̂_in-track (≈ velocity direction projected ⟂ R̂)
      ẑ = +Ĥ  (orbit normal)
    """
    rh = r_eci / max(np.linalg.norm(r_eci), 1e-9)
    h  = np.cross(r_eci, v_eci)
    nh = h  / max(np.linalg.norm(h),  1e-9)
    yh = np.cross(nh, rh)
    yh = yh / max(np.linalg.norm(yh), 1e-9)
    return np.column_stack([rh, yh, nh])


def mean_motion(r_eci: np.ndarray, v_eci: np.ndarray) -> float:
    """Mean motion of the (assumed circular) reference orbit at this state."""
    a = 1.0 / (2.0 / np.linalg.norm(r_eci) - np.dot(v_eci, v_eci) / MU_EARTH)
    a = max(a, R_EARTH + 50.0)
    return float(np.sqrt(MU_EARTH / (a ** 3)))
