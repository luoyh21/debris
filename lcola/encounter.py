"""Encounter geometry: TCA finding and encounter-plane projection.

Short-term encounter assumption (Alfriend & Akella 2000):
  At TCA the relative motion is locally linear → reduces 3D time-integral
  to a 2D static integral on the encounter plane.

Coordinate convention
---------------------
  Encounter plane (B-plane) basis:
    ê_ξ  = v_rel / |v_rel|              (along relative velocity)
    ê_η  = (r_rel × v_rel) / |…|        (out-of-plane)
    ê_ζ  = ê_ξ × ê_η                   (in-plane, perpendicular to v_rel)

  The miss vector projected on (ê_ζ, ê_η) defines (x_m, y_m).
  The 3×3 combined covariance projected on (ê_ζ, ê_η) gives Σ_2×2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, brentq


# ─── encounter result ─────────────────────────────────────────────────────────

@dataclass
class EncounterGeometry:
    """All quantities needed to compute Foster Pc."""
    tca_s:           float          # TCA [s] since some reference epoch
    miss_distance_km: float         # |r_rel| at TCA
    r_rel_km:        np.ndarray     # (3,) relative position at TCA [km]
    v_rel_kms:       np.ndarray     # (3,) relative velocity at TCA [km/s]
    miss_xy_km:      np.ndarray     # (2,) miss vector in encounter plane [km]
    cov_2x2:         np.ndarray     # (2,2) combined covariance in encounter plane
    T_enc:           np.ndarray     # (3,2) projection matrix (col = ê_ζ, ê_η)


# ─── interpolation helpers ────────────────────────────────────────────────────

def _interpolate_position(times: np.ndarray, positions: np.ndarray, t: float) -> np.ndarray:
    """Cubic-spline interpolation.  positions: (N,3) km."""
    cs = CubicSpline(times, positions, bc_type="not-a-knot")
    return cs(t)


def _build_interp(times: np.ndarray, states: np.ndarray):
    """Build cubic spline for positions from (N,) times and (N,3) states."""
    return CubicSpline(times, states, bc_type="not-a-knot")


# ─── TCA finder ───────────────────────────────────────────────────────────────

def find_tca(
    times1:  np.ndarray,    # (N,) seconds (monotonically increasing)
    pos1:    np.ndarray,    # (N,3) km  – object 1 positions
    times2:  np.ndarray,    # (M,) seconds
    pos2:    np.ndarray,    # (M,3) km  – object 2 positions
    t_start: Optional[float] = None,
    t_end:   Optional[float] = None,
    n_coarse: int = 200,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Find the Time of Closest Approach (TCA).

    Returns (tca_s, miss_km, r_rel_km, v_rel_kms).
    """
    t0 = max(times1[0], times2[0]) if t_start is None else t_start
    t1 = min(times1[-1], times2[-1]) if t_end is None else t_end
    if t0 >= t1:
        raise ValueError("Trajectories have no time overlap")

    cs1_p = _build_interp(times1, pos1)
    cs2_p = _build_interp(times2, pos2)

    def dist(t):
        dp = cs1_p(t) - cs2_p(t)
        return float(np.linalg.norm(dp))

    # Coarse scan
    t_coarse = np.linspace(t0, t1, n_coarse)
    d_coarse  = np.array([dist(tc) for tc in t_coarse])
    idx_min   = int(np.argmin(d_coarse))

    # Fine bracket around coarse minimum
    t_lo = t_coarse[max(0, idx_min - 1)]
    t_hi = t_coarse[min(n_coarse - 1, idx_min + 1)]

    result = minimize_scalar(dist, bounds=(t_lo, t_hi), method="bounded",
                             options={"xatol": 1e-3})   # 1 ms precision
    tca  = float(result.x)
    r1   = cs1_p(tca)
    r2   = cs2_p(tca)
    r_rel = r1 - r2

    # Relative velocity via finite difference
    eps   = 1.0    # 1-second FD
    v_rel = (cs1_p(tca + eps) - cs2_p(tca + eps) -
             (cs1_p(tca - eps) - cs2_p(tca - eps))) / (2 * eps)

    return tca, float(np.linalg.norm(r_rel)), r_rel, v_rel


# ─── encounter plane ──────────────────────────────────────────────────────────

def build_encounter_frame(v_rel: np.ndarray) -> np.ndarray:
    """
    Build orthonormal basis for encounter plane.

    Returns T (3×2): columns are ê_ζ and ê_η.
    ê_ξ = v_rel / |v_rel|  (along relative velocity – normal to plane)
    ê_η = cross(r_arb, ê_ξ) normalised
    ê_ζ = cross(ê_ξ, ê_η)
    """
    v_n = float(np.linalg.norm(v_rel))
    if v_n < 1e-12:
        raise ValueError("Relative velocity too small to define encounter plane")

    e_xi = v_rel / v_n

    # Arbitrary vector not parallel to e_xi
    arb = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(arb, e_xi))) > 0.9:
        arb = np.array([1.0, 0.0, 0.0])

    e_eta = np.cross(arb, e_xi)
    e_eta /= np.linalg.norm(e_eta)
    e_zeta = np.cross(e_xi, e_eta)
    e_zeta /= np.linalg.norm(e_zeta)

    T = np.column_stack([e_zeta, e_eta])   # (3,2)
    return T


def project_covariance(C1_3x3: np.ndarray, C2_3x3: np.ndarray,
                       T: np.ndarray) -> np.ndarray:
    """
    Combine and project position covariances to encounter plane.

    C1_3x3, C2_3x3 : 3×3 position covariances [km²]
    T               : (3,2) encounter-plane basis
    Returns         : 2×2 combined covariance in encounter plane
    """
    C_combined = C1_3x3 + C2_3x3     # (3,3)
    return T.T @ C_combined @ T       # (2,2)


# ─── full encounter geometry computation ─────────────────────────────────────

def compute_encounter(
    times1:    np.ndarray,
    pos1:      np.ndarray,
    vel1:      np.ndarray,
    times2:    np.ndarray,
    pos2:      np.ndarray,
    vel2:      np.ndarray,
    cov1_3x3:  Optional[np.ndarray] = None,   # position covariance, object 1
    cov2_3x3:  Optional[np.ndarray] = None,
    sigma_default_km: float = 0.2,             # default 1-σ if no covariance
) -> EncounterGeometry:
    """
    High-level wrapper: find TCA → build encounter geometry → project covariance.

    If covariances are None, an isotropic default σ = sigma_default_km is used.
    """
    tca, miss_km, r_rel, v_rel = find_tca(times1, pos1, times2, pos2)

    T = build_encounter_frame(v_rel)

    # Project miss vector onto encounter plane
    miss_xy = T.T @ r_rel    # (2,)

    # Build covariances
    def _default_cov(sigma):
        return np.diag([sigma**2, sigma**2, sigma**2])

    c1 = cov1_3x3 if cov1_3x3 is not None else _default_cov(sigma_default_km)
    c2 = cov2_3x3 if cov2_3x3 is not None else _default_cov(sigma_default_km)

    cov_2x2 = project_covariance(c1, c2, T)

    # Enforce numerical positive-definiteness
    cov_2x2 = 0.5 * (cov_2x2 + cov_2x2.T)
    eigvals  = np.linalg.eigvalsh(cov_2x2)
    if eigvals.min() < 1e-12:
        cov_2x2 += np.eye(2) * max(1e-12, -eigvals.min() + 1e-12)

    return EncounterGeometry(
        tca_s=tca,
        miss_distance_km=miss_km,
        r_rel_km=r_rel,
        v_rel_kms=v_rel,
        miss_xy_km=miss_xy,
        cov_2x2=cov_2x2,
        T_enc=T,
    )
