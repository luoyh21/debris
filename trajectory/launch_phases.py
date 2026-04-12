"""Launch phase detection and tagging.

Defines four standard phases for LCOLA analysis:

  ASCENT          – liftoff → exo-atmospheric (alt < 200 km, rising)
  PARKING_ORBIT   – first MECO → second ignition (coasting at circular orbit)
  TRANSFER_BURN   – second ignition → apogee / payload sep
  POST_SEPARATION – detritus (fairings, spent stages) for 72 h after sep

Each phase carries:
  - name          phase label
  - t_start_met   mission elapsed time [s], start
  - t_end_met     mission elapsed time [s], end
  - points        sub-list of TrajectoryPoint objects
  - risk_profile  qualitative collision risk characterisation string
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .six_dof import TrajectoryPoint, MU_KM3S2, R_EARTH_KM


# ─── phase names ──────────────────────────────────────────────────────────────

class PhaseName:
    ASCENT          = "ASCENT"
    PARKING_ORBIT   = "PARKING_ORBIT"
    TRANSFER_BURN   = "TRANSFER_BURN"
    POST_SEPARATION = "POST_SEPARATION"


# ─── phase data structure ─────────────────────────────────────────────────────

@dataclass
class LaunchPhase:
    name:          str
    t_start_met:   float
    t_end_met:     float
    points:        List[TrajectoryPoint]
    risk_profile:  str = ""

    @property
    def duration_s(self) -> float:
        return self.t_end_met - self.t_start_met

    @property
    def alt_range_km(self) -> tuple:
        if not self.points:
            return (0.0, 0.0)
        alts = [p.alt_km for p in self.points]
        return (min(alts), max(alts))

    @property
    def mean_speed_kms(self) -> float:
        if not self.points:
            return 0.0
        return float(np.mean([np.linalg.norm(p.vel_eci) for p in self.points]))

    def __repr__(self) -> str:
        a0, a1 = self.alt_range_km
        return (f"<LaunchPhase {self.name} "
                f"MET={self.t_start_met:.0f}–{self.t_end_met:.0f} s "
                f"alt={a0:.0f}–{a1:.0f} km>")


# ─── orbital element helpers ──────────────────────────────────────────────────

def _orbital_elements(r: np.ndarray, v: np.ndarray):
    """Return (sma_km, eccentricity, alt_perigee_km, alt_apogee_km)."""
    r_n = float(np.linalg.norm(r))
    v_n = float(np.linalg.norm(v))
    eps = v_n**2 / 2 - MU_KM3S2 / r_n        # specific energy
    if eps >= 0:
        return None, None, None, None          # hyperbolic / escape
    sma  = -MU_KM3S2 / (2 * eps)
    h    = np.cross(r, v)
    h_n  = float(np.linalg.norm(h))
    ecc_vec = np.cross(v, h) / MU_KM3S2 - r / r_n
    ecc  = float(np.linalg.norm(ecc_vec))
    r_pe = sma * (1 - ecc) - R_EARTH_KM
    r_ap = sma * (1 + ecc) - R_EARTH_KM
    return sma, ecc, r_pe, r_ap


def _is_roughly_circular(r: np.ndarray, v: np.ndarray, ecc_tol: float = 0.05) -> bool:
    _, ecc, _, _ = _orbital_elements(r, v)
    return ecc is not None and ecc < ecc_tol


# ─── phase detector ───────────────────────────────────────────────────────────

def detect_phases(
    points:         List[TrajectoryPoint],
    t_meco1:        Optional[float] = None,
    t_stage_sep:    Optional[float] = None,
    t_meco2:        Optional[float] = None,
    t_payload_sep:  Optional[float] = None,
    post_sep_days:  float = 3.0,
) -> List[LaunchPhase]:
    """
    Segment a trajectory into launch phases.

    Priority order:
      1. Use explicit event MET times if provided (from SimResult).
      2. Fall back to heuristic detection from state vectors.

    Returns list of LaunchPhase in chronological order.
    """
    if not points:
        return []

    t_end = points[-1].t_met_s

    # ── try explicit event times ──────────────────────────────────────────
    if t_meco1 is not None:
        # ASCENT: liftoff → MECO-1
        t_ascent_end   = t_meco1
        t_parking_end  = t_meco2 if t_meco2 else t_end
        t_transfer_end = t_payload_sep if t_payload_sep else t_end
        t_postsep_end  = t_transfer_end + post_sep_days * 86400.0

        boundaries = [
            (PhaseName.ASCENT,          0.0,            t_ascent_end),
            (PhaseName.PARKING_ORBIT,   t_ascent_end,   t_parking_end),
            (PhaseName.TRANSFER_BURN,   t_parking_end,  t_transfer_end),
            (PhaseName.POST_SEPARATION, t_transfer_end, t_postsep_end),
        ]
    else:
        # ── heuristic detection ───────────────────────────────────────────
        boundaries = _heuristic_phases(points, post_sep_days)

    phases: List[LaunchPhase] = []
    for name, t0, t1 in boundaries:
        seg_pts = [p for p in points if t0 <= p.t_met_s < t1]
        risk = _risk_profile(name)
        phases.append(LaunchPhase(
            name=name,
            t_start_met=t0,
            t_end_met=t1,
            points=seg_pts,
            risk_profile=risk,
        ))

    return phases


def _heuristic_phases(points: List[TrajectoryPoint],
                      post_sep_days: float) -> List[tuple]:
    """Fall-back phase boundaries from kinematics."""
    t_end = points[-1].t_met_s

    # Find when altitude first exceeds 200 km (end of "atmospheric" ascent)
    t_exo = t_end
    for p in points:
        if p.alt_km > 200:
            t_exo = p.t_met_s
            break

    # Find first roughly circular point (parking orbit entry)
    t_parking = t_exo
    for p in points:
        if p.t_met_s < t_exo:
            continue
        if _is_roughly_circular(p.pos_eci, p.vel_eci):
            t_parking = p.t_met_s
            break

    # Detect acceleration event after parking (= transfer burn)
    t_transfer = t_parking
    prev_v = None
    for p in points:
        if p.t_met_s < t_parking:
            continue
        v_n = float(np.linalg.norm(p.vel_eci))
        if prev_v is not None and (v_n - prev_v) > 0.05:   # >0.05 km/s increase per step
            t_transfer = p.t_met_s
            break
        prev_v = v_n

    t_sep  = t_transfer + 300.0   # ~5 min after burn start
    t_ps   = t_sep + post_sep_days * 86400.0

    return [
        (PhaseName.ASCENT,          0.0,        t_exo),
        (PhaseName.PARKING_ORBIT,   t_exo,      t_transfer),
        (PhaseName.TRANSFER_BURN,   t_transfer, t_sep),
        (PhaseName.POST_SEPARATION, t_sep,      min(t_ps, t_end)),
    ]


def _risk_profile(phase_name: str) -> str:
    return {
        PhaseName.ASCENT:
            "低碎片密度（VLEO大气层内），主要不确定性来源：气动扰动、推力偏差",
        PhaseName.PARKING_ORBIT:
            "极高碎片密度（LEO高密集区），相对速度可达12.2 km/s，容限极严",
        PhaseName.TRANSFER_BURN:
            "中等密度，主动推力导致开普勒外推失效，时空筛选精度要求高",
        PhaseName.POST_SEPARATION:
            "72小时内发射衍生物（残骸/整流罩）未入公开目录，需预测星历",
    }.get(phase_name, "")
