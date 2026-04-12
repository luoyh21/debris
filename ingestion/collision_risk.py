"""Collision probability calculation (Chan / Foster / Pc).

Uses the standard 2-D probability-of-collision formula from
Alfriend & Akella (2000) / Chan (2008).  For LCOLA (Launch Conjunction
On-orbit Launch Assessment) the per-phase Pc values are summed.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
from sqlalchemy import text

from database.db import session_scope
from propagator.sgp4_propagator import SGP4Propagator, StateVector

log = logging.getLogger(__name__)

# Hard body radius (combined) in km – typical debris + rocket
HBR_KM = 0.020      # 20 m combined
# Default covariance sigma (km) when no CDM available
DEFAULT_SIGMA_KM = 0.200


@dataclass
class ConjunctionEvent:
    norad_cat_id: int
    object_name: str
    object_type: str
    tca: datetime
    miss_distance_km: float
    probability: float
    phase: str
    relative_velocity_km_s: float = 0.0


def _pc_chan(
    r: float,
    sigma_r: float,
    sigma_t: float,
    hbr: float,
) -> float:
    """2-D Chan collision probability approximation.

    Parameters
    ----------
    r        : miss distance (km)
    sigma_r  : combined radial sigma (km)
    sigma_t  : combined transverse sigma (km)
    hbr      : hard-body radius (km)
    """
    if sigma_r <= 0 or sigma_t <= 0:
        return 0.0
    u = r**2 / (2 * sigma_r**2)
    # Series approximation for small Pc
    pc = 0.0
    k_max = 20
    for k in range(k_max):
        a_k = (hbr**2 / (2 * sigma_r * sigma_t)) ** k / math.factorial(k)
        # Modified Bessel function I0 approximation (first term)
        gamma_term = math.exp(-u) * (u**k) / math.factorial(k)
        pc += a_k * gamma_term
    pc *= math.exp(-(hbr**2) / (2 * sigma_t**2))
    return min(pc, 1.0)


def _eci_distance(sv1: StateVector, sv2: StateVector) -> float:
    return float(np.linalg.norm(sv1.pos - sv2.pos))


def assess_launch_risk(
    launch_trajectory: List[tuple],   # (t: datetime, lat, lon, alt_km) waypoints
    window_open: datetime,
    window_close: datetime,
    max_conjunction_km: float = 10.0,
    sigma_km: float = DEFAULT_SIGMA_KM,
    hbr_km: float = HBR_KM,
) -> List[ConjunctionEvent]:
    """Scan PostGIS for debris segments near the launch trajectory in the time window.

    Returns conjunction events sorted by probability descending.
    """
    events: List[ConjunctionEvent] = []

    # Build WKT for the launch trajectory
    if not launch_trajectory:
        return []

    wkt_pts = ", ".join(f"{lon} {lat} {alt}" for _, lat, lon, alt in launch_trajectory)
    launch_wkt = f"SRID=4326;LINESTRINGZ({wkt_pts})"

    query = text("""
        SELECT
            ts.norad_cat_id,
            co.name              AS object_name,
            co.object_type,
            ts.t_start,
            ts.t_end,
            ST_3DDistance(
                ts.geom_geo,
                ST_GeomFromEWKT(:launch_wkt)
            )                    AS miss_km
        FROM trajectory_segments ts
        JOIN catalog_objects co ON co.norad_cat_id = ts.norad_cat_id
        WHERE
            ts.t_start <= :t_close
            AND ts.t_end   >= :t_open
            AND ST_3DDWithin(
                ts.geom_geo,
                ST_GeomFromEWKT(:launch_wkt),
                :threshold_km
            )
        ORDER BY miss_km ASC
        LIMIT 500
    """)

    with session_scope() as sess:
        rows = sess.execute(query, {
            "launch_wkt": launch_wkt,
            "t_open": window_open,
            "t_close": window_close,
            "threshold_km": max_conjunction_km,
        }).fetchall()

    for row in rows:
        miss_km = float(row.miss_km)
        tca = row.t_start + (row.t_end - row.t_start) / 2
        pc = _pc_chan(miss_km, sigma_km, sigma_km, hbr_km)

        # Determine ascent phase
        phase = _classify_phase(tca, window_open, window_close)

        events.append(ConjunctionEvent(
            norad_cat_id=row.norad_cat_id,
            object_name=row.object_name or "UNKNOWN",
            object_type=row.object_type or "UNKNOWN",
            tca=tca,
            miss_distance_km=miss_km,
            probability=pc,
            phase=phase,
        ))

    events.sort(key=lambda e: e.probability, reverse=True)
    return events


def _classify_phase(tca: datetime, t_open: datetime, t_close: datetime) -> str:
    span = (t_close - t_open).total_seconds()
    elapsed = (tca - t_open).total_seconds()
    frac = elapsed / span if span > 0 else 0.5
    if frac < 0.25:
        return "ASCENT"
    elif frac < 0.6:
        return "STAGE_SEPARATION"
    else:
        return "COAST"
