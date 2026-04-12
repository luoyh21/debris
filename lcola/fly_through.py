"""Fly-through screening (LCOLA – Launch Collision Avoidance).

Pipeline
--------
For each launch-time candidate t_L in [window_open, window_close]:

  1. Time-shift OEM segments by (t_L − t_nominal).
  2. PostGIS pre-filter: find catalog objects whose trajectory_segments
     come within a coarse threshold (default 200 km) during the phase.
  3. For survivors: propagate debris with SGP4; find TCA via encounter.py.
  4. Compute 2-D Foster Pc via foster_pc.py.
  5. Classify result; flag t_L as BLACKOUT if max(Pc) > threshold.

Output
------
A ScreeningReport containing:
  - per-launch-time Pc curves
  - blackout windows
  - top-conjunction events (sorted by Pc)
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectory.launch_phases import LaunchPhase, PhaseName
from trajectory.oem_io import OEMSegment
from lcola.encounter import compute_encounter, EncounterGeometry
from lcola.foster_pc import foster_pc as _foster_pc
from lcola.foster_pc import chan_pc as _chan_pc

log = logging.getLogger(__name__)

# ─── LCOLA thresholds (NASA NPR 8715.6B) ──────────────────────────────────────
PC_CREWED        = 1e-6    # crewed vehicles
PC_UNCREWED      = 1e-5    # standard payloads
MISS_ABSOLUTE_KM = 25.0    # hard exclusion zone [km]
HBR_KM           = 0.02    # combined hard-body radius [km] (20 m)
COARSE_THRESH_KM = 200.0   # PostGIS pre-filter radius [km]
FINE_THRESH_KM   = 50.0    # secondary filter for TCA search [km]

# ─── data structures ──────────────────────────────────────────────────────────

@dataclass
class ConjunctionEvent:
    """One close approach event."""
    t_launch_offset_s:  float         # seconds from nominal launch time
    phase:              str
    norad_cat_id:       int
    object_name:        str
    tca:                datetime
    miss_distance_km:   float
    probability:        float         # Foster Pc
    pc_error:           float
    v_rel_kms:          float
    is_blackout:        bool = False

    @property
    def risk_level(self) -> str:
        if self.probability >= PC_UNCREWED:   return "RED"    # ≥ 1e-5
        if self.probability >= PC_CREWED:     return "AMBER"  # ≥ 1e-6
        if self.probability >= 1e-7:          return "YELLOW" # ≥ 1e-7
        return "GREEN"


@dataclass
class LaunchTimeResult:
    """Screening result for one candidate launch time."""
    t_launch_offset_s: float
    launch_time:       datetime
    max_pc:            float
    is_blackout:       bool
    events:            List[ConjunctionEvent] = field(default_factory=list)


@dataclass
class ScreeningReport:
    """Full fly-through screening report."""
    mission_name:        str
    nominal_launch_time: datetime
    window_open:         datetime
    window_close:        datetime
    step_s:              float
    pc_threshold:        float
    hbr_km:              float
    results:             List[LaunchTimeResult] = field(default_factory=list)
    top_events:          List[ConjunctionEvent] = field(default_factory=list)

    @property
    def blackout_windows(self) -> List[Tuple[datetime, datetime]]:
        """Merge consecutive blackout offsets into windows."""
        bo = [r for r in self.results if r.is_blackout]
        if not bo:
            return []
        windows = []
        start = bo[0].launch_time
        prev  = bo[0]
        for r in bo[1:]:
            if (r.launch_time - prev.launch_time).total_seconds() > self.step_s * 2:
                windows.append((start, prev.launch_time))
                start = r.launch_time
            prev = r
        windows.append((start, prev.launch_time))
        return windows

    @property
    def safe_windows(self) -> List[Tuple[datetime, datetime]]:
        safe = [r for r in self.results if not r.is_blackout]
        if not safe:
            return []
        wins = []
        start = safe[0].launch_time
        prev  = safe[0]
        for r in safe[1:]:
            if (r.launch_time - prev.launch_time).total_seconds() > self.step_s * 2:
                wins.append((start, prev.launch_time))
                start = r.launch_time
            prev = r
        wins.append((start, prev.launch_time))
        return wins


# ─── debris propagation helper ───────────────────────────────────────────────

def _propagate_debris(norad_id: int, t_start: datetime, t_end: datetime,
                      dt_s: float = 30.0):
    """
    Use the existing SGP4 propagator to generate debris state vectors.
    Returns (times_s_array, pos_eci_array) or (None, None) on failure.
    """
    try:
        from database.db import session_scope
        from propagator.sgp4_propagator import SGP4Propagator
        from sqlalchemy import text

        with session_scope() as sess:
            row = sess.execute(text(
                "SELECT tle_line1, tle_line2, mean_motion, eccentricity, "
                "       inclination, ra_of_asc_node, arg_of_pericenter, mean_anomaly "
                "FROM gp_elements WHERE norad_cat_id=:nid "
                "ORDER BY epoch DESC LIMIT 1"
            ), {"nid": norad_id}).fetchone()

        if row is None:
            return None, None

        rec = {
            "NORAD_CAT_ID":      str(norad_id),
            "TLE_LINE1":         row[0] or "",
            "TLE_LINE2":         row[1] or "",
            "MEAN_MOTION":       row[2],
            "ECCENTRICITY":      row[3],
            "INCLINATION":       row[4],
            "RA_OF_ASC_NODE":    row[5],
            "ARG_OF_PERICENTER": row[6],
            "MEAN_ANOMALY":      row[7],
        }

        prop = SGP4Propagator(rec)
        times_rel = np.arange(
            0, (t_end - t_start).total_seconds() + dt_s, dt_s
        )
        positions = []
        for dt in times_rel:
            epoch = t_start + timedelta(seconds=float(dt))
            sv = prop.propagate(epoch)
            if sv is None:
                break
            positions.append(sv.pos_eci_km if hasattr(sv, "pos_eci_km")
                              else np.array([sv.x, sv.y, sv.z]))

        if len(positions) < 4:
            return None, None

        n = min(len(times_rel), len(positions))
        return times_rel[:n], np.array(positions[:n])

    except Exception as exc:
        log.debug("Debris propagation failed for %d: %s", norad_id, exc)
        return None, None


# ─── PostGIS spatial pre-filter ───────────────────────────────────────────────

def _spatial_prefilter(phase: LaunchPhase, t_start: datetime, t_end: datetime,
                       threshold_km: float) -> List[Tuple[int, str]]:
    """
    Query PostGIS for debris whose trajectory_segments come within
    threshold_km of the rocket trajectory during the phase time window.
    Returns list of (norad_cat_id, name).
    """
    try:
        from database.db import session_scope
        from sqlalchemy import text

        # Build rocket trajectory as a PostGIS point list for rough bounding box
        if not phase.points:
            return []

        # Coarse bounding box filter
        lats  = [p.lat_deg for p in phase.points]
        lons  = [p.lon_deg for p in phase.points]
        alts  = [p.alt_km  for p in phase.points]
        lat_c = (min(lats) + max(lats)) / 2
        lon_c = (min(lons) + max(lons)) / 2
        alt_c = (min(alts) + max(alts)) / 2

        # Convert threshold to degrees (rough: 1° ≈ 111 km)
        deg_r = threshold_km / 111.0

        with session_scope() as sess:
            rows = sess.execute(text("""
                SELECT DISTINCT co.norad_cat_id, co.name
                FROM trajectory_segments ts
                JOIN catalog_objects co ON co.norad_cat_id = ts.norad_cat_id
                WHERE ts.t_start <= :t_end
                  AND ts.t_end   >= :t_start
                  AND ts.geom_geo && ST_Expand(
                        ST_MakePoint(:lon, :lat)::geography::geometry,
                        :deg_r
                      )
                LIMIT 2000
            """), {
                "t_start": t_start,
                "t_end":   t_end,
                "lon":     lon_c,
                "lat":     lat_c,
                "deg_r":   deg_r,
            }).fetchall()

        return [(r[0], r[1] or str(r[0])) for r in rows]

    except Exception as exc:
        log.debug("Spatial pre-filter failed: %s", exc)
        return []


# ─── fly-through screener ─────────────────────────────────────────────────────

class FlyThroughScreener:
    """
    Main LCOLA fly-through screening engine.

    Usage:
        screener = FlyThroughScreener(oem_segments, phases, config)
        report   = screener.screen(window_open, window_close)
    """

    def __init__(
        self,
        oem_segments:        List[OEMSegment],
        phases:              List[LaunchPhase],
        mission_name:        str   = "MISSION",
        pc_threshold:        float = PC_UNCREWED,
        hbr_km:              float = HBR_KM,
        coarse_threshold_km: float = COARSE_THRESH_KM,
        fine_threshold_km:   float = FINE_THRESH_KM,
        default_sigma_km:    float = 0.2,   # 200 m debris position 1-sigma
        use_fast_pc:         bool  = True,  # use Chan analytical Pc for screening
                                             # (≈100× faster than Foster dblquad)
    ):
        self.oem_segments        = oem_segments
        self.phases              = phases
        self.mission_name        = mission_name
        self.pc_threshold        = pc_threshold
        self.hbr_km              = hbr_km
        self.coarse_threshold_km = coarse_threshold_km
        self.fine_threshold_km   = fine_threshold_km
        self.default_sigma_km    = default_sigma_km
        self.use_fast_pc         = use_fast_pc

    def screen(
        self,
        window_open:   datetime,
        window_close:  datetime,
        nominal_launch: datetime,
        step_s:         float = 60.0,    # 1-minute cadence
        max_events_per_step: int = 500,
        progress_cb    = None,           # optional callback(step, total_steps)
    ) -> ScreeningReport:
        """Run the full LCOLA fly-through screening."""
        report = ScreeningReport(
            mission_name=self.mission_name,
            nominal_launch_time=nominal_launch,
            window_open=window_open,
            window_close=window_close,
            step_s=step_s,
            pc_threshold=self.pc_threshold,
            hbr_km=self.hbr_km,
        )

        t_offsets = np.arange(
            0,
            (window_close - window_open).total_seconds() + step_s,
            step_s,
        )
        nominal_offset = (nominal_launch - window_open).total_seconds()

        for i_off, t_off in enumerate(t_offsets):
            t_launch = window_open + timedelta(seconds=float(t_off))
            dt_from_nominal = t_off - nominal_offset

            result = self._screen_one_launch_time(
                t_launch, dt_from_nominal, max_events_per_step
            )
            report.results.append(result)
            if progress_cb:
                progress_cb(i_off + 1, len(t_offsets))
            log.debug("t_offset=%+.0fs  max_Pc=%.2e  blackout=%s",
                      dt_from_nominal, result.max_pc, result.is_blackout)

        # Top events across all launch times
        all_events = [ev for r in report.results for ev in r.events]
        report.top_events = heapq.nlargest(50, all_events, key=lambda e: e.probability)

        n_bo = sum(1 for r in report.results if r.is_blackout)
        log.info("Screening done: %d launch times, %d blackouts, %d total conjunctions",
                 len(report.results), n_bo, len(all_events))
        return report

    # ── internal helpers ──────────────────────────────────────────────────────

    def _screen_one_launch_time(
        self,
        t_launch:         datetime,
        dt_from_nominal:  float,
        max_events:       int,
    ) -> LaunchTimeResult:
        result = LaunchTimeResult(
            t_launch_offset_s=dt_from_nominal,
            launch_time=t_launch,
            max_pc=0.0,
            is_blackout=False,
        )

        for phase in self.phases:
            # Actual calendar times for this phase
            t_phase_start = t_launch + timedelta(seconds=phase.t_start_met)
            t_phase_end   = t_launch + timedelta(seconds=phase.t_end_met)

            if (t_phase_end - t_phase_start).total_seconds() < 1:
                continue

            # Step 1: spatial pre-filter
            candidates = _spatial_prefilter(
                phase, t_phase_start, t_phase_end, self.coarse_threshold_km
            )
            if not candidates:
                continue

            # Step 2: build rocket trajectory for this phase (time-shifted)
            rk_times, rk_pos, rk_cov = self._rocket_trajectory_for_phase(
                phase, t_launch, dt_from_nominal
            )
            if rk_times is None:
                continue

            n_eval = 0
            for norad_id, obj_name in candidates:
                if n_eval >= max_events:
                    break

                # Step 3: propagate debris
                db_times, db_pos = _propagate_debris(
                    norad_id, t_phase_start, t_phase_end, dt_s=30.0
                )
                if db_times is None:
                    continue

                # Step 4: find TCA + encounter geometry
                try:
                    enc = compute_encounter(
                        rk_times, rk_pos, np.zeros_like(rk_pos),
                        db_times, db_pos, np.zeros_like(db_pos),
                        cov1_3x3=rk_cov,
                        sigma_default_km=self.default_sigma_km,
                    )
                except Exception:
                    continue

                if enc.miss_distance_km > self.fine_threshold_km:
                    continue

                n_eval += 1

                # Step 5: Pc computation
                # Use fast Chan analytical series for batch screening;
                # fall back to Foster dblquad only when use_fast_pc=False.
                if self.use_fast_pc:
                    pc      = _chan_pc(enc.miss_xy_km, enc.cov_2x2, self.hbr_km)
                    pc_err  = 0.0     # analytical — no integration error
                else:
                    pc, pc_err = _foster_pc(enc.miss_xy_km, enc.cov_2x2, self.hbr_km)

                tca_dt = t_phase_start + timedelta(seconds=enc.tca_s)
                is_bo  = (pc >= self.pc_threshold or
                          enc.miss_distance_km < MISS_ABSOLUTE_KM)

                ev = ConjunctionEvent(
                    t_launch_offset_s=dt_from_nominal,
                    phase=phase.name,
                    norad_cat_id=norad_id,
                    object_name=obj_name,
                    tca=tca_dt,
                    miss_distance_km=enc.miss_distance_km,
                    probability=pc,
                    pc_error=pc_err,
                    v_rel_kms=float(np.linalg.norm(enc.v_rel_kms)),
                    is_blackout=is_bo,
                )
                result.events.append(ev)
                result.max_pc = max(result.max_pc, pc)
                if is_bo:
                    result.is_blackout = True

        result.events.sort(key=lambda e: e.probability, reverse=True)
        return result

    def _rocket_trajectory_for_phase(
        self,
        phase:           LaunchPhase,
        t_launch:        datetime,
        dt_nominal:      float,
    ) -> Tuple:
        """Extract rocket ECI positions for a phase, shifted to t_launch."""
        pts = phase.points
        if not pts:
            return None, None, None

        times  = np.array([p.t_met_s for p in pts])
        pos    = np.array([p.pos_eci for p in pts])    # (N,3)

        # Default covariance (isotropic 1-sigma = 0.2 km if no MC result)
        first_cov = pts[0].cov_6x6
        if first_cov is not None:
            cov3 = first_cov[:3, :3]
        else:
            sig = self.default_sigma_km
            cov3 = np.diag([sig**2, sig**2, sig**2])

        return times, pos, cov3


# ─── single launch-time per-phase assessment ─────────────────────────────────

@dataclass
class PhaseRiskSummary:
    """Collision risk summary for one launch phase."""
    phase_name:    str
    t_start_met:   float
    t_end_met:     float
    t_start_utc:   datetime
    t_end_utc:     datetime
    n_candidates:  int          # debris passed spatial pre-filter
    n_evaluated:   int          # with valid TCA
    max_pc:        float
    events:        List[ConjunctionEvent]
    risk_text:     str = ""

    @property
    def risk_level(self) -> str:
        if self.max_pc >= PC_UNCREWED:   return "RED"    # ≥ 1e-5
        if self.max_pc >= PC_CREWED:     return "AMBER"  # ≥ 1e-6
        if self.max_pc >= 1e-7:          return "YELLOW" # ≥ 1e-7
        return "GREEN"


def assess_launch_phases(
    phases:           List[LaunchPhase],
    launch_time:      datetime,
    hbr_km:           float = HBR_KM,
    pc_threshold:     float = 0.0,        # include all events (no cutoff)
    coarse_km:        float = COARSE_THRESH_KM,
    fine_km:          float = FINE_THRESH_KM,
    default_sigma_km: float = 1.5,        # realistic TLE position uncertainty [km]
    max_per_phase:    int   = 500,
    inject_demo:      bool  = False,      # inject synthetic demo threats for UI testing
    progress_cb       = None,             # optional callback(phase_name, i, n)
) -> List[PhaseRiskSummary]:
    """
    Assess collision risk for each phase of a single launch.

    For every phase:
      1. PostGIS spatial pre-filter → candidate debris list.
      2. SGP4 propagate each candidate over the phase time window.
      3. Find TCA (encounter.py) + compute Foster Pc (foster_pc.py).
      4. Return per-phase PhaseRiskSummary (events sorted by Pc desc).

    This is the engine behind the '碰撞风险' Streamlit page.
    """
    summaries: List[PhaseRiskSummary] = []

    for ph_idx, phase in enumerate(phases):
        t_start = launch_time + timedelta(seconds=phase.t_start_met)
        t_end   = launch_time + timedelta(seconds=phase.t_end_met)

        if (t_end - t_start).total_seconds() < 10:
            summaries.append(PhaseRiskSummary(
                phase_name=phase.name, t_start_met=phase.t_start_met,
                t_end_met=phase.t_end_met, t_start_utc=t_start, t_end_utc=t_end,
                n_candidates=0, n_evaluated=0, max_pc=0.0, events=[],
                risk_text=phase.risk_profile,
            ))
            continue

        # Build rocket trajectory arrays for this phase
        pts = phase.points
        if not pts:
            summaries.append(PhaseRiskSummary(
                phase_name=phase.name, t_start_met=phase.t_start_met,
                t_end_met=phase.t_end_met, t_start_utc=t_start, t_end_utc=t_end,
                n_candidates=0, n_evaluated=0, max_pc=0.0, events=[],
                risk_text=phase.risk_profile,
            ))
            continue

        rk_times = np.array([p.t_met_s - phase.t_start_met for p in pts])
        rk_pos   = np.array([p.pos_eci  for p in pts])

        first_cov = pts[0].cov_6x6
        cov_rk = first_cov[:3, :3] if first_cov is not None \
                 else np.diag([default_sigma_km**2]*3)

        # Step 1: spatial pre-filter
        candidates = _spatial_prefilter(phase, t_start, t_end, coarse_km)
        n_cand = len(candidates)

        if progress_cb:
            progress_cb(phase.name, 0, n_cand)

        events: List[ConjunctionEvent] = []
        n_eval = 0

        for i, (norad_id, obj_name) in enumerate(candidates[:max_per_phase]):
            if progress_cb and i % 10 == 0:
                progress_cb(phase.name, i, n_cand)

            # Step 2: propagate debris over phase window
            db_times, db_pos = _propagate_debris(norad_id, t_start, t_end, dt_s=30.0)
            if db_times is None or len(db_times) < 4:
                continue
            if len(rk_times) < 2 or len(db_times) < 2:
                continue

            # Step 3: find TCA
            try:
                enc = compute_encounter(
                    rk_times, rk_pos, np.zeros_like(rk_pos),
                    db_times, db_pos, np.zeros_like(db_pos),
                    cov1_3x3=cov_rk,
                    sigma_default_km=default_sigma_km,
                )
            except Exception:
                continue

            if enc.miss_distance_km > fine_km:
                continue

            n_eval += 1

            # Step 4: Foster Pc
            pc, pc_err = _foster_pc(enc.miss_xy_km, enc.cov_2x2, hbr_km)

            if pc < pc_threshold:
                continue

            tca_dt = t_start + timedelta(seconds=enc.tca_s)
            is_bo  = (pc >= PC_UNCREWED or enc.miss_distance_km < MISS_ABSOLUTE_KM)

            events.append(ConjunctionEvent(
                t_launch_offset_s=0.0,
                phase=phase.name,
                norad_cat_id=norad_id,
                object_name=obj_name,
                tca=tca_dt,
                miss_distance_km=enc.miss_distance_km,
                probability=pc,
                pc_error=pc_err,
                v_rel_kms=float(np.linalg.norm(enc.v_rel_kms)),
                is_blackout=is_bo,
            ))

        events.sort(key=lambda e: e.probability, reverse=True)
        max_pc = events[0].probability if events else 0.0

        summaries.append(PhaseRiskSummary(
            phase_name=phase.name, t_start_met=phase.t_start_met,
            t_end_met=phase.t_end_met, t_start_utc=t_start, t_end_utc=t_end,
            n_candidates=n_cand, n_evaluated=n_eval, max_pc=max_pc, events=events,
            risk_text=phase.risk_profile,
        ))
        log.info("Phase %-20s  cand=%d  eval=%d  max_Pc=%.2e",
                 phase.name, n_cand, n_eval, max_pc)

    # ── optional synthetic demo threats ───────────────────────────────────────
    if inject_demo:
        _inject_demo_threats(summaries, launch_time, hbr_km)

    return summaries


def _inject_demo_threats(
    summaries: List["PhaseRiskSummary"],
    launch_time: datetime,
    hbr_km: float,
) -> None:
    """
    Inject synthetic conjunction events into specific phases for UI demonstration.

    These events are clearly labelled with norad_cat_id=-1 and a name prefix
    「🧪 DEMO」. They represent plausible close-approach geometries that would
    occur in a more densely populated catalog, and let users explore the risk
    display before real DB data is available.
    """
    # Predefined demo threats: (phase_name, met_offset_s, miss_km, sigma_km,
    #                            v_rel_kms, norad, name)
    _DEMO = [
        ("PARKING_ORBIT",   280.0, 2.8, 1.5, 8.2,  -1, "🧪 DEMO: CZ-3C DEB 2023-041C"),
        ("POST_SEPARATION", 680.0, 1.2, 1.5, 7.1,  -2, "🧪 DEMO: SL-16 R/B 2019-088B"),
        ("POST_SEPARATION", 920.0, 8.5, 2.0, 9.4,  -3, "🧪 DEMO: FENGYUN-1C DEB"),
    ]
    phase_map = {s.phase_name: s for s in summaries}

    for phase_name, met_off, miss_km, sigma_km, v_kms, nid, obj_name in _DEMO:
        if phase_name not in phase_map:
            continue
        s = phase_map[phase_name]

        # Skip if already injected (e.g. called twice in same session)
        if any(e.norad_cat_id == nid for e in s.events):
            continue

        miss_xy = np.array([miss_km, 0.0])
        cov_2x2 = np.diag([sigma_km**2, sigma_km**2])
        pc, pc_err = _foster_pc(miss_xy, cov_2x2, hbr_km)

        tca_dt  = launch_time + timedelta(seconds=s.t_start_met + met_off)
        is_bo   = (pc >= PC_UNCREWED or miss_km < MISS_ABSOLUTE_KM)

        ev = ConjunctionEvent(
            t_launch_offset_s=0.0, phase=phase_name,
            norad_cat_id=nid, object_name=obj_name,
            tca=tca_dt, miss_distance_km=miss_km,
            probability=pc, pc_error=pc_err,
            v_rel_kms=v_kms, is_blackout=is_bo,
        )
        s.events.append(ev)
        s.events.sort(key=lambda e: e.probability, reverse=True)
        s.max_pc = s.events[0].probability
