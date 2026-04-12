"""CCSDS Orbit Ephemeris Message (OEM) I/O.

Implements CCSDS 502.0-B-3 (OEM version 2.0) in ASCII/KVN format.

State units  : km, km/s
Covariance   : km², km²/s, (km/s)²  (upper-triangle, 21 elements per epoch)
Time system  : UTC
Reference    : J2000 ECI

Example OEM segment (no covariance):

    CCSDS_OEM_VERS = 2.0
    CREATION_DATE  = 2026-04-10T15:00:00.000
    ORIGINATOR     = SpaceDebrisMonitor

    META_START
    OBJECT_NAME          = CZ5B_STAGE1
    OBJECT_ID            = 2026-001A
    CENTER_NAME          = EARTH
    REF_FRAME            = J2000
    TIME_SYSTEM          = UTC
    START_TIME           = 2026-04-15T06:00:00.000
    STOP_TIME            = 2026-04-15T07:40:00.000
    INTERPOLATION        = LAGRANGE
    INTERPOLATION_DEGREE = 7
    META_STOP

    COMMENT Launch phase: ASCENT
    2026-04-15T06:00:00.000  6395.000  0.000  0.000  0.000  7.789  0.000

    COVARIANCE_START
    EPOCH        = 2026-04-15T06:00:00.000
    COV_REF_FRAME = J2000
    CX_X         = 1.000e-03
    CY_X         = 0.000e+00
    ...
    COVARIANCE_STOP
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

# ─── names of the 21 upper-triangle covariance elements ──────────────────────
_COV_KEYS = [
    "CX_X",
    "CY_X",    "CY_Y",
    "CZ_X",    "CZ_Y",    "CZ_Z",
    "CX_DOT_X","CX_DOT_Y","CX_DOT_Z","CX_DOT_X_DOT",
    "CY_DOT_X","CY_DOT_Y","CY_DOT_Z","CY_DOT_X_DOT","CY_DOT_Y_DOT",
    "CZ_DOT_X","CZ_DOT_Y","CZ_DOT_Z","CZ_DOT_X_DOT","CZ_DOT_Y_DOT","CZ_DOT_Z_DOT",
]
# Map key → (row, col) in 6×6 matrix
_COV_IDX: dict[str, Tuple[int,int]] = {}
_idx_list = [
    (0,0),
    (1,0),(1,1),
    (2,0),(2,1),(2,2),
    (3,0),(3,1),(3,2),(3,3),
    (4,0),(4,1),(4,2),(4,3),(4,4),
    (5,0),(5,1),(5,2),(5,3),(5,4),(5,5),
]
for _k, _ij in zip(_COV_KEYS, _idx_list):
    _COV_IDX[_k] = _ij


def _cov6x6_to_kvn(cov: np.ndarray) -> List[str]:
    """Flatten 6×6 covariance to CCSDS KVN lines (upper triangle)."""
    lines = []
    for k, (i, j) in zip(_COV_KEYS, _idx_list):
        val = cov[i, j]
        lines.append(f"{k:<20s} = {val:.6e}")
    return lines


def _kvn_to_cov6x6(kv: dict) -> np.ndarray:
    """Rebuild 6×6 covariance from parsed KVN dict."""
    C = np.zeros((6, 6))
    for k, (i, j) in _COV_IDX.items():
        if k in kv:
            C[i, j] = float(kv[k])
            C[j, i] = float(kv[k])   # symmetry
    return C


# ─── data structures ───────────────────────────────────────────────────────────

@dataclass
class OEMState:
    epoch:   datetime      # UTC
    pos_km:  np.ndarray    # (3,) km   in REF_FRAME
    vel_kms: np.ndarray    # (3,) km/s in REF_FRAME
    cov_6x6: Optional[np.ndarray] = None   # 6×6, km²/(km/s)² units


@dataclass
class OEMSegment:
    """One logical segment (one META block + its data)."""
    object_name:  str   = "UNKNOWN"
    object_id:    str   = "0000-000A"
    center_name:  str   = "EARTH"
    ref_frame:    str   = "J2000"
    time_system:  str   = "UTC"
    interpolation:      str = "LAGRANGE"
    interp_degree:      int = 7
    phase_comment:      str = ""
    states: List[OEMState] = field(default_factory=list)

    @property
    def start_time(self) -> Optional[datetime]:
        return self.states[0].epoch if self.states else None

    @property
    def stop_time(self) -> Optional[datetime]:
        return self.states[-1].epoch if self.states else None


# ─── formatter helpers ────────────────────────────────────────────────────────

def _fmt_epoch(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]   # millisecond precision


def _parse_epoch(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse OEM epoch: {s!r}")


# ─── writer ────────────────────────────────────────────────────────────────────

def write_oem(
    filepath: str,
    segments: List[OEMSegment],
    originator: str = "SpaceDebrisMonitor",
    creation_date: Optional[datetime] = None,
) -> None:
    """Write one or more OEM segments to an ASCII file."""
    if creation_date is None:
        creation_date = datetime.now(timezone.utc)

    lines: List[str] = []
    lines.append(f"CCSDS_OEM_VERS = 2.0")
    lines.append(f"CREATION_DATE  = {_fmt_epoch(creation_date)}")
    lines.append(f"ORIGINATOR     = {originator}")

    for seg in segments:
        if not seg.states:
            continue
        lines.append("")
        lines.append("META_START")
        lines.append(f"OBJECT_NAME          = {seg.object_name}")
        lines.append(f"OBJECT_ID            = {seg.object_id}")
        lines.append(f"CENTER_NAME          = {seg.center_name}")
        lines.append(f"REF_FRAME            = {seg.ref_frame}")
        lines.append(f"TIME_SYSTEM          = {seg.time_system}")
        lines.append(f"START_TIME           = {_fmt_epoch(seg.start_time)}")
        lines.append(f"STOP_TIME            = {_fmt_epoch(seg.stop_time)}")
        lines.append(f"INTERPOLATION        = {seg.interpolation}")
        lines.append(f"INTERPOLATION_DEGREE = {seg.interp_degree}")
        lines.append("META_STOP")
        lines.append("")

        if seg.phase_comment:
            lines.append(f"COMMENT {seg.phase_comment}")

        # State data block
        for st in seg.states:
            x, y, z   = st.pos_km
            vx, vy, vz = st.vel_kms
            lines.append(
                f"{_fmt_epoch(st.epoch)}  "
                f"{x:16.6f}  {y:16.6f}  {z:16.6f}  "
                f"{vx:14.9f}  {vy:14.9f}  {vz:14.9f}"
            )

        # Covariance blocks (one per state that has a covariance)
        for st in seg.states:
            if st.cov_6x6 is None:
                continue
            lines.append("")
            lines.append("COVARIANCE_START")
            lines.append(f"EPOCH         = {_fmt_epoch(st.epoch)}")
            lines.append(f"COV_REF_FRAME = {seg.ref_frame}")
            for kv_line in _cov6x6_to_kvn(st.cov_6x6):
                lines.append(kv_line)
            lines.append("COVARIANCE_STOP")

    with open(filepath, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")


# ─── reader ────────────────────────────────────────────────────────────────────

def read_oem(filepath: str) -> List[OEMSegment]:
    """Parse a CCSDS OEM 2.0 ASCII file.  Returns list of OEMSegment."""
    with open(filepath, "r", encoding="ascii", errors="replace") as fh:
        raw_lines = fh.readlines()

    segments: List[OEMSegment] = []
    cur_seg:  Optional[OEMSegment] = None
    in_meta = False
    in_cov  = False
    cov_kv:  dict = {}
    cov_epoch: Optional[datetime] = None
    data_re = re.compile(
        r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
        r"\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
        r"\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
    )

    for raw in raw_lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # skip blank + file-header lines
        if not stripped or stripped.startswith("CCSDS_OEM") or stripped.startswith("CREATION") or stripped.startswith("ORIGINATOR"):
            continue

        if stripped == "META_START":
            cur_seg = OEMSegment()
            in_meta = True
            continue

        if stripped == "META_STOP":
            in_meta = False
            segments.append(cur_seg)
            continue

        if stripped == "COVARIANCE_START":
            in_cov = True
            cov_kv = {}
            cov_epoch = None
            continue

        if stripped == "COVARIANCE_STOP":
            in_cov = False
            if cur_seg and cov_epoch and cov_kv:
                C = _kvn_to_cov6x6(cov_kv)
                # Attach to matching state
                for st in reversed(cur_seg.states):
                    if st.epoch == cov_epoch:
                        st.cov_6x6 = C
                        break
            continue

        if in_meta and cur_seg and "=" in stripped:
            k, _, v = stripped.partition("=")
            k, v = k.strip(), v.strip()
            if   k == "OBJECT_NAME":          cur_seg.object_name   = v
            elif k == "OBJECT_ID":            cur_seg.object_id     = v
            elif k == "CENTER_NAME":          cur_seg.center_name   = v
            elif k == "REF_FRAME":            cur_seg.ref_frame     = v
            elif k == "TIME_SYSTEM":          cur_seg.time_system   = v
            elif k == "INTERPOLATION":        cur_seg.interpolation = v
            elif k == "INTERPOLATION_DEGREE": cur_seg.interp_degree = int(v)
            continue

        if in_cov:
            if "=" in stripped:
                k, _, v = stripped.partition("=")
                k, v = k.strip(), v.strip()
                if k == "EPOCH":
                    cov_epoch = _parse_epoch(v)
                else:
                    cov_kv[k] = v
            continue

        if stripped.startswith("COMMENT") and cur_seg:
            cur_seg.phase_comment = stripped[7:].strip()
            continue

        # State data line
        m = data_re.match(line)
        if m and cur_seg:
            epoch = _parse_epoch(m.group(1))
            pos   = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])
            vel   = np.array([float(m.group(5)), float(m.group(6)), float(m.group(7))])
            cur_seg.states.append(OEMState(epoch=epoch, pos_km=pos, vel_kms=vel))

    return segments


# ─── convenience: SimResult → OEM segments ────────────────────────────────────

def sim_result_to_oem_segments(
    result,              # trajectory.rocketpy_sim.SimResult
    phase_segs,          # List[LaunchPhase] from launch_phases
    mission_id: str = "2026-001",
) -> List[OEMSegment]:
    """Convert a SimResult + phase list to OEM segments (one per phase)."""
    from .launch_phases import LaunchPhase

    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    segs: List[OEMSegment] = []

    nom = result.nominal
    mc_dt = result.mc_dt_s

    for ph in phase_segs:
        oem_seg = OEMSegment(
            object_name  = f"{result.config.vehicle_name}_{ph.name}",
            object_id    = f"{mission_id}A",
            phase_comment= f"Launch phase: {ph.name}",
        )
        pts = [p for p in nom
               if ph.t_start_met <= p.t_met_s <= ph.t_end_met]
        for pt in pts:
            epoch = result.config.launch_utc + __import__("datetime").timedelta(seconds=pt.t_met_s)
            epoch = epoch.replace(tzinfo=timezone.utc)

            # Interpolate covariance if available
            cov = None
            if result.covariances is not None:
                idx = int(pt.t_met_s / mc_dt)
                idx = min(idx, len(result.covariances) - 1)
                cov = result.covariances[idx]

            oem_seg.states.append(OEMState(
                epoch   = epoch,
                pos_km  = pt.pos_eci.copy(),
                vel_kms = pt.vel_eci.copy(),
                cov_6x6 = cov,
            ))
        if oem_seg.states:
            segs.append(oem_seg)
    return segs
