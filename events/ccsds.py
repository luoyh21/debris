"""CCSDS Navigation Data Messages (NDM) — KVN parsers + writers.

Implements (subset) of:

* **CDM**  — Conjunction Data Message (CCSDS 508.0-B-1)
* **OPM**  — Orbit Parameter Message (CCSDS 502.0-B-3, single state + maneuvers)
* **OCM**  — Orbit Comprehensive Message (CCSDS 502.0-B-3, mission plan)
* **RDM**  — Re-entry Data Message (CCSDS 508.1-B-1)

The full OEM (orbit ephemeris) round-trip already lives in
``trajectory.oem_io`` — we expose a thin wrapper here for symmetry.

All writers emit the standard *KVN* (key-value-newline) form, which is
the most widely deployed dialect and the one Space-Track/SDC
distributes by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import EventType, SpaceEvent


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000")


def _fmt_epoch(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def _parse_epoch(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%j-%H:%M:%S.%f",
                "%Y-%j-%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"无法解析 CCSDS 时间戳: {s!r}")


# ─── format detection ────────────────────────────────────────────────────────

def detect_format(text: str) -> str:
    """Return one of ``CDM, OPM, OEM, OCM, RDM`` from a CCSDS file body."""
    head = text[:2048].upper()
    for tag, name in [
        ("CCSDS_CDM_VERS", "CDM"),
        ("CCSDS_OPM_VERS", "OPM"),
        ("CCSDS_OEM_VERS", "OEM"),
        ("CCSDS_OCM_VERS", "OCM"),
        ("CCSDS_RDM_VERS", "RDM"),
    ]:
        if tag in head:
            return name
    raise ValueError("未识别的 CCSDS NDM 类型（缺 CCSDS_*_VERS 头）。")


def _kvn_lines(text: str) -> List[Tuple[str, str]]:
    """Yield (key, value) tuples from a KVN body, ignoring blanks & comments."""
    out: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.split("COMMENT", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out.append((k.strip().upper(), v.strip()))
    return out


# ─── CDM parser ──────────────────────────────────────────────────────────────

def _parse_cdm(text: str) -> SpaceEvent:
    kv = dict(_kvn_lines(text))

    tca = _parse_epoch(kv.get("TCA", _now_iso()))
    miss = float(kv.get("MISS_DISTANCE", 0)) / 1000.0   # m → km
    pc   = float(kv.get("COLLISION_PROBABILITY", 0))
    obj1 = kv.get("OBJECT1_OBJECT_DESIGNATOR") or kv.get("OBJECT1_NORAD_CAT_ID")
    obj2 = kv.get("OBJECT2_OBJECT_DESIGNATOR") or kv.get("OBJECT2_NORAD_CAT_ID")

    def _norad(s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        try:
            return int("".join(ch for ch in s if ch.isdigit()))
        except ValueError:
            return None

    return SpaceEvent(
        event_type      = EventType.CDM,
        epoch           = tca,
        name            = f"CDM {obj1 or '?'} ↔ {obj2 or '?'}",
        description     = kv.get("MESSAGE_ID", ""),
        parent_norad    = _norad(obj1),
        secondary_norad = _norad(obj2),
        miss_distance_km= miss,
        probability     = pc,
        source          = "CCSDS-IMPORT",
        source_id       = kv.get("MESSAGE_ID", ""),
        raw             = kv,
    )


# ─── OPM parser ──────────────────────────────────────────────────────────────

def _parse_opm(text: str) -> SpaceEvent:
    kv = dict(_kvn_lines(text))
    epoch = _parse_epoch(kv.get("EPOCH", _now_iso()))
    return SpaceEvent(
        event_type      = EventType.MANEUVER,
        epoch           = epoch,
        name            = f"OPM {kv.get('OBJECT_NAME','?')}",
        description     = kv.get("ORIGINATOR", ""),
        parent_norad    = _safe_int(kv.get("OBJECT_ID")),
        source          = "CCSDS-IMPORT",
        source_id       = kv.get("OBJECT_ID", ""),
        raw             = kv,
    )


# ─── RDM parser ──────────────────────────────────────────────────────────────

def _parse_rdm(text: str) -> SpaceEvent:
    kv = dict(_kvn_lines(text))
    epoch = _parse_epoch(kv.get("REENTRY_EPOCH",
                                  kv.get("REENTRY_EPOCH_TIME", _now_iso())))
    return SpaceEvent(
        event_type      = EventType.REENTRY,
        epoch           = epoch,
        name            = f"RDM {kv.get('OBJECT_NAME','?')}",
        description     = kv.get("REENTRY_INFO_NOTE",
                                  kv.get("ORIGINATOR", "")),
        parent_norad    = _safe_int(kv.get("OBJECT_ID")),
        altitude_km     = _safe_float(kv.get("REENTRY_ALTITUDE")),
        source          = "CCSDS-IMPORT",
        source_id       = kv.get("OBJECT_ID", ""),
        raw             = kv,
    )


# ─── OCM parser (very thin) ──────────────────────────────────────────────────

def _parse_ocm(text: str) -> SpaceEvent:
    kv = dict(_kvn_lines(text))
    epoch = _parse_epoch(kv.get("EPOCH_TZERO", _now_iso()))
    return SpaceEvent(
        event_type      = EventType.MANEUVER,
        epoch           = epoch,
        name            = f"OCM {kv.get('OBJECT_NAME','?')}",
        description     = kv.get("ORIGINATOR", ""),
        parent_norad    = _safe_int(kv.get("OBJECT_DESIGNATOR")),
        source          = "CCSDS-IMPORT",
        source_id       = kv.get("OBJECT_DESIGNATOR", ""),
        raw             = kv,
    )


def _safe_int(s) -> Optional[int]:
    if s is None: return None
    try:
        return int("".join(ch for ch in str(s) if ch.isdigit()) or 0) or None
    except ValueError:
        return None


def _safe_float(s) -> Optional[float]:
    if s is None: return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ─── public dispatcher ───────────────────────────────────────────────────────

def parse_ccsds_message(text: str) -> SpaceEvent:
    """Auto-detect CCSDS NDM type and parse to ``SpaceEvent``."""
    fmt = detect_format(text)
    if fmt == "CDM":  return _parse_cdm(text)
    if fmt == "OPM":  return _parse_opm(text)
    if fmt == "RDM":  return _parse_rdm(text)
    if fmt == "OCM":  return _parse_ocm(text)
    if fmt == "OEM":
        # Reuse trajectory.oem_io for proper parsing → minimal envelope
        from trajectory.oem_io import read_oem
        import tempfile, os
        tmp = tempfile.mktemp(suffix=".oem")
        with open(tmp, "w") as fh: fh.write(text)
        try:
            segs = read_oem(tmp)
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        if not segs:
            raise ValueError("OEM 文件内无可用 segment")
        s0 = segs[0]
        return SpaceEvent(
            event_type   = EventType.MANEUVER,
            epoch        = s0.start_time or datetime.now(timezone.utc),
            name         = f"OEM {s0.object_name}",
            parent_norad = _safe_int(s0.object_id),
            description  = f"{len(segs)} segments / {sum(len(s.states) for s in segs)} states",
            source       = "CCSDS-IMPORT",
            source_id    = s0.object_id,
            raw          = {"segments": [s.object_name for s in segs]},
        )
    raise ValueError(f"暂不支持的 CCSDS 类型: {fmt}")


# ─── writers ─────────────────────────────────────────────────────────────────

def write_cdm(event: SpaceEvent, *, originator: str = "SpaceDebrisMonitor",
              message_id: Optional[str] = None) -> str:
    """Emit a minimum-viable CDM (CCSDS 508.0-B-1) for the given event."""
    msg_id = message_id or f"CDM-{int(event.epoch.timestamp())}"
    pc   = event.probability or 0.0
    miss = (event.miss_distance_km or 0.0) * 1000.0
    pn   = event.parent_norad    or 0
    sn   = event.secondary_norad or 0

    lines = [
        "CCSDS_CDM_VERS  = 1.0",
        f"CREATION_DATE  = {_now_iso()}",
        f"ORIGINATOR     = {originator}",
        f"MESSAGE_ID     = {msg_id}",
        f"TCA            = {_fmt_epoch(event.epoch)}",
        f"MISS_DISTANCE  = {miss:.6f}",
        f"COLLISION_PROBABILITY = {pc:.6e}",
        f"COLLISION_PROBABILITY_METHOD = FOSTER-1992",
        "",
        "OBJECT          = OBJECT1",
        f"OBJECT_DESIGNATOR = {pn}",
        f"OBJECT_NAME     = {event.name or 'PRIMARY'}",
        f"INTERNATIONAL_DESIGNATOR = UNKNOWN",
        "OBJECT_TYPE     = PAYLOAD",
        "OPERATOR_CONTACT_POSITION = OPS",
        "OPERATOR_ORGANIZATION = OPS",
        "EPHEMERIS_NAME  = N/A",
        "COVARIANCE_METHOD = CALCULATED",
        "MANEUVERABLE    = NO",
        "REF_FRAME       = ITRF",
        "",
        "OBJECT          = OBJECT2",
        f"OBJECT_DESIGNATOR = {sn}",
        f"OBJECT_NAME     = SECONDARY",
        f"INTERNATIONAL_DESIGNATOR = UNKNOWN",
        "OBJECT_TYPE     = DEBRIS",
        "OPERATOR_CONTACT_POSITION = N/A",
        "OPERATOR_ORGANIZATION = N/A",
        "EPHEMERIS_NAME  = N/A",
        "COVARIANCE_METHOD = CALCULATED",
        "MANEUVERABLE    = N/A",
        "REF_FRAME       = ITRF",
    ]
    return "\n".join(lines) + "\n"


def write_opm(event: SpaceEvent, *,
              r_eci_km: Optional[np.ndarray] = None,
              v_eci_km_s: Optional[np.ndarray] = None,
              maneuvers: Optional[List[dict]] = None,
              originator: str = "SpaceDebrisMonitor") -> str:
    """Emit a CCSDS 502.0 OPM (state + optional maneuver list)."""
    r = r_eci_km if r_eci_km is not None else np.zeros(3)
    v = v_eci_km_s if v_eci_km_s is not None else np.zeros(3)
    lines = [
        "CCSDS_OPM_VERS  = 2.0",
        f"CREATION_DATE  = {_now_iso()}",
        f"ORIGINATOR     = {originator}",
        "",
        "META_START",
        f"OBJECT_NAME    = {event.name or 'OBJECT'}",
        f"OBJECT_ID      = {event.parent_norad or 0}",
        "CENTER_NAME    = EARTH",
        "REF_FRAME      = TEME",
        "TIME_SYSTEM    = UTC",
        "META_STOP",
        "",
        f"EPOCH          = {_fmt_epoch(event.epoch)}",
        f"X              = {r[0]:.6f}",
        f"Y              = {r[1]:.6f}",
        f"Z              = {r[2]:.6f}",
        f"X_DOT          = {v[0]:.9f}",
        f"Y_DOT          = {v[1]:.9f}",
        f"Z_DOT          = {v[2]:.9f}",
    ]
    for i, mv in enumerate(maneuvers or [], start=1):
        lines.extend([
            "",
            f"MAN_EPOCH_IGNITION = {_fmt_epoch(mv['epoch'])}",
            f"MAN_DURATION       = {mv.get('duration_s', 0.0):.3f}",
            f"MAN_DELTA_MASS     = {mv.get('delta_mass_kg', 0.0):.6f}",
            f"MAN_REF_FRAME      = TNW",
            f"MAN_DV_1           = {mv['dv'][0]:.6f}",
            f"MAN_DV_2           = {mv['dv'][1]:.6f}",
            f"MAN_DV_3           = {mv['dv'][2]:.6f}",
        ])
    return "\n".join(lines) + "\n"


def write_rdm(event: SpaceEvent, *, originator: str = "SpaceDebrisMonitor") -> str:
    """Emit a minimum CCSDS 508.1 RDM (re-entry data message)."""
    lines = [
        "CCSDS_RDM_VERS  = 1.0",
        f"CREATION_DATE  = {_now_iso()}",
        f"ORIGINATOR     = {originator}",
        f"OBJECT_NAME    = {event.name or 'REENTRY'}",
        f"OBJECT_ID      = {event.parent_norad or 0}",
        "OBJECT_TYPE    = PAYLOAD",
        f"REENTRY_EPOCH  = {_fmt_epoch(event.epoch)}",
        f"REENTRY_ALTITUDE = {event.altitude_km or 80.0}",
        "REF_FRAME      = ITRF",
        f"REENTRY_INFO_NOTE = {event.description or 'AUTO-GENERATED'}",
    ]
    return "\n".join(lines) + "\n"


def write_ocm(event: SpaceEvent, *, fragments=None,
              originator: str = "SpaceDebrisMonitor") -> str:
    """Emit a thin CCSDS 502.0 OCM with one TRAJ section + optional MAN block.

    For breakup events we list each generated fragment as a deployed
    object inside USER-DEFINED-PARAMETERS (USER_DEFINED_*).
    """
    n_frag = len(fragments) if fragments else 0
    lines = [
        "CCSDS_OCM_VERS  = 3.0",
        f"CREATION_DATE   = {_now_iso()}",
        f"ORIGINATOR      = {originator}",
        "",
        "META_START",
        f"OBJECT_NAME     = {event.name or 'EVENT'}",
        f"OBJECT_DESIGNATOR = {event.parent_norad or 0}",
        f"INTERNATIONAL_DESIGNATOR = UNKNOWN",
        f"CATALOG_NAME    = SPACE-TRACK",
        f"EPOCH_TZERO     = {_fmt_epoch(event.epoch)}",
        "TIME_SYSTEM     = UTC",
        "META_STOP",
        "",
    ]
    if event.event_type == EventType.MANEUVER:
        lines += [
            "MAN_BEGIN",
            f"MAN_ID                    = AUTO-1",
            f"MAN_PURPOSE               = AVOIDANCE",
            f"MAN_T_START               = 0.0",
            f"MAN_T_STOP                = 0.0",
            "MAN_END",
            "",
        ]
    if n_frag:
        lines.append("USER_DEFINED_PARAMETERS_BEGIN")
        lines.append(f"USER_DEFINED_FRAGMENT_COUNT = {n_frag}")
        for i, fr in enumerate(fragments[:1000]):
            lines.append(f"USER_DEFINED_FRAGMENT_{i+1:04d} = "
                          f"Lc={fr.lc_m:.4f},mass={fr.mass_kg:.4f},"
                          f"AM={fr.am_m2_per_kg:.4f},"
                          f"|dV|={float(np.linalg.norm(fr.delta_v_kms))*1000:.3f} m/s")
        lines.append("USER_DEFINED_PARAMETERS_END")
    return "\n".join(lines) + "\n"
