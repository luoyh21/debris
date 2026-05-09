"""Catalog → CCSDS OEM 2.0 export (STK / GMAT / Orekit compatible).

This module takes a list of NORAD IDs, pulls their latest TLE +
catalog metadata from the database, propagates each via SGP4 in TEME and
writes one CCSDS OEM 2.0 file containing one rich segment per object.

Compared with the lightweight orbit-forecast OEM (in viz_explorer), this
exporter:

* declares ``REF_FRAME = TEME`` (matches SGP4 output – avoids STK applying
  a wrong frame transformation),
* fills ``OBJECT_ID`` with the COSPAR international designator when known,
* emits ``USEABLE_START_TIME`` / ``USEABLE_STOP_TIME`` mirroring start/stop,
* embeds a long block of ``COMMENT`` lines (NORAD ID, mean motion, BSTAR,
  RCS class, country, primary source, generator) that STK preserves and
  shows in the «Object Properties / Description» panel,
* optionally writes a placeholder 6×6 covariance per state so that STK
  Conjunction / Astrogator workflows that *require* a covariance can be
  exercised end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np

try:
    from sgp4.api import Satrec, jday
    _SGP4_OK = True
except Exception:
    _SGP4_OK = False


def _format_intldes(raw: str | None, launch_date) -> str:
    """Normalise COSPAR designator to STK-friendly ``YYYY-NNNP`` form."""
    if raw:
        s = str(raw).strip().upper().replace(" ", "")
        if "-" in s and len(s) >= 7:
            return s
        if len(s) >= 5 and s[:2].isdigit():
            yy = int(s[:2])
            yyyy = 1900 + yy if yy >= 57 else 2000 + yy
            return f"{yyyy}-{s[2:]}"
    if launch_date is not None:
        try:
            return f"{launch_date.year:04d}-000A"
        except Exception:
            pass
    return "0000-000A"


def build_catalog_oem_bytes(norad_ids: Tuple[int, ...],
                             *, n_orbits: float = 2.0,
                             step_s: float = 60.0,
                             with_cov: bool = False,
                             ) -> tuple[bytes, int, int]:
    """Build an OEM file in memory. Returns ``(bytes, n_segments, n_states)``."""
    if not norad_ids or not _SGP4_OK:
        return b"", 0, 0

    import tempfile, os
    from sqlalchemy import text as _text
    from database.db import session_scope as _scope
    from trajectory.oem_io import OEMSegment, OEMState, write_oem

    try:
        with _scope() as sess:
            rows = sess.execute(_text("""
                SELECT DISTINCT ON (g.norad_cat_id)
                    g.norad_cat_id, g.tle_line1, g.tle_line2,
                    g.mean_motion, g.bstar, g.eccentricity,
                    g.inclination, g.epoch,
                    co.name, co.object_type, co.country_code,
                    co.object_id, co.launch_date, co.rcs_size
                FROM gp_elements g
                JOIN catalog_objects co ON co.norad_cat_id = g.norad_cat_id
                WHERE g.norad_cat_id = ANY(:ids)
                ORDER BY g.norad_cat_id, g.epoch DESC
            """), {"ids": list(norad_ids)}).fetchall()
    except Exception:
        return b"", 0, 0

    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    segs: list[OEMSegment] = []
    n_states_total = 0

    for row in rows:
        try:
            sat = Satrec.twoline2rv(str(row.tle_line1), str(row.tle_line2))
        except Exception:
            continue

        period_min = 1440.0 / max(float(row.mean_motion or 15.5), 0.01)
        total_s    = period_min * 60.0 * float(n_orbits)
        n_pts      = max(10, int(round(total_s / max(step_s, 5.0))))

        states: list[OEMState] = []
        for i in range(n_pts + 1):
            t  = t0 + timedelta(seconds=i * step_s)
            jd, fr = jday(t.year, t.month, t.day,
                          t.hour, t.minute,
                          t.second + t.microsecond / 1e6)
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:
                continue
            cov = None
            if with_cov:
                cov = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])
            states.append(OEMState(
                epoch=t,
                pos_km=np.array([float(r[0]), float(r[1]), float(r[2])]),
                vel_kms=np.array([float(v[0]), float(v[1]), float(v[2])]),
                cov_6x6=cov,
            ))
        if not states:
            continue

        # ASCII-only OBJECT_NAME (STK rejects non-ASCII).
        nm = str(row.name or f"NORAD-{row.norad_cat_id}")
        nm_ascii = nm.encode("ascii", errors="ignore").decode() or f"NORAD-{row.norad_cat_id}"
        nm_ascii = nm_ascii[:48]

        intldes = _format_intldes(row.object_id, row.launch_date)
        ep = row.epoch.isoformat() if row.epoch else "n/a"

        comments = [
            f"NORAD_CAT_ID  = {row.norad_cat_id}",
            f"INTLDES       = {intldes}",
            f"OBJECT_TYPE   = {row.object_type or 'UNKNOWN'}",
            f"COUNTRY       = {row.country_code or 'UNK'}",
            f"RCS_SIZE      = {row.rcs_size or 'UNK'}",
            f"PRIMARY_SOURCE= Space-Track (gp_elements)",
            f"TLE_EPOCH     = {ep}",
            f"MEAN_MOTION   = {float(row.mean_motion or 0):.8f} rev/day",
            f"ECCENTRICITY  = {float(row.eccentricity or 0):.7f}",
            f"INCLINATION   = {float(row.inclination or 0):.4f} deg",
            f"BSTAR         = {float(row.bstar or 0):.6e} 1/ER",
            f"PROPAGATOR    = SGP4 (sgp4.api.Satrec)",
            f"GENERATED_BY  = SpaceDebrisMonitor catalog OEM exporter",
            f"UNITS         = km, km/s",
        ]

        segs.append(OEMSegment(
            object_name      = nm_ascii,
            object_id        = intldes,
            center_name      = "EARTH",
            ref_frame        = "TEME",
            time_system      = "UTC",
            interpolation    = "LAGRANGE",
            interp_degree    = 7,
            useable_start_time = states[0].epoch,
            useable_stop_time  = states[-1].epoch,
            comments         = comments,
            states           = states,
        ))
        n_states_total += len(states)

    if not segs:
        return b"", 0, 0

    tmp = tempfile.mktemp(suffix=".oem")
    try:
        write_oem(tmp, segs, originator="SpaceDebrisMonitor (catalog export)")
        with open(tmp, "rb") as fh:
            return fh.read(), len(segs), n_states_total
    finally:
        try: os.unlink(tmp)
        except OSError: pass
