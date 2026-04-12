"""Ingestion pipeline: Space-Track → PostGIS.

Usage:
    python -m ingestion.ingest_gp [--limit 5000] [--types DEBRIS PAYLOAD]
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import SEGMENT_MINUTES, PROPAGATION_HORIZON_DAYS
from database.db import init_db, session_scope
from database.models import CatalogObject, GpElement, TrajectorySegment
from fetcher.spacetrack_client import SpaceTrackClient
from propagator.sgp4_propagator import SGP4Propagator

try:
    from geoalchemy2.shape import from_shape
    from shapely.geometry import LineString
    _SHAPELY = True
except ImportError:
    _SHAPELY = False

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_linestring_wkt(points: list) -> str:
    coord_str = ", ".join(f"{x} {y} {z}" for x, y, z in points)
    return f"LINESTRINGZ ({coord_str})"


def _float_or_none(val):
    try:
        return float(val) if val not in (None, "", "0", 0) else None
    except (TypeError, ValueError):
        return None


def _date_or_none(val):
    if not val:
        return None
    try:
        from dateutil.parser import parse as dparse
        return dparse(val).date()
    except Exception:
        return None


def _upsert_catalog(session, rec: dict):
    # Derive perigee/apogee from semi-major axis + eccentricity when not given
    perigee = _float_or_none(rec.get("PERIAPSIS"))
    apogee  = _float_or_none(rec.get("APOAPSIS"))
    if perigee is None:
        try:
            n = float(rec.get("MEAN_MOTION") or 0)      # rev/day
            ecc = float(rec.get("ECCENTRICITY") or 0)
            if n > 0:
                from math import pi
                mu = 398600.4418   # km³/s²
                n_rad = n * 2 * pi / 86400
                a = (mu / n_rad**2) ** (1/3)
                R_EARTH = 6378.137
                perigee = a * (1 - ecc) - R_EARTH
                apogee  = a * (1 + ecc) - R_EARTH
        except Exception:
            pass

    stmt = pg_insert(CatalogObject).values(
        norad_cat_id=int(rec["NORAD_CAT_ID"]),
        name=rec.get("OBJECT_NAME"),
        object_type=rec.get("OBJECT_TYPE"),
        country_code=rec.get("COUNTRY_CODE"),
        launch_date=_date_or_none(rec.get("LAUNCH_DATE")),
        launch_site=rec.get("SITE"),
        decay_date=_date_or_none(rec.get("DECAY_DATE")),
        inclination=_float_or_none(rec.get("INCLINATION")),
        period_min=_float_or_none(rec.get("PERIOD")),
        apogee_km=apogee,
        perigee_km=perigee,
        rcs_size=rec.get("RCS_SIZE"),
        object_id=rec.get("OBJECT_ID"),
        updated_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        index_elements=["norad_cat_id"],
        set_=dict(
            name=rec.get("OBJECT_NAME"),
            object_type=rec.get("OBJECT_TYPE"),
            inclination=_float_or_none(rec.get("INCLINATION")),
            period_min=_float_or_none(rec.get("PERIOD")),
            perigee_km=perigee,
            apogee_km=apogee,
            updated_at=datetime.now(timezone.utc),
        ),
    )
    session.execute(stmt)


def _upsert_gp(session, rec: dict):
    from dateutil.parser import parse as dparse
    epoch_str = rec.get("EPOCH", "")
    try:
        epoch = dparse(epoch_str).replace(tzinfo=timezone.utc)
    except Exception:
        epoch = datetime.now(timezone.utc)

    stmt = pg_insert(GpElement).values(
        norad_cat_id=int(rec["NORAD_CAT_ID"]),
        epoch=epoch,
        mean_motion=float(rec.get("MEAN_MOTION") or 0),
        eccentricity=float(rec.get("ECCENTRICITY") or 0),
        inclination=float(rec.get("INCLINATION") or 0),
        ra_of_asc_node=float(rec.get("RA_OF_ASC_NODE") or 0),
        arg_of_pericenter=float(rec.get("ARG_OF_PERICENTER") or 0),
        mean_anomaly=float(rec.get("MEAN_ANOMALY") or 0),
        bstar=float(rec.get("BSTAR") or 0),
        tle_line1=rec.get("TLE_LINE1"),
        tle_line2=rec.get("TLE_LINE2"),
        ingested_at=datetime.now(timezone.utc),
    ).on_conflict_do_nothing()
    session.execute(stmt)


def _ingest_segments(session, rec: dict, horizon_days: int, seg_minutes: int):
    """Propagate one object and insert trajectory segments."""
    try:
        prop = SGP4Propagator(rec)
    except Exception as exc:
        log.debug("Cannot build propagator for %s: %s", rec.get("NORAD_CAT_ID"), exc)
        return

    from datetime import timedelta
    t_start = datetime.now(timezone.utc)
    t_end = t_start + timedelta(days=horizon_days)

    segs = prop.generate_segments(t_start, t_end, segment_minutes=seg_minutes)

    for seg in segs:
        if len(seg.points) < 2:
            continue
        wkt_eci = _make_linestring_wkt(seg.points)

        row = dict(
            norad_cat_id=seg.norad_id,
            t_start=seg.t_start,
            t_end=seg.t_end,
            geom_eci=f"SRID=0;{wkt_eci}",
            created_at=datetime.now(timezone.utc),
        )
        if seg.geodetic_points and len(seg.geodetic_points) >= 2:
            # geodetic: (lat, lon, alt) → WKT expects (lon lat alt)
            geo_pts = [(lon, lat, alt) for lat, lon, alt in seg.geodetic_points]
            wkt_geo = _make_linestring_wkt(geo_pts)
            row["geom_geo"] = f"SRID=4326;{wkt_geo}"

        session.execute(
            pg_insert(TrajectorySegment).values(**row).on_conflict_do_nothing()
        )


# ------------------------------------------------------------------
# Main ingestion entry point
# ------------------------------------------------------------------

def ingest(
    limit: int = 0,
    object_types: List[str] | None = None,
    propagate: bool = True,
    horizon_days: int = PROPAGATION_HORIZON_DAYS,
    seg_minutes: int = SEGMENT_MINUTES,
):
    """Fetch GP records from Space-Track and propagate trajectories.

    Parameters
    ----------
    limit : int
        Maximum objects to fetch.  0 (default) = no limit, full catalog
        (~25 000–27 000 objects).
    """
    init_db()
    if object_types is None:
        object_types = ["DEBRIS", "PAYLOAD", "ROCKET BODY"]

    with SpaceTrackClient() as client:
        log.info("Fetching GP records (limit=%s, types=%s)",
                 limit if limit > 0 else "ALL", object_types)
        records = client.get_latest_gp(object_types=object_types, limit=limit)
        log.info("Fetched %d records", len(records))

    batch_size = 200
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        with session_scope() as sess:
            for rec in batch:
                _upsert_catalog(sess, rec)
                _upsert_gp(sess, rec)
                if propagate:
                    _ingest_segments(sess, rec, horizon_days, seg_minutes)
        log.info("Ingested batch %d–%d", i, i + len(batch))

    log.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = no limit (full catalog)")
    parser.add_argument("--types", nargs="+",
                        default=["DEBRIS", "PAYLOAD", "ROCKET BODY"])
    parser.add_argument("--no-propagate", action="store_true")
    parser.add_argument("--horizon-days", type=int, default=PROPAGATION_HORIZON_DAYS)
    parser.add_argument("--seg-minutes", type=int, default=SEGMENT_MINUTES)
    args = parser.parse_args()

    ingest(
        limit=args.limit,
        object_types=args.types,
        propagate=not args.no_propagate,
        horizon_days=args.horizon_days,
        seg_minutes=args.seg_minutes,
    )
