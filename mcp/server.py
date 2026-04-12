"""FastMCP server exposing space-debris analysis tools.

Two tools are registered:
  1. query_debris_in_region  – spatial + altitude + type filter via PostGIS
  2. predict_launch_collision_risk – full 6-DOF sim + Foster Pc per phase

Run the server:
    cd /mnt/space_debris
    python -m mcp.server          # stdio transport (for Claude Desktop / Claude Code)
    python -m mcp.server --http   # SSE transport on port 8888

The server is also importable so the Streamlit agent can call tools directly.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Annotated, Optional

# Add project root to sys.path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from pydantic import Field
from ingestion.tools import (
    query_debris_in_region as _query_debris_in_region_fn,
    predict_launch_collision_risk as _predict_launch_collision_risk_fn,
    get_debris_reentry_forecast as _get_debris_reentry_forecast_fn,
    get_object_tle as _get_object_tle_fn,
    query_debris_by_rcs as _query_debris_by_rcs_fn,
)

log = logging.getLogger(__name__)

mcp = FastMCP(
    name="SpaceDebrisMCP",
    instructions=(
        "Space debris situational awareness tools. "
        "Use query_debris_in_region to find tracked objects near any location. "
        "Use predict_launch_collision_risk to assess conjunction hazards for a planned launch."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 · query_debris_in_region
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def query_debris_in_region(
    lat_deg: Annotated[float, Field(
        description="Centre latitude of the search region (degrees, −90 to 90). "
                    "E.g. 19.61 for Wenchang launch site."
    )],
    lon_deg: Annotated[float, Field(
        description="Centre longitude of the search region (degrees, −180 to 180)."
    )],
    radius_km: Annotated[float, Field(
        description="Search radius in km (recommended 100–2000). "
                    "Only objects whose *ground track* passes within this radius "
                    "in the given time window are returned."
    )] = 500.0,
    alt_min_km: Annotated[float, Field(
        description="Minimum orbital altitude filter (km, perigee ≥ this value). "
                    "Use 200 for LEO lower bound, 0 to include re-entering debris."
    )] = 0.0,
    alt_max_km: Annotated[float, Field(
        description="Maximum orbital altitude filter (km, apogee ≤ this value). "
                    "Use 2000 for LEO, 36000 for GEO."
    )] = 2000.0,
    object_type: Annotated[str, Field(
        description="Object category filter. One of: 'DEBRIS', 'PAYLOAD', "
                    "'ROCKET BODY', 'ALL'. Defaults to 'ALL'."
    )] = "ALL",
    t_start_utc: Annotated[Optional[str], Field(
        description="ISO-8601 UTC start of time window, e.g. '2026-04-11T06:00:00Z'. "
                    "If omitted, defaults to now."
    )] = None,
    hours: Annotated[float, Field(
        description="Duration of time window in hours (1–72). Defaults to 6 hours."
    )] = 6.0,
    limit: Annotated[int, Field(
        description="Maximum number of objects to return (1–200). Defaults to 50."
    )] = 50,
) -> dict:
    """
    Find tracked space objects (debris, payloads, rocket bodies) whose ground
    track passes within `radius_km` of the given point during a time window.

    Uses PostGIS trajectory_segments spatial index for fast lookup, then
    enriches results with orbital parameters from the catalog.

    Returns a structured dict with:
    - region: search parameters
    - count: total matches found (may exceed `limit`)
    - objects: list of matching objects with name, type, altitude, inclination
    - summary: human-readable one-line summary
    """
    return _query_debris_in_region_fn(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        radius_km=radius_km,
        alt_min_km=alt_min_km,
        alt_max_km=alt_max_km,
        object_type=object_type,
        t_start_utc=t_start_utc,
        hours=hours,
        limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 · predict_launch_collision_risk
# ─────────────────────────────────────────────────────────────────────────────

_PRESET_VEHICLES = ["CZ-5B", "Falcon9", "Ariane6"]

@mcp.tool()
def predict_launch_collision_risk(
    vehicle: Annotated[str, Field(
        description=(
            "Launch vehicle preset name OR a description. "
            f"Available presets: {_PRESET_VEHICLES}. "
            "If you provide a name not in presets (e.g. 'Long March 7'), "
            "the closest preset will be selected automatically with a note."
        )
    )] = "CZ-5B",
    launch_lat_deg: Annotated[float, Field(
        description="Launch site latitude (degrees). "
                    "Defaults: Wenchang=19.61, Baikonur=45.92, Kourou=5.23, "
                    "Vandenberg=34.63, Cape Canaveral=28.57."
    )] = 19.61,
    launch_lon_deg: Annotated[float, Field(
        description="Launch site longitude (degrees). "
                    "Defaults: Wenchang=110.95, Baikonur=63.34, Kourou=-52.77, "
                    "Vandenberg=-120.61, Cape Canaveral=-80.58."
    )] = 110.95,
    launch_az_deg: Annotated[Optional[float], Field(
        description="Launch azimuth (degrees, 0=North, 90=East). "
                    "If omitted, defaults to due-East (90°) which gives an "
                    "orbital inclination ≈ launch latitude. "
                    "For ISS/Tiangong rendezvous use ~45° from Wenchang."
    )] = None,
    launch_utc: Annotated[Optional[str], Field(
        description=(
            "Launch date-time in ISO-8601 UTC, e.g. '2026-04-11T06:00:00Z'. "
            "If omitted, defaults to tomorrow at 06:00 UTC (within current DB window). "
            "NOTE: the space-debris DB covers a rolling 3-day propagation window; "
            "dates more than 3 days ahead will find 0 candidates."
        )
    )] = None,
    t_max_s: Annotated[float, Field(
        description="Simulation duration in seconds (600–7200). "
                    "600s covers ascent only; 3600s covers ascent + parking orbit; "
                    "7200s = 2 hours. Longer runs are more thorough but slower."
    )] = 3600.0,
    include_demo_threats: Annotated[bool, Field(
        description="If True (default), three synthetic 🧪 DEMO conjunction events "
                    "are injected to illustrate the risk display even when the "
                    "live catalog is sparse. Set False for production assessment."
    )] = True,
) -> dict:
    """
    Simulate a rocket launch with 6-DOF physics and assess collision probability
    against the tracked debris catalog using the Foster (1992) algorithm.

    The tool:
    1. Selects or approximates the vehicle from presets.
    2. Runs a gravity-turn 6-DOF trajectory (J2 + USSA-76 atmosphere).
    3. Detects launch phases (ASCENT, PARKING_ORBIT, TRANSFER_BURN, POST_SEPARATION).
    4. For each phase: PostGIS spatial pre-filter → SGP4 debris propagation →
       TCA (Time of Closest Approach) via cubic-spline interpolation →
       2-D Foster Pc integral on encounter plane.
    5. Returns structured per-phase and per-event results plus a natural-language
       risk assessment.

    Output fields:
    - assumed_params: what the tool inferred or defaulted for missing inputs
    - vehicle_used: preset name actually used
    - trajectory: altitude/velocity/mass at key milestones
    - phases[]: per-phase summary (duration, candidates, max_Pc, risk_level)
    - top_events[]: up to 10 highest-Pc conjunction events
    - overall_risk: GREEN/YELLOW/AMBER/RED + Pc value
    - recommendation: concise action text in Chinese
    - caveats: list of assumptions or limitations
    """
    return _predict_launch_collision_risk_fn(
        vehicle=vehicle,
        launch_lat_deg=launch_lat_deg,
        launch_lon_deg=launch_lon_deg,
        launch_az_deg=launch_az_deg,
        launch_utc=launch_utc,
        t_max_s=t_max_s,
        include_demo_threats=include_demo_threats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 · get_debris_reentry_forecast
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_debris_reentry_forecast(
    days_ahead: Annotated[float, Field(
        description="Forecast window in days (1–365). "
                    "Objects with confirmed decay_date within this window "
                    "or perigee below alt_max_km are returned."
    )] = 30.0,
    alt_max_km: Annotated[float, Field(
        description="For objects without a confirmed decay date, include those "
                    "whose perigee is at or below this altitude (km). "
                    "300 km = typical re-entry zone; use 500 for broader sweep."
    )] = 300.0,
    object_type: Annotated[str, Field(
        description="Object category filter: 'DEBRIS', 'PAYLOAD', 'ROCKET BODY', or 'ALL'."
    )] = "ALL",
    limit: Annotated[int, Field(
        description="Maximum number of objects to return (1–200). Defaults to 50."
    )] = 50,
) -> dict:
    """
    Forecast space objects predicted to re-enter Earth's atmosphere.

    Queries catalog_objects for:
    - Objects with a confirmed decay_date within the next `days_ahead` days
    - Objects without a confirmed decay_date whose perigee ≤ alt_max_km
      (indicating imminent natural re-entry)

    Returns a structured dict with NORAD ID, name, type, country, decay_date,
    days_to_reentry, perigee_km, apogee_km, inclination.
    """
    return _get_debris_reentry_forecast_fn(
        days_ahead=days_ahead,
        alt_max_km=alt_max_km,
        object_type=object_type,
        limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 · get_object_tle
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_object_tle(
    norad_cat_id: Annotated[int, Field(
        description="NORAD catalog ID of the space object (e.g. 25544 for ISS, "
                    "20580 for Hubble). Use query_debris_in_region to find IDs."
    )],
) -> dict:
    """
    Retrieve the latest Two-Line Element (TLE) set for a tracked space object.

    Returns TLE line 1, TLE line 2, orbital epoch, and key mean elements
    (inclination, eccentricity, mean motion, RAAN, arg. of pericenter,
    mean anomaly, B* drag coefficient) needed for external SGP4 propagation.

    Also returns catalog metadata: name, object type, country, perigee/apogee.
    """
    return _get_object_tle_fn(norad_cat_id=norad_cat_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 · query_debris_by_rcs
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def query_debris_by_rcs(
    rcs_sizes: Annotated[Optional[list], Field(
        description="List of RCS size categories to include. "
                    "Valid values: 'SMALL' (< 0.1 m²), 'MEDIUM' (0.1–1 m²), "
                    "'LARGE' (> 1 m²). Defaults to all three."
    )] = None,
    alt_min_km: Annotated[float, Field(
        description="Minimum orbital altitude filter (km). Default 0."
    )] = 0.0,
    alt_max_km: Annotated[float, Field(
        description="Maximum orbital altitude filter (km). Default 2000 (LEO)."
    )] = 2000.0,
    object_type: Annotated[str, Field(
        description="Object type filter: 'DEBRIS', 'PAYLOAD', 'ROCKET BODY', or 'ALL'."
    )] = "ALL",
    limit: Annotated[int, Field(
        description="Maximum number of objects to return (1–200). Defaults to 50."
    )] = 50,
) -> dict:
    """
    Filter tracked space objects by radar cross-section (RCS) size category.

    RCS size categories:
    - SMALL   (< 0.1 m²): hardest to track, largest uncertainty in miss distance
    - MEDIUM  (0.1–1 m²): moderate trackability
    - LARGE   (> 1 m²):   most trackable, most massive — highest debris generation
                          potential on fragmentation

    Use this to discriminate threats by physical size, e.g. filter to LARGE only
    for crewed-mission collision avoidance, or SMALL to assess catalog completeness.

    Returns: norad_cat_id, name, object_type, country, rcs_size, perigee_km,
    apogee_km, inclination_deg, period_min.
    """
    return _query_debris_by_rcs_fn(
        rcs_sizes=rcs_sizes,
        alt_min_km=alt_min_km,
        alt_max_km=alt_max_km,
        object_type=object_type,
        limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Space Debris MCP Server")
    parser.add_argument("--http", action="store_true",
                        help="Run as HTTP/SSE server on port 8888 instead of stdio")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="sse", host="0.0.0.0", port=args.port)
    else:
        mcp.run()   # stdio transport for Claude Desktop / claude-code
