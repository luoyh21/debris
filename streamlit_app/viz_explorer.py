"""
可视化探索模块 — 三场景沉浸式 3D 仪表盘

Tab 1  全球碎片态势     pydeck GlobeView 交互球体 + 高度直方图
Tab 2  高度分层下钻     Plotly Scatter3d 地球球面 + 5 层轨道带
Tab 3  火箭发射碎片预警 6-DOF 轨迹动画 + 时间轴 + 近地碎片高亮
"""
from __future__ import annotations

import logging
import math
import string
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from streamlit_app.nav_icons import icon_inline, section_title, title_row

log = logging.getLogger(__name__)

try:
    from sgp4.api import Satrec, WGS84, jday
    _SGP4_OK = True
except Exception:
    _SGP4_OK = False

# ─── Keplerian two-body propagation helpers ───────────────────────────────────
_MU_KM3S2  = 398600.4418
_OMEGA_E   = 7.2921150e-5   # rad/s


def _propagate_coast_ecef(
    r_ecef: np.ndarray,
    vel_ecef: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Propagate a coast-phase orbit and return the new ECEF position (km).

    Key insight: we only need the *relative* Earth rotation during the coast
    interval, not the absolute GAST/J2000 angle.  This avoids any dependency
    on launch_utc / session-state and removes the sign-convention ambiguity.

    Algorithm
    ---------
    1. Convert the 6-DOF ECEF velocity (rotating-frame) to the inertial
       velocity expressed in ECEF coordinates at the MECO epoch:
           v_inertial_ecef = vel_ecef + ω × r_ecef
    2. Run Keplerian two-body propagation using (r_ecef, v_inertial_ecef)
       as the initial state.  Treating ECEF as momentarily inertial is valid
       because Keplerian mechanics is rotation-invariant — a pure frame
       rotation of the input just produces the same rotation of the output.
    3. Undo Earth's rotation during the coast: rotate the Keplerian result
       by −ω·dt_s around Z.  This yields the correct new ECEF position.
    """
    omega_vec = np.array([0.0, 0.0, _OMEGA_E])
    v_inertial = vel_ecef + np.cross(omega_vec, r_ecef)
    r_kep, _ = _keplerian_propagate_eci(r_ecef, v_inertial, dt_s)
    # Rotate by -omega*dt to account for Earth's rotation
    _theta = _OMEGA_E * dt_s
    _c, _s = math.cos(_theta), math.sin(_theta)
    return np.array([
         _c * float(r_kep[0]) + _s * float(r_kep[1]),
        -_s * float(r_kep[0]) + _c * float(r_kep[1]),
        float(r_kep[2]),
    ])


def _keplerian_propagate_eci(r0: np.ndarray, v0: np.ndarray,
                              dt_s: float) -> tuple:
    """Propagate ECI state (km, km/s) forward by dt_s seconds under 2-body gravity.

    Returns (r_eci, v_eci).  Falls back to the input state on non-elliptic orbits.
    """
    mu = _MU_KM3S2
    r0n = float(np.linalg.norm(r0))
    v0n = float(np.linalg.norm(v0))
    energy = v0n**2 / 2.0 - mu / r0n
    if energy >= 0.0:
        return r0.copy(), v0.copy()    # hyperbolic / escape

    a = -mu / (2.0 * energy)           # semi-major axis
    h = np.cross(r0, v0)               # specific angular momentum
    e_vec = np.cross(v0, h) / mu - r0 / r0n
    e = float(np.linalg.norm(e_vec))

    # Inclination & RAAN from h
    hn  = float(np.linalg.norm(h))
    inc = math.acos(max(-1.0, min(1.0, float(h[2]) / hn)))
    h_xy = math.hypot(float(h[0]), float(h[1]))
    raan = math.atan2(float(h[0]), -float(h[1])) if h_xy > 1e-9 else 0.0

    # Argument of perigee
    n_hat = np.array([math.cos(raan), math.sin(raan), 0.0])
    if e > 1e-9:
        cos_w = max(-1.0, min(1.0, float(np.dot(n_hat, e_vec)) / e))
        omega = math.acos(cos_w)
        if float(e_vec[2]) < 0:
            omega = 2 * math.pi - omega
    else:
        omega = 0.0

    # True anomaly at t0
    p_orb = a * (1.0 - e**2)
    if e > 1e-9:
        rdotv = float(np.dot(r0, v0))
        cos_nu0 = max(-1.0, min(1.0, (p_orb / r0n - 1.0) / e))
        nu0 = math.acos(cos_nu0)
        if rdotv < 0:
            nu0 = 2 * math.pi - nu0
    else:
        nu0 = math.atan2(float(np.dot(r0, np.cross(h, e_vec))) / (hn * r0n),
                         float(np.dot(r0, e_vec)) / r0n) if e > 1e-12 else \
              math.atan2(float(r0[1]), float(r0[0]))

    # Eccentric anomaly at t0
    E0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu0 / 2.0),
                           math.sqrt(1.0 + e) * math.cos(nu0 / 2.0))
    M0 = E0 - e * math.sin(E0)

    # Propagate mean anomaly
    n_mot = math.sqrt(mu / abs(a) ** 3)
    M = (M0 + n_mot * dt_s) % (2.0 * math.pi)

    # Newton-Raphson Kepler equation
    E = M
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break

    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                           math.sqrt(1.0 - e) * math.cos(E / 2.0))
    r_mag = a * (1.0 - e * math.cos(E))

    # Perifocal frame
    r_pf = r_mag * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p_orb) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    # Rotation matrix: perifocal → ECI
    cO, sO = math.cos(raan),  math.sin(raan)
    cw, sw = math.cos(omega), math.sin(omega)
    ci, si = math.cos(inc),   math.sin(inc)
    Q = np.array([
        [ cO*cw - sO*sw*ci,  -cO*sw - sO*cw*ci,  sO*si],
        [ sO*cw + cO*sw*ci,  -sO*sw + cO*cw*ci, -cO*si],
        [ sw*si,               cw*si,              ci   ],
    ])
    return Q @ r_pf, Q @ v_pf



# ─── Design tokens ────────────────────────────────────────────────────────────
DARK_BG   = "#0e1117"
SCENE_BG  = "#000814"
R_EARTH   = 6371.0          # km

ALTITUDE_LAYERS = [
    {"id": "VLEO", "label": "VLEO <400km",    "alt_min":  0,     "alt_max":  400,
     "color": "#FF6B6B", "note": "ISS/天宫活跃区"},
    {"id": "LEO1", "label": "LEO-I 400-800",  "alt_min":  400,   "alt_max":  800,
     "color": "#FF9F45", "note": "星链/OneWeb星座"},
    {"id": "LEO2", "label": "LEO-II 800-2k",  "alt_min":  800,   "alt_max": 2000,
     "color": "#FFD93D", "note": "碎屑最高密度区"},
    {"id": "MEO",  "label": "MEO 2k-30k",     "alt_min": 2000,   "alt_max": 30000,
     "color": "#6BCB77", "note": "GPS/北斗/Galileo"},
    {"id": "GEO",  "label": "GEO >30000km",   "alt_min": 30000,  "alt_max": 42500,
     "color": "#4D96FF", "note": "通信卫星/墓地轨道"},
]

# object_type → (hex color,  [R,G,B] for pydeck)
_TYPE_HEX = {
    "DEBRIS":       "#FF2244",   # vivid red
    "PAYLOAD":      "#00CCFF",   # cyan-blue  (high contrast vs red)
    "ROCKET BODY":  "#FFEE00",   # pure yellow (distinct from red + cyan)
}
_TYPE_PDK = {
    "DEBRIS":       [255,  34,  68, 200],
    "PAYLOAD":      [  0, 204, 255, 200],
    "ROCKET BODY":  [255, 238,   0, 200],
}
_TYPE_CN  = {"DEBRIS": "碎片", "PAYLOAD": "载荷", "ROCKET BODY": "火箭级"}


# ─── Geometry helpers ──────────────────────────────────────────────────────────
def _earth_mesh(n: int = 80) -> tuple:
    """Return (x, y, z) for an R_EARTH-radius sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi,     n // 2)
    x = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH * np.outer(np.ones(n), np.cos(v))
    return x, y, z


_LAND_POLYGONS: list = [
    # Africa
    [(-17,14),(-16,20),(-13,28),(-5,36),(10,37),(20,35),(30,31),(35,28),
     (40,20),(45,12),(50,12),(48,8),(42,2),(40,-3),(38,-10),(35,-20),
     (33,-28),(28,-34),(18,-35),(20,-30),(28,-22),(35,-12),(40,-2),
     (44,5),(50,10),(42,12),(35,18),(25,22),(15,20),(5,10),(-5,5),
     (-10,8),(-15,12),(-17,14)],
    # Europe
    [(-10,36),(-5,43),(0,46),(3,48),(5,51),(3,54),(5,56),(8,58),
     (12,56),(18,55),(24,56),(28,58),(30,62),(25,65),(18,67),(12,66),
     (8,58),(5,54),(0,50),(-5,44),(-10,38),(-10,36)],
    # Asia
    [(28,42),(32,38),(36,35),(42,30),(48,27),(55,24),(62,24),(68,22),
     (72,18),(77,10),(80,6),(85,12),(90,20),(95,18),(100,5),(104,1),
     (108,10),(114,22),(120,32),(126,40),(133,48),(140,55),(150,60),
     (165,65),(180,66),(180,72),(160,70),(140,68),(120,62),(100,56),
     (80,58),(60,54),(45,50),(35,44),(28,42)],
    # North America
    [(-170,52),(-165,60),(-155,70),(-140,70),(-128,55),(-122,46),
     (-118,38),(-116,30),(-108,25),(-100,18),(-92,15),(-85,12),
     (-80,8),(-82,14),(-87,20),(-84,25),(-80,25),(-78,34),(-72,42),
     (-66,46),(-55,50),(-60,54),(-68,58),(-82,65),(-100,70),
     (-125,72),(-155,70),(-165,60),(-170,55),(-170,52)],
    # South America
    [(-80,10),(-72,12),(-65,10),(-55,4),(-48,0),(-40,-3),(-35,-8),
     (-35,-12),(-40,-20),(-48,-28),(-55,-35),(-62,-40),(-68,-52),
     (-75,-52),(-73,-42),(-72,-28),(-76,-12),(-80,0),(-80,10)],
    # Australia
    [(115,-35),(120,-28),(130,-12),(140,-12),(148,-20),(153,-28),
     (153,-35),(148,-38),(140,-38),(132,-34),(120,-35),(115,-35)],
    # Antarctica
    [(-180,-62),(-120,-68),(-60,-68),(0,-70),(60,-68),(120,-68),
     (180,-62),(180,-90),(-180,-90),(-180,-62)],
    # Greenland
    [(-55,60),(-45,60),(-20,70),(-18,78),(-30,82),(-52,82),(-58,72),(-55,60)],
    # UK/Ireland
    [(-10,50),(-5,50),(-2,53),(-4,57),(-8,56),(-10,52),(-10,50)],
    # Japan
    [(130,31),(135,35),(140,38),(143,44),(145,45),(142,40),(137,34),(130,31)],
    # India subcontinent
    [(68,24),(72,18),(77,10),(80,8),(84,14),(88,22),(90,24),(84,26),(76,28),(68,24)],
    # Arabian Peninsula
    [(35,28),(42,18),(48,14),(55,22),(56,25),(48,28),(40,30),(35,28)],
    # Scandinavia
    [(5,58),(10,62),(16,68),(22,70),(28,66),(30,62),(25,60),(15,56),(5,58)],
    # Italy
    [(8,44),(12,46),(16,42),(16,38),(12,38),(9,42),(8,44)],
    # Indonesia (Sumatra-Java arc)
    [(95,-6),(100,-1),(106,-6),(110,-8),(115,-8),(120,-5),(115,-4),
     (108,-2),(100,-3),(95,-6)],
    # New Zealand
    [(166,-46),(172,-40),(178,-37),(178,-42),(170,-46),(166,-46)],
    # Madagascar
    [(44,-25),(48,-18),(50,-12),(46,-12),(43,-20),(44,-25)],
    # Papua New Guinea
    [(141,-8),(148,-5),(152,-6),(148,-8),(141,-8)],
    # Borneo
    [(109,-2),(115,2),(119,2),(116,-2),(111,-3),(109,-2)],
]


@st.cache_data(show_spinner=False)
def _build_earth_texture(n: int = 110) -> np.ndarray:
    """Build surface-color array matching _earth_mesh orientation using polygon land mask."""
    from matplotlib.path import Path as _MplPath

    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    lon = np.degrees(u)
    lon = np.where(lon > 180, lon - 360, lon)
    lat = 90 - np.degrees(v)

    lon_g, lat_g = np.meshgrid(lon, lat)    # shape (n//2, n)
    pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])

    land = np.zeros(len(pts), dtype=bool)
    for poly in _LAND_POLYGONS:
        closed = list(poly) + [poly[0]]
        path = _MplPath(closed)
        land |= path.contains_points(pts)

    color = np.full(len(pts), 0.12)
    ocean = ~land
    olat = np.abs(lat_g.ravel()[ocean])
    color[ocean] = 0.05 + 0.12 * (1 + np.cos(np.radians(olat) * 1.5)) / 2

    llat = lat_g.ravel()[land]
    llon = lon_g.ravel()[land]
    elev = (0.55 + 0.15 * np.abs(np.sin(np.radians(llat) * 1.5))
            + 0.08 * np.cos(np.radians(llon) * 2))
    color[land] = np.clip(elev, 0.42, 0.85)

    polar = np.abs(lat_g.ravel()) > 72
    color[polar & land] = 0.93
    color[polar & ocean] = 0.90

    return color.reshape(len(lat), len(lon)).T    # shape (n, n//2)


def _gridlines_3d(fig: go.Figure, r: float = None, color: str = "rgba(60,140,200,0.22)",
                  width: float = 0.8):
    """Add latitude / longitude grid lines as Scatter3d traces on a sphere."""
    if r is None:
        # Lift gridlines clearly above the textured surface so they don't
        # z-fight or 'bleed through' from the back side of the globe.
        r = R_EARTH + 30
    n = 120

    # Longitude lines every 30°
    for lon_d in range(-180, 180, 30):
        lat_arr = np.linspace(-90, 90, n)
        lon_arr = np.full(n, lon_d, dtype=float)
        x, y, z = lla_to_ecef(lat_arr, lon_arr, np.zeros(n) + (r - R_EARTH))
        fig.add_trace(go.Scatter3d(
            x=list(x), y=list(y), z=list(z),
            mode="lines", line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
        ))

    # Latitude lines every 30°
    for lat_d in range(-60, 90, 30):
        lon_arr = np.linspace(-180, 180, n)
        lat_arr = np.full(n, lat_d, dtype=float)
        x, y, z = lla_to_ecef(lat_arr, lon_arr, np.zeros(n) + (r - R_EARTH))
        fig.add_trace(go.Scatter3d(
            x=list(x), y=list(y), z=list(z),
            mode="lines", line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
        ))

    # Equator (thicker)
    lon_arr = np.linspace(-180, 180, n)
    lat_arr = np.zeros(n)
    x, y, z = lla_to_ecef(lat_arr, lon_arr, np.zeros(n) + (r - R_EARTH))
    fig.add_trace(go.Scatter3d(
        x=list(x), y=list(y), z=list(z),
        mode="lines", line=dict(color="rgba(100,200,255,0.35)", width=1.2),
        hoverinfo="skip", showlegend=False,
    ))


# Simplified coastline polylines (lon, lat) — major continents only
_COASTLINE_POINTS: list = None

def _get_coastlines() -> list:
    """Return list of (lon_array, lat_array) polylines for major coastlines."""
    global _COASTLINE_POINTS
    if _COASTLINE_POINTS is not None:
        return _COASTLINE_POINTS

    segments = []

    # Africa outline
    segments.append((
        [-17,-12,-5,10,12,32,42,51,44,35,33,12,2,-5,-8,-17,-17],
        [15,5,0,-2,5,10,12,28,37,35,32,33,35,36,28,22,15]
    ))
    # Europe outline
    segments.append((
        [-10,-10,0,10,20,28,32,40,30,25,20,15,5,0,-10],
        [36,40,48,54,58,62,70,68,60,55,48,43,38,36,36]
    ))
    # Asia (simplified)
    segments.append((
        [28,35,40,50,55,65,80,100,110,120,130,140,160,170,170,140,130,
         120,105,100,90,80,70,55,42,35,28],
        [32,30,25,20,10,10,5,5,15,25,35,40,55,60,65,60,50,45,40,35,
         25,30,28,25,20,30,32]
    ))
    # North America
    segments.append((
        [-130,-140,-165,-165,-160,-140,-125,-110,-95,-82,-75,-65,-55,
         -65,-80,-85,-88,-95,-100,-110,-120,-130],
        [40,55,60,65,70,70,72,70,68,65,60,45,28,25,25,30,30,28,25,30,35,40]
    ))
    # South America
    segments.append((
        [-80,-75,-70,-65,-55,-45,-35,-40,-50,-60,-68,-72,-78,-80,-80],
        [10,5,0,-5,-5,-10,-15,-25,-35,-50,-55,-50,-40,-20,10]
    ))
    # Australia
    segments.append((
        [115,130,145,150,153,148,140,130,120,115],
        [-35,-12,-12,-20,-28,-38,-38,-35,-32,-35]
    ))

    _COASTLINE_POINTS = []
    for lons, lats in segments:
        _COASTLINE_POINTS.append((np.array(lons, float), np.array(lats, float)))
    return _COASTLINE_POINTS


def _densify_great_circle(lons: np.ndarray, lats: np.ndarray,
                          steps_per_seg: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide a polyline along great-circle arcs.

    Without this, two adjacent control points on the simplified continent
    polygons are connected by a *3D chord* (a straight line in ECEF), which
    sinks below the sphere surface for long segments and visually appears
    as awkward 'horizontal lines' cutting through the globe.
    """
    if len(lons) < 2:
        return np.asarray(lons, float), np.asarray(lats, float)
    out_lo, out_la = [float(lons[0])], [float(lats[0])]
    for i in range(1, len(lons)):
        lo1, la1 = float(lons[i-1]), float(lats[i-1])
        lo2, la2 = float(lons[i]),   float(lats[i])
        # Convert endpoints to unit vectors on the sphere
        p1 = np.array([
            np.cos(np.radians(la1)) * np.cos(np.radians(lo1)),
            np.cos(np.radians(la1)) * np.sin(np.radians(lo1)),
            np.sin(np.radians(la1)),
        ])
        p2 = np.array([
            np.cos(np.radians(la2)) * np.cos(np.radians(lo2)),
            np.cos(np.radians(la2)) * np.sin(np.radians(lo2)),
            np.sin(np.radians(la2)),
        ])
        dot = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
        omega = np.arccos(dot)
        if omega < 1e-6:
            out_lo.append(lo2); out_la.append(la2); continue
        sin_o = np.sin(omega)
        for k in range(1, steps_per_seg + 1):
            t = k / steps_per_seg
            a = np.sin((1 - t) * omega) / sin_o
            b = np.sin(t * omega) / sin_o
            p = a * p1 + b * p2
            out_lo.append(float(np.degrees(np.arctan2(p[1], p[0]))))
            out_la.append(float(np.degrees(np.arcsin(np.clip(p[2], -1, 1)))))
    return np.asarray(out_lo, float), np.asarray(out_la, float)


def _add_coastlines_3d(fig: go.Figure, r: float = None,
                       color: str = "rgba(80,200,140,0.55)", width: float = 1.5):
    """Render simplified coastline outlines on the sphere."""
    if r is None:
        # Lift slightly off the surface so chords don't z-fight or sink
        # through the textured Earth surface.
        r = R_EARTH + 30
    for lons, lats in _get_coastlines():
        d_lons, d_lats = _densify_great_circle(lons, lats, steps_per_seg=12)
        alt = np.full(len(d_lons), r - R_EARTH)
        x, y, z = lla_to_ecef(d_lats, d_lons, alt)
        fig.add_trace(go.Scatter3d(
            x=list(x), y=list(y), z=list(z),
            mode="lines", line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
        ))


def lla_to_ecef(lat_deg, lon_deg, alt_km):
    """LLA → ECEF (km). Accepts scalars or arrays."""
    lat = np.radians(np.asarray(lat_deg, float))
    lon = np.radians(np.asarray(lon_deg, float))
    r   = R_EARTH + np.asarray(alt_km,   float)
    return r * np.cos(lat) * np.cos(lon), \
           r * np.cos(lat) * np.sin(lon), \
           r * np.sin(lat)


# ─── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_positions_at_time(
    t_utc: datetime,
    alt_min: float = 0.0,
    alt_max: float = 2000.0,
    obj_type: str  = "ALL",
    limit:    int  = 15000,
) -> pd.DataFrame:
    """Propagate catalog objects to a target UTC epoch using latest GP/TLE."""
    from database.db import session_scope
    from sqlalchemy import text
    from propagator.sgp4_propagator import StateVector

    if not _SGP4_OK:
        st.error("缺少 sgp4 依赖，无法进行时刻递推。")
        return pd.DataFrame()

    t_utc = t_utc.replace(tzinfo=timezone.utc) if t_utc.tzinfo is None else t_utc
    obj_u = (obj_type or "ALL").upper()

    sql = text("""
        WITH latest_gp AS (
            SELECT DISTINCT ON (norad_cat_id)
                norad_cat_id,
                epoch,
                mean_motion,
                eccentricity,
                inclination AS gp_inclination,
                ra_of_asc_node,
                arg_of_pericenter,
                mean_anomaly,
                bstar,
                tle_line1,
                tle_line2
            FROM gp_elements
            WHERE norad_cat_id IS NOT NULL
            ORDER BY norad_cat_id, epoch DESC
        )
        SELECT
            co.norad_cat_id                                AS norad_cat_id,
            COALESCE(co.name, '')                          AS name,
            COALESCE(co.object_type, 'UNKNOWN')            AS object_type,
            COALESCE(co.country_code, '?')                 AS country_code,
            COALESCE(co.perigee_km, 0)                     AS perigee_km,
            COALESCE(co.apogee_km,  0)                     AS apogee_km,
            COALESCE(co.inclination, lg.gp_inclination, 0) AS inclination,
            lg.mean_motion                                 AS mean_motion,
            lg.eccentricity                                AS eccentricity,
            lg.ra_of_asc_node                              AS ra_of_asc_node,
            lg.arg_of_pericenter                           AS arg_of_pericenter,
            lg.mean_anomaly                                AS mean_anomaly,
            lg.bstar                                       AS bstar,
            lg.tle_line1                                   AS tle_line1,
            lg.tle_line2                                   AS tle_line2
        FROM catalog_objects co
        JOIN latest_gp lg ON lg.norad_cat_id = co.norad_cat_id
        WHERE co.apogee_km  >= :alt_min
          AND co.perigee_km <= :alt_max_buf
          AND (:obj_type = 'ALL' OR UPPER(co.object_type) = :obj_type)
        ORDER BY co.norad_cat_id
        LIMIT :lim
    """)

    def _build_satrec(row) -> Optional["Satrec"]:
        line1 = row.tle_line1
        line2 = row.tle_line2
        if line1 and line2:
            try:
                return Satrec.twoline2rv(str(line1), str(line2))
            except Exception:
                pass
        try:
            sat = Satrec()
            sat.sgp4init(
                WGS84,
                "i",
                int(row.norad_cat_id),
                float(row.bstar or 0.0),
                0.0,
                0.0,
                float(row.eccentricity),
                math.radians(float(row.arg_of_pericenter)),
                math.radians(float(row.gp_inclination if hasattr(row, "gp_inclination") else row.inclination)),
                math.radians(float(row.mean_anomaly)),
                float(row.mean_motion) * 2 * math.pi / 1440.0,
                math.radians(float(row.ra_of_asc_node)),
            )
            return sat
        except Exception:
            return None

    try:
        with session_scope() as sess:
            rows = sess.execute(sql, {
                "alt_min":     alt_min,
                "alt_max_buf": alt_max + 500,
                "obj_type":    obj_u,
                "lim":         limit,
            }).fetchall()

        jd, fr = jday(
            t_utc.year, t_utc.month, t_utc.day,
            t_utc.hour, t_utc.minute, t_utc.second + t_utc.microsecond / 1e6
        )
        out = []
        for row in rows:
            sat = _build_satrec(row)
            if sat is None:
                continue
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:
                continue
            sv = StateVector(
                epoch=t_utc, x=float(r[0]), y=float(r[1]), z=float(r[2]),
                vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
            )
            lat, lon, alt = sv.to_geodetic()
            if not np.isfinite(alt) or alt < alt_min or alt > alt_max:
                continue
            out.append({
                "norad_cat_id": int(row.norad_cat_id),
                "name": row.name,
                "object_type": row.object_type,
                "country_code": row.country_code,
                "perigee_km": float(row.perigee_km or 0.0),
                "apogee_km": float(row.apogee_km or 0.0),
                "inclination": float(row.inclination or 0.0),
                "lon": float(lon),
                "lat": float(lat),
                "alt_km": float(max(0.0, alt)),
            })

        df = pd.DataFrame(out)
        return df
    except Exception as exc:
        st.error(f"SGP4 递推位置查询错误：{exc}")
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_positions_now(
    alt_min: float = 0.0,
    alt_max: float = 2000.0,
    obj_type: str  = "ALL",
    limit:    int  = 15000,
) -> pd.DataFrame:
    """Backward-compatible wrapper: positions at current UTC."""
    return load_positions_at_time(
        t_utc=datetime.now(timezone.utc),
        alt_min=alt_min, alt_max=alt_max, obj_type=obj_type, limit=limit,
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_layer_stats() -> pd.DataFrame:
    """Count objects per layer using propagated current-time positions."""
    now_utc = datetime.now(timezone.utc)
    rows = []
    try:
        for layer in ALTITUDE_LAYERS:
            dfl = load_positions_at_time(
                t_utc=now_utc,
                alt_min=float(layer["alt_min"]),
                alt_max=float(layer["alt_max"]),
                obj_type="ALL",
                limit=60000,
            )
            rows.append({**layer, "count": int(len(dfl))})
    except Exception:
        rows = [{**l, "count": 0} for l in ALTITUDE_LAYERS]
    return pd.DataFrame(rows)


@st.cache_data(ttl=30, show_spinner=False)
def load_debris_near_point(
    lat: float, lon: float, alt_km: float,
    t_utc: datetime,
    radius_km: float = 500.0,
    limit: int = 400,
) -> pd.DataFrame:
    """Propagate nearby debris at t_utc and return nearest objects."""
    try:
        alt_min = max(0.0, float(alt_km) - float(radius_km) - 400.0)
        alt_max = min(120000.0, float(alt_km) + float(radius_km) + 400.0)
        candidate_cap = int(max(4000, min(15000, limit * 25)))

        cands = load_positions_at_time(
            t_utc=t_utc,
            alt_min=alt_min,
            alt_max=alt_max,
            obj_type="ALL",
            limit=candidate_cap,
        )
        if cands.empty:
            return pd.DataFrame()

        rx, ry, rz = lla_to_ecef(lat, lon, alt_km)
        dx, dy, dz = lla_to_ecef(cands["lat"].values, cands["lon"].values, cands["alt_km"].values)
        dist = np.sqrt((dx - rx) ** 2 + (dy - ry) ** 2 + (dz - rz) ** 2)
        cands = cands.copy()
        cands["dist_km"] = dist
        cands = cands[cands["dist_km"] <= float(radius_km)]
        cands = cands.sort_values("dist_km").head(limit)
        return cands[["norad_cat_id", "name", "object_type", "lon", "lat", "alt_km", "dist_km"]]
    except Exception:
        return pd.DataFrame()


# ─── Figure helpers ────────────────────────────────────────────────────────────
def _hex_color_col(df: pd.DataFrame) -> list:
    return df["object_type"].map(_TYPE_HEX).fillna("#888888").tolist()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#rrggbb' hex string to an 'rgba(r,g,b,a)' string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_3d_layout(fig: go.Figure, height: int = 680, title: str = ""):
    ax = dict(showticklabels=False, showgrid=False, zeroline=False,
              showline=False, showspikes=False, backgroundcolor=SCENE_BG,
              gridcolor="rgba(0,0,0,0)", title="")
    fig.update_layout(
        scene=dict(
            xaxis=ax, yaxis=ax, zaxis=ax,
            bgcolor=SCENE_BG,
            aspectmode="data",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=1.6, y=1.6, z=0.8),
            ),
        ),
        paper_bgcolor=DARK_BG,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        height=height,
        showlegend=False,
        title=dict(text=title, font=dict(color="#cccccc", size=14)) if title else None,
    )


_EARTH_COLORSCALE = [
    [0.0, "#030a15"],     # deep ocean
    [0.08, "#051525"],    # ocean
    [0.18, "#082a40"],    # mid ocean
    [0.30, "#0c3a55"],    # shallow ocean
    [0.42, "#1a5c30"],    # coastal lowland
    [0.50, "#287038"],    # lowland green
    [0.58, "#3a7a35"],    # mid green
    [0.65, "#4a8530"],    # upper green
    [0.72, "#607828"],    # savanna/steppe
    [0.80, "#807020"],    # highland brown
    [0.85, "#956a1a"],    # mountain
    [0.90, "#c8d8e8"],    # ice edge
    [0.95, "#e0eaf4"],    # snow
    [1.0, "#f0f4f8"],     # bright ice
]


def _add_earth_grid_only(fig: go.Figure, n: int = 80):
    """Add a plain dark sphere with lat/lon gridlines only — no land/ocean/coastlines."""
    x_e, y_e, z_e = _earth_mesh(n)
    color_arr = np.full_like(x_e, 0.15)
    fig.add_trace(go.Surface(
        x=x_e, y=y_e, z=z_e,
        surfacecolor=color_arr,
        colorscale=[[0, "#1a3a5e"], [1, "#2a5a8a"]],
        showscale=False, opacity=1.0,
        lighting=dict(ambient=0.7, diffuse=0.4, specular=0.05),
        hoverinfo="skip", name="Earth",
    ))
    _gridlines_3d(fig)


def _add_earth(fig: go.Figure, n: int = 110, opacity: float = 1.0,
               show_grid: bool = True, show_coastlines: bool = False):
    """Add solid Earth sphere with land/ocean coloring and gridlines.

    Coastlines are off by default — the surface already shows the land/ocean
    boundary via its texture; the simplified hand-drawn polylines on top
    add visual clutter (3D chords sinking through the globe) without
    contributing useful information.
    """
    x_e, y_e, z_e = _earth_mesh(n)
    topo = _build_earth_texture(n)

    fig.add_trace(go.Surface(
        x=x_e, y=y_e, z=z_e,
        surfacecolor=topo,
        colorscale=_EARTH_COLORSCALE,
        showscale=False,
        opacity=opacity,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.15, roughness=0.8,
                      fresnel=0.1),
        lightposition=dict(x=100000, y=50000, z=80000),
        hoverinfo="skip",
        name="Earth",
    ))

    if show_grid:
        _gridlines_3d(fig)
    if show_coastlines:
        _add_coastlines_3d(fig)


def _add_earth_local(fig: go.Figure, rx: float, ry: float, rz: float, n: int = 55):
    """Full-size Earth at its correct position in the local rocket frame."""
    ecx, ecy, ecz = -rx, -ry, -rz
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi,     n // 2)
    xe = ecx + R_EARTH * np.outer(np.cos(u), np.sin(v))
    ye = ecy + R_EARTH * np.outer(np.sin(u), np.sin(v))
    ze = ecz + R_EARTH * np.outer(np.ones(n), np.cos(v))

    topo = _build_earth_texture(n)

    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        surfacecolor=topo,
        colorscale=_EARTH_COLORSCALE,
        showscale=False, opacity=1.0,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.15, roughness=0.8),
        lightposition=dict(x=100000, y=50000, z=80000),
        hoverinfo="skip", showlegend=False, name="Earth",
    ))

    _n = 80
    for lon_d in range(-180, 180, 30):
        lat_arr = np.linspace(-90, 90, _n)
        lon_arr = np.full(_n, lon_d, dtype=float)
        gx, gy, gz = lla_to_ecef(lat_arr, lon_arr, np.full(_n, 2.0))
        fig.add_trace(go.Scatter3d(
            x=list(np.asarray(gx) - rx), y=list(np.asarray(gy) - ry),
            z=list(np.asarray(gz) - rz),
            mode="lines", line=dict(color="rgba(60,140,200,0.18)", width=0.6),
            hoverinfo="skip", showlegend=False,
        ))
    for lat_d in range(-60, 90, 30):
        lon_arr = np.linspace(-180, 180, _n)
        lat_arr = np.full(_n, lat_d, dtype=float)
        gx, gy, gz = lla_to_ecef(lat_arr, lon_arr, np.full(_n, 2.0))
        fig.add_trace(go.Scatter3d(
            x=list(np.asarray(gx) - rx), y=list(np.asarray(gy) - ry),
            z=list(np.asarray(gz) - rz),
            mode="lines", line=dict(color="rgba(60,140,200,0.18)", width=0.6),
            hoverinfo="skip", showlegend=False,
        ))
    # Coastlines intentionally omitted — the textured Earth surface already
    # encodes land/ocean colours; drawing simplified outlines on top creates
    # visible 3D chords cutting through the globe.


def _add_altitude_shell(fig: go.Figure, alt_km: float, color: str, opacity: float = 0.08):
    """Add a semi-transparent spherical shell at a given altitude."""
    r = R_EARTH + alt_km
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, np.pi,     45)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(60), np.cos(v))
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        hoverinfo="skip",
    ))


# ─── Tab 1: Globe figure builders ─────────────────────────────────────────────
def make_globe_ortho(df: pd.DataFrame) -> go.Figure:
    """Plotly Scattergeo orthographic projection (always-available globe view)."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=DARK_BG, height=520)
        return fig

    colors = _hex_color_col(df)
    hover = (
        "NORAD: "  + df["norad_cat_id"].astype(str)
        + "<br>"   + df["name"].fillna("").str[:28]
        + "<br>类型: " + df["object_type"].fillna("?")
        + "<br>高度: " + df["alt_km"].round(0).astype(int).astype(str) + " km"
        + "<br>倾角: " + df["inclination"].round(1).astype(str) + "°"
    )
    fig = go.Figure(go.Scattergeo(
        lon=df["lon"], lat=df["lat"],
        mode="markers",
        marker=dict(size=2.5, color=colors, opacity=0.72, line=dict(width=0)),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_geos(
        projection_type="orthographic",
        showcoastlines=True,  coastlinecolor="#3a7a5a", coastlinewidth=1.2,
        showland=True,        landcolor="#1a3525",
        showocean=True,       oceancolor="#071520",
        showlakes=True,       lakecolor="#0a1f30",
        showrivers=True,      rivercolor="#0d2a40", riverwidth=0.5,
        showcountries=True,   countrycolor="#2a4a3a", countrywidth=0.5,
        showframe=False,
        bgcolor=DARK_BG,
        lonaxis=dict(showgrid=True, gridcolor="#1a2e3e", dtick=30, gridwidth=0.5),
        lataxis=dict(showgrid=True, gridcolor="#1a2e3e", dtick=30, gridwidth=0.5),
        projection_rotation=dict(lon=110, lat=20, roll=0),
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG, margin=dict(l=0, r=0, t=0, b=0),
        height=530, showlegend=False,
    )
    return fig


# ─── Orbit propagation trace ───────────────────────────────────────────────────
_TRACE_COLORS = [
    "#FF6B6B", "#00CCFF", "#FFEE00", "#6BCB77", "#4D96FF",
    "#FF9F45", "#C77DFF", "#FF87CA", "#A8E6CF", "#FFC93C",
]


@st.cache_data(ttl=300, show_spinner=False)
def propagate_orbit_traces(norad_ids: tuple, n_orbits: float = 1.5) -> dict:
    """SGP4-propagate N orbits for each NORAD ID; return {nid: {name, object_type, lat, lon, alt_km}}."""
    if not norad_ids or not _SGP4_OK:
        return {}
    from database.db import session_scope as _scope
    from sqlalchemy import text as _text
    from propagator.sgp4_propagator import StateVector as _SV

    try:
        with _scope() as sess:
            rows = sess.execute(_text("""
                SELECT DISTINCT ON (g.norad_cat_id)
                    g.norad_cat_id, g.tle_line1, g.tle_line2, g.mean_motion,
                    co.name, co.object_type
                FROM gp_elements g
                JOIN catalog_objects co ON co.norad_cat_id = g.norad_cat_id
                WHERE g.norad_cat_id = ANY(:ids)
                ORDER BY g.norad_cat_id, g.epoch DESC
            """), {"ids": list(norad_ids)}).fetchall()
    except Exception:
        return {}

    t0 = datetime.now(timezone.utc)
    result: dict = {}
    for row in rows:
        try:
            sat = Satrec.twoline2rv(str(row.tle_line1), str(row.tle_line2))
        except Exception:
            continue
        period_min = 1440.0 / max(float(row.mean_motion), 0.01)
        total_min  = period_min * n_orbits
        n_pts      = max(80, min(600, int(total_min / max(0.2, period_min / 200.0))))
        step_min   = total_min / n_pts

        lats: list = []
        lons: list = []
        alts: list = []
        eci_r: list = []   # [x, y, z] km  (TEME ≈ J2000)
        eci_v: list = []   # [vx, vy, vz] km/s
        epochs: list = []
        for i in range(n_pts + 1):
            t   = t0 + timedelta(minutes=i * step_min)
            jd2, fr2 = jday(t.year, t.month, t.day,
                            t.hour, t.minute, t.second + t.microsecond / 1e6)
            e, r, v = sat.sgp4(jd2, fr2)
            if e != 0:
                continue
            sv = _SV(epoch=t, x=float(r[0]), y=float(r[1]), z=float(r[2]),
                     vx=float(v[0]), vy=float(v[1]), vz=float(v[2]))
            lat2, lon2, alt2 = sv.to_geodetic()
            if not np.isfinite(alt2):
                continue
            lats.append(float(lat2))
            lons.append(float(lon2))
            alts.append(float(alt2))
            eci_r.append([float(r[0]), float(r[1]), float(r[2])])
            eci_v.append([float(v[0]), float(v[1]), float(v[2])])
            epochs.append(t)

        if lats:
            result[int(row.norad_cat_id)] = {
                "name": str(row.name or f"NORAD {row.norad_cat_id}"),
                "object_type": str(row.object_type or "UNKNOWN"),
                "lat": lats, "lon": lons, "alt_km": alts,
                "eci_r": eci_r, "eci_v": eci_v, "epochs": epochs,
            }
    return result


def make_orbit_trace_fig(traces: dict) -> go.Figure:
    """3D Earth + time-ordered gradient line traces for each propagated orbit.

    Each orbit line uses a numeric colour array + per-object colorscale so
    the trajectory fades from dim-grey (start) to the object's full colour
    (end), giving an unambiguous time-progression cue without arrows.
    A white hollow circle marks the current epoch (start), a filled circle
    marks the propagation end-point.
    """
    fig = go.Figure()
    _add_earth(fig, n=100)

    max_alt = 500.0
    for i, (nid, info) in enumerate(traces.items()):
        lats = np.array(info["lat"])
        lons = np.array(info["lon"])
        alts = np.array(info["alt_km"])
        max_alt = max(max_alt, float(alts.max()))
        x, y, z = lla_to_ecef(lats, lons, alts)
        n      = len(x)
        color  = _TRACE_COLORS[i % len(_TRACE_COLORS)]
        name   = info["name"][:22]
        otype  = info.get("object_type", "?")

        # Numeric time index 0…n-1 drives the gradient
        t_idx  = list(range(n))
        fig.add_trace(go.Scatter3d(
            x=list(x), y=list(y), z=list(z),
            mode="lines",
            line=dict(
                color=t_idx,
                # dim grey at t=0, object colour at t=max
                colorscale=[[0.0, "rgba(90,90,90,0.35)"], [1.0, color]],
                cmin=0, cmax=n - 1,
                width=2.5,
            ),
            name=name,
            legendgroup=str(nid),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"NORAD {nid}  |  {_TYPE_CN.get(otype, otype)}<br>"
                "高度: %{text} km<extra></extra>"
            ),
            text=[f"{a:.0f}" for a in alts],
        ))

        # ── Start marker (hollow white circle = "now") ───────────────────────
        fig.add_trace(go.Scatter3d(
            x=[float(x[0])], y=[float(y[0])], z=[float(z[0])],
            mode="markers",
            marker=dict(size=5, color="white", symbol="circle-open",
                        line=dict(color=color, width=1.5)),
            name=f"起始 · {name[:14]}",
            legendgroup=str(nid),
            showlegend=False,
            hovertemplate=f"起始  {name}<br>高度 {alts[0]:.0f} km<extra></extra>",
        ))
        # ── End marker (filled object colour = future) ────────────────────────
        fig.add_trace(go.Scatter3d(
            x=[float(x[-1])], y=[float(y[-1])], z=[float(z[-1])],
            mode="markers",
            marker=dict(size=4, color=color, symbol="circle"),
            name=f"终点 · {name[:14]}",
            legendgroup=str(nid),
            showlegend=False,
            hovertemplate=f"终点  {name}<br>高度 {alts[-1]:.0f} km<extra></extra>",
        ))

    scene_r = R_EARTH + max_alt * 1.25 + 200
    earth_ax = dict(range=[-scene_r, scene_r])
    _apply_3d_layout(fig, height=720)
    fig.update_layout(
        scene=dict(
            xaxis=earth_ax, yaxis=earth_ax, zaxis=earth_ax,
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
        ),
        legend=dict(
            x=0.01, y=0.99, font=dict(color="white", size=10),
            bgcolor="rgba(0,0,0,0.45)", bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1, tracegroupgap=2,
        ),
    )
    return fig


# ─── Orbit propagation trace ───────────────────────────────────────────────────
_TRACE_COLORS = [
    "#FF6B6B", "#00CCFF", "#FFEE00", "#6BCB77", "#4D96FF",
    "#FF9F45", "#C77DFF", "#FF87CA", "#A8E6CF", "#FFC93C",
]



def make_altitude_hist(
    df: pd.DataFrame,
    *,
    x_max: float,
    title: str,
    log_x: bool = False,
    bin_size: float = 40.0,
) -> go.Figure:
    """Stacked altitude histogram by object type with optional log-space binning."""
    import numpy as _np
    fig = go.Figure()
    if df.empty:
        return fig

    for otype, color in _TYPE_HEX.items():
        sub = df[df["object_type"] == otype]
        if sub.empty:
            continue
        if log_x:
            # Bin in log space so MEO/GEO appear as real-width bars, not invisible slivers
            vals = _np.log10(_np.maximum(sub["alt_km"].values, 1.0))
            data_log_min = float(_np.log10(max(1.0, df["alt_km"].clip(lower=1).min())))
            data_log_max = float(_np.log10(x_max))
            nbins = 50
            bsize = (data_log_max - data_log_min) / nbins
            xbins_d = dict(start=data_log_min, end=data_log_max, size=bsize)
        else:
            vals = sub["alt_km"].values
            xbins_d = dict(start=0, end=max(1.0, x_max), size=bin_size)
        fig.add_trace(go.Histogram(
            x=vals,
            name=_TYPE_CN.get(otype, otype),
            marker_color=color,
            opacity=0.88,
            xbins=xbins_d,
        ))

    fig.update_layout(
        barmode="stack",
        yaxis_title="目标数",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_color="#334155",
        title=dict(text=""),      # empty string avoids "undefined" JS artefact
        legend=dict(
            orientation="h",
            x=0.5, y=1.0,
            xanchor="center", yanchor="bottom",
            font=dict(color="#334155", size=11),
            tracegroupgap=0,
        ),
        margin=dict(l=44, r=10, t=44, b=40),
        height=276,
    )

    if log_x:
        data_log_min = float(_np.log10(max(1.0, df["alt_km"].clip(lower=1).min())))
        data_log_max = float(_np.log10(x_max))
        tick_km = [200, 500, 1000, 2000, 5000, 10000, 35786]
        tick_km = [v for v in tick_km if _np.log10(v) <= data_log_max + 0.1]
        tick_lbl = [("GEO" if v == 35786 else (f"{v//1000:.0f}k" if v >= 1000 else str(v)))
                    for v in tick_km]
        fig.update_xaxes(
            tickvals=[_np.log10(v) for v in tick_km],
            ticktext=tick_lbl,
            title_text="轨道高度 (km)",
            range=[data_log_min - 0.05, data_log_max + 0.05],
            gridcolor="rgba(148,163,184,0.22)",
            color="#475569",
        )
    else:
        fig.update_xaxes(
            title_text="轨道高度 (km)",
            range=[0, x_max],
            gridcolor="rgba(148,163,184,0.22)",
            color="#475569",
        )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)", color="#475569")
    return fig


# ─── Tab 2: 3D sphere ─────────────────────────────────────────────────────────
def make_3d_sphere(
    df: pd.DataFrame,
    layer: dict | None = None,
    max_pts: int = 8000,
) -> go.Figure:
    """Earth + debris scatter. If layer given, dim out-of-band points."""
    fig = go.Figure()
    _add_earth(fig, n=110)

    if df.empty:
        _apply_3d_layout(fig)
        return fig

    # Sample if needed
    plot_df = df.sample(min(len(df), max_pts), random_state=42) if len(df) > max_pts else df.copy()
    x, y, z = lla_to_ecef(plot_df["lat"].values, plot_df["lon"].values, plot_df["alt_km"].values)
    base_colors = _hex_color_col(plot_df)

    if layer is not None:
        colors = [_hex_to_rgba(c, 0.90) for c in base_colors]
        sizes  = [2.0] * len(plot_df)
    else:
        colors = [_hex_to_rgba(c, 0.78) for c in base_colors]
        sizes  = [1.8] * len(plot_df)

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=0)),
        text=(
            plot_df["name"].fillna("").str[:28]
            + "<br>NORAD: " + plot_df["norad_cat_id"].astype(str)
            + "<br>Alt: "   + plot_df["alt_km"].round(0).astype(int).astype(str) + " km"
            + "<br>"        + plot_df["object_type"].fillna("?")
        ),
        hovertemplate="%{text}<extra></extra>",
        name="",
    ))

    # Remove altitude-shell edge highlights for cleaner visual.

    # Explicit Earth-centric scene range — prevents camera drift when
    # max_pts > actual point count (Plotly falls back to data bbox which may
    # be very different from Earth scale when few/no points are present)
    scene_r = R_EARTH + (layer["alt_max"] + 500 if layer else 2500)
    earth_ax = dict(range=[-scene_r, scene_r])
    _apply_3d_layout(fig, height=680)
    fig.update_layout(scene=dict(
        xaxis=earth_ax, yaxis=earth_ax, zaxis=earth_ax,
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
    ))
    return fig


# ─── Tab 3: Mission safety ─────────────────────────────────────────────────────
def make_mission_fig(
    traj_points: list,
    t_slider_s:  float,
    nearby_df:   pd.DataFrame,
    risk_events: list,
    rocket_state: Optional[dict] = None,
    traj_half_window: int = 40,   # show N future traj points after current idx
    launch_utc=None,              # kept for API compatibility (unused)
    past_lla: Optional[list] = None,  # pre-computed (lat,lon,alt) arc from caller
) -> go.Figure:
    """3D figure — LOCAL rocket frame + full-size Earth as backdrop.

    Rocket sits at origin (0, 0, 0).  Earth is placed at its correct offset
    (-rx, -ry, -rz) — full size sphere visible.  scene.*.range is set to
    ±r_norm so the whole Earth fits, while the initial camera is positioned
    close to the rocket for a useful starting view.

    past_lla: pre-computed list of (lat_deg, lon_deg, alt_km) tuples for the
    complete past trajectory including any coast extension. The last tuple
    must match rocket_state exactly so the arc ends at local origin (0,0,0).
    When None, only the nominal sim points up to t_slider_s are used.
    """
    fig = go.Figure()

    if not traj_points:
        _apply_3d_layout(fig, height=640)
        return fig

    # ── Current rocket state ─────────────────────────────────────────────────
    times_arr = np.array([p.t_met_s for p in traj_points])
    idx  = int(np.searchsorted(times_arr, t_slider_s).clip(0, len(traj_points) - 1))
    rp_nom = traj_points[idx]
    if rocket_state is None:
        rp_live = {
            "lat_deg": float(rp_nom.lat_deg),
            "lon_deg": float(rp_nom.lon_deg),
            "alt_km": float(rp_nom.alt_km),
            "vel_kms": float(np.linalg.norm(rp_nom.vel_eci)),
        }
    else:
        rp_live = {
            "lat_deg": float(rocket_state["lat_deg"]),
            "lon_deg": float(rocket_state["lon_deg"]),
            "alt_km": float(rocket_state["alt_km"]),
            "vel_kms": float(rocket_state["vel_kms"]),
        }

    rx, ry, rz = (float(v) for v in lla_to_ecef(rp_live["lat_deg"], rp_live["lon_deg"], rp_live["alt_km"]))
    r_norm = float(np.sqrt(rx**2 + ry**2 + rz**2))   # distance from Earth center → rocket
    vel_kms = rp_live["vel_kms"]

    # ── Scene range: large enough to fit the COMPLETE Earth sphere ───────────
    # The Earth centre is at (-rx,-ry,-rz); its far edge is r_norm + R_EARTH
    # from the rocket (origin).  Without enough range the back hemisphere is
    # clipped by the scene bounding box → only "one face" of Earth visible.
    scene_km = (r_norm + R_EARTH) * 1.05

    # ── Earth as full sphere (local frame) ───────────────────────────────────
    _add_earth_local(fig, rx, ry, rz, n=55)

    # ── Full already-flown arc: launch site → current position ───────────────
    def _local(pts):
        la = [p.lat_deg for p in pts]
        lo = [p.lon_deg for p in pts]
        al = [p.alt_km  for p in pts]
        wx, wy, wz = lla_to_ecef(la, lo, al)
        return (np.asarray(wx) - rx,
                np.asarray(wy) - ry,
                np.asarray(wz) - rz)

    in_coast = (t_slider_s > times_arr[-1])

    # ── Past arc: single continuous trace from launch → rocket ───────────────
    # Use pre-computed past_lla when available (includes coast extension).
    # The last entry of past_lla is numerically identical to rocket_state, so
    # lla_to_ecef(last) - (rx,ry,rz) = (0,0,0) → arc ends exactly at rocket.
    if past_lla and len(past_lla) >= 2:
        _lats = [p[0] for p in past_lla]
        _lons = [p[1] for p in past_lla]
        _alts = [p[2] for p in past_lla]
        _px, _py, _pz = lla_to_ecef(_lats, _lons, _alts)
        _arc_x = np.asarray(_px) - rx
        _arc_y = np.asarray(_py) - ry
        _arc_z = np.asarray(_pz) - rz
        fig.add_trace(go.Scatter3d(
            x=_arc_x, y=_arc_y, z=_arc_z,
            mode="lines",
            line=dict(color="#00CCFF", width=2.5),
            hoverinfo="skip", name="已飞航段",
        ))
    else:
        # Fallback: nominal points only up to current index
        past_pts = traj_points[: idx + 1]
        if len(past_pts) >= 2:
            px_l, py_l, pz_l = _local(past_pts)
            fig.add_trace(go.Scatter3d(
                x=px_l, y=py_l, z=pz_l,
                mode="lines",
                line=dict(color="#00CCFF", width=2.5),
                hoverinfo="skip", name="已飞航段",
            ))

    # ── Future window (nominal phase only) ───────────────────────────────────
    if not in_coast:
        i1 = min(len(traj_points), idx + traj_half_window + 1)
        fut_pts = traj_points[idx: i1]
        if len(fut_pts) >= 2:
            fx_l, fy_l, fz_l = _local(fut_pts)
            fig.add_trace(go.Scatter3d(
                x=fx_l, y=fy_l, z=fz_l,
                mode="lines",
                line=dict(color="#1a3a6a", width=1.5),
                opacity=0.65, hoverinfo="skip", name="待飞航段",
            ))

    # ── Rocket marker at local origin ────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=11, color="#00FF55", symbol="diamond",
                    line=dict(color="white", width=1.5)),
        text=[f"T+{t_slider_s:.0f}s"],
        textfont=dict(color="white", size=11),
        textposition="top center",
        hovertemplate=(
            f"LV · T+{t_slider_s:.0f}s<br>"
            f"Alt: {rp_live['alt_km']:.1f} km<br>"
            f"Lat: {rp_live['lat_deg']:.2f}°  Lon: {rp_live['lon_deg']:.2f}°<br>"
            f"V: {vel_kms:.3f} km/s"
            "<extra></extra>"
        ),
        name="火箭",
    ))

    # ── Nearby debris in local frame ─────────────────────────────────────────
    if not nearby_df.empty:
        nd = nearby_df.dropna(subset=["lat", "lon", "alt_km"]).head(300)
        if len(nd):
            ndx, ndy, ndz = lla_to_ecef(nd["lat"].values, nd["lon"].values,
                                         nd["alt_km"].values)
            dx_l = np.asarray(ndx) - rx
            dy_l = np.asarray(ndy) - ry
            dz_l = np.asarray(ndz) - rz
            dist = nd["dist_km"].values.astype(float)
            d_colors = [
                "#FF2222" if d < 20  else
                "#FF8800" if d < 100 else
                "#FFDD00" if d < 300 else "#6699CC"
                for d in dist
            ]
            d_sizes = [7 if d < 20 else 5 if d < 100 else 4 for d in dist]
            fig.add_trace(go.Scatter3d(
                x=dx_l, y=dy_l, z=dz_l, mode="markers",
                marker=dict(size=d_sizes, color=d_colors, opacity=0.85,
                            line=dict(width=0)),
                text=(
                    nd["name"].fillna("").str[:24]
                    + "<br>dist: " + nd["dist_km"].round(1).astype(str) + " km"
                    + "<br>" + nd["object_type"].fillna("?")
                ),
                hovertemplate="%{text}<extra></extra>",
                name="附近目标",
            ))

    # ── Risk conjunction lines ────────────────────────────────────────────────
    if risk_events and not nearby_df.empty:
        for ev in sorted(risk_events, key=lambda e: e.probability, reverse=True)[:5]:
            if ev.probability < 1e-9 or ev.norad_cat_id < 0:
                continue
            row = nearby_df[nearby_df["norad_cat_id"] == ev.norad_cat_id]
            if row.empty:
                continue
            ex, ey, ez = lla_to_ecef(
                float(row["lat"].iloc[0]),
                float(row["lon"].iloc[0]),
                float(row["alt_km"].iloc[0]),
            )
            pc_color = (
                "#FF2222" if ev.probability >= 1e-5 else
                "#FF8800" if ev.probability >= 1e-6 else "#FFDD00"
            )
            fig.add_trace(go.Scatter3d(
                x=[0, ex - rx, None], y=[0, ey - ry, None], z=[0, ez - rz, None],
                mode="lines", line=dict(color=pc_color, width=2),
                hoverinfo="skip", showlegend=False,
            ))

    # ── Layout: no axes, scene range = full Earth, camera near rocket ────────
    # scene_km covers full Earth sphere.  Camera eye at normalized distance
    # ~2.5 → physical dist ≈ 2.5 × scene_km from origin ≈ above rocket.
    # For deep zoom into debris: use scroll wheel to zoom in from this view.
    ax = dict(
        showticklabels=False, showgrid=False, zeroline=False,
        showline=False, showspikes=False,
        backgroundcolor=SCENE_BG,
        gridcolor="rgba(0,0,0,0)", title="",
        range=[-scene_km, scene_km],
    )
    # Initial camera: position slightly above and to the side of rocket origin,
    # looking at origin (rocket).  eye is in scene-normalized units (1 unit = scene_km).
    # We want camera ~2 rocket-altitudes away from rocket → eye magnitude ~2.
    fig.update_layout(
        scene=dict(
            xaxis=ax, yaxis=ax, zaxis=ax,
            bgcolor=SCENE_BG,
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor=DARK_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=640,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(0,8,20,0.60)",
            font=dict(color="#aaaaaa", size=10),
        ),
    )
    return fig



def make_proximity_2d(
    rp,
    nearby_df: pd.DataFrame,
    search_radius_km: float,
    rocket_state: Optional[dict] = None,
) -> go.Figure:
    """2D East-offset × altitude-offset scatter in rocket local frame.

    Rocket sits at origin (0, 0).  x-axis = East offset (km), y-axis = altitude
    offset relative to rocket (km).  Supports unlimited zoom in Streamlit.
    """
    import math

    nd = nearby_df.dropna(subset=["lat", "lon", "alt_km"])
    if nd.empty:
        return None

    lat_r = float(rocket_state["lat_deg"]) if rocket_state else float(rp.lat_deg)
    lon_r = float(rocket_state["lon_deg"]) if rocket_state else float(rp.lon_deg)
    alt_r = float(rocket_state["alt_km"]) if rocket_state else float(rp.alt_km)

    cos_lat = math.cos(math.radians(lat_r))
    KM_PER_DEG = 111.32

    east_km  = (nd["lon"].values - lon_r) * cos_lat * KM_PER_DEG
    alt_off  = nd["alt_km"].values - alt_r

    dist     = nd["dist_km"].values.astype(float)
    colors   = [
        "#FF2222" if d < 20  else
        "#FF8800" if d < 100 else
        "#FFDD00" if d < 300 else "#4477aa"
        for d in dist
    ]
    sizes    = [10 if d < 20 else 7 if d < 100 else 4 for d in dist]
    hover    = (
        nd["name"].fillna("").str[:24]
        + "<br>3D dist: " + nd["dist_km"].round(1).astype(str) + " km"
        + "<br>type: "    + nd["object_type"].fillna("?")
        + "<br>alt: "     + nd["alt_km"].round(1).astype(str) + " km"
    ).tolist()

    fig = go.Figure()

    # ── Reference rings (approximate 3D-distance circles, North offset ignored) ──
    θ = np.linspace(0, 2 * np.pi, 300)
    for r_c, clr, label in [
        (20,               "rgba(255,34,34,0.55)",  "20 km"),
        (100,              "rgba(255,136,0,0.45)",  "100 km"),
        (300,              "rgba(255,221,0,0.30)",  "300 km"),
        (search_radius_km, "rgba(80,120,180,0.20)", f"{search_radius_km:.0f} km"),
    ]:
        if r_c > search_radius_km * 1.15:
            continue
        fig.add_trace(go.Scatter(
            x=r_c * np.cos(θ), y=r_c * np.sin(θ),
            mode="lines",
            line=dict(color=clr, width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            x=r_c * 0.707, y=r_c * 0.707,
            text=label, font=dict(size=9, color="#aaaaaa"),
            showarrow=False, xanchor="left",
        )

    # ── Debris scatter ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=east_km, y=alt_off,
        mode="markers",
        marker=dict(
            size=sizes, color=colors, opacity=0.88,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="附近目标",
    ))

    # ── Rocket at origin ──────────────────────────────────────────────────────
    vel_kms = float(rocket_state["vel_kms"]) if rocket_state else float(np.linalg.norm(rp.vel_eci))
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=16, color="#00FF55", symbol="star",
                    line=dict(color="white", width=1.5)),
        text=["LV"],
        textfont=dict(size=13),
        textposition="top center",
        hovertemplate=(
            f"LV · T+{rp.t_met_s:.0f}s<br>"
            f"Alt: {alt_r:.1f} km<br>"
            f"Lat: {lat_r:.2f}°  Lon: {lon_r:.2f}°<br>"
            f"V: {vel_kms:.3f} km/s"
            "<extra></extra>"
        ),
        name="火箭",
    ))

    lim = search_radius_km * 1.05
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor="#070f1a",
        margin=dict(l=52, r=16, t=36, b=46),
        height=380,
        xaxis=dict(
            title="East 偏移 (km)", color="#8899aa",
            gridcolor="#172030", zerolinecolor="#2a3f55", zerolinewidth=1.2,
            range=[-lim, lim],
        ),
        yaxis=dict(
            title="高度偏移 (km)", color="#8899aa",
            gridcolor="#172030", zerolinecolor="#2a3f55", zerolinewidth=1.2,
            range=[-lim, lim],
            scaleanchor="x", scaleratio=1,
        ),
        showlegend=False,
        title=dict(
            text=(
                f"近场 2D 态势  ·  中心高度 {alt_r:.0f} km  ·  "
                f"搜索半径 {search_radius_km:.0f} km  ·  "
                f"<span style='color:#FF2222'>●</span> &lt;20 km  "
                f"<span style='color:#FF8800'>●</span> &lt;100 km  "
                f"<span style='color:#FFDD00'>●</span> &lt;300 km"
            ),
            font=dict(color="#cccccc", size=12),
            x=0.01, xanchor="left",
        ),
    )
    return fig


# ─── Real-time continuous orbital animation ──────────────────────────────────
@st.cache_data(ttl=600, show_spinner="预计算实时轨道…")
def _compute_realtime_frames(
    t_base_iso: str, n_frames: int = 60, step_s: float = 10.0,
    alt_min: float = 0, alt_max: float = 2000,
    obj_type: str = "ALL", limit: int = 4000,
) -> dict:
    """Pre-compute keyframes for real-time animation.

    Returns {"colors": [...], "texts": [...], "frames": [...], "step_s": float}
    Only debris that succeed for ALL frames are kept (consistent array sizes).
    """
    from database.db import session_scope
    from sqlalchemy import text
    from propagator.sgp4_propagator import StateVector

    if not _SGP4_OK:
        return {"colors": [], "frames": []}

    t_base = datetime.fromisoformat(t_base_iso).replace(tzinfo=timezone.utc)
    obj_u = (obj_type or "ALL").upper()

    sql = text("""
        WITH latest_gp AS (
            SELECT DISTINCT ON (norad_cat_id)
                norad_cat_id, tle_line1, tle_line2
            FROM gp_elements WHERE norad_cat_id IS NOT NULL
            ORDER BY norad_cat_id, epoch DESC
        )
        SELECT co.norad_cat_id, co.object_type, co.name,
               lg.tle_line1, lg.tle_line2
        FROM catalog_objects co
        JOIN latest_gp lg ON lg.norad_cat_id = co.norad_cat_id
        WHERE co.apogee_km  >= :alt_min
          AND co.perigee_km <= :alt_max_buf
          AND (:obj_type = 'ALL' OR UPPER(co.object_type) = :obj_type)
        ORDER BY co.norad_cat_id LIMIT :lim
    """)

    try:
        with session_scope() as sess:
            rows = sess.execute(sql, {
                "alt_min": alt_min, "alt_max_buf": alt_max + 500,
                "obj_type": obj_u, "lim": limit,
            }).fetchall()
    except Exception:
        return {"colors": [], "frames": []}

    sats = []
    for row in rows:
        try:
            sat = Satrec.twoline2rv(str(row.tle_line1), str(row.tle_line2))
            sats.append((sat, str(row.object_type or "UNKNOWN"),
                         int(row.norad_cat_id), str(row.name or "")[:30]))
        except Exception:
            continue

    ns = len(sats)
    if ns == 0:
        return {"colors": [], "frames": []}

    all_data = np.full((ns, n_frames, 5), np.nan)

    for fi in range(n_frames):
        t = t_base + timedelta(seconds=fi * step_s)
        jd2, fr2 = jday(t.year, t.month, t.day,
                         t.hour, t.minute, t.second + t.microsecond / 1e6)
        for si, (sat, _, _nid, _nm) in enumerate(sats):
            e, r, v = sat.sgp4(jd2, fr2)
            if e != 0:
                continue
            sv = StateVector(epoch=t, x=float(r[0]), y=float(r[1]), z=float(r[2]),
                             vx=float(v[0]), vy=float(v[1]), vz=float(v[2]))
            lat2, lon2, alt2 = sv.to_geodetic()
            if not np.isfinite(alt2) or alt2 < alt_min or alt2 > alt_max:
                continue
            ex, ey, ez = lla_to_ecef(lat2, lon2, alt2)
            all_data[si, fi] = [lon2, lat2, float(ex), float(ey), float(ez)]

    valid = ~np.any(np.isnan(all_data[:, :, 0]), axis=1)
    all_data = all_data[valid]
    colors = [_TYPE_HEX.get(sats[i][1], "#888") for i in range(ns) if valid[i]]
    texts = [
        f"{sats[i][3]}<br>NORAD {sats[i][2]}<br>{_TYPE_CN.get(sats[i][1], sats[i][1])}"
        for i in range(ns) if valid[i]
    ]

    frames = []
    for fi in range(n_frames):
        frames.append({
            "lon": np.round(all_data[:, fi, 0], 2).tolist(),
            "lat": np.round(all_data[:, fi, 1], 2).tolist(),
            "x": np.round(all_data[:, fi, 2], 1).tolist(),
            "y": np.round(all_data[:, fi, 3], 1).tolist(),
            "z": np.round(all_data[:, fi, 4], 1).tolist(),
        })
    return {"colors": colors, "texts": texts, "frames": frames, "step_s": step_s}


_RT_JS_TEMPLATE = """
<div id="{div_id}" style="width:100%;height:{height}px"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function(){{
  var fig={fig_json};
  var el=document.getElementById('{div_id}');
  Plotly.newPlot(el,fig.data,fig.layout,{{scrollZoom:true,displayModeBar:false,responsive:true}});
  var F={frames_json};
  var N=F.length,stepMs={step_ms},trIdx={trace_idx};
  if(N<2)return;
  // Pause animation while the user interacts (drag-rotate / wheel-zoom) so
  // Plotly's gesture handler is not fighting our restyle on every frame.
  var paused=false, resumeT=null;
  function pauseFor(ms){{
    paused=true;
    if(resumeT) clearTimeout(resumeT);
    resumeT=setTimeout(function(){{paused=false;resumeT=null;}}, ms);
  }}
  el.addEventListener('mousedown',  function(){{pauseFor(800);}});
  el.addEventListener('mousemove',  function(e){{if(e.buttons) pauseFor(800);}});
  el.addEventListener('touchstart', function(){{pauseFor(800);}}, {{passive:true}});
  el.addEventListener('touchmove',  function(){{pauseFor(800);}}, {{passive:true}});
  el.addEventListener('wheel',      function(){{pauseFor(600);}}, {{passive:true}});
  var t0=performance.now(),last=0;
  function tick(now){{
    if(paused){{requestAnimationFrame(tick);return;}}
    if(now-last<500){{requestAnimationFrame(tick);return;}}
    last=now;
    var elapsed=now-t0,total=N*stepMs;
    var pos=(elapsed%total)/stepMs;
    var i0=Math.floor(pos)%N,i1=(i0+1)%N,f=pos-Math.floor(pos);
    var n=F[i0].{k0}.length;
    var a=new Array(n),b=new Array(n){extra_arr_decl};
    for(var i=0;i<n;i++){{
      var d0=F[i1].{k0}[i]-F[i0].{k0}[i];
      {wrap_fix_0}
      a[i]=F[i0].{k0}[i]+f*d0;
      {wrap_clamp_0}
      var d1=F[i1].{k1}[i]-F[i0].{k1}[i];
      {wrap_fix_1}
      b[i]=F[i0].{k1}[i]+f*d1;
      {extra_interp}
    }}
    Plotly.restyle(el,{restyle_obj},[trIdx]);
    requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
</script>
"""


# ─── 2-D ortho globe with DYNAMIC growing trails ──────────────────────────────
# Uses string.Template (`$name`) instead of str.format to avoid having to double
# every curly brace in embedded JavaScript.  Trails start empty at animation
# start and grow one frame at a time up to the current position; when the
# animation loops (i0 < lastI) the trail naturally restarts because the
# rebuild iterates 0..i0 from scratch.
_RT_JS_TEMPLATE_ORTHO_TRAILS = string.Template("""
<div id="$div_id" style="width:100%;height:${height}px"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function(){
  var fig=$fig_json;
  var el=document.getElementById('$div_id');
  Plotly.newPlot(el,fig.data,fig.layout,{scrollZoom:true,displayModeBar:false,responsive:true});
  var F=$frames_json;
  var TG=$trail_groups_json;        // [{trIdx, indices:[...]}]
  var N=F.length, stepMs=$step_ms, mkIdx=0;
  if(N<2) return;
  // Pause animation while the user is rotating / zooming, otherwise the
  // restyle calls fight Plotly's gesture handler -> jumpy drag.
  var paused=false, resumeT=null;
  function pauseFor(ms){
    paused=true;
    if(resumeT) clearTimeout(resumeT);
    resumeT=setTimeout(function(){paused=false;resumeT=null;}, ms);
  }
  el.addEventListener('mousedown',  function(){pauseFor(800);});
  el.addEventListener('mousemove',  function(e){if(e.buttons) pauseFor(800);});
  el.addEventListener('touchstart', function(){pauseFor(800);}, {passive:true});
  el.addEventListener('touchmove',  function(){pauseFor(800);}, {passive:true});
  el.addEventListener('wheel',      function(){pauseFor(600);}, {passive:true});

  function rebuildTrails(iEnd){
    for(var gi=0; gi<TG.length; gi++){
      var g=TG[gi], flatLon=[], flatLat=[];
      for(var j=0; j<g.indices.length; j++){
        var k=g.indices[j], prev=null;
        for(var fi=0; fi<=iEnd; fi++){
          var lon=F[fi].lon[k], lat=F[fi].lat[k];
          if(prev!==null && Math.abs(lon-prev)>180){
            flatLon.push(null); flatLat.push(null);
          }
          flatLon.push(lon); flatLat.push(lat);
          prev=lon;
        }
        flatLon.push(null); flatLat.push(null);
      }
      Plotly.restyle(el,{lon:[flatLon],lat:[flatLat]},[g.trIdx]);
    }
  }

  var t0=performance.now(), last=0, lastI=-1;
  function tick(now){
    if(paused){requestAnimationFrame(tick); return;}
    if(now-last<500){requestAnimationFrame(tick); return;}
    last=now;
    var elapsed=now-t0, total=N*stepMs;
    var pos=(elapsed%total)/stepMs;
    var i0=Math.floor(pos)%N, i1=(i0+1)%N, f=pos-Math.floor(pos);
    var n=F[i0].lon.length;
    var a=new Array(n), b=new Array(n);
    for(var i=0; i<n; i++){
      var d0=F[i1].lon[i]-F[i0].lon[i];
      if(d0>180) d0-=360; if(d0<-180) d0+=360;
      a[i]=F[i0].lon[i]+f*d0;
      if(a[i]>180) a[i]-=360; if(a[i]<-180) a[i]+=360;
      b[i]=F[i0].lat[i]+f*(F[i1].lat[i]-F[i0].lat[i]);
    }
    Plotly.restyle(el,{lon:[a],lat:[b]},[mkIdx]);
    if(i0!==lastI){
      rebuildTrails(i0);   // loop wrap (i0<lastI) naturally resets from frame 0
      lastI=i0;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>
""")


def _render_realtime_ortho(data: dict, height: int = 530, div_id: str = "rt-globe",
                            show_trails: bool = False):
    """Render continuously animating orthographic globe (auto-play, no interaction needed).

    When ``show_trails`` is True each debris leaves a colour-matched polyline
    that **grows dynamically from animation start to the current frame** (i.e.
    only the *past* portion of the orbit is shown — never the future).  The
    trail resets automatically every time the animation loops back to frame 0.
    Lines are grouped by colour so we use ≤3 extra traces regardless of the
    number of debris.
    """
    import streamlit.components.v1 as components
    import json as _json

    colors = data.get("colors", [])
    texts = data.get("texts", [])
    frames = data.get("frames", [])
    step_s = data.get("step_s", 10)
    if not frames or not frames[0].get("lon"):
        st.info("暂无足够数据用于动画")
        return

    fig = go.Figure(go.Scattergeo(
        lon=frames[0]["lon"], lat=frames[0]["lat"],
        mode="markers",
        marker=dict(size=2.5, color=colors, opacity=0.72, line=dict(width=0)),
        text=texts if texts else None,
        hovertemplate="%{text}<extra></extra>" if texts else None,
        hoverinfo="text" if texts else "skip",
    ))

    # ── Optional: dynamic orbit trails (empty at start, grow every frame) ─
    trail_groups: list[dict] = []
    if show_trails and colors:
        from collections import defaultdict
        color_to_idx: dict[str, list[int]] = defaultdict(list)
        for _i, _c in enumerate(colors):
            color_to_idx[_c].append(_i)
        for _c, _idx_list in color_to_idx.items():
            trail_groups.append({"trIdx": len(fig.data), "indices": _idx_list})
            fig.add_trace(go.Scattergeo(
                lon=[], lat=[],
                mode="lines",
                line=dict(color=_c, width=0.9),
                opacity=0.55,
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.update_geos(
        projection_type="orthographic",
        showcoastlines=True, coastlinecolor="#3a7a5a", coastlinewidth=1.2,
        showland=True, landcolor="#1a3525",
        showocean=True, oceancolor="#071520",
        showlakes=True, lakecolor="#0a1f30",
        showcountries=True, countrycolor="#2a4a3a", countrywidth=0.5,
        showframe=False, bgcolor=DARK_BG,
        lonaxis=dict(showgrid=True, gridcolor="#1a2e3e", dtick=30, gridwidth=0.5),
        lataxis=dict(showgrid=True, gridcolor="#1a2e3e", dtick=30, gridwidth=0.5),
        projection_rotation=dict(lon=110, lat=20, roll=0),
    )
    fig.update_layout(paper_bgcolor=DARK_BG, margin=dict(l=0,r=0,t=0,b=0),
                      height=height, showlegend=False)

    fd = [{"lon": f["lon"], "lat": f["lat"]} for f in frames]

    if show_trails and trail_groups:
        html = _RT_JS_TEMPLATE_ORTHO_TRAILS.substitute(
            div_id=div_id, height=height,
            fig_json=fig.to_json(),
            frames_json=_json.dumps(fd, separators=(',', ':')),
            trail_groups_json=_json.dumps(trail_groups, separators=(',', ':')),
            step_ms=int(step_s * 1000),
        )
    else:
        html = _RT_JS_TEMPLATE.format(
            div_id=div_id, height=height,
            fig_json=fig.to_json(),
            frames_json=_json.dumps(fd, separators=(',', ':')),
            step_ms=int(step_s * 1000),
            trace_idx=0,
            k0="lon", k1="lat",
            extra_arr_decl="",
            wrap_fix_0="if(d0>180)d0-=360;if(d0<-180)d0+=360;",
            wrap_clamp_0="if(a[i]>180)a[i]-=360;if(a[i]<-180)a[i]+=360;",
            wrap_fix_1="",
            extra_interp="",
            restyle_obj="{lon:[a],lat:[b]}",
        )
    components.html(html, height=height + 10, scrolling=False)


def _render_realtime_3d(data: dict, layer: dict, height: int = 680):
    """Render continuously animating 3D sphere with grid-only Earth."""
    import streamlit.components.v1 as components
    import json as _json

    colors = data.get("colors", [])
    texts = data.get("texts", [])
    frames = data.get("frames", [])
    step_s = data.get("step_s", 10)
    if not frames or not frames[0].get("x"):
        st.info("暂无足够数据用于 3D 动画")
        return

    fig = go.Figure()
    _add_earth_grid_only(fig, n=80)
    n_earth_traces = len(fig.data)

    f0 = frames[0]
    fig.add_trace(go.Scatter3d(
        x=f0["x"], y=f0["y"], z=f0["z"],
        mode="markers",
        marker=dict(size=2.0, color=colors, opacity=0.85, line=dict(width=0)),
        text=texts if texts else None,
        hovertemplate="%{text}<extra></extra>" if texts else None,
        hoverinfo="text" if texts else "skip",
        name="",
    ))

    scene_r = R_EARTH + (layer["alt_max"] + 500)
    earth_ax = dict(range=[-scene_r, scene_r])
    _apply_3d_layout(fig, height=height)
    fig.update_layout(scene=dict(
        xaxis=earth_ax, yaxis=earth_ax, zaxis=earth_ax,
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
    ))

    fd = [{"x": f["x"], "y": f["y"], "z": f["z"]} for f in frames]

    html = _RT_JS_TEMPLATE.format(
        div_id="rt-3d", height=height,
        fig_json=fig.to_json(),
        frames_json=_json.dumps(fd, separators=(',', ':')),
        step_ms=int(step_s * 1000),
        trace_idx=n_earth_traces,
        k0="x", k1="y",
        extra_arr_decl=",c=new Array(n)",
        wrap_fix_0="", wrap_clamp_0="",
        wrap_fix_1="",
        extra_interp="var d2=F[i1].z[i]-F[i0].z[i];c[i]=F[i0].z[i]+f*d2;",
        restyle_obj="{x:[a],y:[b],z:[c]}",
    )
    components.html(html, height=height + 10, scrolling=False)


# ─── Tab renderers ─────────────────────────────────────────────────────────────
def _render_global_view():
    col_ctrl, col_map = st.columns([1, 3.2], gap="medium")

    with col_ctrl:
        st.markdown(section_title("sliders", "筛选", level=4, icon_size=20), unsafe_allow_html=True)
        hour_offset = st.slider(
            "时刻偏移（相对当前，小时）",
            -24, 24, 0, step=1, key="gv_hour_offset",
            help="按选定时刻递推所有目标位置，避免仅用观测片段中点导致的条带空洞。",
        )
        alt_range = st.slider("高度范围 (km)", 0, 42000, (0, 2000), step=50,
                              key="gv_alt")
        obj_type  = st.selectbox("目标类型",
                                  ["ALL", "DEBRIS", "PAYLOAD", "ROCKET BODY"],
                                  key="gv_type")
        n_limit   = st.slider("显示上限", 2000, 30000, 12000, step=1000,
                              key="gv_limit")

        if st.button("刷新数据", key="gv_refresh"):
            load_positions_at_time.clear()
            load_debris_near_point.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("**图例**")
        for k, c in _TYPE_HEX.items():
            st.markdown(
                f'<span style="color:{c};font-size:18px">●</span> {_TYPE_CN.get(k, k)}',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        _gv_ortho, _gv_flat = "正射球面", "平面交互地图"
        st.caption("视图模式")
        st.markdown(
            '<div style="display:flex;gap:14px;align-items:center;font-size:12px;'
            'color:#475569;margin:2px 0 6px 0">'
            f'{icon_inline("overview", 17)}<span>正射球面</span>'
            f'{icon_inline("globe_flat", 17)}<span>平面地图</span></div>',
            unsafe_allow_html=True,
        )
        view_mode = st.radio(
            "选项",
            [_gv_ortho, _gv_flat],
            key="gv_view_v2",
            label_visibility="collapsed",
        )

        show_trails = st.toggle(
            "显示轨迹拖尾",
            value=False,
            key="gv_show_trails",
            help="开启后，每个碎片会从当前动画起点开始留下一条与自身同色的轨道拖尾，"
                 "仅显示『已经走过的路径』——不会展示未来位置；动画每次循环回到起点时拖尾自动复位。",
        )

    with col_map:
        t_query = datetime.now(timezone.utc) + timedelta(hours=int(hour_offset))
        with st.spinner("加载当前轨道位置…"):
            df = load_positions_at_time(
                t_utc=t_query,
                alt_min=float(alt_range[0]),
                alt_max=float(alt_range[1]),
                obj_type=obj_type,
                limit=n_limit,
            )

        if df.empty:
            st.warning("无数据。请确认已完成全量摄入并等待 trajectory_segments 覆盖当前时刻。")
            return

        now_str = t_query.strftime("%Y-%m-%d %H:%M UTC")
        _actual = len(df)
        _hit_limit = _actual >= n_limit
        if _hit_limit:
            st.caption(
                f"显示 **{n_limit:,}** / 上限 {n_limit:,} 个目标 · SGP4 递推 · {now_str}  ·  "
                f"碎片 {(df['object_type']=='DEBRIS').sum():,}  "
                f"载荷 {(df['object_type']=='PAYLOAD').sum():,}  "
                f"火箭级 {(df['object_type']=='ROCKET BODY').sum():,}  "
                f"⚠️ 已达上限，可拖动滑块查看更多"
            )
        else:
            st.caption(
                f"显示 **{_actual:,}** / 上限 {n_limit:,} 个目标 · SGP4 递推 · {now_str}  ·  "
                f"碎片 {(df['object_type']=='DEBRIS').sum():,}  "
                f"载荷 {(df['object_type']=='PAYLOAD').sum():,}  "
                f"火箭级 {(df['object_type']=='ROCKET BODY').sum():,}  "
                f"（该范围共 {_actual:,} 个，未达上限）"
            )

        if view_mode == _gv_ortho:
            _gv_alt_min = float(alt_range[0])
            _gv_alt_max = float(alt_range[1])
            _gv_obj_type = obj_type
            # Ortho writes the entire animation as inline HTML inside an
            # iframe; with the full Space-Track catalogue (~30 k objects)
            # 4000-marker × 60-frame payloads grew past 3 MB and browsers
            # were rendering blank.  Cap to ≤1800 markers × 36 frames so
            # the inline payload stays around ~700 KB.
            _gv_limit = min(n_limit, 1800)

            @st.fragment(run_every=timedelta(seconds=600))
            def _gv_animated_ortho():
                t_now = datetime.now(timezone.utc) + timedelta(hours=int(hour_offset))
                rd = _compute_realtime_frames(
                    t_base_iso=t_now.isoformat(),
                    n_frames=36, step_s=10.0,
                    alt_min=_gv_alt_min, alt_max=_gv_alt_max,
                    obj_type=_gv_obj_type, limit=_gv_limit,
                )
                if rd.get("frames"):
                    _render_realtime_ortho(
                        rd, height=530, div_id="rt-globe",
                        show_trails=bool(st.session_state.get("gv_show_trails", False)),
                    )
                else:
                    st.plotly_chart(make_globe_ortho(df), use_container_width=True,
                                    config=dict(scrollZoom=True, displayModeBar=False))
            _gv_animated_ortho()
        else:
            # ── pydeck GlobeView ──────────────────────────────────────────────
            try:
                import pydeck as pdk
                df_p = df.copy()
                df_p["r"] = df_p["object_type"].map(
                    {k: v[0] for k, v in _TYPE_PDK.items()}).fillna(136).astype(int)
                df_p["g"] = df_p["object_type"].map(
                    {k: v[1] for k, v in _TYPE_PDK.items()}).fillna(136).astype(int)
                df_p["b"] = df_p["object_type"].map(
                    {k: v[2] for k, v in _TYPE_PDK.items()}).fillna(136).astype(int)

                scatter = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_p,
                    get_position=["lon", "lat"],
                    get_color=["r", "g", "b", 200],
                    get_radius=25000,
                    radius_min_pixels=1.5,
                    radius_max_pixels=6,
                    pickable=True,
                    opacity=0.75,
                )
                vs = pdk.ViewState(latitude=20, longitude=110, zoom=0,
                                   min_zoom=0, max_zoom=12)
                deck = pdk.Deck(
                    layers=[scatter],
                    initial_view_state=vs,
                    views=[pdk.View("GlobeView", controller=True)],
                    map_style=None,
                    tooltip={
                        "html": "<b>NORAD {norad_cat_id}</b><br/>{name}<br/>"
                                "类型: {object_type}<br/>Alt: {alt_km} km",
                        "style": {"backgroundColor": "#0e1117", "color": "white",
                                  "font-size": "12px"},
                    },
                    parameters={"clearColor": [0.03, 0.06, 0.12, 1]},
                )
                st.pydeck_chart(deck, use_container_width=True, height=540)
            except Exception as e:
                st.error(f"pydeck 渲染失败，回退到正射球面视图：{e}")
                st.plotly_chart(make_globe_ortho(df), use_container_width=True,
                                config=dict(displayModeBar=False))

    # ── Altitude histograms (LEO + full economy space) ──────────────────────
    h1, h2 = st.columns(2, gap="medium")
    with h1:
        leo_df = df[(df["alt_km"] >= 0) & (df["alt_km"] <= 2000)]
        if not leo_df.empty:
            st.markdown(section_title("chart_bar", "高度分布（LEO 0–2000 km）", level=4, icon_size=18), unsafe_allow_html=True)
            st.plotly_chart(
                make_altitude_hist(
                    leo_df,
                    x_max=2000,
                    title="LEO 细分（线性坐标）",
                    log_x=False,
                    bin_size=35,
                ),
                use_container_width=True,
                config=dict(displayModeBar=False),
            )
        else:
            st.info("LEO 范围暂无数据")

    with h2:
        full_df = load_positions_at_time(
            t_utc=t_query,
            alt_min=0.0,
            alt_max=42000.0,
            obj_type=obj_type,
            limit=max(15000, n_limit),
        )
        if not full_df.empty:
            st.markdown(section_title("chart_line", "高度分布（全空间经济范围）", level=4, icon_size=18), unsafe_allow_html=True)
            st.plotly_chart(
                make_altitude_hist(
                    full_df,
                    x_max=42000,
                    title="全轨道范围（log10 坐标）",
                    log_x=True,
                    bin_size=250,
                ),
                use_container_width=True,
                config=dict(displayModeBar=False),
            )
        else:
            st.info("全空间范围暂无数据")


def _render_layer_drilldown():
    # ── Layer metric cards ──────────────────────────────────────────────────
    with st.spinner("加载分层统计…"):
        stats_df = load_layer_stats()

    st.markdown("##### 选择高度层")
    layer_cols = st.columns(len(ALTITUDE_LAYERS))
    sel_id = st.session_state.get("ld_layer", "LEO2")

    for col, layer in zip(layer_cols, ALTITUDE_LAYERS):
        cnt = int(
            stats_df.loc[stats_df["id"] == layer["id"], "count"].iloc[0]
        ) if not stats_df.empty else 0
        selected = sel_id == layer["id"]
        border   = f"border:2px solid {layer['color']};" if selected else "border:2px solid #333;"
        bg       = f"background:{layer['color']}22;" if selected else "background:#111827;"
        with col:
            st.markdown(
                f'<div style="{bg}{border}border-radius:8px;padding:10px 6px;'
                f'text-align:center;height:108px;box-sizing:border-box;overflow:hidden;">'
                f'<span style="color:{layer["color"]};font-size:11px;font-weight:700;'
                f'display:block;margin-bottom:3px;white-space:nowrap;'
                f'overflow:hidden;line-height:1.2">'
                f'{layer["label"]}</span>'
                f'<span style="color:#ccc;font-size:22px;font-weight:bold;'
                f'display:block;margin-bottom:3px">{cnt:,}</span>'
                f'<span style="color:#888;font-size:10px;display:block;'
                f'overflow:hidden;white-space:nowrap">'
                f'{layer["note"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("选择", key=f"lb_{layer['id']}", use_container_width=True):
                st.session_state["ld_layer"] = layer["id"]
                # Reset slider to default for the new layer
                st.session_state["ld_pts"] = 5000
                load_positions_at_time.clear()
                st.rerun()

    st.markdown("---")

    cur = next((l for l in ALTITUDE_LAYERS if l["id"] == sel_id), ALTITUDE_LAYERS[2])
    cur_total = int(
        stats_df.loc[stats_df["id"] == cur["id"], "count"].iloc[0]
    ) if not stats_df.empty else 0

    col_3d, col_side = st.columns([3, 1], gap="medium")

    with col_side:
        st.markdown(
            f'<div style="background:{cur["color"]}18;border-left:4px solid {cur["color"]};'
            f'border-radius:4px;padding:10px;margin-bottom:12px">'
            f'<b>{cur["label"]}</b><br><small>{cur["note"]}</small></div>',
            unsafe_allow_html=True,
        )
        obj_type_3d = st.selectbox(
            "目标类型", ["ALL", "DEBRIS", "PAYLOAD", "ROCKET BODY"],
            key="ld_type",
        )
        max_pts = st.slider("最大点数", 1000, 30000, 5000, step=500, key="ld_pts")
        st.markdown("---")
        st.markdown("**颜色**")
        for k, c in _TYPE_HEX.items():
            st.markdown(
                f'<span style="color:{c}">●</span> {_TYPE_CN.get(k, k)}',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("**高亮带边界**")
        st.markdown(
            f'<span style="color:{cur["color"]}">─ {cur["alt_min"]:,.0f} km</span><br>'
            f'<span style="color:{cur["color"]}">─ {cur["alt_max"]:,.0f} km</span>',
            unsafe_allow_html=True,
        )

    with col_3d:
        t_now = datetime.now(timezone.utc)
        with st.spinner("加载该轨道层数据…"):
            fetch_limit = min(max(cur_total * 5 + 1000, max_pts * 6), 60000)
            df3d = load_positions_at_time(
                t_utc=t_now,
                alt_min=float(cur["alt_min"]), alt_max=float(cur["alt_max"]),
                obj_type=obj_type_3d, limit=fetch_limit,
            )
        if df3d.empty:
            st.info("该轨道层暂无数据")
        else:
            rendered = min(len(df3d), max_pts)
            at_limit = len(df3d) > max_pts
            st.caption(
                f"**{cur['label']}** 层内目标 **{cur_total:,}** 个"
                + (f"（当前渲染 {rendered:,} / {len(df3d):,}，已达渲染上限 {max_pts:,}）" if at_limit
                   else f"（当前渲染 {rendered:,}）")
            )
            # 静态 3D 渲染：直接使用已加载的 df3d，无动画，所有层同步
            _df_render = df3d.head(max_pts)
            _ex, _ey, _ez = lla_to_ecef(
                _df_render["lat"].values,
                _df_render["lon"].values,
                _df_render["alt_km"].values,
            )
            _colors = [_TYPE_HEX.get(str(t), "#888") for t in _df_render["object_type"]]
            _texts = [
                f"{row['name']}<br>NORAD {row['norad_cat_id']}<br>"
                f"{_TYPE_CN.get(str(row['object_type']), str(row['object_type']))}"
                for _, row in _df_render.iterrows()
            ]
            _fig3d = go.Figure()
            _add_earth_grid_only(_fig3d, n=80)
            _fig3d.add_trace(go.Scatter3d(
                x=_ex.tolist(), y=_ey.tolist(), z=_ez.tolist(),
                mode="markers",
                marker=dict(size=2.0, color=_colors, opacity=0.85, line=dict(width=0)),
                text=_texts,
                hovertemplate="%{text}<extra></extra>",
                hoverinfo="text",
                name="",
            ))
            _scene_r = R_EARTH + (cur["alt_max"] + 500)
            _earth_ax = dict(range=[-_scene_r, _scene_r])
            _apply_3d_layout(_fig3d, height=680)
            _fig3d.update_layout(scene=dict(
                xaxis=_earth_ax, yaxis=_earth_ax, zaxis=_earth_ax,
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
            ))
            st.plotly_chart(_fig3d, use_container_width=True)

    # Detail table
    if not df3d.empty:
        with st.expander(f"{cur['label']} 层 · 目标列表（前 30）", expanded=False):
            st.dataframe(
                df3d[["norad_cat_id","name","object_type","alt_km",
                       "perigee_km","apogee_km","inclination","country_code"]]
                .sort_values("alt_km").head(30)
                .rename(columns={
                    "norad_cat_id": "NORAD ID", "name": "名称",
                    "object_type": "类型",      "alt_km": "当前高度(km)",
                    "perigee_km":  "近地点(km)", "apogee_km": "远地点(km)",
                    "inclination": "倾角(°)",    "country_code": "国家",
                }),
                use_container_width=True, hide_index=True,
            )


def _ofp_build_presets():
    """Build presets from whatever is actually in the database."""
    from database.db import session_scope as _scope
    from sqlalchemy import text as _text
    try:
        with _scope() as sess:
            rows = sess.execute(_text("""
                SELECT g.norad_cat_id, co.name, co.object_type
                FROM gp_elements g
                JOIN catalog_objects co ON co.norad_cat_id = g.norad_cat_id
                ORDER BY g.norad_cat_id LIMIT 5000
            """)).fetchall()
    except Exception:
        return {}, "5\n11\n22\n29\n340"
    payloads = [r for r in rows if r[2] == "PAYLOAD"]
    debris = [r for r in rows if r[2] == "DEBRIS"]
    rockets = [r for r in rows if r[2] == "ROCKET BODY"]
    presets = {}
    if payloads:
        sample = payloads[:5]
        presets["经典卫星"] = "\n".join(str(r[0]) for r in sample)
    if debris:
        sample = debris[:5]
        presets["碎片样本"] = "\n".join(str(r[0]) for r in sample)
    if payloads and debris:
        mixed = payloads[:3] + debris[:2]
        presets["混合 (载荷+碎片)"] = "\n".join(str(r[0]) for r in mixed)
    default = "\n".join(str(r[0]) for r in (payloads or rows)[:4])
    return presets, default


@st.cache_data(ttl=600, show_spinner=False)
def _ofp_cached_presets():
    return _ofp_build_presets()


# ─── OEM helpers ──────────────────────────────────────────────────────────────

def _traces_to_oem_bytes(traces: dict) -> bytes:
    """Serialise orbit-forecast traces to a CCSDS OEM 2.0 file (in-memory)."""
    import tempfile, os
    from trajectory.oem_io import OEMSegment, OEMState, write_oem

    segs = []
    for nid, info in traces.items():
        seg = OEMSegment(
            object_name=info["name"][:25],
            object_id=str(abs(nid)) if nid > 0 else info["name"][:16],
        )
        for epoch, r, v in zip(info.get("epochs", []),
                                info.get("eci_r", []),
                                info.get("eci_v", [])):
            seg.states.append(OEMState(
                epoch=epoch,
                pos_km=np.array(r),
                vel_kms=np.array(v),
            ))
        if seg.states:
            segs.append(seg)

    if not segs:
        return b""
    tmp = tempfile.mktemp(suffix=".oem")
    try:
        write_oem(tmp, segs)
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _oem_bytes_to_traces(raw: bytes) -> dict:
    """Parse OEM bytes → traces dict (same schema as propagate_orbit_traces).
    
    Tries to extract a numeric NORAD ID from OBJECT_ID; falls back to
    negative index keys for OEM segments that lack NORAD IDs.
    """
    import tempfile, os
    from trajectory.oem_io import read_oem
    from propagator.sgp4_propagator import StateVector as _SV

    tmp = tempfile.mktemp(suffix=".oem")
    try:
        with open(tmp, "w", encoding="ascii", errors="replace") as fh:
            fh.write(raw.decode("ascii", errors="replace"))
        segs = read_oem(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    traces: dict = {}
    for idx, seg in enumerate(segs):
        # Determine key: prefer numeric OBJECT_ID as NORAD
        obj_id = (seg.object_id or "").strip()
        key: int = -(idx + 1)
        if obj_id.isdigit():
            key = int(obj_id)

        lats, lons, alts, er, ev, eps = [], [], [], [], [], []
        for state in seg.states:
            try:
                sv = _SV(epoch=state.epoch,
                         x=float(state.pos_km[0]), y=float(state.pos_km[1]), z=float(state.pos_km[2]),
                         vx=float(state.vel_kms[0]), vy=float(state.vel_kms[1]), vz=float(state.vel_kms[2]))
                lat2, lon2, alt2 = sv.to_geodetic()
                if not np.isfinite(alt2):
                    continue
                lats.append(float(lat2)); lons.append(float(lon2)); alts.append(float(alt2))
                er.append([float(state.pos_km[0]), float(state.pos_km[1]), float(state.pos_km[2])])
                ev.append([float(state.vel_kms[0]), float(state.vel_kms[1]), float(state.vel_kms[2])])
                eps.append(state.epoch)
            except Exception:
                continue

        if lats:
            traces[key] = {
                "name": (seg.object_name or f"SEG-{idx+1}")[:25],
                "object_type": "PAYLOAD",
                "lat": lats, "lon": lons, "alt_km": alts,
                "eci_r": er, "eci_v": ev, "epochs": eps,
            }
    return traces


# ─── OEM helpers ──────────────────────────────────────────────────────────────



def _render_orbit_forecast():
    """Tab: select objects by NORAD ID → SGP4-propagate N orbits → 3D Earth + orbit traces."""
    # ── Apply any preset that was chosen in the PREVIOUS run ─────────────────
    # Must happen BEFORE widgets are rendered so that st.text_area sees the new value.
    if "_ofp_preset_pending" in st.session_state:
        st.session_state["ofp_nids"] = st.session_state.pop("_ofp_preset_pending")
        st.session_state.pop("orbit_forecast_data", None)

    _OFP_PRESETS, _OFP_DEFAULT = _ofp_cached_presets()

    st.caption(
        "输入 NORAD ID，SGP4 传播 N 圈，在三维地球上查看真实轨迹形状"
        "（圆轨道 / 椭圆轨道、高低起伏一目了然）。"
        "浅色为起始时刻，深色为末端；空心圆 = 当前历元，实心圆 = 传播终点。"
    )

    col_ctrl, col_3d = st.columns([1, 3], gap="medium")
    with col_ctrl:
        norad_raw = st.text_area(
            "NORAD ID（每行一个或逗号分隔）",
            value=_OFP_DEFAULT,
            height=130,
            key="ofp_nids",
        )
        n_orbits = st.slider("预报圈数", 0.5, 6.0, 1.5, 0.5, key="ofp_n")
        btn = st.button("开始预报", type="primary", key="ofp_run", use_container_width=True)

        st.markdown("---")
        st.caption("**快速示例**")
        for label, nids_str in _OFP_PRESETS.items():
            if st.button(label, key=f"ofp_pre_{label}", use_container_width=True):
                # Store in a staging key; applied at the top of the NEXT run
                st.session_state["_ofp_preset_pending"] = nids_str
                st.rerun()

    # Parse NORAD IDs (max 15 to keep performance reasonable)
    nids_parsed: list[int] = []
    for part in norad_raw.replace(",", "\n").split("\n"):
        p = part.strip()
        if p.isdigit():
            nids_parsed.append(int(p))
    nids_parsed = list(dict.fromkeys(nids_parsed))[:15]

    cache_key = (tuple(nids_parsed), float(n_orbits))
    need_run = btn or st.session_state.get("ofp_cache_key") != cache_key

    if need_run:
        if not nids_parsed:
            st.warning("请输入至少一个有效的 NORAD ID。")
            return
        with st.spinner(f"SGP4 传播 {len(nids_parsed)} 个目标 × {n_orbits:.1f} 圈…"):
            traces = propagate_orbit_traces(tuple(nids_parsed), float(n_orbits))
        st.session_state["orbit_forecast_data"] = traces
        st.session_state["ofp_cache_key"] = cache_key
    else:
        traces = st.session_state.get("orbit_forecast_data", {})

    if not traces:
        with col_3d:
            st.info("未在数据库中找到任何目标，请检查 NORAD ID 或等待数据摄入完成。")
        # Still show the OEM import section even when no SGP4 results
        _render_oem_panel(col_ctrl=col_ctrl)
        return

    with col_3d:
        fig = make_orbit_trace_fig(traces)
        st.plotly_chart(fig, use_container_width=True,
                        config=dict(scrollZoom=True, displayModeBar=True))

    # Orbit info table
    rows_info = []
    for nid, info in traces.items():
        alts = info["alt_km"]
        rows_info.append({
            "NORAD ID": nid if nid > 0 else "OEM",
            "名称": info["name"][:28],
            "类型": _TYPE_CN.get(info["object_type"], info["object_type"]),
            "近地点(km)": f"{min(alts):.0f}",
            "远地点(km)": f"{max(alts):.0f}",
            "高度差(km)": f"{max(alts) - min(alts):.0f}",
        })
    st.dataframe(pd.DataFrame(rows_info), use_container_width=True, hide_index=True)

    # OEM export / import panel (below the table)
    _render_oem_panel(traces=traces)


def _render_oem_panel(traces: dict | None = None, col_ctrl=None):
    """Collapsible OEM export (from current traces) + import (upload → visualise)."""
    with st.expander("📄 OEM 导出 / 导入（CCSDS 502.0-B-3）", expanded=False):
        tab_exp, tab_imp = st.tabs(["导出 OEM", "导入 OEM"])

        # ── Export ───────────────────────────────────────────────────────────
        with tab_exp:
            # Source 1: current SGP4 propagation traces
            if traces:
                oem_bytes = _traces_to_oem_bytes(traces)
                if oem_bytes:
                    st.download_button(
                        "⬇ 下载轨道预报 OEM",
                        data=oem_bytes,
                        file_name="orbit_forecast.oem",
                        mime="text/plain",
                        key="oem_dl_forecast",
                        use_container_width=True,
                    )
                    st.caption(f"包含 {len(traces)} 个目标的传播轨迹")
                else:
                    st.info("当前轨迹缺少 ECI 状态向量，无法导出（需重新传播）。")
            else:
                st.info("请先在上方运行轨道预报，再导出 OEM。")

            # Source 2: rocket sim trajectory (if available)
            if "sim_result" in st.session_state:
                st.markdown("---")
                st.caption("**也可导出火箭仿真轨迹：**")
                mission_id = st.text_input("任务编号", "2026-001", key="oem_exp_mid")
                if st.button("⬇ 下载仿真轨迹 OEM", key="oem_dl_sim", use_container_width=True):
                    try:
                        import tempfile, os as _os
                        from trajectory.oem_io import sim_result_to_oem_segments, write_oem
                        _result = st.session_state["sim_result"]
                        _phases = st.session_state["sim_phases"]
                        _segs   = sim_result_to_oem_segments(_result, _phases, mission_id=mission_id)
                        _tmp    = tempfile.mktemp(suffix=".oem")
                        write_oem(_tmp, _segs)
                        with open(_tmp, "rb") as fh:
                            _data = fh.read()
                        try:
                            _os.unlink(_tmp)
                        except OSError:
                            pass
                        st.download_button(
                            "点此下载",
                            data=_data,
                            file_name="sim_trajectory.oem",
                            mime="text/plain",
                            key="oem_dl_sim_actual",
                        )
                    except Exception as exc:
                        st.error(f"导出失败：{exc}")

        # ── Import ───────────────────────────────────────────────────────────
        with tab_imp:
            st.caption("上传 CCSDS OEM 2.0 文件，解析后直接在三维地球上显示轨迹。")
            uploaded = st.file_uploader(
                "选择 OEM 文件",
                type=["oem", "txt", "aem"],
                key="oem_upload",
            )
            if uploaded is not None:
                try:
                    raw_bytes = uploaded.read()
                    oem_traces = _oem_bytes_to_traces(raw_bytes)
                    if not oem_traces:
                        st.warning("未能从 OEM 文件中解析出有效轨迹。")
                    else:
                        # Merge into current session traces
                        st.session_state["orbit_forecast_data"] = oem_traces
                        # Populate NORAD ID field via staging key (widget already rendered)
                        positive_ids = [k for k in oem_traces if k > 0]
                        if positive_ids:
                            st.session_state["_ofp_preset_pending"] = "\n".join(
                                str(k) for k in positive_ids[:15]
                            )
                        # Extract display info
                        n_segs = len(oem_traces)
                        all_objs = [info["name"] for info in oem_traces.values()]
                        st.success(
                            f"✅ 解析成功：{n_segs} 个轨迹段 — "
                            + ", ".join(all_objs[:5])
                            + ("…" if n_segs > 5 else "")
                        )
                        st.info("轨迹已加载，请点击上方「开始预报」按钮（或切换其他标签再切回来）刷新 3D 视图。")
                        # Show parse summary
                        _parse_rows = []
                        for nid, info in oem_traces.items():
                            alts = info["alt_km"]
                            _parse_rows.append({
                                "OBJECT_ID": nid if nid > 0 else "N/A",
                                "名称": info["name"][:25],
                                "状态点数": len(info["lat"]),
                                "近地点(km)": f"{min(alts):.0f}" if alts else "—",
                                "远地点(km)": f"{max(alts):.0f}" if alts else "—",
                            })
                        st.dataframe(pd.DataFrame(_parse_rows), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"解析失败：{exc}")


def _render_mission_slider():
    st.caption(
        "运行 6-DOF 仿真 → 时间轴拖动 → 实时查询火箭附近碎片 → "
        "3D 近场态势（火箭坐标系，火箭固定在原点，支持深度缩放）+ 2D 近场切片。"
        "**红色**目标距离 < 20 km，**橙色** < 100 km，**黄色** < 300 km。"
    )

    # ── Sim config ──────────────────────────────────────────────────────────
    with st.expander("仿真参数", expanded=not bool(st.session_state.get("ms_result"))):
        c1, c2, c3, c4 = st.columns(4)
        vehicle   = c1.selectbox("运载火箭", ["CZ-5B", "Falcon9", "Ariane6"], key="ms_veh")
        lat_l     = c2.number_input("发射纬度 (°)", value=19.61, step=0.1, key="ms_lat")
        lon_l     = c3.number_input("发射经度 (°)", value=110.95, step=0.1, key="ms_lon")
        az        = c4.number_input("方位角 (°)", value=90.0, min_value=0.0,
                                    max_value=360.0, step=1.0, key="ms_az")
        c5, c6, c7, c8 = st.columns(4)
        t_max_cfg = c5.number_input("仿真时长 (s)", value=6000, min_value=300,
                                    max_value=10800, step=300, key="ms_tmax",
                                    help="CZ-5B / Falcon9 入轨约 600 s；Ariane6 约 950 s；建议 ≥ 6000 s 以覆盖完整入轨后轨道弧。")
        dt_cfg    = c6.number_input("输出步长 (s)", value=10, min_value=1,
                                    max_value=30, step=1, key="ms_dt")
        radius_km = c7.number_input("搜索半径 (km)", value=500, min_value=50,
                                    max_value=2000, step=50, key="ms_rad")
        auto_stop = c8.checkbox("自动到轨停止",  value=True, key="ms_auto_stop",
                                help="检测到入轨（近地点 > 150 km）后立即停止仿真，避免多余的轨道圈。")
        # ── Launch time (UTC) ────────────────────────────────────────────
        _default_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        cd1, cd2, cd3 = st.columns([2, 1, 1])
        launch_date = cd1.date_input(
            "发射日期 (UTC)", value=_default_date, key="ms_launch_date",
            help="碎片位置将按此日期 + 时刻（UTC）传播，反映实际在轨目标的分布。",
        )
        launch_hour = cd2.number_input(
            "发射时 (UTC)", value=6, min_value=0, max_value=23, step=1, key="ms_launch_h",
        )
        launch_minute = cd3.number_input(
            "发射分 (UTC)", value=0, min_value=0, max_value=59, step=1, key="ms_launch_m",
        )
        run_btn   = st.button("运行仿真", type="primary", key="ms_run",
                              use_container_width=True)

    # ── Build launch_utc from user inputs (outside expander so always available)
    _ld = st.session_state.get("ms_launch_date", _default_date)
    _lh = int(st.session_state.get("ms_launch_h", 6))
    _lm = int(st.session_state.get("ms_launch_m", 0))
    _launch_utc_input = datetime(
        _ld.year, _ld.month, _ld.day, _lh, _lm, 0, tzinfo=timezone.utc
    )

    # ── Run simulation ──────────────────────────────────────────────────────
    if run_btn or "ms_result" not in st.session_state:
        with st.spinner(f"运行 6-DOF 仿真（{vehicle}，请稍候）…"):
            try:
                from trajectory.rocketpy_sim import SimConfig, simulate
                launch_utc = _launch_utc_input
                cfg = SimConfig(
                    vehicle_name=vehicle,
                    launch_lat_deg=float(lat_l),
                    launch_lon_deg=float(lon_l),
                    launch_az_deg=float(az),
                    launch_utc=launch_utc,
                    t_max_s=float(t_max_cfg),
                    dt_out_s=float(dt_cfg),
                    run_mc=False,
                    auto_stop_orbit=bool(auto_stop),
                )
                result = simulate(cfg)
                st.session_state["ms_result"]     = result
                st.session_state["ms_launch_utc"] = launch_utc
                st.session_state["ms_vehicle"]    = vehicle
                st.session_state["ms_radius"]     = float(radius_km)
                # ── Sync to app.py session keys so collision / LCOLA pages can use this trajectory
                try:
                    from trajectory.launch_phases import detect_phases as _dp
                    _phases = _dp(
                        result.nominal,
                        t_meco1=result.t_meco1, t_stage_sep=result.t_stage_sep,
                        t_meco2=result.t_meco2, t_payload_sep=result.t_payload_sep,
                    )
                    st.session_state["sim_result"] = result
                    st.session_state["sim_phases"] = _phases
                except Exception:
                    pass
                st.success(
                    f"{vehicle} 仿真完成 — "
                    f"{len(result.nominal)} 个轨迹点 · "
                    f"最终高度 {result.nominal[-1].alt_km:.1f} km · "
                    f"最终速度 {float(np.linalg.norm(result.nominal[-1].vel_eci)):.3f} km/s  "
                    f"· 轨迹已同步，可在侧边栏「碰撞风险」/ 「LCOLA 飞越筛选」中直接使用。"
                )
            except Exception as exc:
                st.error(f"仿真失败：{exc}")
                return

    sim    = st.session_state.get("ms_result")
    if sim is None or not sim.nominal:
        st.info("点击「运行仿真」开始")
        return

    nom        = sim.nominal
    launch_utc = st.session_state.get("ms_launch_utc", _launch_utc_input)
    t_max_act  = int(nom[-1].t_met_s)
    t_max_cfg  = int(st.session_state.get("ms_tmax", t_max_act))
    rad        = st.session_state.get("ms_radius", float(radius_km))

    # ── Phase event markers for the timeline ───────────────────────────────
    phase_marks: dict[int, str] = {0: "起飞"}
    if sim.t_meco1:        phase_marks[int(sim.t_meco1)]       = "MECO1"
    if sim.t_stage_sep:    phase_marks[int(sim.t_stage_sep)]   = "级间分离"
    if sim.t_meco2:        phase_marks[int(sim.t_meco2)]       = "MECO2"
    if sim.t_payload_sep:  phase_marks[int(sim.t_payload_sep)] = "载荷分离"

    # ── Time slider (full-width) ─────────────────────────────────────────────
    st.markdown("---")
    _veh_label = st.session_state.get("ms_vehicle", "火箭")
    _lu_str    = launch_utc.strftime("%Y-%m-%d %H:%M UTC")
    st.caption(
        f"**{_veh_label}** · 发射时刻 **{_lu_str}** · "
        "碎片/卫星位置按 UTC 时间实时传播，拖动滑块即可查看不同飞行阶段的近场态势。"
    )
    t_slider = st.slider(
        "⏱ 飞行时刻  T+",
        min_value=0, max_value=t_max_cfg, value=0,
        step=max(1, t_max_cfg // 200),
        format="%d s",
        key="ms_slider",
    )

    # ── Rocket state at current time ─────────────────────────────────────────
    times_arr = np.array([p.t_met_s for p in nom])
    idx = int(np.searchsorted(times_arr, t_slider).clip(0, len(nom) - 1))
    rp  = nom[idx]
    vel_kms = float(np.linalg.norm(rp.vel_eci))
    in_orbit_coast = (t_slider > t_max_act)

    # Keplerian propagation for coast phase (t_slider > MECO)
    disp_alt_km  = rp.alt_km
    disp_vel_kms = vel_kms
    disp_lat     = rp.lat_deg
    disp_lon     = rp.lon_deg

    # Pre-compute the trajectory arc for the 3-D figure.
    #
    # Design rationale: Plotly Scatter3d has no depth-occlusion against Surface
    # traces, so any line segment that passes *behind* the Earth sphere will
    # visually appear to cut through it.  To avoid this:
    #
    # • Nominal phase  – show only the recent ascent window (last _TRAIL_PTS
    #   trajectory points up to current index).  All points are near the rocket
    #   and in the same "hemisphere" of the local view.
    # • Coast phase    – show ONLY the Keplerian coast arc from MECO → current
    #   rocket.  The early-ascent nominal points (near Earth's surface on the
    #   far side of the globe) are intentionally omitted.
    #
    # The final `_past_lla` entry is always (disp_lat, disp_lon, disp_alt_km),
    # which is numerically identical to rocket_state, so the arc ends exactly
    # at the rocket marker (local origin = 0, 0, 0).
    _TRAIL_PTS = 200   # max nominal points shown in non-coast phase

    if in_orbit_coast:
        # ── Coast phase: Keplerian propagation in ECEF-relative frame ────────
        # _propagate_coast_ecef() avoids any dependency on launch_utc / J2000:
        #   1. Compute inertial velocity at MECO: v_iner = vel_ecef + ω × r_ecef
        #   2. Keplerian-propagate (r_ecef, v_iner) forward dt_s seconds
        #      (ECEF treated as momentarily inertial — valid due to rotation-
        #       invariance of two-body mechanics)
        #   3. Undo Earth rotation: rotate result by −ω·dt around Z
        _coast_dt = float(t_slider - t_max_act)
        # Start arc at MECO (above Earth sphere, no occlusion issues)
        _past_lla: list[tuple[float, float, float]] = [
            (float(nom[-1].lat_deg), float(nom[-1].lon_deg), float(nom[-1].alt_km))
        ]
        try:
            from trajectory.six_dof import ecef_to_geodetic as _eg

            # ── Final rocket position ─────────────────────────────────────────
            _r_ecef_new = _propagate_coast_ecef(
                nom[-1].pos_ecef, nom[-1].vel_ecef, _coast_dt
            )
            _lat, _lon, _alt = _eg(_r_ecef_new)
            disp_alt_km = float(_alt)
            disp_lat    = float(_lat)
            disp_lon    = float(_lon)
            # Inertial speed (magnitude preserved by rotation)
            _v_iner = nom[-1].vel_ecef + np.cross(
                np.array([0.0, 0.0, _OMEGA_E]), nom[-1].pos_ecef
            )
            disp_vel_kms = float(np.linalg.norm(_v_iner))  # const for 2-body

            # Dense intermediate coast arc
            _ns = max(20, int(_coast_dt / 20))
            for _k in range(1, _ns):
                _dtk = _coast_dt * _k / _ns
                _rk_ecef = _propagate_coast_ecef(
                    nom[-1].pos_ecef, nom[-1].vel_ecef, _dtk
                )
                _latk, _lonk, _altk = _eg(_rk_ecef)
                _past_lla.append((float(_latk), float(_lonk), float(_altk)))
            # Final point = exact rocket state
            _past_lla.append((disp_lat, disp_lon, disp_alt_km))
        except Exception:
            # Fall back: just MECO + current (short straight line, acceptable)
            _past_lla.append((disp_lat, disp_lon, disp_alt_km))
    else:
        # ── Nominal phase: recent trajectory window only ──────────────────────
        _start = max(0, idx + 1 - _TRAIL_PTS)
        _past_lla = [
            (float(p.lat_deg), float(p.lon_deg), float(p.alt_km))
            for p in nom[_start: idx + 1]
        ]

    phase_tag = (
        "轨道惯性飞行" if in_orbit_coast else
        "大气层内"     if rp.alt_km < 100 else
        "上升末段"     if rp.alt_km < 200 else
        "亚轨道"       if vel_kms < 5    else
        "轨道注入"
    )

    # Nearest phase event for info table
    near_marks = sorted(phase_marks.keys(), key=lambda m: abs(m - t_slider))
    nearest_ev = near_marks[0] if near_marks else None
    nearest_ev_str = (
        f"{phase_marks[nearest_ev]} T+{nearest_ev}s"
        if nearest_ev is not None else "—"
    )

    # ── Query nearby debris ─────────────────────────────────────────────────
    t_query = launch_utc + timedelta(seconds=float(t_slider))
    with st.spinner("查询附近目标…"):
        nearby = load_debris_near_point(
            lat=disp_lat, lon=disp_lon, alt_km=disp_alt_km,
            t_utc=t_query, radius_km=rad, limit=400,
        )

    # Risk events (from session state if available)
    risk_events = []
    if "phase_summaries" in st.session_state:
        for s in st.session_state["phase_summaries"]:
            risk_events.extend(s.events)

    # ── 3D mission figure (left 70%) + info table (right 30%) ───────────────
    rocket_state = {
        "lat_deg": disp_lat,
        "lon_deg": disp_lon,
        "alt_km": disp_alt_km,
        "vel_kms": disp_vel_kms,
    }
    col_3d, col_info = st.columns([7, 3])
    with col_3d:
        st.markdown(
            "<small style='color:#8899aa'>▲ 3D 近场态势（火箭坐标系，地球为背景参考）— 鼠标拖动旋转 / 滚轮缩放 / 双击复位</small>",
            unsafe_allow_html=True,
        )
        fig_m = make_mission_fig(
            nom,
            float(t_slider),
            nearby,
            risk_events,
            rocket_state=rocket_state,
            past_lla=_past_lla,
        )
        st.plotly_chart(fig_m, use_container_width=True,
                        config=dict(scrollZoom=True, displayModeBar=True))
    with col_info:
        st.markdown(f"""| | |
|---|---|
|**阶段**|{phase_tag}|
|**T+**|{t_slider} s|
|**高度**|{disp_alt_km:.1f} km|
|**速度**|{disp_vel_kms:.3f} km/s|
|**纬度**|{disp_lat:.2f}°|
|**经度**|{disp_lon:.2f}°|
|**最近事件**|{nearest_ev_str}|
""")

    # ── 2D proximity view (unlimited zoom) ──────────────────────────────────
    if not nearby.empty:
        fig_2d = make_proximity_2d(rp, nearby, rad, rocket_state=rocket_state)
        if fig_2d is not None:
            st.markdown(
                "<small style='color:#8899aa'>▼ 近场 2D 视图</small>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_2d, use_container_width=True,
                            config=dict(scrollZoom=True, displayModeBar=True))

    # ── Stats row ───────────────────────────────────────────────────────────
    if not nearby.empty:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("附近目标总数", len(nearby))
        mc2.metric("距离最近 (km)", f"{nearby['dist_km'].min():.1f}")
        _deb_dist = nearby[nearby["object_type"] == "DEBRIS"]["dist_km"]
        mc3.metric("最近碎片",
                   f"{_deb_dist.min():.1f} km" if not _deb_dist.empty else "—",
                   delta=None)
        high_risk = (nearby["dist_km"] < 20).sum()
        mc4.metric("< 20 km 目标", high_risk,
                   delta="需关注" if high_risk > 0 else None,
                   delta_color="inverse")

        with st.expander(f"附近目标（{len(nearby)} 个，半径 {rad:.0f} km）", expanded=False):
            disp = nearby[["norad_cat_id","name","object_type","alt_km","dist_km"]].copy()
            disp.columns = ["NORAD ID","名称","类型","高度(km)","距离(km)"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info(f"T+{t_slider}s 附近 {rad:.0f} km 内暂无目标（超出 DB 覆盖窗口或高度过低）。")


# ─── Tab 5: 发射趋势 ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_lt_cached():
    from streamlit_app.launch_trend import load_launch_history
    return load_launch_history()


def _render_launch_trend():
    """Historical launch trend visualization tab."""
    import pandas as pd
    from streamlit_app.launch_trend import (
        make_annual_launch_fig,
        make_cumulative_fig,
        make_country_trend_fig,
        make_decade_summary_fig,
        make_recent_country_bar,
        make_unoosa_comparison_fig,
        make_ucs_purpose_fig,
        make_ucs_users_fig,
        make_ucs_orbit_fig,
    )

    st.caption(
        "融合 GCAT（McDowell）、UNOOSA、UCS、ESA DISCOS 等多源数据，"
        "展示 1957 年至今的航天发射统计与在轨卫星分析。"
    )

    lt = _load_lt_cached()
    if not lt:
        st.warning("暂无历史数据，请先运行数据摄入。")
        return

    by_region = lt.get("annual_by_region", pd.DataFrame())
    cumulative = lt.get("cumulative", pd.DataFrame())

    # ── 顶部 KPI ──────────────────────────────────────────────────────────────
    if not by_region.empty:
        total_all = int(by_region["n"].sum())
        current_yr = pd.Timestamp.now().year
        this_yr = int(by_region[by_region["yr"] == current_yr]["n"].sum())
        prev_yr = int(by_region[by_region["yr"] == current_yr - 1]["n"].sum())
        top_country = by_region.groupby("region")["n"].sum().idxmax()

        k1, k2, k3, k4 = st.columns(4)

        def _kcard(col, label, val, sub=""):
            sub_h = f"<div style='font-size:0.72em;color:#94A3B8;margin-top:2px'>{sub}</div>" if sub else ""
            col.markdown(
                f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;"
                f"padding:12px 10px;text-align:center'>"
                f"<div style='font-size:0.76em;color:#475569;margin-bottom:4px'>{label}</div>"
                f"<div style='font-size:1.3em;font-weight:700;color:#1E293B'>{val}</div>{sub_h}</div>",
                unsafe_allow_html=True,
            )

        _kcard(k1, "历史发射总量", f"{total_all:,}颗", "有效载荷 · 1957—今")
        _kcard(k2, f"{current_yr} 年发射量", f"{this_yr:,}颗", "仅统计已入库数据")
        _kcard(k3, f"{current_yr-1} 年发射量", f"{prev_yr:,}颗", "完整年份数据")
        _kcard(k4, "最大发射国/地区", top_country, "历史累计")

    st.markdown("---")

    # ── 年代汇总 + 近年逐年 ────────────────────────────────────────────────────
    col_dec, col_yr = st.columns(2)
    with col_dec:
        st.markdown(section_title("chart_bar", "年代发射量汇总（1957—今）",
                                  level=4, icon_size=18), unsafe_allow_html=True)
        if not by_region.empty:
            fig_dec = make_decade_summary_fig(by_region)
            st.plotly_chart(fig_dec, use_container_width=True)

    with col_yr:
        st.markdown(section_title("chart_bar", "近年逐年载荷发射量（2010—今）",
                                  level=4, icon_size=18), unsafe_allow_html=True)
        if not by_region.empty:
            fig_ann = make_annual_launch_fig(by_region)
            st.plotly_chart(fig_ann, use_container_width=True)

    # ── 近年折线 + 地区对比 ──────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(section_title("chart_line", "近年分国别发射趋势（2000—今）",
                                  level=4, icon_size=18), unsafe_allow_html=True)
        if not by_region.empty:
            fig_trend = make_country_trend_fig(by_region, start_year=2000)
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown(section_title("chart_bar", "2020 年后各地区发射量对比",
                                  level=4, icon_size=18), unsafe_allow_html=True)
        if not by_region.empty:
            fig_recent = make_recent_country_bar(by_region, start_year=2020)
            st.plotly_chart(fig_recent, use_container_width=True)

    # ── 在轨目标历史演化 ──────────────────────────────────────────────────────
    st.markdown(section_title("layers", "在轨目标数量历史演化（按类型累计）",
                              level=4, icon_size=18), unsafe_allow_html=True)
    if not cumulative.empty:
        fig_cum = make_cumulative_fig(cumulative)
        st.plotly_chart(fig_cum, use_container_width=True)

    # ── UNOOSA 交叉验证 + UCS 卫星用途 ────────────────────────────────────────
    unoosa_world = lt.get("unoosa_world", pd.DataFrame())
    ucs = lt.get("ucs", pd.DataFrame())

    has_extra = (not unoosa_world.empty) or (not ucs.empty)
    if has_extra:
        st.markdown("---")
        st.markdown(section_title("chart_line", "多源数据分析",
                                  level=4, icon_size=18), unsafe_allow_html=True)

    if not unoosa_world.empty and not by_region.empty:
        col_cmp, col_note = st.columns([2, 1])
        with col_cmp:
            st.markdown("**GCAT vs UNOOSA 年度发射量对比**")
            fig_cmp = make_unoosa_comparison_fig(by_region, unoosa_world)
            st.plotly_chart(fig_cmp, use_container_width=True)
        with col_note:
            st.info(
                "GCAT 统计有效载荷（PAYLOAD），UNOOSA 统计所有已登记发射物体"
                "（含碎片、箭体），因此 UNOOSA 数字通常更高。"
                "两条曲线形态一致说明数据趋势可靠。"
            )

    if not ucs.empty:
        _ucs_total = len(ucs)
        st.markdown("---")
        st.markdown(section_title("chart_bar", f"UCS 在轨卫星分析（{_ucs_total:,} 颗）",
                                  level=4, icon_size=18), unsafe_allow_html=True)

        st.markdown("**卫星用途分布**")
        fig_p = make_ucs_purpose_fig(ucs)
        st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("**轨道类别分布**")
        fig_o = make_ucs_orbit_fig(ucs)
        st.plotly_chart(fig_o, use_container_width=True)

        st.markdown("**军/民/商属性**")
        fig_u = make_ucs_users_fig(ucs)
        st.plotly_chart(fig_u, use_container_width=True)

        # Top-7 countries as pie chart, everything else collapsed into "其他"
        _all_counts = ucs["country"].value_counts()
        _top_n = 7
        _top_part = _all_counts.head(_top_n)
        _rest_part = _all_counts.iloc[_top_n:]
        _pie_labels = [str(l) for l in _top_part.index]
        _pie_values = [int(v) for v in _top_part.values]
        if not _rest_part.empty:
            _pie_labels.append("其他")
            _pie_values.append(int(_rest_part.sum()))
        st.markdown("**主要运营国/地区卫星数量**")
        _ctotal = sum(_pie_values)
        _cpos = ["outside" if v / _ctotal < 0.05 else "inside" for v in _pie_values]
        _fig_country = go.Figure(go.Pie(
            labels=_pie_labels,
            values=_pie_values,
            hole=0.4,
            sort=False,
            textinfo="label+value+percent",
            textfont=dict(size=12),
            textposition=_cpos,
            insidetextorientation="horizontal",
            marker=dict(colors=[
                "#3B82F6", "#EF4444", "#F59E0B", "#10B981", "#8B5CF6",
                "#EC4899", "#06B6D4", "#F97316", "#6366F1", "#84CC16", "#94A3B8",
            ]),
        ))
        _fig_country.update_layout(
            template="plotly_white", height=380,
            margin=dict(t=10, b=30, l=60, r=60),
            showlegend=False,
        )
        st.plotly_chart(_fig_country, use_container_width=True)




# ─── Main entry point ─────────────────────────────────────────────────────────
def render_viz_explorer():
    st.markdown(title_row("viz", "空间碎片可视化探索"), unsafe_allow_html=True)
    st.caption(
        "三维沉浸式探索平台  ·  "
        "全球碎片实时态势  ·  "
        "高度分层 3D 下钻  ·  "
        "火箭发射碎片预警"
    )
    st.markdown(
        """
        <style>
        div.stButton > button {
            white-space: pre-line !important;
            text-align: center;
            line-height: 1.3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "viz_sub_idx" not in st.session_state:
        st.session_state.viz_sub_idx = 0

    _viz_tabs = [
        ("overview",  "全球碎片\n态势"),
        ("layers",    "高度分层\n下钻"),
        ("orbit",     "目标轨道\n预报"),
        ("timeline",  "火箭发射\n碎片预警"),
        ("rocket",    "发射历史\n趋势分析"),
    ]

    def _render_tab_btn(col, i, icon_id, label):
        with col:
            ic_col, btn_col = st.columns([0.15, 0.85], gap="small")
            with ic_col:
                st.markdown(
                    '<div style="display:flex;align-items:center;justify-content:center;'
                    'min-height:2.5rem">'
                    f"{icon_inline(icon_id, 22)}</div>",
                    unsafe_allow_html=True,
                )
            with btn_col:
                _on = st.session_state.viz_sub_idx == i
                if st.button(
                    label,
                    key=f"viz_sub_{i}",
                    use_container_width=True,
                    type="primary" if _on else "secondary",
                ):
                    if st.session_state.viz_sub_idx != i:
                        st.session_state.viz_sub_idx = i
                        st.rerun()

    # Row 1: first 3 tabs
    _row1 = st.columns(3, gap="small")
    for col, i in zip(_row1, range(3)):
        _render_tab_btn(col, i, _viz_tabs[i][0], _viz_tabs[i][1])
    # Row 2: last 2 tabs, centered via padding columns
    _pad, _c3, _c4, _pad2 = st.columns([0.5, 1, 1, 0.5], gap="small")
    _render_tab_btn(_c3, 3, _viz_tabs[3][0], _viz_tabs[3][1])
    _render_tab_btn(_c4, 4, _viz_tabs[4][0], _viz_tabs[4][1])

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    _idx = int(st.session_state.viz_sub_idx)
    if _idx == 0:
        _render_global_view()
    elif _idx == 1:
        _render_layer_drilldown()
    elif _idx == 2:
        _render_orbit_forecast()
    elif _idx == 3:
        _render_mission_slider()
    else:
        _render_launch_trend()
