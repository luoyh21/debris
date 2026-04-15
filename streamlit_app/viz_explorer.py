"""
可视化探索模块 — 三场景沉浸式 3D 仪表盘

Tab 1  全球碎片态势     pydeck GlobeView 交互球体 + 高度直方图
Tab 2  高度分层下钻     Plotly Scatter3d 地球球面 + 5 层轨道带
Tab 3  火箭发射碎片预警 6-DOF 轨迹动画 + 时间轴 + 近地碎片高亮
"""
from __future__ import annotations

import logging
import math
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
    n_hat = np.array([-math.sin(raan), math.cos(raan), 0.0])
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


def _eci_to_ecef_r(r_eci: np.ndarray, t_met_s: float,
                   launch_utc) -> np.ndarray:
    """Rotate ECI position to ECEF using Earth rotation."""
    from datetime import timezone
    if launch_utc is None:
        return r_eci
    j2000 = __import__('datetime').datetime(2000, 1, 1, 12, 0, 0,
                                             tzinfo=timezone.utc)
    t_j2k = (launch_utc - j2000).total_seconds() + t_met_s
    theta  = _OMEGA_E * t_j2k
    c, s   = math.cos(theta), math.sin(theta)
    # ECI → ECEF: rotate by -theta around Z
    x = c * float(r_eci[0]) + s * float(r_eci[1])
    y = -s * float(r_eci[0]) + c * float(r_eci[1])
    z = float(r_eci[2])
    return np.array([x, y, z])

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


def _add_earth(fig: go.Figure, n: int = 110, opacity: float = 0.92):
    """Add Earth sphere + thin atmosphere layer to a 3D figure."""
    x_e, y_e, z_e = _earth_mesh(n)
    # Lighting direction from the Sun (top-right)
    fig.add_trace(go.Surface(
        x=x_e, y=y_e, z=z_e,
        surfacecolor=np.ones_like(z_e),
        colorscale=[[0, "#071a2e"], [0.4, "#0d3560"], [1, "#123f70"]],
        showscale=False,
        opacity=opacity,
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.1, roughness=0.85),
        lightposition=dict(x=100000, y=50000, z=100000),
        hoverinfo="skip",
        name="Earth",
    ))
    # Thin atmosphere glow
    r_atm = R_EARTH + 80
    x_a = x_e * (r_atm / R_EARTH)
    y_a = y_e * (r_atm / R_EARTH)
    z_a = z_e * (r_atm / R_EARTH)
    fig.add_trace(go.Surface(
        x=x_a, y=y_a, z=z_a,
        surfacecolor=np.ones_like(z_a),
        colorscale=[[0, "rgba(80,160,255,0)"], [1, "rgba(80,160,255,0.06)"]],
        showscale=False,
        opacity=0.18,
        hoverinfo="skip",
        name="Atmosphere",
    ))


def _add_earth_local(fig: go.Figure, rx: float, ry: float, rz: float, n: int = 55):
    """Full-size Earth at its correct position in the local rocket frame.

    In the local frame the rocket sits at (0,0,0), so Earth's centre is at
    (-rx, -ry, -rz).  Only the portion inside the scene range viewport will
    be visible — it appears as a curved backdrop below/behind the rocket.
    The scene range (set externally) keeps the near-clip tight, enabling
    deep zoom even though the full sphere data is present.
    """
    ecx, ecy, ecz = -rx, -ry, -rz
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi,     n // 2)
    xe = ecx + R_EARTH * np.outer(np.cos(u), np.sin(v))
    ye = ecy + R_EARTH * np.outer(np.sin(u), np.sin(v))
    ze = ecz + R_EARTH * np.outer(np.ones(n), np.cos(v))

    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        surfacecolor=np.ones_like(ze),
        colorscale=[[0, "#071a2e"], [0.4, "#0d3560"], [1, "#123f70"]],
        showscale=False, opacity=0.90,
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.1, roughness=0.85),
        lightposition=dict(x=100000, y=50000, z=100000),
        hoverinfo="skip", showlegend=False, name="Earth",
    ))
    # Atmosphere glow ring
    sc = (R_EARTH + 80) / R_EARTH
    fig.add_trace(go.Surface(
        x=ecx + (xe - ecx) * sc,
        y=ecy + (ye - ecy) * sc,
        z=ecz + (ze - ecz) * sc,
        surfacecolor=np.ones_like(ze),
        colorscale=[[0, "rgba(80,160,255,0)"], [1, "rgba(80,160,255,0.06)"]],
        showscale=False, opacity=0.18,
        hoverinfo="skip", showlegend=False,
    ))


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
        showcoastlines=True,  coastlinecolor="#2a3d55",
        showland=True,        landcolor="#132030",
        showocean=True,       oceancolor="#07111e",
        showframe=False,
        bgcolor=DARK_BG,
        lonaxis=dict(showgrid=True, gridcolor="#1c2a38", dtick=30),
        lataxis=dict(showgrid=True, gridcolor="#1c2a38", dtick=30),
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
) -> go.Figure:
    """3D figure — LOCAL rocket frame + full-size Earth as backdrop.

    Rocket sits at origin (0, 0, 0).  Earth is placed at its correct offset
    (-rx, -ry, -rz) — full size sphere visible.  scene.*.range is set to
    ±r_norm so the whole Earth fits, while the initial camera is positioned
    close to the rocket for a useful starting view.
    Already-flown arc spans from launch site (traj[0]) all the way to the
    current rocket position for maximum trajectory context.
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

    # ── Scene range: large enough to fit full Earth sphere ───────────────────
    # r_norm ≈ R_EARTH + alt_km.  Adding 20% margin ensures sphere fits fully.
    scene_km = r_norm * 1.20

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

    # Past arc: all points from start up to (and including) current index
    past_pts = traj_points[: idx + 1]
    if len(past_pts) >= 2:
        px_l, py_l, pz_l = _local(past_pts)
        fig.add_trace(go.Scatter3d(
            x=px_l, y=py_l, z=pz_l,
            mode="lines",
            line=dict(color="#00CCFF", width=2.5),
            hoverinfo="skip", name="已飞航段",
        ))

    # Future window: current index → next N points
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
        st.caption(
            f"显示 **{len(df):,}** 个目标（SGP4 递推） · {now_str}  ·  "
            f"碎片 {(df['object_type']=='DEBRIS').sum():,}  "
            f"载荷 {(df['object_type']=='PAYLOAD').sum():,}  "
            f"火箭级 {(df['object_type']=='ROCKET BODY').sum():,}"
        )

        if view_mode == _gv_ortho:
            fig_globe = make_globe_ortho(df)
            st.plotly_chart(fig_globe, use_container_width=True,
                            config=dict(scrollZoom=True, displayModeBar=False))
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
        with st.spinner("渲染 3D 轨道球面…"):
            # Fetch enough propagation candidates so that after strict-altitude filtering
            # we end up with ≈cur_total points, then 3D sampling caps at max_pts.
            fetch_limit = min(max(cur_total * 5 + 1000, max_pts * 6), 60000)
            df3d = load_positions_at_time(
                t_utc=datetime.now(timezone.utc),
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
            fig3d = make_3d_sphere(df3d, layer=cur, max_pts=max_pts)
            st.plotly_chart(fig3d, use_container_width=True,
                            config=dict(scrollZoom=True, displayModeBar=False))

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


_OFP_PRESETS = {
    "ISS + 碎片云":       "25544\n33751\n22675\n4632\n16609",
    "GPS 星座样本":       "24876\n25933\n26360\n27663\n28474",
    "Cosmos-2251 碎片群": "\n".join(str(i) for i in range(33751, 33771)),  # 20 objects
}
_OFP_DEFAULT = "25544\n49445\n43013\n20580"


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


def _render_orbit_forecast():
    """Tab: select objects by NORAD ID → SGP4-propagate N orbits → 3D Earth + orbit traces."""
    # ── Apply any preset that was chosen in the PREVIOUS run ─────────────────
    # Must happen BEFORE widgets are rendered so that st.text_area sees the new value.
    if "_ofp_preset_pending" in st.session_state:
        st.session_state["ofp_nids"] = st.session_state.pop("_ofp_preset_pending")
        st.session_state.pop("orbit_forecast_data", None)

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
            help="ISS=25544  Hubble=20580  Starlink=49445",
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
        run_btn   = st.button("运行仿真", type="primary", key="ms_run",
                              use_container_width=True)

    # ── Run simulation ──────────────────────────────────────────────────────
    if run_btn or "ms_result" not in st.session_state:
        with st.spinner("运行 6-DOF 仿真（CZ-5B，请稍候）…"):
            try:
                from trajectory.rocketpy_sim import SimConfig, simulate
                launch_utc = (
                    datetime.now(timezone.utc)
                    .replace(hour=6, minute=0, second=0, microsecond=0)
                    + timedelta(days=1)
                )
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
    launch_utc = st.session_state["ms_launch_utc"]
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
    if in_orbit_coast:
        # Propagate last nominal point forward as a 2-body Keplerian orbit
        _coast_dt = float(t_slider - t_max_act)
        _r0 = nom[-1].pos_eci.copy()
        _v0 = nom[-1].vel_eci.copy()
        try:
            _r_kep, _v_kep = _keplerian_propagate_eci(_r0, _v0, _coast_dt)
            disp_vel_kms = float(np.linalg.norm(_v_kep))
            _r_ecef = _eci_to_ecef_r(_r_kep, float(nom[-1].t_met_s + _coast_dt),
                                      st.session_state.get("ms_launch_utc"))
            from trajectory.six_dof import ecef_to_geodetic
            _lat, _lon, _alt = ecef_to_geodetic(_r_ecef)
            disp_alt_km = float(_alt)
            disp_lat    = float(_lat)
            disp_lon    = float(_lon)
        except Exception:
            pass   # fall back to stale MECO values

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
        ("overview", "全球碎片\n态势"),
        ("layers",   "高度分层\n下钻"),
        ("orbit",    "目标轨道\n预报"),
        ("timeline", "火箭发射\n碎片预警"),
    ]
    tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 1], gap="small")
    for col, i, (icon_id, label) in zip((tc1, tc2, tc3, tc4), range(4), _viz_tabs):
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

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    _idx = int(st.session_state.viz_sub_idx)
    if _idx == 0:
        _render_global_view()
    elif _idx == 1:
        _render_layer_drilldown()
    elif _idx == 2:
        _render_orbit_forecast()
    else:
        _render_mission_slider()
