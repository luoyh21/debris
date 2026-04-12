"""
可视化探索模块 — 三场景沉浸式 3D 仪表盘

Tab 1  全球碎片态势     pydeck GlobeView 交互球体 + 高度直方图
Tab 2  高度分层下钻     Plotly Scatter3d 地球球面 + 5 层轨道带
Tab 3  发射安全时滑     6-DOF 轨迹动画 + 时间轴 + 近地碎片高亮
"""
from __future__ import annotations

import logging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import List, Optional

log = logging.getLogger(__name__)

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
    {"id": "VLEO", "label": "VLEO  < 400 km",     "alt_min":  0,     "alt_max":  400,
     "color": "#FF6B6B", "note": "空间站活跃区（ISS / 天宫）"},
    {"id": "LEO1", "label": "LEO-I  400–800 km",  "alt_min":  400,   "alt_max":  800,
     "color": "#FF9F45", "note": "Starlink / OneWeb 大型星座层"},
    {"id": "LEO2", "label": "LEO-II 800–2000 km", "alt_min":  800,   "alt_max": 2000,
     "color": "#FFD93D", "note": "历史碰撞碎屑最高密度区"},
    {"id": "MEO",  "label": "MEO  2000–30000 km", "alt_min": 2000,   "alt_max": 30000,
     "color": "#6BCB77", "note": "GPS / 北斗 / Galileo 导航星座"},
    {"id": "GEO",  "label": "GEO   > 30000 km",   "alt_min": 30000,  "alt_max": 42500,
     "color": "#4D96FF", "note": "通信卫星与墓地轨道"},
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
def load_positions_now(
    alt_min: float = 0.0,
    alt_max: float = 2000.0,
    obj_type: str  = "ALL",
    limit:    int  = 15000,
) -> pd.DataFrame:
    """
    Query trajectory_segments midpoint (lon, lat, alt_km) for all objects
    whose segment overlaps the current UTC time.
    """
    from database.db import session_scope
    from sqlalchemy import text

    type_sql = (
        ""
        if obj_type == "ALL"
        else f" AND UPPER(co.object_type) = '{obj_type.upper()}'"
    )
    sql = text(f"""
        SELECT DISTINCT ON (co.norad_cat_id)
            co.norad_cat_id                                                     AS norad_cat_id,
            co.name                                                             AS name,
            COALESCE(co.object_type, 'UNKNOWN')                                AS object_type,
            COALESCE(co.country_code, '?')                                     AS country_code,
            COALESCE(co.perigee_km, 0)                                         AS perigee_km,
            COALESCE(co.apogee_km,  0)                                         AS apogee_km,
            COALESCE(ge.inclination, 0)                                        AS inclination,
            ST_X(ST_LineInterpolatePoint(ts.geom_geo::geometry, 0.5))          AS lon,
            ST_Y(ST_LineInterpolatePoint(ts.geom_geo::geometry, 0.5))          AS lat,
            ST_Z(ST_LineInterpolatePoint(ts.geom_geo::geometry, 0.5))          AS alt_km
        FROM trajectory_segments ts
        JOIN  catalog_objects co ON co.norad_cat_id = ts.norad_cat_id
        LEFT JOIN gp_elements ge  ON ge.norad_cat_id = co.norad_cat_id
        WHERE ts.t_start  <= NOW()
          AND ts.t_end    >= NOW()
          AND ts.geom_geo IS NOT NULL
          AND co.perigee_km >= :alt_min
          AND co.apogee_km  <= :alt_max_buf
          {type_sql}
        ORDER BY co.norad_cat_id, ts.t_start DESC
        LIMIT :lim
    """)
    try:
        from database.db import session_scope
        with session_scope() as sess:
            rows = sess.execute(sql, {
                "alt_min":     alt_min,
                "alt_max_buf": alt_max + 500,
                "lim":         limit,
            }).fetchall()
        df = pd.DataFrame(rows, columns=[
            "norad_cat_id", "name", "object_type", "country_code",
            "perigee_km", "apogee_km", "inclination",
            "lon", "lat", "alt_km",
        ])
        df.dropna(subset=["lat", "lon", "alt_km"], inplace=True)
        df["alt_km"] = df["alt_km"].clip(lower=0)
        return df
    except Exception as exc:
        st.error(f"位置查询错误：{exc}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_layer_stats() -> pd.DataFrame:
    """Count catalog_objects per altitude layer."""
    from database.db import session_scope
    from sqlalchemy import text
    rows = []
    try:
        with session_scope() as sess:
            for layer in ALTITUDE_LAYERS:
                r = sess.execute(text("""
                    SELECT COUNT(*) FROM catalog_objects
                    WHERE perigee_km >= :lo AND perigee_km < :hi
                """), {"lo": layer["alt_min"], "hi": layer["alt_max"]}).fetchone()
                rows.append({**layer, "count": int(r[0]) if r else 0})
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
    """Spatial + temporal query for objects near (lat, lon) at time t_utc."""
    from database.db import session_scope
    from sqlalchemy import text
    sql = text("""
        SELECT DISTINCT ON (co.norad_cat_id)
            co.norad_cat_id,
            co.name,
            co.object_type,
            ST_X(ST_LineInterpolatePoint(ts.geom_geo::geometry, 0.5))   AS lon,
            ST_Y(ST_LineInterpolatePoint(ts.geom_geo::geometry, 0.5))   AS lat,
            ST_Z(ST_LineInterpolatePoint(ts.geom_geo::geometry, 0.5))   AS alt_km,
            ROUND(CAST(
                ST_Distance(
                    ST_MakePoint(:lon, :lat)::geography,
                    ST_Centroid(ts.geom_geo)::geography
                ) / 1000.0 AS numeric), 1
            ) AS dist_km
        FROM trajectory_segments ts
        JOIN catalog_objects co ON co.norad_cat_id = ts.norad_cat_id
        WHERE ts.t_start  <= :t
          AND ts.t_end    >= :t
          AND ts.geom_geo IS NOT NULL
          AND ts.geom_geo && ST_Expand(
                ST_MakePoint(:lon, :lat)::geography::geometry, :deg_r)
        ORDER BY co.norad_cat_id, dist_km
        LIMIT :lim
    """)
    try:
        with session_scope() as sess:
            rows = sess.execute(sql, {
                "lat": lat, "lon": lon,
                "deg_r": radius_km / 111.0,
                "t":    t_utc,
                "lim":  limit,
            }).fetchall()
        df = pd.DataFrame(rows, columns=[
            "norad_cat_id", "name", "object_type", "lon", "lat", "alt_km", "dist_km"
        ])
        df.dropna(subset=["lat", "lon", "alt_km"], inplace=True)
        df["dist_km"] = df["dist_km"].astype(float)
        df = df[df["dist_km"] <= radius_km].sort_values("dist_km")
        return df
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


def _apply_3d_layout(fig: go.Figure, height: int = 620, title: str = ""):
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


def _add_earth(fig: go.Figure, n: int = 80, opacity: float = 0.90):
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


def _add_altitude_shell(fig: go.Figure, alt_km: float, color: str, opacity: float = 0.07):
    """Add a semi-transparent spherical shell at a given altitude."""
    r = R_EARTH + alt_km
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi,     30)
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


def make_altitude_hist(df: pd.DataFrame) -> go.Figure:
    """Stacked altitude histogram by object type."""
    fig = go.Figure()
    if df.empty:
        return fig
    for otype, color in _TYPE_HEX.items():
        sub = df[df["object_type"] == otype]
        if sub.empty:
            continue
        fig.add_trace(go.Histogram(
            x=sub["alt_km"],
            name=_TYPE_CN.get(otype, otype),
            marker_color=color,
            opacity=0.80,
            xbins=dict(start=0, end=2100, size=40),
        ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="轨道高度 (km)",
        yaxis_title="目标数",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SCENE_BG,
        font_color="#aaaaaa",
        legend=dict(
            orientation="h", y=1.05, x=1, xanchor="right",
            font=dict(color="#aaaaaa"),
        ),
        margin=dict(l=45, r=10, t=5, b=40),
        height=195,
    )
    fig.update_xaxes(gridcolor="#1a2535", color="#aaaaaa")
    fig.update_yaxes(gridcolor="#1a2535", color="#aaaaaa")
    return fig


# ─── Tab 2: 3D sphere ─────────────────────────────────────────────────────────
def make_3d_sphere(
    df: pd.DataFrame,
    layer: dict | None = None,
    max_pts: int = 8000,
) -> go.Figure:
    """Earth + debris scatter. If layer given, dim out-of-band points."""
    fig = go.Figure()
    _add_earth(fig, n=80)

    if df.empty:
        _apply_3d_layout(fig)
        return fig

    # Sample if needed
    plot_df = df.sample(min(len(df), max_pts), random_state=42) if len(df) > max_pts else df.copy()
    x, y, z = lla_to_ecef(plot_df["lat"].values, plot_df["lon"].values, plot_df["alt_km"].values)
    base_colors = _hex_color_col(plot_df)

    if layer is not None:
        # All loaded points are in-layer (query filtered by altitude range)
        colors = [_hex_to_rgba(c, 0.90) for c in base_colors]
        sizes  = [3.2] * len(plot_df)
    else:
        colors = [_hex_to_rgba(c, 0.70) for c in base_colors]
        sizes  = [2.2] * len(plot_df)

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

    # Altitude band shells (only for LEO-range layers to avoid huge geometry)
    if layer is not None and layer["alt_max"] <= 2500:
        _add_altitude_shell(fig, layer["alt_min"], layer["color"], opacity=0.06)
        _add_altitude_shell(fig, layer["alt_max"], layer["color"], opacity=0.10)

    # Explicit Earth-centric scene range — prevents camera drift when
    # max_pts > actual point count (Plotly falls back to data bbox which may
    # be very different from Earth scale when few/no points are present)
    scene_r = R_EARTH + (layer["alt_max"] + 500 if layer else 2500)
    earth_ax = dict(range=[-scene_r, scene_r])
    _apply_3d_layout(fig, height=600)
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
    rp   = traj_points[idx]
    rx, ry, rz = (float(v) for v in lla_to_ecef(rp.lat_deg, rp.lon_deg, rp.alt_km))
    r_norm = float(np.sqrt(rx**2 + ry**2 + rz**2))   # distance from Earth center → rocket
    vel_kms = float(np.linalg.norm(rp.vel_eci))

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
            f"🚀 T+{t_slider_s:.0f}s<br>"
            f"Alt: {rp.alt_km:.1f} km<br>"
            f"Lat: {rp.lat_deg:.2f}°  Lon: {rp.lon_deg:.2f}°<br>"
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
                "#FFDD00" if d < 300 else "#445566"
                for d in dist
            ]
            d_sizes = [7 if d < 20 else 5 if d < 100 else 3 for d in dist]
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



def make_proximity_2d(rp, nearby_df: pd.DataFrame, search_radius_km: float) -> go.Figure:
    """2D East-offset × altitude-offset scatter in rocket local frame.

    Rocket sits at origin (0, 0).  x-axis = East offset (km), y-axis = altitude
    offset relative to rocket (km).  Supports unlimited zoom in Streamlit.
    """
    import math

    nd = nearby_df.dropna(subset=["lat", "lon", "alt_km"])
    if nd.empty:
        return None

    lat_r = rp.lat_deg
    lon_r = rp.lon_deg
    alt_r = rp.alt_km

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
    vel_kms = float(np.linalg.norm(rp.vel_eci))
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=16, color="#00FF55", symbol="star",
                    line=dict(color="white", width=1.5)),
        text=["🚀"],
        textfont=dict(size=13),
        textposition="top center",
        hovertemplate=(
            f"🚀  T+{rp.t_met_s:.0f}s<br>"
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
        st.markdown("##### 🎛️ 筛选")
        alt_range = st.slider("高度范围 (km)", 0, 42000, (0, 2000), step=50,
                              key="gv_alt")
        obj_type  = st.selectbox("目标类型",
                                  ["ALL", "DEBRIS", "PAYLOAD", "ROCKET BODY"],
                                  key="gv_type")
        n_limit   = st.slider("显示上限", 2000, 30000, 12000, step=1000,
                              key="gv_limit")

        if st.button("🔄 刷新", key="gv_refresh"):
            load_positions_now.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("**图例**")
        for k, c in _TYPE_HEX.items():
            st.markdown(
                f'<span style="color:{c};font-size:18px">●</span> {_TYPE_CN.get(k, k)}',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        view_mode = st.radio(
            "视图模式",
            ["🌍 正射球面", "🗺️ 平面交互地图"],
            key="gv_view",
        )

    with col_map:
        with st.spinner("加载当前轨道位置…"):
            df = load_positions_now(
                alt_min=float(alt_range[0]),
                alt_max=float(alt_range[1]),
                obj_type=obj_type,
                limit=n_limit,
            )

        if df.empty:
            st.warning("无数据。请确认已完成全量摄入并等待 trajectory_segments 覆盖当前时刻。")
            return

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(
            f"显示 **{len(df):,}** 个目标  ·  {now_str}  ·  "
            f"碎片 {(df['object_type']=='DEBRIS').sum():,}  "
            f"载荷 {(df['object_type']=='PAYLOAD').sum():,}  "
            f"火箭级 {(df['object_type']=='ROCKET BODY').sum():,}"
        )

        if view_mode == "🌍 正射球面":
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

    # ── Altitude histogram (full width) ─────────────────────────────────────
    leo_df = df[df["alt_km"] <= 2100]
    if not leo_df.empty:
        st.markdown("##### 高度分布（LEO 段）")
        st.plotly_chart(make_altitude_hist(leo_df), use_container_width=True,
                        config=dict(displayModeBar=False))


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
                f'text-align:center;height:116px;box-sizing:border-box;">'
                f'<span style="color:{layer["color"]};font-size:13px;font-weight:bold;'
                f'display:block;margin-bottom:4px">'
                f'{layer["label"]}</span>'
                f'<span style="color:#ccc;font-size:22px;font-weight:bold;'
                f'display:block;margin-bottom:4px">{cnt:,}</span>'
                f'<span style="color:#888;font-size:11px;display:block;'
                f'overflow:hidden;white-space:nowrap;text-overflow:ellipsis">'
                f'{layer["note"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("选择", key=f"lb_{layer['id']}", use_container_width=True):
                st.session_state["ld_layer"] = layer["id"]
                load_positions_now.clear()
                st.rerun()

    st.markdown("---")

    cur = next((l for l in ALTITUDE_LAYERS if l["id"] == sel_id), ALTITUDE_LAYERS[2])

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
        max_pts = st.slider("最大点数", 1000, 20000, 5000, step=500, key="ld_pts")
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
            df3d = load_positions_now(
                alt_min=float(cur["alt_min"]), alt_max=float(cur["alt_max"]),
                obj_type=obj_type_3d, limit=max_pts,
            )
        if df3d.empty:
            st.info("该轨道层暂无数据")
        else:
            st.caption(
                f"**{cur['label']}** 层内目标 {len(df3d):,} 个"
                + (f"（已达上限 {max_pts:,}，实际更多）" if len(df3d) >= max_pts else "（全部显示）")
            )
            fig3d = make_3d_sphere(df3d, layer=cur, max_pts=max_pts)
            st.plotly_chart(fig3d, use_container_width=True,
                            config=dict(scrollZoom=True, displayModeBar=False))

    # Detail table
    if not df3d.empty:
        with st.expander(f"📋 {cur['label']} 层目标列表（前 30）", expanded=False):
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


def _render_mission_slider():
    st.markdown("#### 🚀 发射任务安全时滑")
    st.caption(
        "运行 6-DOF 仿真 → 时间轴拖动 → 实时查询火箭附近碎片 → "
        "3D 近场态势（火箭坐标系，火箭固定在原点，支持深度缩放）+ 2D 近场切片。"
        "**红色**目标距离 < 20 km，**橙色** < 100 km，**黄色** < 300 km。"
    )

    # ── Sim config ──────────────────────────────────────────────────────────
    with st.expander("⚙️ 仿真参数", expanded=not bool(st.session_state.get("ms_result"))):
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
        auto_stop = c8.checkbox("🛰 自动到轨停止",  value=True, key="ms_auto_stop",
                                help="检测到入轨（近地点 > 150 km）后立即停止仿真，避免多余的轨道圈。")
        run_btn   = st.button("▶ 运行仿真", type="primary", key="ms_run",
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
                st.success(
                    f"✅ {vehicle} 仿真完成 — "
                    f"{len(result.nominal)} 个轨迹点 · "
                    f"最终高度 {result.nominal[-1].alt_km:.1f} km · "
                    f"最终速度 {float(np.linalg.norm(result.nominal[-1].vel_eci)):.3f} km/s"
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
    phase_marks: dict[int, str] = {0: "🚀 起飞"}
    if sim.t_meco1:        phase_marks[int(sim.t_meco1)]       = "✂️ MECO1"
    if sim.t_stage_sep:    phase_marks[int(sim.t_stage_sep)]   = "🔩 级间分离"
    if sim.t_meco2:        phase_marks[int(sim.t_meco2)]       = "✂️ MECO2"
    if sim.t_payload_sep:  phase_marks[int(sim.t_payload_sep)] = "🛰 载荷分离"

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
        "🛰️ 轨道惯性飞行" if in_orbit_coast else
        "🔥 大气层内"     if rp.alt_km < 100 else
        "🚀 上升末段"     if rp.alt_km < 200 else
        "🌌 亚轨道"       if vel_kms < 5    else
        "🛰️ 轨道注入"
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
            lat=rp.lat_deg, lon=rp.lon_deg, alt_km=rp.alt_km,
            t_utc=t_query, radius_km=rad, limit=400,
        )

    # Risk events (from session state if available)
    risk_events = []
    if "phase_summaries" in st.session_state:
        for s in st.session_state["phase_summaries"]:
            risk_events.extend(s.events)

    # ── 3D mission figure (left 70%) + info table (right 30%) ───────────────
    col_3d, col_info = st.columns([7, 3])
    with col_3d:
        st.markdown(
            "<small style='color:#8899aa'>▲ 3D 近场态势（火箭坐标系，地球为背景参考）— 鼠标拖动旋转 / 滚轮缩放 / 双击复位</small>",
            unsafe_allow_html=True,
        )
        fig_m = make_mission_fig(nom, float(t_slider), nearby, risk_events)
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
        fig_2d = make_proximity_2d(rp, nearby, rad)
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
        mc3.metric("最近碎片",
                   nearby[nearby["object_type"] == "DEBRIS"]["dist_km"].min()
                   if (nearby["object_type"] == "DEBRIS").any() else "—",
                   delta=None)
        high_risk = (nearby["dist_km"] < 20).sum()
        mc4.metric("< 20 km 目标", high_risk,
                   delta="⚠️ 需关注" if high_risk > 0 else None,
                   delta_color="inverse")

        with st.expander(f"📋 附近目标（{len(nearby)} 个，半径 {rad:.0f} km）", expanded=False):
            disp = nearby[["norad_cat_id","name","object_type","alt_km","dist_km"]].copy()
            disp.columns = ["NORAD ID","名称","类型","高度(km)","距离(km)"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info(f"T+{t_slider}s 附近 {rad:.0f} km 内暂无目标（超出 DB 覆盖窗口或高度过低）。")


# ─── Main entry point ─────────────────────────────────────────────────────────
def render_viz_explorer():
    st.title("🌐 空间碎片可视化探索")
    st.caption(
        "三维沉浸式探索平台  ·  "
        "全球碎片实时态势  ·  "
        "高度分层 3D 下钻  ·  "
        "发射任务安全时滑"
    )

    tab1, tab2, tab3 = st.tabs([
        "🌍 全球碎片态势",
        "📊 高度分层下钻",
        "🚀 发射安全时滑",
    ])
    with tab1:
        _render_global_view()
    with tab2:
        _render_layer_drilldown()
    with tab3:
        _render_mission_slider()
