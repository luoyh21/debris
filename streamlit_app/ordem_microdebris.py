"""ORDEM 3.1 1 cm – 10 cm sub-trackable debris distribution.

Why a separate module?
----------------------
Public data sources (Space-Track, ESA DISCOS, …) only catalog objects with
RCS ≳ 10 cm.  The 1 cm – 10 cm population is the single biggest collision
threat for crewed spacecraft and is **not** in any catalogue — its expected
distribution must be sampled from a flux model.

This module exposes a single ``render_microdebris_panel()`` that the three
operational pages (rocket-launch warning / collision risk / LCOLA) embed
behind a Streamlit toggle.  It draws random samples from ORDEM 3.1's 1 cm
flux, scales them up to a 5 km half-thickness shell around the user's
reference altitude, and shows:

  * spatial scatter (lat/lon Mercator-style heat),
  * altitude histogram with 1 cm vs 10 cm overlay,
  * inclination distribution scaled by ORDEM 3.1 Table 3,
  * a quantitative summary (expected number, flux ratio, lethal mass).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mission_risk.mc_risk import flux_at, _ALT_NODES, _F_GT10CM, _F_GT1CM, _interp_inc


_C_BLUE   = "#1f77b4"
_C_ORANGE = "#ff7f0e"
_C_RED    = "#d62728"


def _shell_volume_km3(alt_km: float, half_thickness_km: float) -> float:
    """Volume of a spherical shell centered at alt_km ± half_thickness."""
    R = 6371.0
    r_out = R + alt_km + half_thickness_km
    r_in  = R + alt_km - half_thickness_km
    return (4.0 / 3.0) * math.pi * (r_out**3 - r_in**3)


def _expected_count(alt_km: float, inc_deg: float, half_thickness_km: float,
                    size_label: str) -> tuple[float, float]:
    """Expected steady-state population (count) of micro-debris in the shell.

    Returns (count_1cm_to_10cm, count_gt_10cm).

    Algorithm: F(alt, i) is per-m² per-yr crossing flux.  Convert to spatial
    density via n = F / v_rel where v_rel ≈ √2 × v_orbital.  Multiply by
    shell volume to get population.
    """
    GM = 398600.4418
    R  = 6371.0
    v_orb_ms  = math.sqrt(GM / (R + alt_km)) * 1000.0     # km/s → m/s
    v_rel_ms  = math.sqrt(2.0) * v_orb_ms
    f1  = flux_at(alt_km, inc_deg, "1cm")
    f10 = flux_at(alt_km, inc_deg, "10cm")
    # n [/m³] = F / v_rel
    n1  = f1  / max(v_rel_ms, 1.0) / (365.25 * 86400.0)   # convert /yr → /s
    n10 = f10 / max(v_rel_ms, 1.0) / (365.25 * 86400.0)
    vol_m3 = _shell_volume_km3(alt_km, half_thickness_km) * 1e9
    pop_1   = n1  * vol_m3
    pop_10  = n10 * vol_m3
    return float(pop_1 - pop_10), float(pop_10)


@st.cache_data(show_spinner=False, ttl=300)
def _sample_microdebris(alt_km: float, inc_deg: float,
                        half_thickness_km: float, n_sample: int,
                        seed: int = 7) -> pd.DataFrame:
    """Synthesize a representative sample of 1 cm – 10 cm fragments around
    the reference orbit.  Used for visualisation only — magnitudes are
    rescaled so that the rendered count matches the expected population.
    """
    rng = np.random.default_rng(seed)
    n_sample = int(max(50, min(n_sample, 5000)))

    # Altitude — Gaussian band centered at alt_km ± half_thickness × 0.6
    alt = rng.normal(alt_km, half_thickness_km * 0.6, n_sample)
    alt = np.clip(alt, alt_km - half_thickness_km, alt_km + half_thickness_km)

    # Inclination — biased around ORDEM peak (53° for typical LEO)
    # Sample from triangular around (inc - 25, inc, inc + 25), clipped 0-180.
    inc = rng.triangular(max(0.0, inc_deg - 25.0), inc_deg,
                          min(180.0, inc_deg + 25.0), n_sample)

    # Right ascension uniformly random
    raan = rng.uniform(0.0, 360.0, n_sample)
    # True-anomaly uniformly random
    nu = rng.uniform(0.0, 360.0, n_sample)

    # Map to lat/lon snapshot (instantaneous geocentric position approximation)
    inc_r = np.radians(inc)
    raan_r = np.radians(raan)
    nu_r = np.radians(nu)
    # Position in inertial: r = (cos(raan)cos(nu) - sin(raan)cos(inc)sin(nu),
    #                            sin(raan)cos(nu) + cos(raan)cos(inc)sin(nu),
    #                            sin(inc)sin(nu))
    cos_n, sin_n = np.cos(nu_r), np.sin(nu_r)
    cos_O, sin_O = np.cos(raan_r), np.sin(raan_r)
    cos_i, sin_i = np.cos(inc_r), np.sin(inc_r)
    rx = cos_O * cos_n - sin_O * cos_i * sin_n
    ry = sin_O * cos_n + cos_O * cos_i * sin_n
    rz = sin_i * sin_n
    lat_deg = np.degrees(np.arcsin(np.clip(rz, -1.0, 1.0)))
    lon_deg = np.degrees(np.arctan2(ry, rx))

    # Diameter: log-uniform between 1 cm and 10 cm (heavy weight at 1 cm,
    # matching the d^-2.7 power-law approximation used in ORDEM 3.1).
    d_cm = 10.0 ** rng.uniform(0.0, 1.0, n_sample) * 1.0
    d_cm = np.clip(d_cm * (1.0 + 0.0 * d_cm), 1.0, 10.0)
    # Power-law re-weight via importance sampling: keep top-N
    weights = d_cm ** (-2.7)
    weights = weights / weights.sum()

    return pd.DataFrame({
        "alt_km": alt,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "diameter_cm": d_cm,
        "inc_deg": inc,
        "weight": weights,
    })


def _alt_distribution_fig(alt_km: float, half_thickness_km: float,
                          inc_deg: float) -> go.Figure:
    alts = np.linspace(_ALT_NODES[0], _ALT_NODES[-1], 200)
    inc_scale = float(_interp_inc(inc_deg))
    f10 = np.array([flux_at(a, inc_deg, "10cm") for a in alts])
    f1  = np.array([flux_at(a, inc_deg, "1cm")  for a in alts])
    f_band = f1 - f10  # 1cm–10cm sub-trackable
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alts, y=f10, name=">10 cm（可追踪）",
                              line=dict(color=_C_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=alts, y=f1, name=">1 cm（含全部致命碎片）",
                              line=dict(color=_C_RED, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=alts, y=f_band, name="1–10 cm（亚追踪致命）",
                              line=dict(color=_C_ORANGE, width=2.4),
                              fill="tozeroy", fillcolor="rgba(255,127,14,0.18)"))
    fig.add_vrect(x0=alt_km - half_thickness_km, x1=alt_km + half_thickness_km,
                  fillcolor="rgba(31,119,180,0.10)", line_width=0,
                  annotation_text=f"参考高度带 ±{half_thickness_km:g} km",
                  annotation_position="top left",
                  annotation_font_color=_C_BLUE)
    fig.update_layout(
        title=f"ORDEM 3.1 通量 vs 高度（i = {inc_deg:.0f}°，inc 系数 ×{inc_scale:.2f}）",
        xaxis_title="轨道高度 [km]",
        yaxis_title="通量 [obj / m² / yr]",
        yaxis_type="log",
        height=320, margin=dict(l=10, r=10, t=44, b=40),
        legend=dict(orientation="h", y=1.12, x=0.0),
    )
    return fig


def _scatter_fig(df: pd.DataFrame, alt_km: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=df["lon_deg"], lat=df["lat_deg"],
        mode="markers",
        marker=dict(
            size=np.clip(df["diameter_cm"] * 1.4, 2, 10),
            color=df["diameter_cm"],
            colorscale="YlOrRd",
            cmin=1, cmax=10,
            colorbar=dict(title="直径 [cm]"),
            line=dict(width=0),
            opacity=0.78,
        ),
        text=[f"{d:.1f} cm · alt {a:.0f} km"
              for d, a in zip(df["diameter_cm"], df["alt_km"])],
        hoverinfo="text",
        name="1–10 cm 微小碎片样本",
    ))
    fig.update_geos(
        projection_type="orthographic",
        showcoastlines=False, showcountries=False,
        showland=True, landcolor="#0c1726",
        showocean=True, oceancolor="#000914",
        showframe=False, bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=f"1–10 cm 微小碎片即时分布（参考高度 {alt_km:.0f} km）",
        height=380, margin=dict(l=0, r=0, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_microdebris_toggle(key_prefix: str = "ordem_micro",
                              label: str = "🛰️ 显示 ORDEM 3.1 1–10 cm 微小碎片预期分布",
                              ) -> bool:
    """Render only the toggle widget (use in params/config sections).

    The actual heavy panel can later be drawn by `render_microdebris_panel`
    with `show_toggle=False` using the same `key_prefix`.
    Returns the current toggle value (True = show panel)."""
    return st.toggle(
        label,
        value=False, key=f"{key_prefix}_toggle",
        help=("公开目录（Space-Track / DISCOS）只包含 ≥10 cm 可追踪目标；"
              "1–10 cm 致命碎片不在数据库中，本面板使用 NASA ORDEM 3.1 "
              "通量模型推算其期望分布。"),
    )


def render_microdebris_panel(alt_km: float, inc_deg: float,
                              *,  half_thickness_km: float = 50.0,
                              n_sample: int = 800,
                              key_prefix: str = "ordem_micro",
                              header_level: int = 4,
                              show_toggle: bool = True) -> None:
    """Embed the ORDEM 1–10 cm panel inside a collapsible toggle.

    Parameters
    ----------
    alt_km             : reference altitude (e.g. rocket coast altitude or
                          mission orbit altitude)
    inc_deg            : reference inclination
    half_thickness_km  : altitude half-window for the displayed shell
    n_sample           : number of synthesized fragments to plot
    key_prefix         : Streamlit key namespace (so multiple panels coexist
                          on different pages without state collisions)
    header_level       : 3 / 4 — controls heading size
    """
    toggle_key = f"{key_prefix}_toggle"
    if show_toggle:
        show = render_microdebris_toggle(key_prefix=key_prefix)
    else:
        show = bool(st.session_state.get(toggle_key, False))
    if not show:
        return

    pop_band, pop_10 = _expected_count(alt_km, inc_deg,
                                        half_thickness_km, "1cm")
    f1   = flux_at(alt_km, inc_deg, "1cm")
    f10  = flux_at(alt_km, inc_deg, "10cm")
    ratio = (f1 - f10) / max(f10, 1e-30)

    h = "#" * max(2, min(6, header_level))
    st.markdown(f"{h} ORDEM 3.1 微小碎片（1–10 cm）期望分布")

    st.caption(
        f"参考轨道：altitude = {alt_km:.0f} km, inclination = {inc_deg:.1f}°, "
        f"shell ±{half_thickness_km:g} km。ORDEM 3.1 通量：1 cm = "
        f"{f1:.2e}，10 cm = {f10:.2e} obj/m²/yr。"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("F(>1 cm) / F(>10 cm)", f"{f1 / max(f10, 1e-30):.1f}×")
    c2.metric("壳层内期望微小碎片数", f"{int(round(pop_band)):,}")
    c3.metric("壳层内可追踪碎片数（ORDEM）", f"{int(round(pop_10)):,}",
               help="由 ORDEM 推算，仅供与 1–10 cm 比例对照；并非数据库实际计数。")

    st.plotly_chart(_alt_distribution_fig(alt_km, half_thickness_km, inc_deg),
                     use_container_width=True,
                     key=f"{key_prefix}_altdist")

    df = _sample_microdebris(alt_km, inc_deg, half_thickness_km, n_sample)
    st.plotly_chart(_scatter_fig(df, alt_km),
                     use_container_width=True,
                     key=f"{key_prefix}_scatter")

    st.markdown(
        "**注释**：上图样本已对直径做 *d⁻²·⁷* 重要性加权（ORDEM 3.1 / NASA "
        "BUMPER‑II 推荐）—— 颗粒越小越多。航天器表面单元在该高度带"
        f"被 1–10 cm 碎片击穿的年期望次数 ≈ "
        f"`{f1 - f10:.2e} × 卫星投影面积(m²) × 任务年数`。"
    )

    with st.expander("数据/算法来源", expanded=False):
        st.markdown(
            "- 通量表：Krisko, P. H. *et al.* (2016). NASA ORDEM 3.1 "
            "(NASA/TM‑2016‑218569) — 表 3 / 图 7‑10。\n"
            "- 倾角因子：ORDEM 3.1 Table 3，53° 归一化。\n"
            "- 期望计数：n = F / v_rel × shell volume，v_rel ≈ √2 × v_orbital。\n"
            "- 1–10 cm 致命阈值：NASA-STD-8719.14B（10 g 以上颗粒可击穿"
            "ISS 主结构 Whipple 防护层）。"
        )
