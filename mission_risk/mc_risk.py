"""
Long-term mission collision-risk assessment.

Algorithm: ORDEM 3.1-calibrated Flux Model + Poisson Monte Carlo
=================================================================

Core formula (NASA ORDEM 3.1 / ESA MASTER-8 standard):

    P_c = 1 − exp(−F · A_cross · Δt)

where
    F       : debris flux [objects / m² / yr] at the target altitude & inclination
    A_cross : satellite effective cross-sectional area [m²]
    Δt      : mission lifetime [yr]

Flux source
-----------
Built-in table calibrated from NASA ORDEM 3.1 (Krisko et al. 2016,
NASA/TM-2016-218569) and ESA MASTER-8 public data, for two size thresholds:
  - > 10 cm  : trackable by ground radar (Space Surveillance Network)
  - >  1 cm  : includes un-trackable lethal debris
               (estimated ~50× the > 10 cm flux; ORDEM 3.1 Fig. 10)

Inclination scaling
-------------------
Polar / retrograde orbits encounter more crossing debris → higher flux.
A simple multiplicative factor f(i) is applied (from ORDEM 3.1 Table 3).

Conjunction counting
--------------------
From the flux we derive a conjunction rate (encounters within d_thresh km):

    λ_conj = F_10cm · π · d_thresh_m²      [conjunctions / yr]

This scales the effective "capture area" from collision (satellite cross-section)
to the conjunction sphere (π · d_thresh²).  The ratio is purely geometric.

Monte Carlo
-----------
For each of n_mc trials:
  1. N_conj ~ Poisson(λ_conj · T)
  2. For each conjunction i:
       d_i  ~ Uniform-area(0, d_thresh)  →  d = d_thresh · √U
       σ_i  ~ |Normal(σ₀, 0.2·σ₀)|
       Pc_i  = ncx2.cdf(HBR²/σ², 2, d²/σ²)   [isotropic 2-D Gaussian]
  3. Pc_agg,k = 1 − ∏(1 − Pc_i)

The headline collision probability P_c_orbit uses the direct ORDEM flux formula,
which is the authoritative engineering result.  The MC aggregate Pc gives the
uncertainty distribution around it.

References
----------
Krisko, P.H. et al. (2016). ORDEM 3.1. NASA/TM-2016-218569.
Flegel, S. et al. (2013). MASTER-8. ESA/ESOC.
NASA NPR 8715.6B: Launch Collision Avoidance Requirements.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ncx2, poisson
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)

# ─── Physical constants ───────────────────────────────────────────────────────
GM_KM3_S2   = 3.986004418e5   # km³/s²
RE_KM       = 6371.0
SECS_PER_YR = 3.156e7         # s/yr

# ─── ORDEM 3.1 built-in flux table ───────────────────────────────────────────
# Approximate flux [obj / m² / yr] at reference inclination i = 53°
# Altitude nodes [km] and corresponding fluxes sourced from:
#   ORDEM 3.1 (Krisko 2016) Figs 7-10; ESA MASTER-8 cross-validated.
#
# Key features captured:
#   • Rapid decay below 400 km (atmospheric drag removes debris quickly)
#   • Strong peak at 800–1200 km (ASAT/Iridium-Cosmos 2009 debris)
#   • Sharp drop in MEO 2000–20 000 km (sparse region)
#   • GEO ring concentration near 36 000 km

_ALT_NODES = np.array([
    200,   300,   400,   500,   600,   700,   800,   900,
    1000,  1200,  1400,  1500,  2000,  5000,  20000, 36000,
], dtype=float)

# Flux for objects > 10 cm (directly trackable by radar)
_F_GT10CM = np.array([
    3.0e-9,  3.0e-8,  8.0e-8,  1.5e-7, 2.5e-7, 3.5e-7, 4.5e-7, 5.2e-7,
    5.8e-7,  8.5e-7,  1.1e-6,  1.0e-6, 1.5e-8, 2.0e-9, 5.0e-10, 2.0e-10,
], dtype=float)

# Flux for objects > 1 cm (lethal un-trackable; ≈ 50× the >10 cm flux)
_F_GT1CM = _F_GT10CM * 50.0

# Interpolators (log-linear in flux, linear in altitude)
_interp_10cm = interp1d(
    _ALT_NODES, np.log10(_F_GT10CM),
    kind="linear", bounds_error=False,
    fill_value=(np.log10(_F_GT10CM[0]), np.log10(_F_GT10CM[-1])),
)
_interp_1cm = interp1d(
    _ALT_NODES, np.log10(_F_GT1CM),
    kind="linear", bounds_error=False,
    fill_value=(np.log10(_F_GT1CM[0]), np.log10(_F_GT1CM[-1])),
)

# Inclination scaling factors at reference inclinations (from ORDEM 3.1 Table 3)
_INC_NODES  = np.array([0,    20,   40,   53,   70,   80,   90,   100,  120,  140,  160,  180], dtype=float)
_INC_SCALE  = np.array([0.38, 0.55, 0.80, 1.00, 1.25, 1.38, 1.45, 1.38, 1.10, 0.80, 0.55, 0.38], dtype=float)
_interp_inc = interp1d(_INC_NODES, _INC_SCALE, kind="linear",
                        bounds_error=False, fill_value=(0.38, 0.38))

# Annual debris growth rates (simplified from NASA LEGEND)
_GROWTH_RATE: Dict[str, float] = {
    "LEO": 0.020, "MEO": 0.005, "GEO": 0.008, "HEO": 0.010,
}


# ─── Flux look-up ─────────────────────────────────────────────────────────────

def flux_at(alt_km: float, inc_deg: float,
            size_threshold: str = "10cm") -> float:
    """
    Return debris flux [obj / m² / yr] at the given altitude and inclination.

    Parameters
    ----------
    alt_km          : orbital altitude [km]
    inc_deg         : orbital inclination [deg]
    size_threshold  : '10cm' (trackable) or '1cm' (all lethal)
    """
    inc_deg = float(np.clip(inc_deg, 0, 180))
    alt_km  = float(np.clip(alt_km, _ALT_NODES[0], _ALT_NODES[-1]))

    log_f = _interp_10cm(alt_km) if size_threshold == "10cm" else _interp_1cm(alt_km)
    f0    = 10.0 ** float(log_f)
    f_inc = float(_interp_inc(inc_deg))
    return f0 * f_inc


def _orbital_velocity_kms(alt_km: float) -> float:
    return math.sqrt(GM_KM3_S2 / (RE_KM + alt_km))


def _shell_volume_km3(alt_km: float, band_km: float) -> float:
    r_out = RE_KM + alt_km + band_km / 2
    r_in  = RE_KM + alt_km - band_km / 2
    return (4 / 3) * math.pi * (r_out ** 3 - r_in ** 3)


def _growth_rate(alt_km: float) -> float:
    if alt_km < 1_000:   return _GROWTH_RATE["LEO"]
    if alt_km < 35_000:  return _GROWTH_RATE["MEO"]
    if alt_km < 40_000:  return _GROWTH_RATE["GEO"]
    return _GROWTH_RATE["HEO"]


def _integrated(rate_per_yr: float, growth: float, years: float) -> float:
    """∫₀ᵀ rate·(1+g)ᵗ dt"""
    if growth <= 0:
        return rate_per_yr * years
    return rate_per_yr * ((1 + growth) ** years - 1) / math.log(1 + growth)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DebrisEnvironment:
    alt_km:            float
    band_km:           float
    n_objects:         int          # from live DB (may be 0 if DB is empty)
    shell_volume_km3:  float
    spatial_density:   float        # from DB [obj / km³]
    inclinations:      np.ndarray
    v_sat_kms:         float
    mean_v_rel_kms:    float
    # ORDEM-derived
    flux_10cm:         float        # [obj / m² / yr]
    flux_1cm:          float
    growth_rate:       float
    top_objects:       pd.DataFrame
    closest_objects:   pd.DataFrame  # 30 closest objects by mean altitude


@dataclass
class MissionRiskReport:
    # Scenario
    altitude_km:        float
    inclination_deg:    float
    mission_years:      float
    launch_date:        datetime
    hbr_km:             float
    conjunction_km:     float
    sigma_km:           float
    n_mc:               int
    sat_area_m2:        float

    # Environment
    env:                DebrisEnvironment

    # ORDEM direct result (primary)
    pc_orbit_10cm:      float   # Pc from trackable objects only
    pc_orbit_1cm:       float   # Pc including all lethal debris
    collision_rate_per_year: float

    # Conjunction statistics (MC)
    lambda_conj_per_year: float
    total_lambda_conj:  float
    n_conj_mean:        float
    n_conj_p95:         float

    # MC aggregate Pc (secondary)
    agg_pc_mean:        float
    agg_pc_p50:         float
    agg_pc_p95:         float

    # Miss-distance statistics
    min_miss_p50_km:    float
    min_miss_p95_km:    float
    min_miss_absolute_km: float

    # Time series (monthly)
    ts_months:          np.ndarray
    ts_pc_flux:         np.ndarray   # ORDEM Pc_orbit curve (mean)
    ts_pc_flux_1cm:     np.ndarray   # ORDEM Pc_orbit_1cm curve
    ts_n_conj_mean:     np.ndarray

    # MC distributions
    pc_agg_samples:     np.ndarray
    conj_count_samples: np.ndarray
    min_miss_samples:   np.ndarray


# ─── Database query ───────────────────────────────────────────────────────────

def fetch_debris_environment(
    alt_km:   float,
    inc_deg:  float,
    band_km:  float = 200.0,
) -> DebrisEnvironment:
    """
    Query the catalog for debris at the target altitude shell, and compute
    the ORDEM 3.1 flux at the specified orbital parameters.
    """
    alt_min = max(0.0, alt_km - band_km / 2)
    alt_max = alt_km + band_km / 2

    # gp_elements does NOT store perigee_km / apogee_km directly.
    # Altitude is derived on-the-fly from mean_motion + eccentricity:
    #   a  = cbrt(GM / n_rad²)  where n_rad = mean_motion × 2π / 86400
    #   perigee = a(1-e) − RE,  apogee = a(1+e) − RE
    _ALT_SQL = """
        cbrt(398600.4418 / power(g.mean_motion * 2 * pi() / 86400, 2))
    """
    _PERI_SQL = f"({_ALT_SQL} * (1 - g.eccentricity) - 6371.0)"
    _APO_SQL  = f"({_ALT_SQL} * (1 + g.eccentricity) - 6371.0)"
    _MEAN_SQL = f"({_ALT_SQL} - 6371.0)"   # mean altitude ≈ semi-major axis − RE

    rows = []
    closest_rows = []
    try:
        from database.db import session_scope
        from sqlalchemy import text

        # Main query: objects whose orbit crosses the altitude band
        with session_scope() as sess:
            rows = sess.execute(text(f"""
                WITH latest AS (
                    SELECT DISTINCT ON (g.norad_cat_id)
                        g.norad_cat_id,
                        g.inclination,
                        g.eccentricity,
                        g.mean_motion,
                        {_PERI_SQL} AS perigee_km,
                        {_APO_SQL}  AS apogee_km
                    FROM gp_elements g
                    WHERE g.mean_motion > 0
                    ORDER BY g.norad_cat_id, g.epoch DESC
                )
                SELECT l.norad_cat_id,
                       COALESCE(co.name, l.norad_cat_id::text) AS name,
                       l.inclination, l.perigee_km, l.apogee_km,
                       l.eccentricity, co.object_type
                FROM latest l
                LEFT JOIN catalog_objects co ON co.norad_cat_id = l.norad_cat_id
                WHERE l.perigee_km <= :hi AND l.apogee_km >= :lo
                LIMIT 5000
            """), {"lo": alt_min, "hi": alt_max}).fetchall()

        # Closest-objects query: top 30 by mean-altitude proximity, no band filter
        with session_scope() as sess:
            closest_rows = sess.execute(text(f"""
                WITH latest AS (
                    SELECT DISTINCT ON (g.norad_cat_id)
                        g.norad_cat_id,
                        g.inclination,
                        g.eccentricity,
                        {_PERI_SQL}  AS perigee_km,
                        {_APO_SQL}   AS apogee_km,
                        {_MEAN_SQL}  AS mean_alt_km,
                        ABS({_MEAN_SQL} - :alt_km) AS dist_km
                    FROM gp_elements g
                    WHERE g.mean_motion > 0
                    ORDER BY g.norad_cat_id, g.epoch DESC
                )
                SELECT l.norad_cat_id,
                       COALESCE(co.name, l.norad_cat_id::text) AS name,
                       l.inclination, l.perigee_km, l.apogee_km,
                       l.mean_alt_km, l.dist_km, co.object_type
                FROM latest l
                LEFT JOIN catalog_objects co ON co.norad_cat_id = l.norad_cat_id
                ORDER BY l.dist_km ASC
                LIMIT 30
            """), {"alt_km": alt_km}).fetchall()

    except Exception as exc:
        log.warning("DB query failed: %s", exc)

    n_obj = len(rows)
    incs  = np.array([float(r[2]) for r in rows], dtype=float)

    vol     = _shell_volume_km3(alt_km, band_km)
    density = n_obj / vol if vol > 0 else 0.0
    v_sat   = _orbital_velocity_kms(alt_km)

    # Mean relative velocity from actual inclination distribution
    if len(incs) > 0:
        di     = np.radians(np.abs(incs - inc_deg))
        v_rel2 = 2 * v_sat ** 2 * (1 - np.cos(di))
        v_rel  = float(np.mean(np.sqrt(np.maximum(v_rel2, 1e-6))))
    else:
        v_rel  = v_sat * math.sqrt(2)

    # ORDEM fluxes (independent of DB)
    f10 = flux_at(alt_km, inc_deg, "10cm")
    f1  = flux_at(alt_km, inc_deg, "1cm")

    # Build top_objects DataFrame
    records = [{
        "NORAD ID":    int(r[0]),
        "名称":         str(r[1]),
        "类型":         str(r[6] or "DEBRIS"),
        "倾角(°)":     round(float(r[2]), 1),
        "近地点(km)":  int(round(float(r[3]))),
        "远地点(km)":  int(round(float(r[4]))),
    } for r in rows]
    top_df = pd.DataFrame(records) if records else pd.DataFrame()

    # Build closest_objects DataFrame
    close_records = [{
        "NORAD ID":     int(r[0]),
        "名称":          str(r[1]),
        "类型":          str(r[7] or "DEBRIS"),
        "倾角(°)":      round(float(r[2]), 1),
        "近地点(km)":   int(round(float(r[3]))),
        "远地点(km)":   int(round(float(r[4]))),
        "均值高度(km)": int(round(float(r[5]))),
        "与目标偏差(km)": round(float(r[6]), 1),
    } for r in closest_rows]
    closest_df = pd.DataFrame(close_records) if close_records else pd.DataFrame()

    return DebrisEnvironment(
        alt_km=alt_km, band_km=band_km,
        n_objects=n_obj, shell_volume_km3=vol, spatial_density=density,
        inclinations=incs, v_sat_kms=v_sat, mean_v_rel_kms=v_rel,
        flux_10cm=f10, flux_1cm=f1,
        growth_rate=_growth_rate(alt_km),
        top_objects=top_df,
        closest_objects=closest_df,
    )


# ─── Vectorised Pc ────────────────────────────────────────────────────────────

def _pc_isotropic(d_km: np.ndarray, sigma_km: np.ndarray,
                   hbr_km: float) -> np.ndarray:
    s = np.maximum(sigma_km, 1e-6)
    return ncx2.cdf((hbr_km / s) ** 2, df=2, nc=(d_km / s) ** 2)


# ─── Main simulation ──────────────────────────────────────────────────────────

def run_monte_carlo(
    env:            DebrisEnvironment,
    inc_deg:        float,
    mission_years:  float,
    hbr_km:         float  = 0.010,    # 10 m combined HBR
    conjunction_km: float  = 5.0,
    sigma_km:       float  = 1.5,
    n_mc:           int    = 2000,
    sat_area_m2:    float  = 10.0,     # satellite effective cross-section [m²]
    rng_seed:       int    = 42,
) -> MissionRiskReport:
    """
    Compute long-term collision risk using ORDEM 3.1 flux model + Poisson MC.

    Primary result: P_c_orbit_10cm = 1 − exp(−F_10cm · A · Δt)
    MC result:      P_c_agg distribution from Poisson-sampled conjunctions
    """
    rng = np.random.default_rng(rng_seed)
    gr  = env.growth_rate

    # ── ORDEM flux-based collision probability ────────────────────────────────
    # Incorporate growth: P_c = 1 − exp(−F·A·∫(1+g)ᵗ dt)
    eff_years_10cm = _integrated(1.0, gr, mission_years)   # effective exposure factor
    eff_years_1cm  = _integrated(1.0, gr, mission_years)

    pc_orbit_10cm = 1.0 - math.exp(
        -env.flux_10cm * sat_area_m2 * eff_years_10cm * mission_years
        / mission_years  # = 1 - exp(-F·A·∫dt) where ∫dt = eff_years*T/T = eff_years
    )
    # Simpler: use integrated exposure
    total_fluence_10cm = env.flux_10cm * sat_area_m2 * _integrated(1.0, gr, mission_years)
    total_fluence_1cm  = env.flux_1cm  * sat_area_m2 * _integrated(1.0, gr, mission_years)
    pc_orbit_10cm = 1.0 - math.exp(-total_fluence_10cm)
    pc_orbit_1cm  = 1.0 - math.exp(-total_fluence_1cm)

    coll_rate_per_yr = env.flux_10cm * sat_area_m2   # [/yr] trackable objects

    # ── Conjunction rate ─────────────────────────────────────────────────────
    # Scale flux from collision cross-section to conjunction sphere
    # λ_conj = F_10cm · π · d_thresh_m²
    d_thresh_m = conjunction_km * 1000.0   # km → m
    lambda_conj_per_yr = env.flux_10cm * math.pi * d_thresh_m ** 2

    total_lambda_conj = _integrated(lambda_conj_per_yr, gr, mission_years)

    log.info(
        "MCRA flux model: alt=%.0f km  inc=%.0f°  F_10cm=%.2e /m²/yr"
        "  Pc_10cm=%.2e  Pc_1cm=%.2e  λ_conj=%.2f/yr  Λ_total=%.1f",
        env.alt_km, inc_deg, env.flux_10cm,
        pc_orbit_10cm, pc_orbit_1cm, lambda_conj_per_yr, total_lambda_conj,
    )

    # ── Monte Carlo (vectorised) ──────────────────────────────────────────────
    conj_counts = rng.poisson(total_lambda_conj, size=n_mc)
    max_conj    = max(int(conj_counts.max()), 1)

    d_miss    = conjunction_km * np.sqrt(rng.random((n_mc, max_conj)))
    sigma_arr = np.abs(rng.normal(sigma_km, 0.2 * sigma_km, (n_mc, max_conj)))
    sigma_arr = np.clip(sigma_arr, 0.05, 20.0)

    pc_all = _pc_isotropic(d_miss, sigma_arr, hbr_km)
    mask   = np.arange(max_conj)[None, :] < conj_counts[:, None]

    log1m          = np.where(mask, np.log1p(-np.minimum(pc_all, 1.0 - 1e-15)), 0.0)
    pc_agg_samples = 1.0 - np.exp(log1m.sum(axis=1))

    d_miss_masked    = np.where(mask, d_miss, np.inf)
    min_miss_samples = np.where(conj_counts > 0, d_miss_masked.min(axis=1), conjunction_km)

    agg_pc_mean = float(np.mean(pc_agg_samples))
    agg_pc_p50  = float(np.percentile(pc_agg_samples, 50))
    agg_pc_p95  = float(np.percentile(pc_agg_samples, 95))

    n_conj_mean  = float(np.mean(conj_counts))
    n_conj_p95   = float(np.percentile(conj_counts, 95))
    min_miss_p50 = float(np.percentile(min_miss_samples, 50))
    min_miss_p95 = float(np.percentile(min_miss_samples, 95))
    fin          = min_miss_samples[np.isfinite(min_miss_samples)]
    min_miss_abs = float(fin.min()) if len(fin) > 0 else conjunction_km

    # ── Time series (analytical, fast) ────────────────────────────────────────
    T_months   = int(round(mission_years * 12))
    ts_months  = np.arange(T_months + 1, dtype=float)
    ts_pc_flux = np.zeros(T_months + 1)
    ts_pc_1cm  = np.zeros(T_months + 1)
    ts_n_conj  = np.zeros(T_months + 1)

    for m in range(1, T_months + 1):
        t_yr          = m / 12.0
        fl_10         = env.flux_10cm * sat_area_m2 * _integrated(1.0, gr, t_yr)
        fl_1          = env.flux_1cm  * sat_area_m2 * _integrated(1.0, gr, t_yr)
        ts_pc_flux[m] = 1.0 - math.exp(-fl_10)
        ts_pc_1cm[m]  = 1.0 - math.exp(-fl_1)
        ts_n_conj[m]  = _integrated(lambda_conj_per_yr, gr, t_yr)

    log.info(
        "MCRA MC done: Pc_agg mean=%.2e  p95=%.2e  n_conj mean=%.1f  "
        "min_miss p50=%.2f km",
        agg_pc_mean, agg_pc_p95, n_conj_mean, min_miss_p50,
    )

    return MissionRiskReport(
        altitude_km=env.alt_km, inclination_deg=inc_deg,
        mission_years=mission_years,
        launch_date=datetime.now(timezone.utc),
        hbr_km=hbr_km, conjunction_km=conjunction_km,
        sigma_km=sigma_km, n_mc=n_mc, sat_area_m2=sat_area_m2,
        env=env,
        pc_orbit_10cm=pc_orbit_10cm,
        pc_orbit_1cm=pc_orbit_1cm,
        collision_rate_per_year=coll_rate_per_yr,
        lambda_conj_per_year=lambda_conj_per_yr,
        total_lambda_conj=total_lambda_conj,
        n_conj_mean=n_conj_mean, n_conj_p95=n_conj_p95,
        agg_pc_mean=agg_pc_mean, agg_pc_p50=agg_pc_p50, agg_pc_p95=agg_pc_p95,
        min_miss_p50_km=min_miss_p50, min_miss_p95_km=min_miss_p95,
        min_miss_absolute_km=min_miss_abs,
        ts_months=ts_months,
        ts_pc_flux=ts_pc_flux, ts_pc_flux_1cm=ts_pc_1cm,
        ts_n_conj_mean=ts_n_conj,
        pc_agg_samples=pc_agg_samples,
        conj_count_samples=conj_counts.astype(float),
        min_miss_samples=min_miss_samples,
    )
