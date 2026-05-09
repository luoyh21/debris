"""Validate the SBM debris-trajectory algorithm against historical events.

For each well-documented benchmark event (FY-1C ASAT 2007, Iridium-Cosmos
2009, Cosmos-1408 ASAT 2021, Briz-M 2007 explosion, NOAA-16 2015
explosion) we:

1. Build a synthetic ``SpaceEvent`` with the *publicly reported* parent
   mass, altitude, inclination, relative velocity, and event type.
2. Run :func:`events.nasa_sbm.simulate_breakup` exactly as the production
   pipeline does.
3. Compare the model output (≥10 cm tracked, ≥1 cm lethal, catastrophic
   flag) against the **post-event NASA ODPO catalog** numbers (from the
   NASA Orbital Debris Quarterly News archive).
4. Print a table and a JSON dump suitable for the validation document.

Run::

    docker exec debris-api-1 python scripts/validate_sbm.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from events.types import EventType, SpaceEvent
from events.nasa_sbm import simulate_breakup, _n_collision, _n_explosion


# ──────────────────────────────────────────────────────────────────────────
# Canonical historical events (NASA ODQN public sources)
# ──────────────────────────────────────────────────────────────────────────
# Each entry stores both the *inputs* used by the SBM and the *observed*
# tracked-fragment counts published by NASA ODPO years after the event,
# i.e. the "ground truth" against which we are validating.
# References:
#   • Pardini & Anselmo, "Long-term evolution of the FY-1C debris cloud",
#     Acta Astronautica 137 (2017).
#   • NASA ODQN 13-2 (Apr 2009) – Iridium 33 / Cosmos 2251 collision.
#   • NASA ODQN 26-1 (Mar 2022) – Cosmos 1408 ASAT.
#   • NASA ODQN 11-2 (Apr 2007) – Briz-M tank failure.
#   • NASA ODQN 19-4 (Oct 2015) – NOAA-16 battery explosion.
BENCHMARKS = [
    {
        "id":       "FY1C-2007",
        "name":     "FY-1C ASAT (2007-01-11)",
        "type":     EventType.COLLISION,
        "alt_km":   865.0,
        "inc_deg":  98.7,
        "m_parent": 880.0,    # FY-1C dry mass
        "m_target": 600.0,    # SC-19 KKV (publicly estimated)
        "v_rel_km_s": 9.0,
        "obs_tracked_ge_10cm": 3438,   # NASA ODPO Apr 2024 catalog count
        "obs_remaining_2024":  2737,   # still on orbit (slow LEO decay)
        "ref": "Pardini & Anselmo 2017; NASA ODQN 28-2 (2024).",
    },
    {
        "id":       "IRIDIUM-COSMOS-2009",
        "name":     "Iridium-33 ↔ Cosmos-2251 (2009-02-10)",
        "type":     EventType.COLLISION,
        "alt_km":   789.0,
        "inc_deg":  86.0,
        "m_parent": 560.0,    # Iridium-33
        "m_target": 900.0,    # Cosmos-2251
        "v_rel_km_s": 11.7,
        "obs_tracked_ge_10cm": 2296,   # 1825 (Cos) + 471 (Irid) cumulative
        "ref": "NASA ODQN 13-2 (Apr 2009); ODQN 26-2 update.",
    },
    {
        "id":       "COSMOS1408-2021",
        "name":     "Cosmos-1408 ASAT (2021-11-15)",
        "type":     EventType.COLLISION,
        "alt_km":   480.0,
        "inc_deg":  82.6,
        "m_parent": 1750.0,
        "m_target":   24.0,    # Nudol PL-19 KKV (estimated)
        "v_rel_km_s": 6.0,
        "obs_tracked_ge_10cm": 1786,   # cataloged in months after event
        "obs_remaining_2024":   267,   # most decayed (low alt)
        "ref": "NASA ODQN 26-1 (Mar 2022).",
    },
    {
        "id":       "BRIZ-M-2007",
        "name":     "Briz-M propellant tank explosion (2007-02-19)",
        "type":     EventType.FRAGMENTATION,
        "alt_km":   495.0,    # apogee at time of breakup
        "inc_deg":  49.9,
        "m_parent": 2370.0,
        "obs_tracked_ge_10cm": 1130,
        "ref": "NASA ODQN 11-2 (Apr 2007), 12-3 update.",
    },
    {
        "id":       "NOAA16-2015",
        "name":     "NOAA-16 battery explosion (2015-11-25)",
        "type":     EventType.FRAGMENTATION,
        "alt_km":   855.0,
        "inc_deg":  98.7,
        "m_parent": 1457.0,
        "obs_tracked_ge_10cm": 458,    # cumulative through 2024
        "ref": "NASA ODQN 19-4 (Oct 2015).",
    },
]


def _make_event(b: dict) -> SpaceEvent:
    """Convert benchmark spec to a SpaceEvent."""
    e_to_m = None
    if b["type"] == EventType.COLLISION:
        # E/M  =  ½ · μ_red · v_rel² / m_total  →  J/g
        m1, m2 = b["m_parent"], b["m_target"]
        mu_red = (m1 * m2) / (m1 + m2)
        v_ms   = b["v_rel_km_s"] * 1000.0
        e_to_m = 0.5 * mu_red * v_ms * v_ms / (m1 + m2) / 1000.0   # J/g
    return SpaceEvent(
        event_type      = b["type"],
        epoch           = dt.datetime.utcnow(),
        name            = b["name"],
        parent_norad    = None,
        altitude_km     = b["alt_km"],
        inclination_deg = b["inc_deg"],
        mass_parent_kg  = b["m_parent"],
        mass_target_kg  = b.get("m_target"),
        energy_to_mass  = e_to_m,
        source          = "validation",
        source_id       = b["id"],
    )


def _theoretical(b: dict, lc_min: float = 0.10, lc_max: float = 1.0) -> int:
    """Closed-form Johnson 2001 prediction at ≥ 10 cm (no sampling)."""
    if b["type"] == EventType.COLLISION:
        m_total = b["m_parent"] + b.get("m_target", 0.0)
        return _n_collision(m_total, lc_min, lc_max, catastrophic=True)
    return _n_explosion(lc_min, lc_max)


def _validate_pc():
    """Cross-check Foster (numerical) vs Chan (series) Pc on canonical cases.

    The benchmarks here come from:

    * Alfriend & Akella, "Probability of Collision Error Analysis"
      Space Debris J. (1999)  — Tab. 1 isotropic test cases.
    * Hall, D., "Implementation recommendations and usage boundaries
      for the 2-D Pc calculation", AAS-19-642 (2019) — NASA CARA test
      vectors (anisotropic).
    """
    import numpy as np
    from lcola.foster_pc import foster_pc, chan_pc

    cases = [
        # ── Isotropic, miss/σ = 0  (centred secondary)
        {"id": "Iso-centred", "miss_km": [0.0, 0.0],
         "cov":  [[0.1**2, 0], [0, 0.1**2]], "hbr_km": 0.020,
         "ref": "1 - exp(-HBR²/2σ²) = 0.01980",
         "Pc_ref": 1.0 - np.exp(- (0.02**2) / (2 * 0.1**2))},
        # ── Isotropic, miss = 1 σ
        {"id": "Iso-1sigma",  "miss_km": [0.1, 0.0],
         "cov":  [[0.1**2, 0], [0, 0.1**2]], "hbr_km": 0.020,
         "ref": "ncx2-CDF analytic",
         "Pc_ref": float(__import__('scipy.stats', fromlist=['ncx2'])
                          .ncx2.cdf(0.02**2/0.1**2, df=2, nc=1.0))},
        # ── Anisotropic Hall 2019 test #1
        {"id": "Hall2019-T1", "miss_km": [0.5, 0.0],
         "cov":  [[1.0, 0.0], [0.0, 0.04]], "hbr_km": 0.020,
         "ref": "NASA CARA 2-D Pc test #1 (Hall 2019)",
         "Pc_ref": None},   # we use foster as the reference here
        # ── Iridium-Cosmos style geometry (miss = 0.584 km, σ ≈ 0.2 km)
        {"id": "Iridium-Cosmos-est", "miss_km": [0.584, 0.0],
         "cov":  [[0.20**2, 0], [0, 0.05**2]], "hbr_km": 0.025,
         "ref": "NASA pre-event CDM Pc ~ 5e-5 .. 1e-3 range",
         "Pc_ref": None},
    ]

    rows = []
    for c in cases:
        miss = np.asarray(c["miss_km"]); cov = np.asarray(c["cov"])
        pc_f, _ = foster_pc(miss, cov, c["hbr_km"])
        pc_c    = chan_pc(miss, cov, c["hbr_km"])
        rows.append({
            "id":      c["id"],
            "Pc_foster": pc_f,
            "Pc_chan":   pc_c,
            "Pc_analytic_or_ref": c["Pc_ref"],
            "rel_err_pct": (None if c["Pc_ref"] in (None, 0)
                            else round(100*(pc_f - c["Pc_ref"])/c["Pc_ref"], 2)),
            "ref": c["ref"],
        })
    return rows


def main():
    rows = []
    for b in BENCHMARKS:
        evt = _make_event(b)
        # Sample the model (≥ 1 cm to get a richer histogram, but we count
        # both tracked and lethal explicitly).
        res = simulate_breakup(
            evt, lc_min_m=0.01, lc_max_m=1.0,
            max_fragments=20000, seed=42,
        )
        n_theo_10 = _theoretical(b, 0.10, 1.0)
        obs10 = b["obs_tracked_ge_10cm"]
        rows.append({
            "id":           b["id"],
            "event":        b["name"],
            "type":         b["type"].value,
            "obs_tracked_ge_10cm":     obs10,
            "sbm_theoretical_ge_10cm": n_theo_10,
            "sbm_sampled_total":       res.n_total,
            "sbm_tracked_ge_10cm":     res.n_tracked_ge_10cm,
            "sbm_lethal_ge_1cm":       res.n_lethal_ge_1cm,
            "catastrophic":            res.catastrophic,
            "rel_err_theo_pct": round(100.0 * (n_theo_10 - obs10) / obs10, 1),
            "ref": b["ref"],
        })

    # Print human table
    hdr = ("ID", "obs≥10cm", "SBM≥10cm", "Δ%", "1cm≥", "Cata?")
    print(f"{hdr[0]:<22} {hdr[1]:>9} {hdr[2]:>9} {hdr[3]:>7} {hdr[4]:>8} {hdr[5]:>6}")
    print("─" * 70)
    for r in rows:
        print(f"{r['id']:<22} {r['obs_tracked_ge_10cm']:>9} "
              f"{r['sbm_theoretical_ge_10cm']:>9} "
              f"{r['rel_err_theo_pct']:>6.1f}% "
              f"{r['sbm_lethal_ge_1cm']:>8} "
              f"{'Y' if r['catastrophic'] else 'N':>6}")

    pc_rows = _validate_pc()
    print("\n══ Pc (Foster vs Chan vs analytic) ══")
    print(f"{'case':<22} {'Pc_foster':>12} {'Pc_chan':>12} "
          f"{'analytic':>12} {'Δ%':>7}")
    print("─" * 75)
    for r in pc_rows:
        ana = r["Pc_analytic_or_ref"]
        ana_s = f"{ana:.3e}" if ana is not None else "—"
        de    = f"{r['rel_err_pct']:.2f}" if r['rel_err_pct'] is not None else "—"
        print(f"{r['id']:<22} {r['Pc_foster']:>12.3e} {r['Pc_chan']:>12.3e} "
              f"{ana_s:>12} {de:>7}")

    out_path = os.path.join(os.path.dirname(__file__), "..",
                             "data", "validation",
                             "sbm_validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sbm_rows": rows,
                   "pc_rows":  pc_rows,
                   "generated_at": dt.datetime.utcnow().isoformat() + "Z"},
                   f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON 写入 {out_path}")


if __name__ == "__main__":
    main()
