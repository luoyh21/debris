"""Ingest Asterank asteroid/NEO data into the database.

Source:
  - Primary API: http://www.asterank.com/api/asterank
    (documented on asterank.com; returns a JSON list with MongoDB-style query)
  - Fallback: data/external/asterank.json (local cached copy)

Schema created:
  external_asterank  (one row per asteroid, sorted by descending ``profit``)

Key fields kept (if present):
  full_name / name        object name
  a / e / i               semi-major axis (AU), eccentricity, inclination (deg)
  H                       absolute magnitude
  diameter                estimated diameter (km)
  GM                      gravitational parameter (m³/s²)
  spec / class            spectral / orbital class
  moid                    minimum orbit-intersection distance w.r.t. Earth
  price                   estimated resource value (USD, heuristic)
  profit                  estimated mining profitability (USD, heuristic)
  dv                      round-trip Δv from LEO (km/s)

Usage
-----
    python3 scripts/ingest_asterank.py                 # default 5000 rows
    python3 scripts/ingest_asterank.py --limit 500     # only first 500

The script performs:
  1. API fetch (with local-cache fallback if the upstream is unreachable)
  2. Data cleaning (numeric coercion, whitespace strip)
  3. Dedup by ``full_name``
  4. ``CREATE TABLE external_asterank`` (drop-and-recreate)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import requests
from sqlalchemy import text

from database.db import get_engine


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_PATH = os.path.join(DATA_DIR, "asterank.json")

_ASTERANK_URL = "http://www.asterank.com/api/asterank"

# Columns we try to persist (only those present will be kept).
_KEEP_COLS = [
    "full_name", "prov_des", "a", "e", "i", "om", "w",
    "H", "diameter", "GM", "rot_per", "albedo",
    "spec", "class", "moid", "n", "per", "price",
    "profit", "dv",
]

# Columns that should be numeric (missing → NaN).
_NUMERIC_COLS = {
    "a", "e", "i", "om", "w", "H", "diameter", "GM", "rot_per",
    "albedo", "moid", "n", "per", "price", "profit", "dv",
}


def fetch_asterank(limit: int = 5000) -> pd.DataFrame:
    """Fetch asteroid records from asterank.com.

    On any network failure we fall back to a locally cached JSON file so
    reruns stay offline-friendly.
    """
    params = {"query": "{}", "limit": int(limit) if limit else 5000}
    try:
        log.info("Requesting Asterank API (limit=%s)…", params["limit"])
        r = requests.get(_ASTERANK_URL, params=params, timeout=60,
                         headers={"User-Agent": "space_debris-ingest/1.0"})
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            # Some Asterank endpoints wrap the results; normalise to list.
            data = data.get("data") or data.get("results") or []
        if data:
            with open(CACHE_PATH, "w") as fh:
                json.dump(data, fh)
            log.info("  Cached %d rows → %s", len(data), CACHE_PATH)
        else:
            log.warning("  API returned empty payload")
    except Exception as exc:
        if os.path.exists(CACHE_PATH):
            log.warning("  API unreachable (%s) — loading local cache", exc)
            with open(CACHE_PATH) as fh:
                data = json.load(fh)
        else:
            log.error("  Asterank API failed and no local cache: %s", exc)
            return pd.DataFrame()

    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def ingest_asterank(limit: int = 5000) -> int:
    """Ingest Asterank rows into ``external_asterank``; returns row count."""
    t0 = time.time()
    log.info("=" * 60)
    log.info("=== Asterank (asteroid / NEO catalogue) ===")

    df = fetch_asterank(limit=limit)
    if df.empty:
        log.warning("  No Asterank rows to ingest — aborting")
        return 0

    raw_count = len(df)
    log.info("  Raw rows: %d | columns(sample): %s",
             raw_count, list(df.columns)[:15])

    # ── Clean: select + coerce types ─────────────────────────────────────
    keep = [c for c in _KEEP_COLS if c in df.columns]
    if "full_name" not in keep and "prov_des" in df.columns:
        df = df.rename(columns={"prov_des": "full_name"})
        keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep].copy() if keep else df

    if "full_name" in df.columns:
        df["full_name"] = (
            df["full_name"].astype(str).str.strip().replace({"nan": pd.NA})
        )
    for col in _NUMERIC_COLS & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("spec", "class"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().replace({"nan": pd.NA})
            )

    # ── Dedup by full_name ───────────────────────────────────────────────
    before = len(df)
    if "full_name" in df.columns:
        has_name = df["full_name"].notna()
        named = df[has_name].drop_duplicates(subset=["full_name"], keep="first")
        unnamed = df[~has_name]
        df = pd.concat([named, unnamed], ignore_index=True)
    dropped = before - len(df)
    if dropped:
        log.info("  Dedup by full_name: dropped %d duplicates", dropped)

    # ── Apply final limit after dedup ────────────────────────────────────
    if limit and limit > 0 and len(df) > limit:
        df = df.head(limit).reset_index(drop=True)

    # ── Summary log ──────────────────────────────────────────────────────
    n_final = len(df)
    cls_dist = (
        df["class"].value_counts().head(5).to_dict()
        if "class" in df.columns else {}
    )
    neo_count = (
        (df["moid"] < 0.05).sum()
        if "moid" in df.columns else "?"
    )
    log.info(
        "  Final rows: %d | near-Earth (moid<0.05 AU): %s | top classes: %s",
        n_final, neo_count, cls_dist,
    )

    # ── Write to DB (drop + replace) ─────────────────────────────────────
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_asterank CASCADE"))

    df.assign(source="Asterank").to_sql(
        "external_asterank", engine,
        if_exists="replace", index=False, method="multi", chunksize=500,
    )
    elapsed = time.time() - t0
    log.info("  Inserted %d Asterank rows into external_asterank (%.1fs)",
             n_final, elapsed)
    log.info("=" * 60)
    return n_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Asterank asteroid data")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max rows to ingest (default 5000; Asterank also"
                             " caps server-side)")
    args = parser.parse_args()
    ingest_asterank(limit=args.limit)
