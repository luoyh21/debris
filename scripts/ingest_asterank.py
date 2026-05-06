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


# 已知 Asterank 支持的轨道类别
_ORBIT_CLASSES = [
    "MBA", "OMB", "MCA", "TJN", "AMO", "APO", "ATE",
    "IEO", "CEN", "TNO", "IMB", "HYA", "PAA", "JFC",
    "JFc", "CTc", "HTC", "ETc", "PAR", "HYP",
]

# 每类对应的半长轴（a，AU）分段区间，用于在同一类别内多次请求拉取不同子集
# 格式: (a_min, a_max)，None 表示无界
_CLASS_A_SEGMENTS: dict[str, list[tuple]] = {
    "MBA": [(None, 2.2), (2.2, 2.5), (2.5, 2.8), (2.8, 3.1), (3.1, None)],
    "OMB": [(None, 2.8), (2.8, 3.2), (3.2, None)],
    "IMB": [(None, 2.2), (2.2, 2.5), (2.5, None)],
    "MCA": [(None, 2.5), (2.5, None)],
    "AMO": [(None, 1.5), (1.5, 2.0), (2.0, None)],
    "APO": [(None, 1.1), (1.1, 1.5), (1.5, None)],
    "ATE": [(None, 0.8), (0.8, 1.0), (1.0, None)],
    "TJN": [(None, 5.1), (5.1, 5.3), (5.3, None)],
    "CEN": [(None, 15.0), (15.0, None)],
    "TNO": [(None, 45.0), (45.0, None)],
    "PAR": [(None, 5.0), (5.0, None)],
}
_DEFAULT_SEGMENTS: list[tuple] = [(None, None)]  # 无分段的类别只取一次


def _fetch_one(query: dict, label: str, retry: int = 3) -> list:
    """单次 HTTP 请求，失败自动重试；返回记录列表。"""
    for attempt in range(retry):
        try:
            r = requests.get(
                _ASTERANK_URL,
                params={"query": json.dumps(query), "limit": 1000},
                timeout=60,
                headers={"User-Agent": "space_debris-ingest/1.0"},
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = data.get("data") or data.get("results") or []
            log.info("  %-20s → %d rows", label, len(data))
            return data or []
        except Exception as exc:
            wait = 3 * (attempt + 1)
            log.warning("  %-20s → 失败(%s)，%ds后重试", label, exc, wait)
            time.sleep(wait)
    return []


def fetch_asterank(limit: int = 0) -> pd.DataFrame:
    """按类别 × 半长轴分段逐批抓取，合并去重，覆盖尽可能多的 Asterank 数据。

    limit=0 表示不截断，保留全部去重后数据。
    """
    all_records: list = []

    # 1. 默认全量（无过滤），兜底未分类目标
    all_records.extend(_fetch_one({}, "default"))
    time.sleep(2)

    # 2. 按轨道类别 × a 分段逐批请求
    for cls in _ORBIT_CLASSES:
        segments = _CLASS_A_SEGMENTS.get(cls, _DEFAULT_SEGMENTS)
        for a_min, a_max in segments:
            query: dict = {"class": cls}
            a_filter: dict = {}
            if a_min is not None:
                a_filter["$gte"] = a_min
            if a_max is not None:
                a_filter["$lt"] = a_max
            if a_filter:
                query["a"] = a_filter
            label = f"{cls} a=[{a_min or '-∞'},{a_max or '+∞'})"
            all_records.extend(_fetch_one(query, label))
            time.sleep(1.5)

    if not all_records:
        if os.path.exists(CACHE_PATH):
            log.warning("所有请求失败，加载本地缓存 %s", CACHE_PATH)
            with open(CACHE_PATH) as fh:
                all_records = json.load(fh)
        else:
            log.error("无网络且无本地缓存，摄入中止")
            return pd.DataFrame()

    with open(CACHE_PATH, "w") as fh:
        json.dump(all_records, fh)
    log.info("  合计原始行（含重复）: %d，已缓存至 %s", len(all_records), CACHE_PATH)

    return pd.DataFrame(all_records)


def ingest_asterank(limit: int = 0) -> int:
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

    # ── Apply final limit after dedup（0 = 不截断）────────────────────────
    if limit and limit > 0 and len(df) > limit:
        df = df.head(limit).reset_index(drop=True)
        log.info("  截断至 limit=%d 行", limit)

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
    parser.add_argument("--limit", type=int, default=0,
                        help="最终写入行数上限，0=不截断全量写入（默认）")
    args = parser.parse_args()
    ingest_asterank(limit=args.limit)
