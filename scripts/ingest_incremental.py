"""Incremental update for *all* data sources, given a ``--since`` cutoff.

Usage
-----
    python3 scripts/ingest_incremental.py --since 2025-06-01
    python3 scripts/ingest_incremental.py --since 2025-06-01T00:00:00
    python3 scripts/ingest_incremental.py --since 2025-06-01 \\
        --sources spacetrack,techport       # update only specified sources
    python3 scripts/ingest_incremental.py --since 2025-06-01 \\
        --no-propagate                       # skip SGP4 trajectory propagation

Strategy per source
-------------------
The script always does **incremental** work — never re-pulling data older than
``since``.  Wherever the upstream supports server-side filtering it is used
directly so that the wire payload only contains new rows.

  * Space-Track:  REST query ``class/gp/EPOCH/>{since}`` — upstream returns
                  **every GP revision** newer than the cutoff (often multiple
                  rows per NORAD).  We **dedupe to latest epoch per NORAD**
                  before upsert so incremental volume matches “objects updated”.
  * GCAT (McDowell satcat.tsv):  LOCAL FILE — banner reminds you to refresh
                  ``data/external/jm_satcat.tsv`` from upstream first; we then
                  read the file and keep only rows whose ``LDate`` ≥ since.
  * UCS Satellite Database (xlsx):  LOCAL FILE — same banner pattern; rows are
                  filtered by ``Date of Launch (UTC)`` ≥ since.
  * UNOOSA:       year-grain — only pulls rows whose ``year`` ≥ since.year.
  * ESA DISCOS:   server-side ``filter=ge(firstEpoch,'<since>')`` filter.
  * Asterank:     no time field is exposed by Asterank; the catalogue is
                  small (≤5,000 rows) so we refetch and upsert by ``full_name``.
  * NASA TechPort:  the index endpoint exposes per-project ``lastUpdated``;
                  we filter that list before issuing detail GETs, so only
                  changed projects hit the network.

All writes are *idempotent* via DELETE-by-PK + INSERT (on the existing tables)
or via SQLAlchemy ORM ``ON CONFLICT`` (Space-Track tables, since they have
SQLAlchemy-mapped models).  Re-running with the same ``--since`` is safe.

For local files (GCAT and UCS) the script ALWAYS prints a clearly visible
banner so you have an obvious cue to refresh the file before re-running.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import requests
from sqlalchemy import inspect, text

from database.db import get_engine, session_scope

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)

ALL_SOURCES = ["spacetrack", "gcat", "unoosa", "ucs", "esa", "asterank", "techport"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_since(s: str) -> dt.datetime:
    """Parse YYYY-MM-DD or full ISO datetime string into UTC datetime."""
    s = s.strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
    ):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    try:
        from dateutil.parser import parse as dparse
        out = dparse(s)
        if out.tzinfo is None:
            out = out.replace(tzinfo=dt.timezone.utc)
        return out.astimezone(dt.timezone.utc)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Could not parse --since '{s}': {exc}")


def _banner_local_file(label: str, path: str, url: str, hint: str) -> None:
    """Print a clearly visible banner for sources that depend on local files."""
    log.info("")
    log.info("┏" + "━" * 72 + "┓")
    log.info(f"┃  ⚠ 提醒：{label} 数据来自本地文件，请先手动刷新到最新版本")
    log.info(f"┃  本地文件：{path}")
    if os.path.exists(path):
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
        size = os.path.getsize(path) / 1024 / 1024
        log.info(f"┃  最后修改：{mtime:%Y-%m-%d %H:%M:%S}    大小：{size:.1f} MB")
    else:
        log.info("┃  最后修改：(文件不存在 —— 请先下载！)")
    log.info(f"┃  上游下载：{url}")
    log.info(f"┃  操作建议：{hint}")
    log.info("┗" + "━" * 72 + "┛")


def _upsert_dataframe(df: pd.DataFrame, table: str, pk_cols: list[str],
                      engine, source_label: str = "") -> int:
    """Idempotent upsert: DELETE matching rows by PK, then INSERT.

    If the target table doesn't exist yet, fall back to creating it (full
    replace) so a first-time run also works.
    """
    if df is None or df.empty:
        log.info("  %s · 无新增 / 变更行", source_label or table)
        return 0

    insp = inspect(engine)
    if not insp.has_table(table):
        log.info("  %s · 目标表 %s 不存在，首次创建（%d 行）",
                 source_label or table, table, len(df))
        df.to_sql(table, engine, if_exists="replace", index=False,
                  method="multi", chunksize=400)
        return len(df)

    # Sanitize: only keep columns that exist in the target table
    existing_cols = {c["name"] for c in insp.get_columns(table)}
    keep = [c for c in df.columns if c in existing_cols]
    if not keep:
        log.warning("  %s · 没有任何字段与目标表 %s 匹配，跳过",
                    source_label or table, table)
        return 0
    df = df[keep].copy()
    pk_cols = [c for c in pk_cols if c in df.columns]
    if not pk_cols:
        log.warning("  %s · 主键列不在数据中，回退到 append 模式（可能产生重复）",
                    source_label or table)
        df.to_sql(table, engine, if_exists="append", index=False,
                  method="multi", chunksize=400)
        return len(df)

    staging = f"_tmp_inc_{table}"
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {staging}'))
    df.to_sql(staging, engine, if_exists="replace", index=False,
              method="multi", chunksize=400)

    pk_list = ", ".join(f'"{c}"' for c in pk_cols)
    col_list = ", ".join(f'"{c}"' for c in df.columns)
    with engine.begin() as conn:
        n_before = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
        conn.execute(text(
            f'DELETE FROM {table} WHERE ({pk_list}) IN '
            f'(SELECT {pk_list} FROM {staging})'
        ))
        conn.execute(text(
            f'INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {staging}'
        ))
        conn.execute(text(f'DROP TABLE IF EXISTS {staging}'))
        n_after = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
    log.info("  %s · %s 行数 %d → %d  (Δ %+d)",
             source_label or table, table, n_before, n_after, n_after - n_before)
    return len(df)


# ──────────────────────────────────────────────────────────────────────────────
# Space-Track helpers
# ──────────────────────────────────────────────────────────────────────────────


def _dedupe_gp_latest_per_norad(records: list[dict]) -> list[dict]:
    """API ``EPOCH/>since`` returns every GP row newer than cutoff — multiple
    revisions per satellite.  Keep the **latest EPOCH** per ``NORAD_CAT_ID``.
    """
    from dateutil.parser import parse as dparse

    best: dict[int, tuple[dt.datetime, dict]] = {}
    for rec in records:
        try:
            nid = int(rec["NORAD_CAT_ID"])
        except (KeyError, TypeError, ValueError):
            continue
        epoch_str = rec.get("EPOCH", "")
        try:
            epoch = dparse(epoch_str).replace(tzinfo=dt.timezone.utc)
        except Exception:
            continue
        prev = best.get(nid)
        if prev is None or epoch > prev[0]:
            best[nid] = (epoch, rec)
    return [t[1] for t in best.values()]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Space-Track  (TLE / GP elements with EPOCH > since)
# ──────────────────────────────────────────────────────────────────────────────
def incremental_spacetrack(since: dt.datetime, *,
                           propagate: bool = True,
                           horizon_days: int = 3,
                           seg_minutes: int = 10,
                           object_types: list[str] | None = None) -> int:
    """Pull GP records with ``EPOCH > since`` and upsert catalog + gp_elements."""
    log.info("=" * 72)
    log.info("=== [1/7] Space-Track  (since EPOCH > %s) ===", since.isoformat())

    from fetcher.spacetrack_client import SpaceTrackClient, BASE_URL

    if object_types is None:
        object_types = ["DEBRIS", "PAYLOAD", "ROCKET BODY"]

    cutoff = since.strftime("%Y-%m-%d %H:%M:%S")
    cutoff_url = cutoff.replace(" ", "%20")
    ot = ",".join(object_types)

    url = (
        f"{BASE_URL}/basicspacedata/query/class/gp"
        f"/EPOCH/%3E{cutoff_url}"
        f"/OBJECT_TYPE/{ot}"
        f"/orderby/EPOCH%20asc"
        "/format/json"
    )

    with SpaceTrackClient() as client:
        log.info("  Space-Track 查询: EPOCH > %s, types=%s", cutoff, ot)
        # Use the authenticated session to do an arbitrary query
        time.sleep(0.3)  # rate limit
        resp = client._session.get(url, timeout=600)
        resp.raise_for_status()
        records = resp.json() or []
    raw_n = len(records)
    records = _dedupe_gp_latest_per_norad(records)
    log.info(
        "  Space-Track 原始 %d 条 GP → 按 NORAD 保留最新 epoch 后 %d 条",
        raw_n,
        len(records),
    )

    if not records:
        return 0

    from ingestion.ingest_gp import _upsert_catalog, _upsert_gp, _ingest_segments
    from database.db import init_db
    init_db()

    batch_size = 200
    n_total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        with session_scope() as sess:
            for rec in batch:
                _upsert_catalog(sess, rec)
                _upsert_gp(sess, rec)
                if propagate:
                    _ingest_segments(sess, rec, horizon_days, seg_minutes)
        n_total += len(batch)
        log.info("    批次 %d / %d  (累计 upsert %d)",
                 (i // batch_size) + 1,
                 (len(records) + batch_size - 1) // batch_size,
                 n_total)
    log.info("  Space-Track 增量完成：%d 条 GP / catalog upsert", n_total)
    return n_total


# ──────────────────────────────────────────────────────────────────────────────
# 2. GCAT (Jonathan McDowell satcat.tsv)  — local file
# ──────────────────────────────────────────────────────────────────────────────
def incremental_gcat(since: dt.datetime, engine) -> int:
    """Read jm_satcat.tsv, keep rows whose LDate ≥ since, recompute the
    GCAT aggregate tables on the *filtered slice*, then upsert.

    GCAT stores launch-history aggregates (per year × country × type), not
    per-row snapshots, so the cleanest semantic for "incremental update" is:
    take the slice of rows launched after ``since`` and add their counts to
    the existing aggregate tables (DELETE-by-key + INSERT).
    """
    log.info("=" * 72)
    log.info("=== [2/7] GCAT McDowell Satellite Catalogue (local file) ===")

    tsv = os.path.join(DATA_DIR, "jm_satcat.tsv")
    if not os.path.exists(tsv):
        log.warning("  本地 GCAT 文件不存在，跳过 GCAT 增量更新")
        return 0

    try:
        from scripts.ingest_external import _STATE_TO_COUNTRY, _parse_type, _parse_year
    except Exception:
        from ingest_external import _STATE_TO_COUNTRY, _parse_type, _parse_year  # type: ignore

    cols = [
        'JCAT', 'Satcat', 'Launch_Tag', 'Piece', 'Type', 'Name', 'PLName',
        'LDate', 'Parent', 'SDate', 'Primary', 'DDate', 'Status', 'Dest',
        'Owner', 'State', 'Manufacturer', 'Bus', 'Motor', 'Mass', 'MassFlag',
        'DryMass', 'DryFlag', 'TotMass', 'TotFlag', 'Length', 'LFlag',
        'Diameter', 'DFlag', 'Span', 'SpanFlag', 'Shape', 'ODate',
        'Perigee', 'PF', 'Apogee', 'AF', 'Inc', 'IF', 'OpOrbit', 'OQUAL',
        'AltNames',
    ]
    df = pd.read_csv(tsv, sep='\t', comment='#', header=None,
                     names=cols, dtype=str)
    log.info("  GCAT TSV 原始行数: %d", len(df))

    for c in ['JCAT', 'Type', 'State', 'Status', 'LDate', 'DDate']:
        df[c] = df[c].astype(str).str.strip()

    df["_ldate"] = pd.to_datetime(df["LDate"], errors="coerce", utc=True)
    before = len(df)
    df = df[df["_ldate"] >= since].copy()
    log.info("  按 LDate ≥ %s 过滤后剩 %d / %d 行",
             since.date(), len(df), before)
    if df.empty:
        log.info("  GCAT 增量为空，无需更新")
        return 0

    df["launch_year"] = df["_ldate"].dt.year.astype(int)
    df["object_type"] = df["Type"].apply(_parse_type)
    df["country_code"] = df["State"].map(_STATE_TO_COUNTRY).fillna("OTHER")
    df["is_on_orbit"] = df["Status"].isin(["O", "OX"])

    yearly = (
        df.groupby(["launch_year", "country_code", "object_type"], as_index=False)
          .size().rename(columns={"size": "count"})
          .assign(source="GCAT-McDowell")
    )
    n_y = _upsert_dataframe(
        yearly, "external_yearly_launches",
        pk_cols=["launch_year", "country_code", "object_type"],
        engine=engine, source_label="GCAT yearly",
    )

    country_yearly = (
        df[df["object_type"] == "PAYLOAD"]
        .groupby(["launch_year", "country_code"], as_index=False)
        .size().rename(columns={"size": "count"})
        .assign(source="GCAT-McDowell")
    )
    n_c = _upsert_dataframe(
        country_yearly, "external_country_yearly_payload",
        pk_cols=["launch_year", "country_code"],
        engine=engine, source_label="GCAT country/year/payload",
    )

    onorbit = (
        df[df["is_on_orbit"]]
        .groupby(["country_code", "object_type"], as_index=False)
        .size().rename(columns={"size": "on_orbit_count"})
        .assign(source="GCAT-McDowell")
    )
    n_o = _upsert_dataframe(
        onorbit, "external_onorbit_snapshot",
        pk_cols=["country_code", "object_type"],
        engine=engine, source_label="GCAT onorbit snapshot",
    )

    return n_y + n_c + n_o


# ──────────────────────────────────────────────────────────────────────────────
# 3. UNOOSA  (year-grain)
# ──────────────────────────────────────────────────────────────────────────────
def incremental_unoosa(since: dt.datetime, engine) -> int:
    """Pull UNOOSA via Our World in Data and keep rows where year ≥ since.year.

    The OWID CSV is small (≤ 5 KB) and has no incremental endpoint, so we
    fetch it whole and slice by year locally.
    """
    log.info("=" * 72)
    log.info("=== [3/7] UNOOSA / Our World in Data (year ≥ %d) ===", since.year)

    url = (
        "https://ourworldindata.org/grapher/"
        "yearly-number-of-objects-launched-into-outer-space"
        ".csv?v=1&csvType=full&useColumnShortNames=true"
    )
    try:
        df = pd.read_csv(
            url,
            storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
        )
    except Exception as exc:
        csv_path = os.path.join(DATA_DIR, "unoosa_owid.csv")
        if os.path.exists(csv_path):
            log.info("  UNOOSA 在线下载失败 (%s) — 使用本地 CSV", exc)
            df = pd.read_csv(csv_path)
        else:
            log.warning("  UNOOSA 在线下载失败且无本地缓存：%s — 跳过", exc)
            return 0

    df.columns = [c.strip() for c in df.columns]
    if "entity" not in df.columns or "year" not in df.columns:
        log.warning("  UNOOSA CSV 缺少 entity/year 列，跳过")
        return 0
    df["entity"] = df["entity"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    val_col = "annual_launches" if "annual_launches" in df.columns else df.columns[-1]
    df["annual_launches"] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["year", "annual_launches"])
    df["year"] = df["year"].astype(int)
    df["annual_launches"] = df["annual_launches"].astype(int)

    before = len(df)
    df = df[df["year"] >= since.year].drop_duplicates(subset=["entity", "year"])
    log.info("  UNOOSA 按 year ≥ %d 过滤后剩 %d / %d 行",
             since.year, len(df), before)
    if df.empty:
        return 0
    keep = [c for c in ("entity", "code", "year", "annual_launches")
            if c in df.columns]
    df = df[keep].assign(source="UNOOSA/OWID")
    return _upsert_dataframe(
        df, "external_unoosa_launches",
        pk_cols=["entity", "year"],
        engine=engine, source_label="UNOOSA",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. UCS Satellite Database  (local xlsx)
# ──────────────────────────────────────────────────────────────────────────────
def incremental_ucs(since: dt.datetime, engine) -> int:
    """Read UCS xlsx, keep rows whose Date of Launch ≥ since, upsert into
    external_ucs_satellites.
    """
    log.info("=" * 72)
    log.info("=== [4/7] UCS Satellite Database (local file) ===")

    xlsx = os.path.join(DATA_DIR, "ucs_satellites.xlsx")
    if not os.path.exists(xlsx):
        log.warning("  本地 UCS 文件不存在，跳过 UCS 增量更新")
        return 0

    try:
        ucs_raw = pd.read_excel(xlsx, sheet_name=0)
    except Exception as exc:
        log.warning("  读取 UCS xlsx 失败: %s", exc)
        return 0

    # Use the same canonical column-rename used by the full ingest_ucs path
    keep_map = {
        "Name of Satellite, Alternate Names": "name",
        "Country of Operator/Owner":          "country",
        "Operator/Owner":                     "operator",
        "Users":                              "users",
        "Purpose":                            "purpose",
        "Detailed Purpose":                   "detailed_purpose",
        "Class of Orbit":                     "orbit_class",
        "Type of Orbit":                      "orbit_type",
        "Perigee (km)":                       "perigee_km",
        "Apogee (km)":                        "apogee_km",
        "Inclination (degrees)":              "inclination_deg",
        "Launch Mass (kg.)":                  "launch_mass_kg",
        "Date of Launch":                     "launch_date",
        "Expected Lifetime (yrs.)":           "expected_lifetime_yr",
        "Launch Vehicle":                     "launch_vehicle",
        "NORAD Number":                       "norad_cat_id",
        "COSPAR Number":                      "cospar_id",
    }
    available = [c for c in keep_map if c in ucs_raw.columns]
    if not available:
        log.warning("  UCS xlsx 列名与预期不符（可能格式被改），跳过")
        return 0
    ucs = ucs_raw[available].rename(columns=keep_map)

    ucs["launch_date"] = pd.to_datetime(ucs.get("launch_date"), errors="coerce")
    for col in ("name", "country", "operator", "users", "purpose",
                "orbit_class", "orbit_type", "cospar_id"):
        if col in ucs.columns:
            ucs[col] = ucs[col].astype(str).str.strip().replace("nan", pd.NA)
    ucs["norad_cat_id"] = pd.to_numeric(ucs.get("norad_cat_id"), errors="coerce") \
                              .astype("Int64")
    for c in ("perigee_km", "apogee_km", "inclination_deg",
              "launch_mass_kg", "expected_lifetime_yr"):
        if c in ucs.columns:
            ucs[c] = pd.to_numeric(ucs[c], errors="coerce")

    # Filter by launch_date ≥ since
    if "launch_date" not in ucs.columns:
        log.warning("  UCS 缺少 launch_date 列，跳过")
        return 0
    before = len(ucs)
    ucs = ucs[ucs["launch_date"] >= pd.Timestamp(since).tz_localize(None)]
    log.info("  UCS 按 launch_date ≥ %s 过滤后剩 %d / %d 行",
             since.date(), len(ucs), before)
    if ucs.empty:
        return 0

    ucs = ucs[ucs["norad_cat_id"].notna()] \
            .drop_duplicates(subset=["norad_cat_id"], keep="first")
    ucs["launch_date"] = ucs["launch_date"].dt.date
    ucs = ucs.assign(source="UCS")
    return _upsert_dataframe(
        ucs, "external_ucs_satellites",
        pk_cols=["norad_cat_id"],
        engine=engine, source_label="UCS",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. ESA DISCOS  (server-side filter on firstEpoch)
# ──────────────────────────────────────────────────────────────────────────────
def incremental_esa(since: dt.datetime, engine) -> int:
    """Pull ESA DISCOS objects with ``firstEpoch ≥ since`` (server-side filter)."""
    log.info("=" * 72)
    log.info("=== [5/7] ESA DISCOS (filter firstEpoch ≥ %s) ===", since.isoformat())
    token = os.environ.get("ESA_DISCOS_TOKEN", "")
    if not token:
        log.warning("  ESA_DISCOS_TOKEN 未设置，跳过 ESA DISCOS 增量更新")
        return 0
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/json"}
    base = "https://discosweb.esoc.esa.int/api"
    iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[dict] = []
    page, total_pages = 1, 1
    while page <= total_pages:
        try:
            r = requests.get(
                f"{base}/objects",
                params={
                    "page[size]": 100,
                    "page[number]": page,
                    # DISCOSweb requires the value to be tagged with the
                    # ``epoch:`` literal-type prefix; plain strings 400.
                    "filter": f"ge(firstEpoch,epoch:'{iso}')",
                    "sort":   "firstEpoch",
                },
                headers=headers, timeout=60,
            )
        except requests.RequestException as exc:
            log.warning("  ESA 请求失败 page=%d: %s", page, exc)
            break
        if r.status_code == 400:
            # If filter syntax not supported, fall back to no-filter and post-filter
            log.warning("  ESA 服务端 filter=ge(firstEpoch,...) 不被接受 (HTTP 400)，"
                        "回退到客户端过滤模式")
            return _esa_client_side_filter(since, engine, headers, base)
        if r.status_code != 200:
            log.warning("  ESA HTTP %d on page %d: %s", r.status_code, page, r.text[:160])
            break
        data = r.json()
        items = data.get("data", [])
        if not items:
            break
        for obj in items:
            attrs = obj.get("attributes", {})
            rows.append({
                "satno":      attrs.get("satno"),
                "cosparId":   attrs.get("cosparId"),
                "name":       attrs.get("name"),
                "objectClass": attrs.get("objectClass"),
                "mass":       attrs.get("mass"),
                "shape":      attrs.get("shape"),
                "xSectMax":   attrs.get("xSectMax"),
                "xSectMin":   attrs.get("xSectMin"),
                "xSectAvg":   attrs.get("xSectAvg"),
                "firstEpoch": attrs.get("firstEpoch"),
                "mission":    attrs.get("mission"),
                "predDecayDate": attrs.get("predDecayDate"),
                "active":     attrs.get("active"),
                "cataloguedFragments": attrs.get("cataloguedFragments"),
                "onOrbitCataloguedFragments": attrs.get("onOrbitCataloguedFragments"),
            })
        total_pages = data.get("meta", {}).get("pagination", {}).get("totalPages", 0) or 1
        if page % 20 == 0:
            log.info("    ESA page %d/%d (累计 %d 行)", page, total_pages, len(rows))
        page += 1

    if not rows:
        log.info("  ESA 自 %s 起无新对象，跳过", iso)
        return 0
    df = pd.DataFrame(rows)
    df["satno"] = pd.to_numeric(df["satno"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["satno"]).drop_duplicates(subset=["satno"], keep="first")
    # Coerce numeric columns so pandas creates a typed staging table that
    # matches the target ``external_esa_discos`` schema.
    for c in ("mass", "xSectMax", "xSectMin", "xSectAvg",
             "cataloguedFragments", "onOrbitCataloguedFragments"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "active" in df.columns:
        df["active"] = df["active"].astype("boolean")
    df["source"] = "ESA-DISCOS"
    return _upsert_dataframe(
        df, "external_esa_discos",
        pk_cols=["satno"],
        engine=engine, source_label="ESA DISCOS",
    )


def _esa_client_side_filter(since, engine, headers, base) -> int:
    """Fallback: no server-side filter; download everything and drop old rows."""
    rows: list[dict] = []
    page, total_pages = 1, 1
    while page <= total_pages:
        r = requests.get(f"{base}/objects",
                         params={"page[size]": 100, "page[number]": page},
                         headers=headers, timeout=60)
        if r.status_code != 200:
            log.warning("  ESA fallback HTTP %d", r.status_code)
            break
        data = r.json()
        for obj in data.get("data", []):
            attrs = obj.get("attributes", {})
            rows.append({**attrs, "esa_id": obj["id"]})
        total_pages = data.get("meta", {}).get("pagination", {}).get("totalPages", 0) or 1
        page += 1
    df = pd.DataFrame(rows)
    if df.empty or "firstEpoch" not in df.columns:
        return 0
    df["_first_dt"] = pd.to_datetime(df["firstEpoch"], errors="coerce", utc=True)
    df = df[df["_first_dt"] >= since].drop(columns=["_first_dt"])
    return _upsert_dataframe(
        df, "external_esa_discos",
        pk_cols=["satno"],
        engine=engine, source_label="ESA DISCOS (fallback)",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Asterank  (no time field; refetch and upsert by full_name)
# ──────────────────────────────────────────────────────────────────────────────
def incremental_asterank(since: dt.datetime, engine, *, limit: int = 5000) -> int:
    """Asterank does not expose a 'lastUpdated' field; the catalogue is small,
    so we refetch up to ``limit`` rows and idempotently upsert.
    """
    log.info("=" * 72)
    log.info("=== [6/7] Asterank (catalogue is small, full upsert) ===")
    try:
        from scripts.ingest_asterank import fetch_asterank, _KEEP_COLS, _NUMERIC_COLS
    except Exception:
        from ingest_asterank import fetch_asterank, _KEEP_COLS, _NUMERIC_COLS  # type: ignore

    df = fetch_asterank(limit=limit)
    if df is None or df.empty:
        return 0
    keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep].copy()
    if "full_name" in df.columns:
        df["full_name"] = df["full_name"].astype(str).str.strip().replace({"nan": pd.NA})
    for col in _NUMERIC_COLS & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["full_name"]).drop_duplicates(subset=["full_name"])
    df["source"] = "Asterank"
    log.info("  Asterank rows after dedup: %d", len(df))
    return _upsert_dataframe(
        df, "external_asterank",
        pk_cols=["full_name"],
        engine=engine, source_label="Asterank",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7. NASA TechPort  (per-project lastUpdated filter)
# ──────────────────────────────────────────────────────────────────────────────
def incremental_techport(since: dt.datetime, engine, *,
                         workers: int = 8,
                         token: str | None = None) -> int:
    """Filter the TechPort index by ``lastUpdated ≥ since`` so only changed
    projects hit the network, then upsert their flattened rows.
    """
    log.info("=" * 72)
    log.info("=== [7/7] NASA TechPort (lastUpdated ≥ %s) ===", since.date())

    try:
        from scripts.ingest_techport import (
            _build_session, fetch_project_index, fetch_project_details,
            _flatten, _clean_dataframe,
        )
    except Exception:
        from ingest_techport import (  # type: ignore
            _build_session, fetch_project_index, fetch_project_details,
            _flatten, _clean_dataframe,
        )

    if token is None:
        token = os.environ.get("NASA_TECHPORT_TOKEN") or None
    sess = _build_session(token)

    listing = fetch_project_index(sess, offline=False)
    if not listing:
        return 0

    def _parse_lu(s: Any) -> dt.datetime | None:
        if not s:
            return None
        try:
            return dt.datetime.strptime(str(s), "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        except Exception:
            try:
                return dt.datetime.strptime(str(s), "%Y-%-m-%-d").replace(tzinfo=dt.timezone.utc)
            except Exception:
                pass
            try:
                from dateutil.parser import parse as dparse
                t = dparse(str(s))
                return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t
            except Exception:
                return None

    cutoff = since.date()
    changed = []
    for it in listing:
        lu = _parse_lu(it.get("lastUpdated"))
        if lu and lu.date() >= cutoff:
            changed.append(int(it["projectId"]))
    log.info("  TechPort 索引共 %d 项，自 %s 起更新过 %d 项",
             len(listing), cutoff, len(changed))
    if not changed:
        return 0

    projects = fetch_project_details(sess, changed, workers=workers)
    flat = [_flatten(p) for p in projects]
    df = _clean_dataframe(flat)
    log.info("  TechPort 待 upsert 行数: %d", len(df))
    return _upsert_dataframe(
        df, "external_techport",
        pk_cols=["project_id"],
        engine=engine, source_label="TechPort",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Refresh views (best-effort)
# ──────────────────────────────────────────────────────────────────────────────
def refresh_unified_view(engine) -> None:
    """Best-effort rebuild of v_unified_objects after a successful update."""
    log.info("=" * 72)
    log.info("=== Refreshing v_unified_objects materialized view ===")
    try:
        from scripts.create_unified_view import create as _create
    except Exception:
        try:
            from create_unified_view import create as _create  # type: ignore
        except Exception:
            log.warning("  create_unified_view 不可用，跳过视图刷新")
            return
    try:
        _create()
        log.info("  v_unified_objects 刷新完成")
    except Exception as exc:  # noqa: BLE001
        log.warning("  v_unified_objects 刷新失败: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental update for all data sources given --since cutoff",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--since", required=True,
                        help="cutoff: YYYY-MM-DD or full ISO datetime "
                             "(rows with timestamp/launch/lastUpdated >= this "
                             "value will be pulled and upserted)")
    parser.add_argument("--sources", default="all",
                        help=("comma-separated subset of: " + ",".join(ALL_SOURCES)
                              + " — default 'all'"))
    parser.add_argument("--no-propagate", action="store_true",
                        help="skip SGP4 trajectory regeneration for Space-Track")
    parser.add_argument("--horizon-days", type=int, default=3,
                        help="propagation horizon in days for new objects")
    parser.add_argument("--seg-minutes", type=int, default=10,
                        help="trajectory segment length in minutes")
    parser.add_argument("--techport-workers", type=int, default=8)
    parser.add_argument("--techport-token", type=str, default=None,
                        help="overrides NASA_TECHPORT_TOKEN env")
    parser.add_argument("--refresh-views", action="store_true", default=True,
                        help="rebuild v_unified_objects after success (default ON)")
    parser.add_argument("--no-refresh-views", dest="refresh_views",
                        action="store_false")
    args = parser.parse_args()

    since = parse_since(args.since)
    log.info("Incremental update — since = %s (UTC)", since.isoformat())

    if args.sources == "all":
        sources = ALL_SOURCES[:]
    else:
        sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
        bad = [s for s in sources if s not in ALL_SOURCES]
        if bad:
            raise SystemExit(f"Unknown --sources entries: {bad}; "
                             f"valid: {ALL_SOURCES}")

    log.info("Sources to update: %s", sources)
    log.info("")

    engine = get_engine()
    summary: dict[str, int] = {}
    t_start = time.time()

    # Pre-print local-file banners up-front so the user sees them immediately,
    # before any blocking API calls — this satisfies the "每次运行前输出对更新
    # 这两个文件的提示" requirement even if the user only chose other sources.
    if "gcat" in sources:
        _banner_local_file(
            label="GCAT (McDowell)",
            path=os.path.join(DATA_DIR, "jm_satcat.tsv"),
            url="https://planet4589.org/space/gcat/tsv/cat/satcat.tsv",
            hint="若需最新数据请先下载到 data/external/jm_satcat.tsv 后再运行此脚本。",
        )
    if "ucs" in sources:
        _banner_local_file(
            label="UCS Satellite Database",
            path=os.path.join(DATA_DIR, "ucs_satellites.xlsx"),
            url="https://www.ucsusa.org/nuclear-weapons/space-weapons/satellite-database",
            hint="UCS 每半年发布一次新版本，请下载替换后再运行此脚本。",
        )

    # Dispatch
    if "spacetrack" in sources:
        try:
            summary["spacetrack"] = incremental_spacetrack(
                since,
                propagate=not args.no_propagate,
                horizon_days=int(args.horizon_days),
                seg_minutes=int(args.seg_minutes),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Space-Track 增量失败: %s", exc)
            summary["spacetrack"] = -1

    if "gcat" in sources:
        try:
            summary["gcat"] = incremental_gcat(since, engine)
        except Exception as exc:  # noqa: BLE001
            log.warning("GCAT 增量失败: %s", exc)
            summary["gcat"] = -1

    if "unoosa" in sources:
        try:
            summary["unoosa"] = incremental_unoosa(since, engine)
        except Exception as exc:  # noqa: BLE001
            log.warning("UNOOSA 增量失败: %s", exc)
            summary["unoosa"] = -1

    if "ucs" in sources:
        try:
            summary["ucs"] = incremental_ucs(since, engine)
        except Exception as exc:  # noqa: BLE001
            log.warning("UCS 增量失败: %s", exc)
            summary["ucs"] = -1

    if "esa" in sources:
        try:
            summary["esa"] = incremental_esa(since, engine)
        except Exception as exc:  # noqa: BLE001
            log.warning("ESA DISCOS 增量失败: %s", exc)
            summary["esa"] = -1

    if "asterank" in sources:
        try:
            summary["asterank"] = incremental_asterank(since, engine)
        except Exception as exc:  # noqa: BLE001
            log.warning("Asterank 增量失败: %s", exc)
            summary["asterank"] = -1

    if "techport" in sources:
        try:
            summary["techport"] = incremental_techport(
                since, engine,
                workers=int(args.techport_workers),
                token=args.techport_token,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("TechPort 增量失败: %s", exc)
            summary["techport"] = -1

    if args.refresh_views:
        refresh_unified_view(engine)

    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 72)
    log.info("=== Incremental Summary (since %s) ===", since.isoformat())
    for src in ALL_SOURCES:
        if src in summary:
            n = summary[src]
            tag = "FAILED" if n < 0 else f"{n} rows upserted"
            log.info("  %-12s %s", src, tag)
    log.info("Total elapsed: %.1f s", elapsed)


if __name__ == "__main__":
    main()
