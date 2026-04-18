"""Ingest external data sources into the database.

Supports:
  - Jonathan McDowell GCAT (jm_satcat.tsv)
  - UNOOSA / Our World in Data (API fetch)
  - UCS Satellite Database (xlsx)
  - ESA DISCOS (API fetch)

All sources perform:
  1. Data cleaning (strip whitespace, normalise types, drop invalid rows)
  2. Within-source deduplication (by primary key: NORAD/JCAT/entity+year/satno)
  3. Cross-source overlap report (NORAD IDs shared between sources)

Usage:
    python3 scripts/ingest_external.py            # all sources
    python3 scripts/ingest_external.py --gcat      # GCAT only
    python3 scripts/ingest_external.py --unoosa    # UNOOSA only
    python3 scripts/ingest_external.py --ucs       # UCS only
    python3 scripts/ingest_external.py --esa       # ESA DISCOS only
"""
import os
import sys
import re
import logging
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import requests
from sqlalchemy import text
from database.db import session_scope, get_engine

engine = get_engine()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)

_STATS: dict[str, dict] = {}  # source → {rows_raw, rows_deduped, rows_inserted}

# ── State mapping (McDowell → standard region) ───────────────────────────────
_STATE_TO_COUNTRY = {
    "US": "US", "SU": "CIS", "RU": "CIS", "CN": "PRC",
    "J": "JPN", "F": "FR", "UK": "UK", "IN": "IND",
    "I-ESA": "ESA", "NZ": "NZ", "IL": "ISR", "KR": "KOR",
    "D": "D", "I": "I", "BR": "BR", "AU": "AU",
    "NL": "NL", "IR": "IR", "KP": "KP",
}

# ── Type mapping (McDowell Type → Space-Track object_type) ────────────────────
_TYPE_MAP = {
    "P": "PAYLOAD",   # Payload
    "R": "ROCKET BODY",
    "D": "DEBRIS",
    "C": "DEBRIS",    # Catalogue debris
}


def _parse_type(raw: str) -> str:
    """Map McDowell type code to standard object_type."""
    if not isinstance(raw, str):
        return "UNKNOWN"
    code = raw.strip()[:1]
    return _TYPE_MAP.get(code, "UNKNOWN")


def _parse_year(date_str: str) -> int | None:
    if not isinstance(date_str, str):
        return None
    m = re.match(r'\s*(\d{4})', date_str.strip())
    return int(m.group(1)) if m else None


def ingest_satcat():
    """Parse jm_satcat.tsv → create external_objects table."""
    t0 = time.time()
    path = os.path.join(DATA_DIR, "jm_satcat.tsv")
    if not os.path.exists(path):
        log.warning("jm_satcat.tsv not found at %s — skipping GCAT", path)
        return

    log.info("Reading %s …", path)
    cols = [
        'JCAT', 'Satcat', 'Launch_Tag', 'Piece', 'Type', 'Name', 'PLName',
        'LDate', 'Parent', 'SDate', 'Primary', 'DDate', 'Status', 'Dest',
        'Owner', 'State', 'Manufacturer', 'Bus', 'Motor', 'Mass', 'MassFlag',
        'DryMass', 'DryFlag', 'TotMass', 'TotFlag', 'Length', 'LFlag',
        'Diameter', 'DFlag', 'Span', 'SpanFlag', 'Shape', 'ODate',
        'Perigee', 'PF', 'Apogee', 'AF', 'Inc', 'IF', 'OpOrbit', 'OQUAL',
        'AltNames',
    ]
    df = pd.read_csv(path, sep='\t', comment='#', header=None,
                     names=cols, dtype=str)
    raw_count = len(df)
    log.info("Raw satcat rows: %d", raw_count)

    # ── Clean: strip whitespace on key columns ─────────────────────────────
    for c in ['JCAT', 'Satcat', 'Type', 'State', 'Status', 'LDate', 'DDate']:
        df[c] = df[c].str.strip()

    # Derive fields
    df["launch_year"] = df["LDate"].apply(_parse_year)
    df["object_type"] = df["Type"].apply(_parse_type)
    df["country_code"] = df["State"].map(_STATE_TO_COUNTRY).fillna("OTHER")
    df["is_on_orbit"] = df["Status"].isin(["O", "OX"])
    df["norad_cat_id"] = pd.to_numeric(df["Satcat"], errors="coerce")

    # Keep only rows with valid launch year
    df = df[df["launch_year"].notna() & (df["launch_year"] >= 1957)]
    df["launch_year"] = df["launch_year"].astype(int)

    # ── Dedup: by JCAT (McDowell unique ID) ────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["JCAT"], keep="first")
    dedup_dropped = before_dedup - len(df)
    if dedup_dropped:
        log.info("  Dedup by JCAT: dropped %d duplicates", dedup_dropped)
    log.info("Valid rows after cleaning + dedup: %d", len(df))

    # Build summary tables
    # 1. Yearly by country by type
    yearly = (
        df.groupby(["launch_year", "country_code", "object_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    # 2. On-orbit count by country by type (current snapshot)
    onorbit = (
        df[df["is_on_orbit"]]
        .groupby(["country_code", "object_type"], as_index=False)
        .size()
        .rename(columns={"size": "on_orbit_count"})
    )

    # Write to DB
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_yearly_launches CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS external_onorbit_snapshot CASCADE"))

        conn.execute(text("""
            CREATE TABLE external_yearly_launches (
                id SERIAL PRIMARY KEY,
                launch_year INT NOT NULL,
                country_code TEXT NOT NULL,
                object_type TEXT NOT NULL,
                count INT NOT NULL,
                source TEXT DEFAULT 'GCAT-McDowell'
            )
        """))
        conn.execute(text("""
            CREATE TABLE external_onorbit_snapshot (
                id SERIAL PRIMARY KEY,
                country_code TEXT NOT NULL,
                object_type TEXT NOT NULL,
                on_orbit_count INT NOT NULL,
                source TEXT DEFAULT 'GCAT-McDowell'
            )
        """))

    # Insert via pandas
    yearly.assign(source="GCAT-McDowell").to_sql(
        "external_yearly_launches", engine,
        if_exists="append", index=False, method="multi", chunksize=500,
    )
    onorbit.assign(source="GCAT-McDowell").to_sql(
        "external_onorbit_snapshot", engine,
        if_exists="append", index=False, method="multi", chunksize=500,
    )

    log.info("Inserted %d yearly rows, %d onorbit rows", len(yearly), len(onorbit))

    # Also create a cumulative on-orbit-over-time table
    # by computing cumulative launches - cumulative decays per year
    df["decay_year"] = df["DDate"].apply(_parse_year)
    launch_agg = df.groupby(["launch_year", "object_type"]).size().reset_index(name="launched")
    decay_agg = (
        df[df["decay_year"].notna()]
        .assign(decay_year=lambda x: x["decay_year"].astype(int))
        .groupby(["decay_year", "object_type"])
        .size()
        .reset_index(name="decayed")
        .rename(columns={"decay_year": "launch_year"})
    )

    years = range(1957, pd.Timestamp.now().year + 1)
    types = ["PAYLOAD", "DEBRIS", "ROCKET BODY"]
    rows = []
    for ot in types:
        cum_launch = 0
        cum_decay = 0
        for yr in years:
            l_val = launch_agg[(launch_agg["launch_year"] == yr) &
                               (launch_agg["object_type"] == ot)]
            d_val = decay_agg[(decay_agg["launch_year"] == yr) &
                               (decay_agg["object_type"] == ot)]
            cum_launch += int(l_val["launched"].sum()) if not l_val.empty else 0
            cum_decay += int(d_val["decayed"].sum()) if not d_val.empty else 0
            rows.append({
                "year": yr,
                "object_type": ot,
                "cumulative_launched": cum_launch,
                "cumulative_decayed": cum_decay,
                "on_orbit": cum_launch - cum_decay,
            })

    cum_df = pd.DataFrame(rows)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_cumulative_onorbit CASCADE"))
        conn.execute(text("""
            CREATE TABLE external_cumulative_onorbit (
                id SERIAL PRIMARY KEY,
                year INT NOT NULL,
                object_type TEXT NOT NULL,
                cumulative_launched INT NOT NULL,
                cumulative_decayed INT NOT NULL,
                on_orbit INT NOT NULL,
                source TEXT DEFAULT 'GCAT-McDowell'
            )
        """))
    cum_df.assign(source="GCAT-McDowell").to_sql(
        "external_cumulative_onorbit", engine,
        if_exists="append", index=False, method="multi", chunksize=500,
    )
    log.info("Inserted %d cumulative rows", len(cum_df))

    # Country yearly for launch trend chart
    country_yearly = (
        df[df["object_type"] == "PAYLOAD"]
        .groupby(["launch_year", "country_code"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_country_yearly_payload CASCADE"))
        conn.execute(text("""
            CREATE TABLE external_country_yearly_payload (
                id SERIAL PRIMARY KEY,
                launch_year INT NOT NULL,
                country_code TEXT NOT NULL,
                count INT NOT NULL,
                source TEXT DEFAULT 'GCAT-McDowell'
            )
        """))
    country_yearly.assign(source="GCAT-McDowell").to_sql(
        "external_country_yearly_payload", engine,
        if_exists="append", index=False, method="multi", chunksize=500,
    )
    log.info("Inserted %d country yearly payload rows", len(country_yearly))

    # Print summary stats
    total = len(df)
    on_orbit = df["is_on_orbit"].sum()
    elapsed = time.time() - t0
    log.info(
        "=== GCAT Summary (%.1fs) ===\n"
        "  Raw rows:         %d\n"
        "  After dedup:      %d\n"
        "  Currently on-orbit: %d\n"
        "  Year range:       %d – %d\n"
        "  Countries:        %d unique\n"
        "  Tables written:   external_yearly_launches (%d), "
        "external_onorbit_snapshot, external_cumulative_onorbit (%d), "
        "external_country_yearly_payload (%d)",
        elapsed, raw_count, total, on_orbit,
        df["launch_year"].min(), df["launch_year"].max(),
        df["country_code"].nunique(),
        len(yearly), len(cum_df), len(country_yearly),
    )
    _STATS["GCAT"] = {"rows_raw": raw_count, "rows_deduped": total,
                       "tables": 4, "elapsed": elapsed}


def ingest_unoosa():
    """Fetch UNOOSA annual launch data from Our World in Data API."""
    t0 = time.time()
    log.info("=== UNOOSA / Our World in Data ===")
    url = (
        "https://ourworldindata.org/grapher/"
        "yearly-number-of-objects-launched-into-outer-space"
        ".csv?v=1&csvType=full&useColumnShortNames=true"
    )
    try:
        df = pd.read_csv(url, storage_options={"User-Agent": "Our World In Data data fetch/1.0"})
        # Cache locally for future offline use
        df.to_csv(os.path.join(DATA_DIR, "unoosa_owid.csv"), index=False)
        log.info("  Fetched from API and cached locally")
    except Exception as e:
        csv_path = os.path.join(DATA_DIR, "unoosa_owid.csv")
        if os.path.exists(csv_path):
            log.info("  API fetch failed (%s), falling back to local CSV", e)
            df = pd.read_csv(csv_path)
        else:
            log.error("  UNOOSA fetch failed and no local CSV: %s", e)
            return

    raw_count = len(df)

    # ── Clean ──────────────────────────────────────────────────────────────
    df["entity"] = df["entity"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["annual_launches"] = pd.to_numeric(df.get("annual_launches", df.columns[-1]),
                                           errors="coerce")
    df = df.dropna(subset=["year", "annual_launches"])
    df["year"] = df["year"].astype(int)
    df["annual_launches"] = df["annual_launches"].astype(int)

    # ── Dedup: entity + year should be unique ──────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["entity", "year"], keep="first")
    dropped = before - len(df)
    if dropped:
        log.info("  Dedup by (entity, year): dropped %d duplicates", dropped)

    log.info("  Rows: %d (raw %d), entities: %d, years: %d–%d",
             len(df), raw_count, df["entity"].nunique(),
             df["year"].min(), df["year"].max())

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_unoosa_launches CASCADE"))
        conn.execute(text("""CREATE TABLE external_unoosa_launches (
            id SERIAL PRIMARY KEY, entity TEXT NOT NULL, code TEXT,
            year INT NOT NULL, annual_launches INT NOT NULL,
            source TEXT DEFAULT 'UNOOSA/OWID',
            UNIQUE(entity, year))"""))

    df.assign(source="UNOOSA/OWID").to_sql(
        "external_unoosa_launches", engine,
        if_exists="append", index=False, method="multi", chunksize=500,
    )
    elapsed = time.time() - t0
    log.info("  Inserted %d UNOOSA rows (%.1fs)", len(df), elapsed)
    _STATS["UNOOSA"] = {"rows_raw": raw_count, "rows_deduped": len(df),
                          "tables": 1, "elapsed": elapsed}


def ingest_ucs():
    """Parse UCS Satellite Database (xlsx) → external_ucs_satellites table."""
    t0 = time.time()
    log.info("=== UCS Satellite Database ===")
    path = os.path.join(DATA_DIR, "ucs_satellites.xlsx")
    if not os.path.exists(path):
        log.warning("  ucs_satellites.xlsx not found at %s — skipping", path)
        return

    ucs_raw = pd.read_excel(path, sheet_name=0)
    raw_count = len(ucs_raw)
    keep = {
        "Name of Satellite, Alternate Names": "name",
        "Country of Operator/Owner": "country",
        "Operator/Owner": "operator",
        "Users": "users",
        "Purpose": "purpose",
        "Detailed Purpose": "detailed_purpose",
        "Class of Orbit": "orbit_class",
        "Type of Orbit": "orbit_type",
        "Perigee (km)": "perigee_km",
        "Apogee (km)": "apogee_km",
        "Inclination (degrees)": "inclination_deg",
        "Launch Mass (kg.)": "launch_mass_kg",
        "Date of Launch": "launch_date",
        "Expected Lifetime (yrs.)": "expected_lifetime_yr",
        "Launch Vehicle": "launch_vehicle",
        "NORAD Number": "norad_cat_id",
        "COSPAR Number": "cospar_id",
    }
    available = [c for c in keep if c in ucs_raw.columns]
    ucs = ucs_raw[available].rename(columns=keep)

    # ── Clean ──────────────────────────────────────────────────────────────
    ucs["launch_date"] = pd.to_datetime(ucs.get("launch_date"), errors="coerce")
    for col in ["name", "country", "operator", "users", "purpose",
                "orbit_class", "orbit_type", "cospar_id"]:
        if col in ucs.columns:
            ucs[col] = ucs[col].astype(str).str.strip().replace("nan", pd.NA)
    ucs["norad_cat_id"] = pd.to_numeric(ucs.get("norad_cat_id"), errors="coerce")
    for num_col in ["perigee_km", "apogee_km", "inclination_deg",
                    "launch_mass_kg", "expected_lifetime_yr"]:
        if num_col in ucs.columns:
            ucs[num_col] = pd.to_numeric(ucs[num_col], errors="coerce")

    # ── Dedup: by NORAD number (primary key for cross-referencing) ────────
    has_norad = ucs["norad_cat_id"].notna()
    ucs_with_norad = ucs[has_norad].drop_duplicates(subset=["norad_cat_id"], keep="first")
    ucs_without_norad = ucs[~has_norad]
    dedup_dropped = len(ucs) - len(ucs_with_norad) - len(ucs_without_norad)
    ucs = pd.concat([ucs_with_norad, ucs_without_norad], ignore_index=True)
    if dedup_dropped:
        log.info("  Dedup by NORAD: dropped %d duplicates", dedup_dropped)

    log.info("  Rows: %d (raw %d), countries: %d, with NORAD: %d",
             len(ucs), raw_count, ucs["country"].nunique(),
             ucs["norad_cat_id"].notna().sum())
    log.info("  Purpose distribution: %s",
             ucs["purpose"].value_counts().head(5).to_dict())
    log.info("  Users distribution: %s",
             ucs["users"].value_counts().head(5).to_dict())

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_ucs_satellites CASCADE"))

    ucs.assign(source="UCS").to_sql(
        "external_ucs_satellites", engine,
        if_exists="replace", index=False, method="multi", chunksize=500,
    )
    elapsed = time.time() - t0
    log.info("  Inserted %d UCS rows (%.1fs)", len(ucs), elapsed)
    _STATS["UCS"] = {"rows_raw": raw_count, "rows_deduped": len(ucs),
                      "tables": 1, "elapsed": elapsed}


def _esa_sma_ecc_to_peri_apo(sma_m, ecc):
    """Convert semi-major axis (metres) + eccentricity → (perigee_km, apogee_km)."""
    R_EARTH_KM = 6371.0
    sma_km = sma_m / 1000.0
    peri = sma_km * (1 - ecc) - R_EARTH_KM
    apo = sma_km * (1 + ecc) - R_EARTH_KM
    return round(peri, 2), round(apo, 2)


def ingest_esa_discos():
    """Fetch ESA DISCOS objects + initialOrbits via API → external_esa_discos table."""
    t0 = time.time()
    log.info("=== ESA DISCOS ===")
    token = os.environ.get("ESA_DISCOS_TOKEN", "")
    if not token:
        csv_path = os.path.join(DATA_DIR, "esa_discos_objects.csv")
        if os.path.exists(csv_path):
            log.info("  No ESA_DISCOS_TOKEN, loading from local CSV (%s)", csv_path)
            esa = pd.read_csv(csv_path)
        else:
            log.warning("  No ESA_DISCOS_TOKEN set and no local CSV — skipping")
            return
    else:
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        base = "https://discosweb.esoc.esa.int/api"

        # ── Phase 1: Fetch all objects (fast, no include) ─────────────────
        all_objects = {}  # esa_id → dict
        page, max_pages = 1, 2000
        while page <= max_pages:
            try:
                r = requests.get(
                    f"{base}/objects?page[size]=100&page[number]={page}",
                    headers=headers, timeout=60,
                )
            except requests.RequestException as e:
                log.error("  ESA objects API error on page %d: %s", page, e)
                break
            if r.status_code != 200:
                log.error("  ESA objects API error on page %d: %d %s", page, r.status_code, r.text[:200])
                break
            data = r.json()
            items = data.get("data", [])
            if not items:
                break
            for obj in items:
                esa_id = obj["id"]
                attrs = obj.get("attributes", {})
                all_objects[esa_id] = {
                    "satno": attrs.get("satno"),
                    "cosparId": attrs.get("cosparId"),
                    "name": attrs.get("name"),
                    "objectClass": attrs.get("objectClass"),
                    "mass": attrs.get("mass"),
                    "shape": attrs.get("shape"),
                    "xSectMax": attrs.get("xSectMax"),
                    "xSectMin": attrs.get("xSectMin"),
                    "xSectAvg": attrs.get("xSectAvg"),
                    "firstEpoch": attrs.get("firstEpoch"),
                    "mission": attrs.get("mission"),
                    "predDecayDate": attrs.get("predDecayDate"),
                    "active": attrs.get("active"),
                    "cataloguedFragments": attrs.get("cataloguedFragments"),
                    "onOrbitCataloguedFragments": attrs.get("onOrbitCataloguedFragments"),
                }
            total_pages = data.get("meta", {}).get("pagination", {}).get("totalPages", 0)
            if page >= total_pages:
                break
            page += 1
            if page % 50 == 0:
                log.info("    objects page %d/%d …", page, total_pages)
        log.info("  Phase 1: fetched %d objects in %d pages", len(all_objects), page)

        # ── Phase 2: Fetch initialOrbits (include=object for parent link) ──
        orbit_for_obj = {}  # esa_id → best orbit dict
        page, max_pages = 1, 2000
        while page <= max_pages:
            try:
                r = requests.get(
                    f"{base}/initial-orbits?page[size]=100&page[number]={page}"
                    "&include=object",
                    headers=headers, timeout=60,
                )
            except requests.RequestException as e:
                log.error("  ESA orbits API error on page %d: %s", page, e)
                break
            if r.status_code != 200:
                log.error("  ESA orbits API error on page %d: %d %s", page, r.status_code, r.text[:200])
                break
            data = r.json()
            items = data.get("data", [])
            if not items:
                break
            for orb_item in items:
                attrs = orb_item.get("attributes", {})
                obj_rel = (orb_item.get("relationships", {})
                           .get("object", {}).get("data") or {})
                esa_id = obj_rel.get("id")
                if not esa_id:
                    continue
                sma = attrs.get("sma")
                ecc = attrs.get("ecc")
                if sma is None or ecc is None:
                    continue
                if esa_id not in orbit_for_obj:
                    orbit_for_obj[esa_id] = {
                        "sma": sma, "ecc": ecc,
                        "inc": attrs.get("inc"),
                    }
            total_pages = data.get("meta", {}).get("pagination", {}).get("totalPages", 0)
            if page >= total_pages:
                break
            page += 1
            if page % 50 == 0:
                log.info("    orbits page %d/%d …", page, total_pages)
        log.info("  Phase 2: fetched orbits for %d objects in %d pages", len(orbit_for_obj), page)

        # ── Merge orbit data into objects ─────────────────────────────────
        for esa_id, row in all_objects.items():
            orb = orbit_for_obj.get(esa_id)
            if orb:
                row["inclination"] = orb.get("inc")
                try:
                    peri, apo = _esa_sma_ecc_to_peri_apo(float(orb["sma"]), float(orb["ecc"]))
                    row["perigee_km"] = peri
                    row["apogee_km"] = apo
                except (ValueError, TypeError):
                    pass

        esa = pd.DataFrame(list(all_objects.values()))
        esa.to_csv(os.path.join(DATA_DIR, "esa_discos_objects.csv"), index=False)
        log.info("  Fetched %d objects, %d with orbits from ESA DISCOS API",
                 len(esa), esa.get("perigee_km", pd.Series()).notna().sum())

    raw_count = len(esa)

    # ── Clean ──────────────────────────────────────────────────────────────
    esa["name"] = esa["name"].astype(str).str.strip().replace("nan", pd.NA)
    esa["objectClass"] = esa["objectClass"].astype(str).str.strip()
    esa["satno"] = pd.to_numeric(esa["satno"], errors="coerce")
    for nc in ["mass", "xSectAvg", "xSectMax", "xSectMin",
               "inclination", "perigee_km", "apogee_km"]:
        if nc in esa.columns:
            esa[nc] = pd.to_numeric(esa[nc], errors="coerce")

    # ── Dedup: by satno (NORAD catalogue number) ──────────────────────────
    has_satno = esa["satno"].notna()
    esa_w = esa[has_satno].drop_duplicates(subset=["satno"], keep="first")
    esa_wo = esa[~has_satno]
    dedup_dropped = len(esa) - len(esa_w) - len(esa_wo)
    esa = pd.concat([esa_w, esa_wo], ignore_index=True)
    if dedup_dropped:
        log.info("  Dedup by satno: dropped %d duplicates", dedup_dropped)

    has_orbit = esa["perigee_km"].notna().sum() if "perigee_km" in esa.columns else 0
    log.info("  Rows: %d (raw %d), with orbits: %d, classes: %s",
             len(esa), raw_count, has_orbit,
             esa["objectClass"].value_counts().to_dict())

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_esa_discos CASCADE"))

    esa.assign(source="ESA-DISCOS").to_sql(
        "external_esa_discos", engine,
        if_exists="replace", index=False, method="multi", chunksize=500,
    )
    elapsed = time.time() - t0
    log.info("  Inserted %d ESA DISCOS rows (%.1fs)", len(esa), elapsed)
    _STATS["ESA"] = {"rows_raw": raw_count, "rows_deduped": len(esa),
                      "tables": 1, "elapsed": elapsed}


def cross_source_report():
    """Print cross-source overlap report using NORAD IDs."""
    log.info("=" * 60)
    log.info("=== Cross-Source Overlap Report ===")

    norad_sets: dict[str, set] = {}
    queries = [
        ("Space-Track", "SELECT DISTINCT norad_cat_id FROM catalog_objects WHERE norad_cat_id IS NOT NULL"),
        ("UCS", "SELECT DISTINCT norad_cat_id FROM external_ucs_satellites WHERE norad_cat_id IS NOT NULL"),
        ("ESA-DISCOS", "SELECT DISTINCT satno FROM external_esa_discos WHERE satno IS NOT NULL"),
        ("GCAT", "SELECT DISTINCT norad_cat_id FROM external_yearly_launches LIMIT 0"),
    ]

    try:
        with engine.connect() as conn:
            for label, sql in queries:
                try:
                    rows = conn.execute(text(sql)).fetchall()
                    ids = {int(r[0]) for r in rows if r[0] is not None}
                    norad_sets[label] = ids
                    log.info("  %s: %d unique NORAD IDs", label, len(ids))
                except Exception:
                    pass

            # Also count GCAT total objects from yearly launches
            try:
                gcat_total = conn.execute(text(
                    "SELECT SUM(count) FROM external_yearly_launches"
                )).scalar()
                log.info("  GCAT total objects (yearly sum): %s", f"{gcat_total:,}" if gcat_total else "N/A")
            except Exception:
                pass
    except Exception as e:
        log.warning("  Cross-source report failed: %s", e)
        return

    # Pairwise overlap
    labels = list(norad_sets.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            overlap = norad_sets[a] & norad_sets[b]
            if overlap:
                log.info("  %s ∩ %s: %d shared NORAD IDs", a, b, len(overlap))
            else:
                log.info("  %s ∩ %s: no overlap", a, b)

    # Final DB table sizes
    log.info("")
    log.info("=== Final Database Table Sizes ===")
    tables = [
        "catalog_objects", "gp_elements", "trajectory_segments",
        "external_yearly_launches", "external_onorbit_snapshot",
        "external_cumulative_onorbit", "external_country_yearly_payload",
        "external_unoosa_launches", "external_ucs_satellites", "external_esa_discos",
    ]
    try:
        with engine.connect() as conn:
            for t in tables:
                try:
                    cnt = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                    log.info("  %-40s %s rows", t, f"{cnt:,}")
                except Exception:
                    log.info("  %-40s (not found)", t)
    except Exception as e:
        log.warning("  Table count failed: %s", e)

    # Per-source summary
    if _STATS:
        log.info("")
        log.info("=== Per-Source Summary ===")
        total_raw = 0
        total_dedup = 0
        total_elapsed = 0.0
        for src, s in _STATS.items():
            log.info("  %-12s  raw=%s  deduped=%s  tables=%d  time=%.1fs",
                     src,
                     f"{s['rows_raw']:,}",
                     f"{s['rows_deduped']:,}",
                     s.get("tables", 0),
                     s.get("elapsed", 0))
            total_raw += s["rows_raw"]
            total_dedup += s["rows_deduped"]
            total_elapsed += s.get("elapsed", 0)
        log.info("  %-12s  raw=%s  deduped=%s  total_time=%.1fs",
                 "TOTAL", f"{total_raw:,}", f"{total_dedup:,}", total_elapsed)

    log.info("=" * 60)
    log.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest external data sources")
    parser.add_argument("--gcat", action="store_true", help="Ingest GCAT only")
    parser.add_argument("--unoosa", action="store_true", help="Ingest UNOOSA only")
    parser.add_argument("--ucs", action="store_true", help="Ingest UCS only")
    parser.add_argument("--esa", action="store_true", help="Ingest ESA DISCOS only")
    args = parser.parse_args()

    run_all = not (args.gcat or args.unoosa or args.ucs or args.esa)
    t_start = time.time()

    if run_all or args.gcat:
        ingest_satcat()
    if run_all or args.unoosa:
        ingest_unoosa()
    if run_all or args.ucs:
        ingest_ucs()
    if run_all or args.esa:
        ingest_esa_discos()

    if run_all:
        cross_source_report()
    else:
        log.info("Partial run complete (%.1fs)", time.time() - t_start)
