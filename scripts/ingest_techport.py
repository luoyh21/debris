"""Ingest NASA TechPort technology-project catalogue into the database.

Source
------
NASA TechPort (https://techport.nasa.gov/help/api):

  GET https://techport.nasa.gov/api/projects        – list of project ids
  GET https://techport.nasa.gov/api/projects/{id}   – per-project JSON

The endpoints are publicly readable; an *API token* is only required by
TechPort for authenticated session-bound calls (e.g. private/draft data).
The token "reflects an active session, you'll need a new token each time
you access the TechPort API" — so we treat it as an optional, refreshable
secret.  Set ``NASA_TECHPORT_TOKEN`` in ``.env`` (or ``--token`` on the
command line) and we'll send it as ``Authorization: Bearer <token>``.

Schema created
--------------
``external_techport``  (one row per project)

Columns kept (flattened from the deeply-nested project JSON):

  project_id            int    primary identifier
  title                 text   project title
  status                text   Active / Completed / …
  release_status        text   Released / Draft / …
  start_date            date   project start
  end_date              date   project end
  start_year, end_year  int    convenience for analytics
  trl_begin, trl_current, trl_end   int   Technology Readiness Levels (1–9)
  view_count            int    public view counter
  description           text   prose description (HTML stripped)
  benefits              text   benefit/impact text (HTML stripped)
  destination_types     text   comma-separated mission destinations (Earth, Moon, Mars …)
  program_id            int    parent program id
  program_acronym       text   e.g. STRG
  program_title         text   e.g. Space Technology Research Grants
  lead_org_id           int    lead organisation NASA TechPort id
  lead_org_name         text   e.g. "Arizona State University-Tempe"
  lead_org_acronym      text   e.g. "ASU"
  lead_org_type         text   Academia / Industry / NASA_Center / …
  lead_org_country      text   ISO/abbreviation (US, FR …)
  lead_org_state        text   US state abbreviation (AZ, CA …)
  primary_tx_code       text   e.g. TX06.1.2
  primary_tx_title      text   e.g. Water Recovery and Management
  primary_tx_level      int    taxonomy depth
  states                text   comma-separated US state abbreviations
  last_updated          text   raw mm/dd/yy string from TechPort
  source                text   constant "TechPort"

Usage
-----
    python3 scripts/ingest_techport.py                       # fetch all ~20k projects
    python3 scripts/ingest_techport.py --limit 500           # quick subset
    python3 scripts/ingest_techport.py --workers 8           # parallel detail fetch
    python3 scripts/ingest_techport.py --token "<api-token>" # explicit token
    python3 scripts/ingest_techport.py --offline             # use local cache only

The script does:
  1. GET /api/projects to get the full id list (cached locally)
  2. Concurrent GETs for /api/projects/{id} (configurable workers, retries on 429/5xx)
  3. Project JSON cached on disk so re-runs are fast and offline-friendly
  4. Schema-aware flattening + numeric/date coercion + HTML strip
  5. Drop-and-replace ``external_techport`` table
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

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

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)
LIST_CACHE   = os.path.join(DATA_DIR, "techport_index.json")
DETAIL_CACHE = os.path.join(DATA_DIR, "techport_projects.jsonl")

_BASE = "https://techport.nasa.gov/api"
_HDR  = {"User-Agent": "space_debris-techport-ingest/1.0",
         "Accept":     "application/json"}

# ── Helpers ──────────────────────────────────────────────────────────────────
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s+")

def _strip_html(s: Any, max_len: int = 4000) -> str | None:
    """Remove HTML tags + collapse whitespace; truncate to *max_len* chars."""
    if s is None or s == "":
        return None
    txt = _HTML_RE.sub(" ", str(s))
    txt = _WS_RE.sub(" ", txt).strip()
    if not txt:
        return None
    return txt[:max_len]


def _safe_get(d: dict | None, *keys, default=None):
    """Safely descend ``d[k1][k2]…``; returns *default* if any step is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur not in (None, "") else default


def _to_int(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _build_session(token: str | None) -> requests.Session:
    s = requests.Session()
    s.headers.update(_HDR)
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _http_get_json(sess: requests.Session, url: str,
                   retries: int = 4, backoff: float = 1.5) -> Any | None:
    """GET *url* with limited retries on 429/5xx."""
    delay = 1.0
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(delay)
                delay *= backoff
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if r.status_code == 404:
                return None
            log.debug("  GET %s failed (%s) attempt %d/%d", url, e, attempt + 1, retries)
        except Exception as e:
            log.debug("  GET %s exception (%s) attempt %d/%d", url, e, attempt + 1, retries)
        time.sleep(delay)
        delay *= backoff
    return None


# ── Step 1: list of project IDs ──────────────────────────────────────────────
def fetch_project_index(sess: requests.Session, *, offline: bool = False) -> list[dict]:
    """Return the full project listing (cached on disk)."""
    if offline and os.path.exists(LIST_CACHE):
        with open(LIST_CACHE) as fh:
            return json.load(fh)

    log.info("Fetching project index from %s/projects", _BASE)
    data = _http_get_json(sess, f"{_BASE}/projects")
    if not data and os.path.exists(LIST_CACHE):
        log.warning("  API failed — falling back to local cache")
        with open(LIST_CACHE) as fh:
            return json.load(fh)
    projects = (data or {}).get("projects", [])
    if projects:
        with open(LIST_CACHE, "w") as fh:
            json.dump(projects, fh)
        log.info("  Cached %d project ids → %s", len(projects), LIST_CACHE)
    return projects


# ── Step 2: per-project detail fetch ─────────────────────────────────────────
def _load_detail_cache() -> dict[int, dict]:
    """Load existing detail cache (jsonl) → {project_id: full project dict}."""
    cache: dict[int, dict] = {}
    if not os.path.exists(DETAIL_CACHE):
        return cache
    with open(DETAIL_CACHE) as fh:
        for line in fh:
            try:
                row = json.loads(line)
                pid = row.get("projectId")
                if pid is not None:
                    cache[int(pid)] = row
            except Exception:
                pass
    return cache


def _append_detail_cache(rows: Iterable[dict]) -> None:
    with open(DETAIL_CACHE, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_project_details(sess: requests.Session, ids: list[int],
                          workers: int = 8,
                          *, offline: bool = False) -> list[dict]:
    """Fetch each project's full JSON, leveraging the on-disk cache."""
    cache = _load_detail_cache()
    log.info("  Detail cache has %d projects on disk", len(cache))

    missing = [i for i in ids if i not in cache]
    if offline:
        if missing:
            log.warning("  Offline mode: %d ids missing from cache, skipping", len(missing))
        return [cache[i] for i in ids if i in cache]

    if missing:
        log.info("  Fetching %d new project details (workers=%d)…", len(missing), workers)
        new_rows: list[dict] = []
        completed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_http_get_json, sess, f"{_BASE}/projects/{pid}"): pid
                for pid in missing
            }
            for fut in as_completed(futures):
                pid = futures[fut]
                payload = fut.result()
                if isinstance(payload, dict) and "project" in payload:
                    proj = payload["project"]
                    proj.setdefault("projectId", pid)
                    new_rows.append(proj)
                    cache[pid] = proj
                completed += 1
                if completed % 200 == 0:
                    rate = completed / max(time.time() - t0, 1e-3)
                    eta = (len(missing) - completed) / max(rate, 0.1)
                    log.info("    progress %d/%d (%.1f /s, ETA %.0fs)",
                             completed, len(missing), rate, eta)

        _append_detail_cache(new_rows)
        log.info("  Wrote %d new rows → %s", len(new_rows), DETAIL_CACHE)

    return [cache[i] for i in ids if i in cache]


# ── Step 3: flatten + clean ──────────────────────────────────────────────────
def _flatten(p: dict) -> dict:
    """Pick a stable, analytics-friendly subset of fields from a project."""
    lead = p.get("leadOrganization") or {}
    program = p.get("program") or {}
    primary_tx = p.get("primaryTx") or {}

    # destinationType is a list of dicts ({name: "..."}) or strings
    dt = p.get("destinationType") or []
    if dt and isinstance(dt[0], dict):
        dt_str = ", ".join(filter(None, (d.get("name") or d.get("title") for d in dt)))
    else:
        dt_str = ", ".join(str(x) for x in dt if x)

    states = p.get("states") or []
    state_abbrs = ", ".join(filter(None, (s.get("abbreviation") for s in states))) \
        if states and isinstance(states[0], dict) else ""

    return {
        "project_id":         _to_int(p.get("projectId")),
        "title":              (p.get("title") or "").strip() or None,
        "status":             p.get("status"),
        "release_status":     p.get("releaseStatus"),
        "start_date":         p.get("startDate"),
        "end_date":           p.get("endDate"),
        "start_year":         _to_int(p.get("startYear")),
        "end_year":           _to_int(p.get("endYear")),
        "trl_begin":          _to_int(p.get("trlBegin")),
        "trl_current":        _to_int(p.get("trlCurrent")),
        "trl_end":            _to_int(p.get("trlEnd")),
        "view_count":         _to_int(p.get("viewCount")),
        "description":        _strip_html(p.get("description")),
        "benefits":           _strip_html(p.get("benefits")),
        "destination_types":  dt_str or None,

        "program_id":         _to_int(program.get("programId")),
        "program_acronym":    program.get("acronym"),
        "program_title":      program.get("title"),

        "lead_org_id":        _to_int(lead.get("organizationId")),
        "lead_org_name":      lead.get("organizationName"),
        "lead_org_acronym":   lead.get("acronym"),
        "lead_org_type":      lead.get("organizationType"),
        "lead_org_country":   _safe_get(lead, "country", "abbreviation"),
        "lead_org_state":     _safe_get(lead, "stateTerritory", "abbreviation"),

        "primary_tx_code":    primary_tx.get("code"),
        "primary_tx_title":   primary_tx.get("title"),
        "primary_tx_level":   _to_int(primary_tx.get("level")),

        "states":             state_abbrs or None,
        "last_updated":       p.get("lastUpdated"),
        "source":             "TechPort",
    }


def _clean_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # Coerce date columns (strings like "2013-08-20" → date; bad values → NaT)
    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Dedup by project_id (should already be unique, but be defensive)
    if "project_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["project_id"], keep="last")
        dropped = before - len(df)
        if dropped:
            log.info("  Dedup by project_id: dropped %d duplicates", dropped)

    return df.reset_index(drop=True)


# ── Public API ───────────────────────────────────────────────────────────────
def ingest_techport(limit: int | None = None,
                    workers: int = 8,
                    token: str | None = None,
                    offline: bool = False) -> int:
    """Ingest TechPort projects into ``external_techport``; returns row count."""
    t0 = time.time()
    log.info("=" * 60)
    log.info("=== NASA TechPort (technology project portfolio) ===")

    if token is None:
        token = os.environ.get("NASA_TECHPORT_TOKEN") or None
    if token:
        log.info("  Using NASA TechPort API token (length=%d)", len(token))

    sess = _build_session(token)

    # Step 1: index
    listing = fetch_project_index(sess, offline=offline)
    if not listing:
        log.warning("  Empty project index — aborting")
        return 0
    log.info("  Index size: %d projects", len(listing))

    ids = [int(it["projectId"]) for it in listing if it.get("projectId") is not None]

    # Apply --limit early to skip needless detail fetches
    if limit and limit > 0 and len(ids) > limit:
        ids = ids[:int(limit)]
        log.info("  Truncated id list to first %d (--limit)", limit)

    # Step 2: details
    projects = fetch_project_details(sess, ids, workers=workers, offline=offline)
    log.info("  Successfully fetched %d / %d project details",
             len(projects), len(ids))
    if not projects:
        log.warning("  No project details available — aborting")
        return 0

    # Step 3: flatten + clean
    flat = [_flatten(p) for p in projects]
    df = _clean_dataframe(flat)
    log.info("  Final rows: %d  | columns: %d", len(df), len(df.columns))

    # Quick distribution log
    if "status" in df.columns:
        log.info("  Status distribution: %s",
                 df["status"].fillna("(null)").value_counts().head(6).to_dict())
    if "trl_current" in df.columns:
        trl = df["trl_current"].dropna()
        if not trl.empty:
            log.info("  TRL current range: %d–%d (median=%.0f)",
                     int(trl.min()), int(trl.max()), float(trl.median()))

    # Step 4: write to DB (drop + replace)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS external_techport CASCADE"))

    df.to_sql("external_techport", engine,
              if_exists="replace", index=False, method="multi", chunksize=400)

    # Add helpful indexes
    with engine.begin() as conn:
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS "
                          "ix_external_techport_pid ON external_techport(project_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS "
                          "ix_external_techport_status ON external_techport(status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS "
                          "ix_external_techport_trl ON external_techport(trl_current)"))

    elapsed = time.time() - t0
    log.info("  Inserted %d TechPort rows into external_techport (%.1fs)",
             len(df), elapsed)
    log.info("=" * 60)
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest NASA TechPort technology project portfolio")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap number of projects (0 = no limit, default = all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent detail-fetch workers (default 8)")
    parser.add_argument("--token", type=str, default=None,
                        help="NASA TechPort API token (overrides NASA_TECHPORT_TOKEN env)")
    parser.add_argument("--offline", action="store_true",
                        help="Use only the on-disk cache; no network calls")
    args = parser.parse_args()

    lim = int(args.limit) if args.limit and args.limit > 0 else None
    ingest_techport(limit=lim, workers=args.workers,
                    token=args.token, offline=args.offline)
