"""Comprehensive space-event ingestion script.

Data sources implemented
========================
1. **ESA DISCOSweb** ``/api/fragmentations``    解体 / 推进剂爆炸 / 反卫星等历史事件
   (1957→今, 共 ~670 条).  需要 ``ESA_DISCOS_TOKEN``.

2. **Space-Track ``cdm_public``**               未来 7 天预测的 CDM 碰撞预警
   (USSPACECOM 公开版本).  需要 ``SPACETRACK_USERNAME / SPACETRACK_PASSWORD``.

3. **Space-Track ``decay``**                    再入 (TIP / Decay) 历史 + 未来预报.

4. **CelesTrak SOCRATES**                       CelesTrak 实时合取预警 (公开, 无需账号).
   每 8 小时刷新; CSV 16 MB / ~150k 条; 我们截取 ``MIN_PROB``-排序前若干条.

5. **Jonathan McDowell GCAT ``ecat``**          GCAT 事件目录 (CC-BY).
   状态码 ``AR / AR IN / AL / AL IN`` => REENTRY 事件.

Usage
=====
    python scripts/ingest_events.py --all                            # 全量
    python scripts/ingest_events.py --discos --cdm                   # 选若干
    python scripts/ingest_events.py --since 2026-04-01 --max 1000    # 增量

Each fetcher gracefully degrades when the source is unavailable
(token missing / network failure / 4xx-5xx) and returns ``[]``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import logging
import os
import sys
from typing import List, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_db
from events.types import EventType, SpaceEvent
from events.crud import upsert_event

log = logging.getLogger("ingest_events")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# ════════════════════════════════════════════════════════════════════════════
#                              Helpers
# ════════════════════════════════════════════════════════════════════════════

def _parse_iso(x) -> Optional[dt.datetime]:
    if not x: return None
    try:
        s = str(x).strip().replace("Z", "+00:00")
        # Space-Track uses ``YYYY-MM-DD HH:MM:SS.ssssss``
        if "T" not in s and " " in s:
            s = s.replace(" ", "T", 1)
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def _since_filter(evt: SpaceEvent, since: Optional[dt.datetime]) -> bool:
    if since is None: return True
    if evt.epoch is None: return False
    e = evt.epoch
    if e.tzinfo is None and since.tzinfo is not None:
        e = e.replace(tzinfo=since.tzinfo)
    if since.tzinfo is None and e.tzinfo is not None:
        e = e.replace(tzinfo=None)
    return e >= since


# ════════════════════════════════════════════════════════════════════════════
# 1. ESA DISCOS – fragmentations
# ════════════════════════════════════════════════════════════════════════════

def fetch_discos_fragmentations(max_rows: int = 1000,
                                since: Optional[dt.datetime] = None,
                                ) -> List[SpaceEvent]:
    token = os.environ.get("ESA_DISCOS_TOKEN", "").strip()
    if not token:
        log.warning("[DISCOS] ESA_DISCOS_TOKEN 未配置, 跳过")
        return []
    headers = {"Authorization": f"Bearer {token}",
               "DiscosWeb-Api-Version": "2",
               "Accept": "application/json"}
    base = "https://discosweb.esoc.esa.int/api"

    out: List[SpaceEvent] = []
    page = 1
    while len(out) < max_rows:
        try:
            r = requests.get(
                f"{base}/fragmentations",
                params={"page[size]": 100, "page[number]": page,
                        "sort": "-epoch",
                        "include": "objects"},
                headers=headers, timeout=60)
        except requests.RequestException as exc:
            log.warning("[DISCOS] HTTP error: %s", exc)
            break
        if r.status_code != 200:
            log.warning("[DISCOS] HTTP %d: %s", r.status_code, r.text[:160])
            break
        body = r.json()
        items = body.get("data", []) or []
        if not items: break

        # Build map id -> (norad, name, mass) from included objects
        included = {}
        for inc in body.get("included", []) or []:
            if inc.get("type") == "object":
                a = inc.get("attributes", {}) or {}
                included[str(inc.get("id"))] = {
                    "satno": a.get("satno"),
                    "name":  a.get("name"),
                    "mass":  a.get("mass"),
                }

        stop = False
        for it in items:
            attr = it.get("attributes", {}) or {}
            epoch = _parse_iso(attr.get("epoch"))
            if epoch is None: continue

            # Walk first related object (the parent) if present
            parent_norad = None; parent_name = ""; parent_mass = None
            rel = (it.get("relationships", {}) or {}).get("objects", {})
            data_links = (rel.get("data") or [])
            if data_links:
                first = str(data_links[0].get("id"))
                if first in included:
                    parent_norad = included[first].get("satno")
                    parent_name  = included[first].get("name") or ""
                    parent_mass  = included[first].get("mass")

            cause = (attr.get("eventType") or "").upper()
            if "COLLISION" in cause:
                etype = EventType.COLLISION
            elif "EXPLOSION" in cause or "PROPULSION" in cause or "BREAKUP" in cause:
                etype = EventType.FRAGMENTATION
            else:
                etype = EventType.FRAGMENTATION  # default for DISCOS frag list

            evt = SpaceEvent(
                event_type      = etype,
                epoch           = epoch,
                name            = parent_name or f"DISCOS-FRAG-{it.get('id','?')}",
                description     = (attr.get("comment") or attr.get("eventType") or "")[:1000],
                parent_norad    = parent_norad,
                altitude_km     = attr.get("altitude"),
                mass_parent_kg  = parent_mass,
                source          = "DISCOS",
                source_id       = str(it.get("id", "")),
                raw             = attr,
            )
            if not _since_filter(evt, since):
                # epoch sorted desc → done
                stop = True; break
            out.append(evt)
            if len(out) >= max_rows:
                stop = True; break

        if stop: break
        meta = body.get("meta", {}).get("pagination", {})
        if page >= int(meta.get("totalPages", page)): break
        page += 1

    log.info("[DISCOS] fragmentations: %d", len(out))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. Space-Track  cdm_public
# ════════════════════════════════════════════════════════════════════════════

def _spacetrack_session() -> Optional[requests.Session]:
    user = os.environ.get("SPACETRACK_USERNAME") or os.environ.get("SPACE_TRACK_USERNAME")
    pwd  = os.environ.get("SPACETRACK_PASSWORD") or os.environ.get("SPACE_TRACK_PASSWORD")
    if not (user and pwd):
        log.warning("[Space-Track] credentials missing, skipping")
        return None
    s = requests.Session()
    try:
        r = s.post("https://www.space-track.org/ajaxauth/login",
                   data={"identity": user, "password": pwd}, timeout=60)
        r.raise_for_status()
        if "Login" in r.text:
            log.warning("[Space-Track] login refused"); return None
    except requests.RequestException as exc:
        log.warning("[Space-Track] login failed: %s", exc); return None
    return s


def fetch_spacetrack_cdms(max_rows: int = 500,
                          since: Optional[dt.datetime] = None,
                          ) -> List[SpaceEvent]:
    sess = _spacetrack_session()
    if sess is None: return []
    url = ("https://www.space-track.org/basicspacedata/query/class/cdm_public/"
           f"orderby/TCA%20desc/limit/{int(max_rows)}/format/json")
    try:
        r = sess.get(url, timeout=120); r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[Space-Track CDM] %s", exc); return []
    rows = r.json() or []
    out: List[SpaceEvent] = []
    for row in rows:
        tca = _parse_iso(row.get("TCA"))
        if tca is None: continue
        evt = SpaceEvent(
            event_type      = EventType.CDM,
            epoch           = tca,
            name            = f"CDM {row.get('SAT_1_NAME','?')} ↔ {row.get('SAT_2_NAME','?')}",
            description     = row.get("MESSAGE_FOR", "")[:200],
            parent_norad    = int(row["SAT_1_ID"])    if row.get("SAT_1_ID") else None,
            secondary_norad = int(row["SAT_2_ID"])    if row.get("SAT_2_ID") else None,
            miss_distance_km= float(row["MISS_DISTANCE"])/1000.0
                              if row.get("MISS_DISTANCE") else None,
            probability     = float(row["PC"])        if row.get("PC") else None,
            source          = "SPACETRACK_CDM",
            source_id       = str(row.get("CDM_ID","")),
            raw             = row,
        )
        if _since_filter(evt, since):
            out.append(evt)
    log.info("[Space-Track CDM] %d", len(out))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Space-Track  decay  (re-entry / TIP)
# ════════════════════════════════════════════════════════════════════════════

def fetch_spacetrack_decays(max_rows: int = 500,
                            since: Optional[dt.datetime] = None,
                            ) -> List[SpaceEvent]:
    sess = _spacetrack_session()
    if sess is None: return []
    url = ("https://www.space-track.org/basicspacedata/query/class/decay/"
           f"orderby/DECAY_EPOCH%20desc/limit/{int(max_rows)}/format/json")
    try:
        r = sess.get(url, timeout=120); r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[Space-Track Decay] %s", exc); return []
    rows = r.json() or []
    out: List[SpaceEvent] = []
    for row in rows:
        ep = _parse_iso(row.get("DECAY_EPOCH") or row.get("MSG_EPOCH"))
        if ep is None: continue
        norad = row.get("NORAD_CAT_ID")
        evt = SpaceEvent(
            event_type      = EventType.REENTRY,
            epoch           = ep,
            name            = f"REENTRY {row.get('OBJECT_NAME','?')}",
            description     = (row.get("RCS_SIZE") or "") + " | " + (row.get("OBJECT_TYPE") or ""),
            parent_norad    = int(norad) if norad else None,
            source          = "SPACETRACK_DECAY",
            source_id       = str(row.get("MSG_EPOCH","")) + "-" + str(norad or ""),
            raw             = row,
        )
        if _since_filter(evt, since):
            out.append(evt)
    log.info("[Space-Track Decay/TIP] %d", len(out))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 4. CelesTrak SOCRATES (public conjunction predictions)
# ════════════════════════════════════════════════════════════════════════════

def fetch_celestrak_socrates(max_rows: int = 500,
                             since: Optional[dt.datetime] = None,
                             ) -> List[SpaceEvent]:
    """Fetch CelesTrak SOCRATES top-N close-approach predictions (CSV)."""
    url = "https://celestrak.org/SOCRATES/sort-minRange.csv"
    try:
        r = requests.get(url, timeout=120); r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[SOCRATES] %s", exc); return []
    text = r.text
    out: List[SpaceEvent] = []
    rdr = csv.DictReader(io.StringIO(text))
    for row in rdr:
        tca = _parse_iso(row.get("TCA"))
        if tca is None: continue
        try:
            n1 = int(row["NORAD_CAT_ID_1"]); n2 = int(row["NORAD_CAT_ID_2"])
        except Exception:
            n1 = n2 = None
        try: miss_km = float(row["TCA_RANGE"])
        except Exception: miss_km = None
        try: pc = float(row["MAX_PROB"])
        except Exception: pc = None
        evt = SpaceEvent(
            event_type      = EventType.CDM,
            epoch           = tca,
            name            = f"SOCRATES {row.get('OBJECT_NAME_1','?')} ↔ {row.get('OBJECT_NAME_2','?')}",
            description     = f"v_rel={row.get('TCA_RELATIVE_SPEED','?')} km/s, dilution={row.get('DILUTION','?')}",
            parent_norad    = n1, secondary_norad = n2,
            miss_distance_km= miss_km, probability = pc,
            source          = "CELESTRAK_SOCRATES",
            source_id       = f"{n1}-{n2}-{tca.strftime('%Y%m%d%H%M%S')}",
            raw             = row,
        )
        if _since_filter(evt, since):
            out.append(evt)
        if len(out) >= max_rows:
            break
    log.info("[SOCRATES] %d", len(out))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 5. Jonathan McDowell  GCAT ecat (event catalog)
# ════════════════════════════════════════════════════════════════════════════

GCAT_ECAT_URL = "https://planet4589.org/space/gcat/tsv/cat/ecat.tsv"


def fetch_gcat_events(max_rows: int = 1000,
                      since: Optional[dt.datetime] = None,
                      ) -> List[SpaceEvent]:
    """Parse GCAT ``ecat.tsv``; map status codes to event types.

    Mappings:
      * ``AR``, ``AR IN``     → REENTRY (atmospheric reentry, intact / fragments)
      * ``AL``, ``AL IN``     → REENTRY (atmospheric loss, low altitude)
      * ``D``, ``DK``, ``DSO`` → REENTRY (decayed / disposal)
      * ``GRP``               → FRAGMENTATION (group released - debris cloud)
      * other status codes are skipped (EVA / TFR / leasing / attached etc.)
    """
    try:
        r = requests.get(GCAT_ECAT_URL, timeout=120); r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[GCAT] %s", exc); return []

    text = r.text
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("# ")]
    if not lines: return []
    header = lines[0].lstrip("#").split("\t")
    # Map column index
    idx = {col.strip(): i for i, col in enumerate(header)}
    def g(parts, key):
        i = idx.get(key); return parts[i].strip() if i is not None and i < len(parts) else ""

    REENTRY = {"AR", "AR IN", "AL", "AL IN", "D", "DK", "DSO"}
    FRAG    = {"GRP"}

    out: List[SpaceEvent] = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        status = g(parts, "Status").upper()
        if status not in REENTRY and status not in FRAG:
            continue
        sd = g(parts, "SDate") or g(parts, "DDate") or g(parts, "ODate")
        epoch = _gcat_date(sd)
        if epoch is None: continue
        norad = g(parts, "Satcat") or ""
        try: norad_i = int(norad) if norad and norad.isdigit() else None
        except Exception: norad_i = None

        etype = EventType.FRAGMENTATION if status in FRAG else EventType.REENTRY
        evt = SpaceEvent(
            event_type   = etype,
            epoch        = epoch,
            name         = f"GCAT {g(parts, 'Name')} ({status})".strip(),
            description  = f"GCAT phase {status} – piece={g(parts, 'Piece')} primary={g(parts, 'Primary')}",
            parent_norad = norad_i,
            altitude_km  = _gcat_float(g(parts, "Perigee")),
            inclination_deg = _gcat_float(g(parts, "Inc")),
            mass_parent_kg  = _gcat_float(g(parts, "Mass")),
            source       = "GCAT_ECAT",
            source_id    = f"{g(parts,'JCAT')}-{status}-{sd}",
            raw          = {h: p for h, p in zip(header, parts)},
        )
        if _since_filter(evt, since):
            out.append(evt)
        if len(out) >= max_rows:
            break
    log.info("[GCAT ecat] %d", len(out))
    return out


def _gcat_date(s: str) -> Optional[dt.datetime]:
    """Parse GCAT date format ``YYYY MMM DD HHMM[ss]``."""
    if not s or s in {"-", "?"}: return None
    s = s.strip()
    fmts = ["%Y %b %d %H%M:%S", "%Y %b %d %H%M", "%Y %b %d", "%Y %b"]
    for f in fmts:
        try: return dt.datetime.strptime(s, f)
        except Exception: continue
    return None


def _gcat_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s or s in {"-", "?"}: return None
    try: return float(s)
    except Exception: return None


# ════════════════════════════════════════════════════════════════════════════
#                              Driver
# ════════════════════════════════════════════════════════════════════════════

ALL_FETCHERS = {
    "discos":   fetch_discos_fragmentations,
    "cdm":      fetch_spacetrack_cdms,
    "decay":    fetch_spacetrack_decays,
    "socrates": fetch_celestrak_socrates,
    "gcat":     fetch_gcat_events,
}


def run_ingest(*, sources: List[str], max_rows: int,
               since: Optional[dt.datetime] = None,
               ) -> tuple[List[SpaceEvent], dict[str, int]]:
    """Fetch + upsert.  Returns (events_inserted, counts_per_source)."""
    init_db()
    fetched: List[SpaceEvent] = []
    counts: dict[str, int] = {}
    for s in sources:
        fn = ALL_FETCHERS.get(s)
        if fn is None: continue
        items = fn(max_rows=max_rows, since=since)
        counts[s] = len(items)
        fetched.extend(items)

    n_ok = 0
    for evt in fetched:
        try:
            upsert_event(evt); n_ok += 1
        except Exception as exc:
            log.warning("upsert failed (%s/%s): %s", evt.source, evt.source_id, exc)
    counts["upserted"] = n_ok
    log.info("总计 upsert: %d 条", n_ok)
    return fetched, counts


def main():
    p = argparse.ArgumentParser(description="拉取多源太空事件 → space_events 表")
    p.add_argument("--discos",   action="store_true")
    p.add_argument("--cdm",      action="store_true", help="Space-Track CDM")
    p.add_argument("--decay",    action="store_true", help="Space-Track Decay/TIP")
    p.add_argument("--socrates", action="store_true", help="CelesTrak SOCRATES")
    p.add_argument("--gcat",     action="store_true", help="McDowell GCAT ecat")
    p.add_argument("--all",      action="store_true")
    p.add_argument("--max",      type=int, default=500)
    p.add_argument("--since",    type=str, default=None,
                   help="只拉取该 ISO 日期之后(>=)的事件，例如 2026-04-01")
    args = p.parse_args()

    if args.all:
        sources = list(ALL_FETCHERS.keys())
    else:
        sources = [k for k in ALL_FETCHERS.keys() if getattr(args, k)]
    if not sources:
        print("请指定 --discos/--cdm/--decay/--socrates/--gcat/--all"); return

    since = _parse_iso(args.since) if args.since else None
    _, counts = run_ingest(sources=sources, max_rows=args.max, since=since)
    log.info("汇总: %s", counts)


if __name__ == "__main__":
    main()
