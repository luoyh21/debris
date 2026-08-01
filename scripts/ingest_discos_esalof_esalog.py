#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISCOSweb v2 · EsaLOF / EsaLOG 摄入。

EsaLOF（历史解体/爆炸/碰撞）
  GET /api/fragmentations?include=objects
  → external_discos_esalof
  → 同步 upsert space_events（与 ``scripts/ingest_events.py --discos`` 兼容）

EsaLOG（GEO 带物体 + 质量/RCS）
  GET /api/initial-orbits?filter=and(ge(sma,40000000),le(sma,45000000))&include=object
  （SMA 单位为米；GEO ≈ 42 164 km）
  → external_discos_esalog

必须请求头：
  Authorization: Bearer <ESA_DISCOS_TOKEN>
  DiscosWeb-Api-Version: 2
  Accept: application/vnd.api+json

用法：
  python scripts/ingest_discos_esalof_esalog.py
  python scripts/ingest_discos_esalof_esalog.py --esalof-only
  python scripts/ingest_discos_esalof_esalog.py --esalog-only --cache-dir data/monitoring_network
  python scripts/ingest_discos_esalof_esalog.py --sync-events   # 同时写入 space_events
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from database.db import get_engine, init_db
from scripts.discos_client import discos_headers, discos_paginate

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CACHE_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "monitoring_network"
# GEO 半长轴带（米）：≈ 40 000–45 000 km
GEO_SMA_FILTER = "and(ge(sma,40000000),le(sma,45000000))"


def _apply_migration(engine) -> None:
    mig = (
        Path(__file__).resolve().parent.parent
        / "database" / "migrations" / "004_discos_esalof_esalog.sql"
    )
    with open(mig, encoding="utf-8") as f:
        sql = f.read()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def _write_table(engine, table: str, rows: list[dict]) -> int:
    if not rows:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {table} RESTART IDENTITY CASCADE"))
        return 0
    df = pd.DataFrame(rows)
    with engine.connect() as conn:
        cols = [
            r[0]
            for r in conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_name='{table}' "
                "AND column_name NOT IN ('id','updated_at') "
                "ORDER BY ordinal_position"
            ))
        ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {table} RESTART IDENTITY CASCADE"))
    df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=500)
    with engine.begin() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    return int(n or 0)


def _included_map(payload: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for inc in payload.get("included") or []:
        if not isinstance(inc, dict):
            continue
        out[str(inc.get("id"))] = inc
    return out


# ── EsaLOF ───────────────────────────────────────────────────────────────────
def fetch_esalof(*, cache: Optional[Path] = None, use_cache: bool = True) -> list[dict]:
    """拉取全部碎片化事件 + 关联物体物理属性。"""
    cache_fp = (cache or CACHE_DEFAULT) / "discos_esalof.json"
    if use_cache and cache_fp.exists() and cache_fp.stat().st_size > 100:
        print(f"  [cache] EsaLOF {cache_fp}")
        return json.loads(cache_fp.read_text(encoding="utf-8"))

    headers = discos_headers()
    rows: list[dict] = []
    page = 0
    for payload in discos_paginate(
        "fragmentations",
        headers=headers,
        params={"include": "objects", "sort": "-epoch"},
        page_size=100,
    ):
        page += 1
        inc = _included_map(payload)
        for it in payload.get("data") or []:
            attr = it.get("attributes") or {}
            rel = ((it.get("relationships") or {}).get("objects") or {}).get("data") or []
            obj_a: dict[str, Any] = {}
            obj_id = None
            if rel:
                obj_id = str(rel[0].get("id"))
                obj_a = (inc.get(obj_id) or {}).get("attributes") or {}
            rows.append({
                "discos_id": str(it.get("id")),
                "epoch": attr.get("epoch"),
                "event_type": attr.get("eventType"),
                "comment": (attr.get("comment") or "")[:4000] or None,
                "latitude": attr.get("latitude"),
                "longitude": attr.get("longitude"),
                "altitude_km": attr.get("altitude"),
                "object_discos_id": obj_id,
                "satno": obj_a.get("satno"),
                "cospar_id": obj_a.get("cosparId"),
                "object_name": obj_a.get("name"),
                "object_class": obj_a.get("objectClass"),
                "mass_kg": obj_a.get("mass"),
                "shape": obj_a.get("shape"),
                "xsect_avg_m2": obj_a.get("xSectAvg"),
                "xsect_max_m2": obj_a.get("xSectMax"),
                "xsect_min_m2": obj_a.get("xSectMin"),
                "catalogued_fragments": obj_a.get("cataloguedFragments"),
                "onorbit_fragments": obj_a.get("onOrbitCataloguedFragments"),
                "source": "DISCOSweb:/api/fragmentations",
            })
        pag = (payload.get("meta") or {}).get("pagination") or {}
        print(f"  EsaLOF page {page}/{pag.get('totalPages', '?')}: {len(rows)}", flush=True)

    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    cache_fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  EsaLOF cached → {cache_fp} ({len(rows)})")
    return rows


def sync_esalof_to_space_events(rows: list[dict]) -> int:
    """将 EsaLOF 行同步到 space_events（复用现有事件模型）。"""
    from datetime import datetime, timezone

    from events.crud import upsert_event
    from events.types import EventType, SpaceEvent

    n = 0
    for r in rows:
        epoch_s = r.get("epoch")
        if not epoch_s:
            continue
        try:
            epoch = datetime.fromisoformat(str(epoch_s)).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        cause = (r.get("event_type") or "").upper()
        etype = EventType.COLLISION if "COLLISION" in cause else EventType.FRAGMENTATION
        evt = SpaceEvent(
            event_type=etype,
            epoch=epoch,
            name=r.get("object_name") or f"DISCOS-FRAG-{r.get('discos_id')}",
            description=(r.get("comment") or r.get("event_type") or "")[:1000],
            parent_norad=r.get("satno"),
            altitude_km=r.get("altitude_km"),
            mass_parent_kg=r.get("mass_kg"),
            source="DISCOS",
            source_id=str(r.get("discos_id")),
            raw={
                "eventType": r.get("event_type"),
                "xSectAvg": r.get("xsect_avg_m2"),
                "shape": r.get("shape"),
                "cataloguedFragments": r.get("catalogued_fragments"),
            },
        )
        try:
            upsert_event(evt)
            n += 1
        except Exception as exc:
            print(f"  upsert space_events fail {r.get('discos_id')}: {exc}")
    return n


# ── EsaLOG ───────────────────────────────────────────────────────────────────
def fetch_esalog(*, cache: Optional[Path] = None, use_cache: bool = True) -> list[dict]:
    """拉取 GEO 带初始轨道关联物体（质量 / RCS）。按 object id 去重，保留最接近 GEO 的轨道。"""
    cache_fp = (cache or CACHE_DEFAULT) / "discos_esalog_geo.json"
    if use_cache and cache_fp.exists() and cache_fp.stat().st_size > 100:
        print(f"  [cache] EsaLOG {cache_fp}")
        return json.loads(cache_fp.read_text(encoding="utf-8"))

    headers = discos_headers()
    best: dict[str, dict] = {}  # object_discos_id → row
    page = 0
    GEO_SMA = 42_164_000.0
    for payload in discos_paginate(
        "initial-orbits",
        headers=headers,
        params={"filter": GEO_SMA_FILTER, "include": "object"},
        page_size=100,
    ):
        page += 1
        inc = _included_map(payload)
        for it in payload.get("data") or []:
            oattr = it.get("attributes") or {}
            rel = ((it.get("relationships") or {}).get("object") or {}).get("data") or {}
            oid = str(rel.get("id") or "")
            if not oid:
                continue
            obj = (inc.get(oid) or {}).get("attributes") or {}
            sma = oattr.get("sma")
            try:
                sma_f = float(sma) if sma is not None else None
            except Exception:
                sma_f = None
            row = {
                "discos_id": oid,
                "satno": obj.get("satno"),
                "cospar_id": obj.get("cosparId"),
                "name": obj.get("name"),
                "object_class": obj.get("objectClass"),
                "mass_kg": obj.get("mass"),
                "shape": obj.get("shape"),
                "xsect_avg_m2": obj.get("xSectAvg"),
                "xsect_max_m2": obj.get("xSectMax"),
                "xsect_min_m2": obj.get("xSectMin"),
                "active": obj.get("active"),
                "pred_decay_date": obj.get("predDecayDate"),
                "orbit_epoch": oattr.get("epoch"),
                "sma_m": sma_f,
                "ecc": oattr.get("ecc"),
                "inc_deg": oattr.get("inc"),
                "raan_deg": oattr.get("raan"),
                "source": "DISCOSweb:/api/initial-orbits (GEO)",
            }
            prev = best.get(oid)
            if prev is None:
                best[oid] = row
            elif sma_f is not None and prev.get("sma_m") is not None:
                if abs(sma_f - GEO_SMA) < abs(float(prev["sma_m"]) - GEO_SMA):
                    best[oid] = row
            elif sma_f is not None and prev.get("sma_m") is None:
                best[oid] = row
        pag = (payload.get("meta") or {}).get("pagination") or {}
        print(
            f"  EsaLOG page {page}/{pag.get('totalPages', '?')}: "
            f"orbits scanned, unique objects={len(best)}",
            flush=True,
        )

    rows = list(best.values())
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    cache_fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  EsaLOG cached → {cache_fp} ({len(rows)})")
    return rows


def enrich_external_esa_discos(engine, esalog_rows: list[dict]) -> int:
    """用 EsaLOG 的质量/RCS 回填 external_esa_discos（按 satno 匹配，仅填空）。"""
    n = 0
    with engine.begin() as conn:
        for r in esalog_rows:
            satno = r.get("satno")
            if satno is None:
                continue
            res = conn.execute(text("""
                UPDATE external_esa_discos SET
                    mass = COALESCE(mass, :mass),
                    shape = COALESCE(shape, :shape),
                    "xSectAvg" = COALESCE("xSectAvg", :xa),
                    "xSectMax" = COALESCE("xSectMax", :xmax),
                    "xSectMin" = COALESCE("xSectMin", :xmin)
                WHERE satno = :satno
                  AND (mass IS NULL OR "xSectAvg" IS NULL)
            """), {
                "mass": r.get("mass_kg"),
                "shape": r.get("shape"),
                "xa": r.get("xsect_avg_m2"),
                "xmax": r.get("xsect_max_m2"),
                "xmin": r.get("xsect_min_m2"),
                "satno": int(satno),
            })
            n += res.rowcount or 0
    return n


def ingest(
    *,
    esalof: bool = True,
    esalog: bool = True,
    sync_events: bool = False,
    enrich_discos: bool = True,
    refresh: bool = False,
    cache: Optional[Path] = None,
) -> dict[str, int]:
    cache = cache or CACHE_DEFAULT
    cache.mkdir(parents=True, exist_ok=True)
    if refresh:
        for name in ("discos_esalof.json", "discos_esalog_geo.json"):
            fp = cache / name
            if fp.exists():
                fp.unlink()
                print(f"  refresh: removed {fp.name}")

    init_db()
    engine = get_engine()
    _apply_migration(engine)
    counts: dict[str, int] = {}

    if esalof:
        print("→ EsaLOF (fragmentations + objects) …")
        rows = fetch_esalof(cache=cache, use_cache=not refresh)
        counts["external_discos_esalof"] = _write_table(engine, "external_discos_esalof", rows)
        if sync_events:
            print("→ sync space_events …")
            counts["space_events_upserted"] = sync_esalof_to_space_events(rows)

    if esalog:
        print("→ EsaLOG (GEO initial-orbits + object mass/RCS) …")
        rows = fetch_esalog(cache=cache, use_cache=not refresh)
        counts["external_discos_esalog"] = _write_table(engine, "external_discos_esalog", rows)
        if enrich_discos and rows:
            print("→ enrich external_esa_discos (fill null mass/RCS) …")
            counts["external_esa_discos_enriched"] = enrich_external_esa_discos(engine, rows)

    return counts


def main():
    ap = argparse.ArgumentParser(description="DISCOSweb v2 EsaLOF / EsaLOG 摄入")
    ap.add_argument("--esalof-only", action="store_true")
    ap.add_argument("--esalog-only", action="store_true")
    ap.add_argument("--sync-events", action="store_true",
                    help="EsaLOF 同步写入 space_events")
    ap.add_argument("--no-enrich", action="store_true",
                    help="不回填 external_esa_discos")
    ap.add_argument("--refresh", action="store_true", help="忽略本地缓存，重新拉取 API")
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()
    esalof = not args.esalog_only
    esalog = not args.esalof_only
    if args.esalof_only:
        esalog = False
    if args.esalog_only:
        esalof = False
    counts = ingest(
        esalof=esalof,
        esalog=esalog,
        sync_events=args.sync_events,
        enrich_discos=not args.no_enrich,
        refresh=args.refresh,
        cache=args.cache_dir,
    )
    print("\n=== EsaLOF / EsaLOG 入库完成 ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
