"""Incremental space-event ingest with snapshot packaging.

Pulls new events from all configured sources (DISCOS / Space-Track CDM /
Space-Track Decay-TIP / CelesTrak SOCRATES / GCAT ecat) since a cutoff
``--since``, upserts them into ``space_events`` AND writes a self-contained
JSONL+manifest zip to ``data/event_packages/``.  The zip can be replayed on a
remote system via :mod:`scripts.apply_events_package`.

Usage
-----
    python3 scripts/ingest_events_incremental.py --since 2026-04-01
    python3 scripts/ingest_events_incremental.py --since 2026-04-01 \\
            --sources discos,cdm,socrates                     # 选择子集
    python3 scripts/ingest_events_incremental.py --since 2026-04-01 \\
            --max 1000 --no-upsert                              # 只打包

Output package layout (zip)::

    manifest.json            run metadata + per-source counts
    events.jsonl             one event per line, full SpaceEvent dataclass
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import io
import json
import logging
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

from events.types import EventType, SpaceEvent
from events.crud import upsert_event
from database.db import init_db
from scripts.ingest_events import ALL_FETCHERS, _parse_iso

log = logging.getLogger("events_incremental")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "event_packages")
os.makedirs(PKG_DIR, exist_ok=True)


def parse_since(s: str) -> dt.datetime:
    out = _parse_iso(s)
    if out is None:
        raise ValueError(f"无法解析 --since: {s!r}")
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out


def _serialize(evt: SpaceEvent) -> dict:
    d = dataclasses.asdict(evt)
    d["event_type"] = evt.event_type.value
    if evt.epoch is not None:
        d["epoch"] = evt.epoch.isoformat()
    return d


def main() -> str | None:
    """Run one cycle.  Returns the zip path produced (or ``None``)."""
    ap = argparse.ArgumentParser(description="增量太空事件拉取 + 压缩包打包")
    ap.add_argument("--since", required=True, help="ISO 起点时间")
    ap.add_argument("--max", type=int, default=1000, help="每个源最多记录数")
    ap.add_argument("--sources", default="all",
                    help=f"逗号分隔的源（{','.join(ALL_FETCHERS)}）或 all")
    ap.add_argument("--no-upsert", action="store_true",
                    help="只生成压缩包，不写入数据库")
    ap.add_argument("--out-dir", default=PKG_DIR)
    args = ap.parse_args()

    since = parse_since(args.since)
    if args.sources == "all":
        sources = list(ALL_FETCHERS.keys())
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    invalid = [s for s in sources if s not in ALL_FETCHERS]
    if invalid:
        log.error("未知数据源: %s; 可用: %s", invalid, list(ALL_FETCHERS))
        return None

    init_db()
    counts: dict[str, int] = {}
    all_events: list[SpaceEvent] = []
    for s in sources:
        items = ALL_FETCHERS[s](max_rows=args.max, since=since)
        counts[s] = len(items)
        all_events.extend(items)

    n_ok = 0
    if not args.no_upsert:
        for evt in all_events:
            try:
                upsert_event(evt); n_ok += 1
            except Exception as exc:
                log.warning("upsert failed (%s/%s): %s",
                            evt.source, evt.source_id, exc)
        log.info("upsert 完成: %d / %d", n_ok, len(all_events))
    counts["upserted"] = n_ok
    counts["total"] = len(all_events)

    if not all_events:
        log.info("无新增事件，未生成压缩包")
        return None

    run_at = dt.datetime.now(dt.timezone.utc)
    fname = (f"events_{since:%Y%m%dT%H%M%S}_{run_at:%Y%m%dT%H%M%S}.zip")
    os.makedirs(args.out_dir, exist_ok=True)
    zpath = os.path.join(args.out_dir, fname)

    manifest = {
        "kind": "space-events-incremental",
        "since": since.isoformat(),
        "generated_at": run_at.isoformat(),
        "sources": sources,
        "counts": counts,
    }
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",
                   json.dumps(manifest, indent=2, ensure_ascii=False,
                              default=str))
        buf = io.StringIO()
        for evt in all_events:
            buf.write(json.dumps(_serialize(evt), ensure_ascii=False,
                                 default=str) + "\n")
        z.writestr("events.jsonl", buf.getvalue())

    log.info("生成压缩包: %s (%.2f KB, %d 条事件)",
             zpath, os.path.getsize(zpath) / 1024.0, len(all_events))
    return zpath


if __name__ == "__main__":
    main()
