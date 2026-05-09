"""Replay a space-event incremental zip into the local DB.

Usage
-----
    python3 scripts/apply_events_package.py data/event_packages/events_xxx.zip
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.db import init_db
from events.types import EventType, SpaceEvent
from events.crud import upsert_event

log = logging.getLogger("apply_events_package")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def _from_dict(d: dict) -> SpaceEvent:
    et_raw = d.get("event_type", "OTHER")
    try: et = EventType(et_raw)
    except ValueError: et = EventType.OTHER
    ep = d.get("epoch")
    if isinstance(ep, str):
        try: ep = datetime.fromisoformat(ep.replace("Z", "+00:00"))
        except Exception: ep = None
    return SpaceEvent(
        event_type      = et,
        epoch           = ep,
        name            = d.get("name", "") or "",
        description     = d.get("description", "") or "",
        parent_norad    = d.get("parent_norad"),
        secondary_norad = d.get("secondary_norad"),
        altitude_km     = d.get("altitude_km"),
        inclination_deg = d.get("inclination_deg"),
        energy_j        = d.get("energy_j"),
        energy_to_mass  = d.get("energy_to_mass"),
        mass_parent_kg  = d.get("mass_parent_kg"),
        mass_target_kg  = d.get("mass_target_kg"),
        miss_distance_km= d.get("miss_distance_km"),
        probability     = d.get("probability"),
        n_fragments_obs = d.get("n_fragments_obs"),
        source          = d.get("source", "package"),
        source_id       = d.get("source_id", ""),
        raw             = d.get("raw"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", help="Path to events_*.zip produced by ingest_events_incremental.py")
    args = ap.parse_args()

    if not os.path.exists(args.zip):
        log.error("文件不存在: %s", args.zip); sys.exit(1)
    init_db()

    with zipfile.ZipFile(args.zip, "r") as z:
        try:
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
            log.info("manifest: %s", manifest)
        except KeyError:
            log.warning("zip 无 manifest.json")

        try:
            data = z.read("events.jsonl").decode("utf-8")
        except KeyError:
            log.error("zip 缺少 events.jsonl"); sys.exit(2)

    n_ok = n_err = 0
    for line in data.splitlines():
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
            upsert_event(_from_dict(d))
            n_ok += 1
        except Exception as exc:
            n_err += 1
            log.warning("行入库失败: %s (%s)", exc, line[:120])
    log.info("应用完成: ok=%d, err=%d", n_ok, n_err)


if __name__ == "__main__":
    main()
