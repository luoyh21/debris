"""Apply an incremental-update snapshot zip (produced by
``scripts/ingest_incremental.py``) onto the local database.

Usage
-----
    python3 scripts/apply_incremental_package.py path/to/incremental_*.zip
    python3 scripts/apply_incremental_package.py path/to/incremental_*.zip \\
        --no-propagate            # skip Space-Track SGP4 segment regeneration
    python3 scripts/apply_incremental_package.py path/to/incremental_*.zip \\
        --refresh-views           # rebuild v_unified_objects after import

The zip contains one CSV per upserted (source, table) plus a ``manifest.json``
listing each file's primary key columns.  All operations are idempotent.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from database.db import get_engine, session_scope
from sqlalchemy import inspect, text

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    raw = zf.read(name)
    return pd.read_csv(io.BytesIO(raw))


def _upsert_dataframe(df: pd.DataFrame, table: str, pk_cols: list[str],
                      engine, label: str = "") -> int:
    """Same idempotent DELETE-by-PK + INSERT used by the producer side.

    Mirrors the implementation in ``ingest_incremental._upsert_dataframe``;
    duplicated here so this script is self-contained when shipped alone with
    a snapshot zip.
    """
    if df is None or df.empty:
        return 0
    insp = inspect(engine)
    if not insp.has_table(table):
        log.info("  %s · 目标表 %s 不存在，首次创建（%d 行）",
                 label or table, table, len(df))
        df.to_sql(table, engine, if_exists="replace", index=False,
                  method="multi", chunksize=400)
        return len(df)
    existing_cols = {c["name"] for c in insp.get_columns(table)}
    keep = [c for c in df.columns if c in existing_cols]
    if not keep:
        log.warning("  %s · 没有任何字段与目标表 %s 匹配，跳过",
                    label or table, table)
        return 0
    df = df[keep].copy()
    pk_cols = [c for c in pk_cols if c in df.columns]
    if not pk_cols:
        log.warning("  %s · 主键列不在数据中，回退到 append 模式（可能产生重复）",
                    label or table)
        df.to_sql(table, engine, if_exists="append", index=False,
                  method="multi", chunksize=400)
        return len(df)

    staging = f"_tmp_apply_{table}"
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
    log.info("  %s · %s rows %d → %d  (Δ %+d)",
             label or table, table, n_before, n_after, n_after - n_before)
    return len(df)


def apply_spacetrack_records(df: pd.DataFrame, *, propagate: bool,
                             horizon_days: int, seg_minutes: int) -> int:
    """Replay raw GP records (originating from spacetrack incremental) by
    feeding them through ingestion.ingest_gp helpers."""
    if df is None or df.empty:
        return 0
    from ingestion.ingest_gp import _upsert_catalog, _upsert_gp, _ingest_segments
    from database.db import init_db
    init_db()
    records = df.to_dict(orient="records")
    n = 0
    batch = 200
    for i in range(0, len(records), batch):
        with session_scope() as sess:
            for rec in records[i: i + batch]:
                _upsert_catalog(sess, rec)
                _upsert_gp(sess, rec)
                if propagate:
                    _ingest_segments(sess, rec, horizon_days, seg_minutes)
        n += len(records[i: i + batch])
        log.info("    Space-Track replay batch %d / %d  (累计 %d)",
                 (i // batch) + 1,
                 (len(records) + batch - 1) // batch, n)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply an incremental snapshot zip onto the database",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("zip_path", help="path to the incremental_*.zip produced "
                                     "by scripts/ingest_incremental.py")
    ap.add_argument("--no-propagate", action="store_true",
                    help="skip SGP4 trajectory regeneration on Space-Track replay")
    ap.add_argument("--horizon-days", type=int, default=3)
    ap.add_argument("--seg-minutes",  type=int, default=10)
    ap.add_argument("--refresh-views", action="store_true",
                    help="rebuild v_unified_objects after import")
    args = ap.parse_args()

    if not os.path.exists(args.zip_path):
        raise SystemExit(f"package not found: {args.zip_path}")

    engine = get_engine()
    with zipfile.ZipFile(args.zip_path, "r") as zf:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except KeyError:
            raise SystemExit("zip is missing manifest.json — not a valid "
                             "incremental snapshot")
        log.info("Applying snapshot: %s", os.path.basename(args.zip_path))
        log.info("  since=%s  generated_at=%s  files=%d",
                 manifest.get("since"), manifest.get("generated_at"),
                 len(manifest.get("files", [])))

        n_total = 0
        for entry in manifest.get("files", []):
            inner   = entry["path"]
            source  = entry["source"]
            table   = entry["table"]
            pk_cols = entry["pk_cols"]
            df = _read_csv_from_zip(zf, inner)
            log.info("→ %s :: %s  (%d rows, pk=%s)",
                     source, table, len(df), pk_cols)
            if source == "spacetrack" and table == "gp_records":
                n_total += apply_spacetrack_records(
                    df,
                    propagate=not args.no_propagate,
                    horizon_days=int(args.horizon_days),
                    seg_minutes=int(args.seg_minutes),
                )
            else:
                n_total += _upsert_dataframe(df, table, pk_cols, engine,
                                             label=source)

    log.info("Snapshot applied — %d rows written across %d tables",
             n_total, len(manifest.get("files", [])))

    if args.refresh_views:
        try:
            from scripts.create_unified_view import create as _create
        except Exception:
            from create_unified_view import create as _create  # type: ignore
        try:
            _create()
            log.info("v_unified_objects refreshed")
        except Exception as exc:
            log.warning("view refresh failed: %s", exc)


if __name__ == "__main__":
    main()
