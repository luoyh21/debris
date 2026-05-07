"""单一数据源：系统概况 / 多源摄入统计 / API `/api/v1/stats`。

所有数值均为查询数据库时的实时结果（不做持久化缓存）。
Streamlit 可对封装函数使用短时 `@st.cache_data`；静态文档页通过浏览器请求同一 API 刷新。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

# ── SQL（与 Streamlit 概览「摄入状态」逻辑一致）───────────────────────────────

_SQL_UNIFIED_AGG = """
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN object_type = 'PAYLOAD' THEN 1 ELSE 0 END) AS payloads,
        SUM(CASE WHEN object_type = 'DEBRIS' THEN 1 ELSE 0 END) AS debris,
        SUM(CASE WHEN object_type = 'ROCKET BODY' THEN 1 ELSE 0 END) AS rockets,
        COUNT(DISTINCT primary_source) AS n_primary_sources
    FROM v_unified_objects
"""

_SQL_INGESTION_BY_TYPE = """
    SELECT 'Space-Track' AS src, COALESCE(object_type,'UNKNOWN') AS obj_type,
           COUNT(*) AS cnt
    FROM catalog_objects GROUP BY 1, 2
    UNION ALL
    SELECT 'UCS', 'PAYLOAD', COUNT(*)
    FROM external_ucs_satellites
    UNION ALL
    SELECT 'ESA-DISCOS',
        CASE
            WHEN "objectClass" IN ('Payload','Payload Mission Related Object')
                THEN 'PAYLOAD'
            WHEN "objectClass" = 'Rocket Body'
                THEN 'ROCKET BODY'
            WHEN "objectClass" IN ('Payload Fragmentation Debris',
                 'Rocket Fragmentation Debris','Payload Debris','Rocket Debris',
                 'Rocket Mission Related Object','Other Debris')
                THEN 'DEBRIS'
            ELSE 'UNKNOWN'
        END,
        COUNT(*)
    FROM external_esa_discos GROUP BY 2
    ORDER BY 1, 3 DESC
"""

_SQL_INGESTION_SOURCE_TOTALS = """
    SELECT 'Space-Track' AS src, COUNT(*) AS cnt FROM catalog_objects
    UNION ALL SELECT 'UCS', COUNT(*) FROM external_ucs_satellites
    UNION ALL SELECT 'ESA-DISCOS', COUNT(*) FROM external_esa_discos
"""

_EXTERNAL_TABLES: tuple[tuple[str, str], ...] = (
    ("ucs", "external_ucs_satellites"),
    ("esa_discos", "external_esa_discos"),
    ("gcat_yearly_launches", "external_yearly_launches"),
    ("gcat_cumulative_onorbit", "external_cumulative_onorbit"),
    ("gcat_country_yearly_payload", "external_country_yearly_payload"),
    ("gcat_onorbit_snapshot", "external_onorbit_snapshot"),
    ("unoosa", "external_unoosa_launches"),
    ("asterank", "external_asterank"),
    ("techport", "external_techport"),
)


def _safe_count(sess: Session, table: str) -> int:
    try:
        return int(sess.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0)
    except Exception:
        return 0


def _row_counts_external(sess: Session) -> dict[str, int]:
    return {key: _safe_count(sess, tbl) for key, tbl in _EXTERNAL_TABLES}


def compute_system_snapshot(sess: Session) -> dict[str, Any]:
    """在同一 Session 内聚合全部指标；供 REST API 与 Streamlit 共用。"""
    generated_at = datetime.now(timezone.utc).isoformat()

    # ── Space-Track 原始目录 ────────────────────────────────────────────
    cat_total = _safe_count(sess, "catalog_objects")
    try:
        rows = sess.execute(text(
            "SELECT COALESCE(object_type,'UNKNOWN'), COUNT(*) "
            "FROM catalog_objects GROUP BY 1 ORDER BY 2 DESC"
        )).fetchall()
        catalog_by_type = {str(r[0]): int(r[1]) for r in rows}
    except Exception:
        catalog_by_type = {}

    gp_count = _safe_count(sess, "gp_elements")
    traj_count = _safe_count(sess, "trajectory_segments")

    external_sources = _row_counts_external(sess)

    # ── 融合视图 v_unified_objects ─────────────────────────────────────
    unified_total = 0
    unified_by_type: dict[str, int] = {}
    n_primary_sources = 0
    payloads = debris = rockets = 0
    try:
        urow = sess.execute(text(_SQL_UNIFIED_AGG)).mappings().first()
        if urow:
            unified_total = int(urow["total"] or 0)
            payloads = int(urow["payloads"] or 0)
            debris = int(urow["debris"] or 0)
            rockets = int(urow["rockets"] or 0)
            n_primary_sources = int(urow["n_primary_sources"] or 0)
        trows = sess.execute(text(
            "SELECT COALESCE(object_type,'UNKNOWN'), COUNT(*) "
            "FROM v_unified_objects GROUP BY 1 ORDER BY 2 DESC"
        )).fetchall()
        unified_by_type = {str(r[0]): int(r[1]) for r in trows}
    except Exception:
        pass

    # ── 摄入状态明细（原始表行数 × 类型），供前端透视 ─────────────────────
    ingestion_typed_rows: list[dict[str, Any]] = []
    try:
        for r in sess.execute(text(_SQL_INGESTION_BY_TYPE)).fetchall():
            ingestion_typed_rows.append({
                "数据源": r[0],
                "目标类型": r[1],
                "数量": int(r[2]),
            })
    except Exception:
        pass

    ingestion_source_totals: dict[str, int] = {}
    try:
        for r in sess.execute(text(_SQL_INGESTION_SOURCE_TOTALS)).fetchall():
            ingestion_source_totals[str(r[0])] = int(r[1])
    except Exception:
        pass

    # 三个轨道目录源的原始行合计（不等于融合去重条数）
    raw_catalog_rows_sum = sum(
        ingestion_source_totals.get(k, 0)
        for k in ("Space-Track", "UCS", "ESA-DISCOS")
    )

    return {
        "generated_at_utc": generated_at,
        # 兼容旧版 API 字段名
        "catalog_total": cat_total,
        "by_type": catalog_by_type,
        "gp_elements_count": gp_count,
        "trajectory_segments_count": traj_count,
        "unified_total": unified_total,
        "unified_primary_sources": n_primary_sources,
        "external_sources": external_sources,
        # 扩展（单一事实来源）
        "unified_by_type": unified_by_type,
        "unified_payloads": payloads,
        "unified_debris": debris,
        "unified_rocket_bodies": rockets,
        "ingestion_typed_rows": ingestion_typed_rows,
        "ingestion_source_totals": ingestion_source_totals,
        "raw_catalog_rows_space_ucs_esa_sum": raw_catalog_rows_sum,
    }
