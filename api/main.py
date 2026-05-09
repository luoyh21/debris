"""FastAPI service exposing space-debris analysis tools as REST endpoints.

Run standalone:  uvicorn api.main:app --host 0.0.0.0 --port 8502
In Docker:       integrated via run.py
"""
from __future__ import annotations

import os, sys
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI(
    title="空间碎片监测系统 API",
    description=(
        "Space Debris Monitoring & Launch Collision Risk Assessment system API.\n\n"
        "提供区域碎片查询、发射碰撞风险预测、再入预报、TLE 检索和 RCS 筛选等接口。"
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── Pydantic models ──────────────────────────────────────────────────────────

class RegionQuery(BaseModel):
    lat_deg: float = Field(..., description="中心纬度 (°)", examples=[19.61])
    lon_deg: float = Field(..., description="中心经度 (°)", examples=[110.95])
    radius_km: float = Field(500.0, description="搜索半径 (km)")
    alt_min_km: float = Field(0.0, description="最低高度 (km)")
    alt_max_km: float = Field(2000.0, description="最高高度 (km)")
    object_type: str = Field("ALL", description="目标类型: ALL / DEBRIS / PAYLOAD / ROCKET BODY")
    t_start_utc: Optional[str] = Field(None, description="起始时间 ISO-8601，默认当前")
    hours: float = Field(6.0, description="时间窗口 (小时)")
    limit: int = Field(50, description="最大返回条数", ge=1, le=200)


class LaunchRiskQuery(BaseModel):
    vehicle: str = Field("CZ-5B", description="运载火箭型号", examples=["CZ-5B", "Falcon9", "Ariane6"])
    launch_lat_deg: float = Field(19.61, description="发射场纬度 (°)")
    launch_lon_deg: float = Field(110.95, description="发射场经度 (°)")
    launch_az_deg: Optional[float] = Field(None, description="发射方位角 (°)，默认 90°")
    launch_utc: Optional[str] = Field(None, description="发射时间 ISO-8601")
    t_max_s: float = Field(3600.0, description="仿真最长时间 (秒)", ge=600, le=7200)
    include_demo_threats: bool = Field(True, description="是否包含演示合成目标")


class ReentryQuery(BaseModel):
    days_ahead: float = Field(30.0, description="预报天数", ge=1, le=365)
    alt_max_km: float = Field(300.0, description="近地点阈值 (km)")
    object_type: str = Field("ALL", description="目标类型")
    limit: int = Field(50, ge=1, le=200)


class RCSQuery(BaseModel):
    rcs_sizes: Optional[list[str]] = Field(None, description="RCS 类别列表: SMALL / MEDIUM / LARGE")
    alt_min_km: float = Field(0.0, description="最低高度 (km)")
    alt_max_km: float = Field(2000.0, description="最高高度 (km)")
    object_type: str = Field("ALL", description="目标类型")
    limit: int = Field(50, ge=1, le=200)


# ── User-defined data source CRUD models ─────────────────────────────────────

class UserDataObjectIn(BaseModel):
    """One row written into a user-managed data source.

    Schema mirrors ``catalog_objects`` (Space-Track) so the rows merge
    seamlessly into the unified view; ``norad_cat_id`` + ``source`` form the
    primary key and are required.
    """
    norad_cat_id: int = Field(..., description="NORAD ID（必填）", examples=[990001])
    name:         Optional[str]   = None
    object_type:  Optional[str]   = Field(None, description="PAYLOAD / DEBRIS / ROCKET BODY / UNKNOWN")
    country_code: Optional[str]   = None
    launch_date:  Optional[str]   = Field(None, description="ISO date (YYYY-MM-DD)")
    decay_date:   Optional[str]   = None
    launch_site:  Optional[str]   = None
    inclination:  Optional[float] = None
    period_min:   Optional[float] = None
    apogee_km:    Optional[float] = None
    perigee_km:   Optional[float] = None
    rcs_size:     Optional[str]   = Field(None, description="SMALL / MEDIUM / LARGE")
    object_id:    Optional[str]   = Field(None, description="COSPAR / 国际编号")
    extra:        Optional[dict]  = Field(None, description="附加自定义字段（JSON）")


class DataSourceUpsert(BaseModel):
    source: str = Field(..., description="数据源名称；不存在时自动创建", examples=["myCustomSrc"])
    priority: int = Field(..., description="优先级（数字越小越优先，与 Space-Track=10 同序）", ge=1, le=9999)
    description: Optional[str] = None


class DataSourcePriorityIn(BaseModel):
    priority: int = Field(..., description="新优先级（数字越小越优先）", ge=1, le=9999)
    description: Optional[str] = None


# ── API endpoints ────────────────────────────────────────────────────────────

@app.post("/api/v1/debris/region", tags=["碎片查询"],
          summary="区域碎片查询",
          description="查找指定地理坐标周围一定范围内的在轨空间物体，支持 PostGIS 空间索引和 SGP4 实时传播两种策略。")
def query_region(q: RegionQuery):
    from ingestion.tools import query_debris_in_region
    return query_debris_in_region(
        lat_deg=q.lat_deg, lon_deg=q.lon_deg, radius_km=q.radius_km,
        alt_min_km=q.alt_min_km, alt_max_km=q.alt_max_km,
        object_type=q.object_type, t_start_utc=q.t_start_utc,
        hours=q.hours, limit=q.limit,
    )


@app.post("/api/v1/launch/risk", tags=["发射风险"],
          summary="发射碰撞风险预测",
          description="模拟火箭发射 6-DOF 轨迹，对各飞行阶段进行碎片碰撞概率（Pc）评估。使用 Foster (1992) 算法。")
def predict_risk(q: LaunchRiskQuery):
    from ingestion.tools import predict_launch_collision_risk
    return predict_launch_collision_risk(
        vehicle=q.vehicle, launch_lat_deg=q.launch_lat_deg,
        launch_lon_deg=q.launch_lon_deg, launch_az_deg=q.launch_az_deg,
        launch_utc=q.launch_utc, t_max_s=q.t_max_s,
        include_demo_threats=q.include_demo_threats,
    )


@app.post("/api/v1/debris/reentry", tags=["再入预报"],
          summary="再入预报",
          description="预测即将再入大气层的空间物体，基于 decay_date 和近地点高度筛选。")
def reentry_forecast(q: ReentryQuery):
    from ingestion.tools import get_debris_reentry_forecast
    return get_debris_reentry_forecast(
        days_ahead=q.days_ahead, alt_max_km=q.alt_max_km,
        object_type=q.object_type, limit=q.limit,
    )


@app.get("/api/v1/tle/{norad_cat_id}", tags=["TLE 查询"],
         summary="获取 TLE 轨道根数",
         description="根据 NORAD 编号检索最新 TLE (两行根数)，包含轨道参数和目录信息。")
def get_tle(norad_cat_id: int):
    from ingestion.tools import get_object_tle
    return get_object_tle(norad_cat_id=norad_cat_id)


@app.post("/api/v1/debris/rcs", tags=["RCS 筛选"],
          summary="按 RCS 大小筛选碎片",
          description="按雷达截面积 (RCS) 类别筛选空间物体：SMALL (<0.1 m²)、MEDIUM (0.1–1 m²)、LARGE (>1 m²)。")
def query_by_rcs(q: RCSQuery):
    from ingestion.tools import query_debris_by_rcs
    return query_debris_by_rcs(
        rcs_sizes=q.rcs_sizes, alt_min_km=q.alt_min_km,
        alt_max_km=q.alt_max_km, object_type=q.object_type,
        limit=q.limit,
    )


@app.get("/api/v1/stats", tags=["系统统计"],
         summary="系统概况统计",
         description="返回数据库中各类目标的总数、数据源统计等概况信息，"
                     "包括多源融合目录（Space-Track / UCS / ESA DISCOS）与"
                     "独立专题库（GCAT / UNOOSA / Asterank / NASA TechPort）的入库条数。")
def system_stats():
    from database.db import session_scope
    from database.system_snapshot import compute_system_snapshot
    try:
        with session_scope() as sess:
            data = compute_system_snapshot(sess)
        return data
    except Exception as exc:
        return {"error": str(exc)}


# ── Data source CRUD endpoints ───────────────────────────────────────────────

_USER_OBJECT_FIELDS = (
    "name", "object_type", "country_code", "launch_date", "decay_date",
    "launch_site", "inclination", "period_min", "apogee_km", "perigee_km",
    "rcs_size", "object_id",
)


def _refresh_unified_view_safe(sess) -> None:
    """Best-effort rebuild of v_unified_objects after a write."""
    try:
        from sqlalchemy import text as _t
        sess.execute(_t("REFRESH MATERIALIZED VIEW v_unified_objects"))
    except Exception:
        try:
            from scripts.create_unified_view import create as _c
            _c()
        except Exception:
            pass


@app.get("/api/v1/datasources", tags=["数据源管理"],
         summary="列出全部数据源及其优先级",
         description=(
             "返回所有已知数据源的优先级、是否用户自建、当前自建源行数。"
             "数字越小优先级越高；当不同源对同一 NORAD 编号都有数据时，"
             "v_unified_objects 在刷新后会保留优先级最高的那条。"
         ))
def list_datasources():
    from database.db import session_scope, init_db
    init_db()
    from sqlalchemy import text
    with session_scope() as sess:
        rows = sess.execute(text(
            "SELECT source, priority, description, is_user_defined, updated_at "
            "FROM datasource_priority ORDER BY priority"
        )).fetchall()
        counts: dict[str, int] = {}
        try:
            for r in sess.execute(text(
                "SELECT source, COUNT(*) FROM user_data_objects GROUP BY source"
            )).fetchall():
                counts[r[0]] = int(r[1])
        except Exception:
            pass
        return {
            "sources": [
                {
                    "source": r[0],
                    "priority": int(r[1]),
                    "description": r[2],
                    "is_user_defined": bool(r[3]),
                    "user_row_count": counts.get(r[0], 0),
                    "updated_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows
            ]
        }


@app.put("/api/v1/datasources/{source}/priority", tags=["数据源管理"],
         summary="修改 / 创建一个数据源的优先级",
         description=(
             "修改指定 source 的优先级（数字越小越优先）。\n\n"
             "若 source 不存在则会以 ``is_user_defined=true`` 创建新记录，\n"
             "**即使该源当前无任何数据**，方便提前规划合并策略；之后通过\n"
             "``POST /api/v1/datasources/{source}/objects`` 写入的数据将立刻\n"
             "按此优先级参与 v_unified_objects 去重。"
         ))
def set_datasource_priority(
    source: str = Path(..., description="数据源名称（区分大小写）"),
    body:  DataSourcePriorityIn = ...,
):
    from database.db import session_scope, init_db
    init_db()
    from sqlalchemy import text
    with session_scope() as sess:
        existed = sess.execute(text(
            "SELECT 1 FROM datasource_priority WHERE source=:s"
        ), {"s": source}).scalar()
        sess.execute(text(
            "INSERT INTO datasource_priority "
            "(source, priority, description, is_user_defined, updated_at) "
            "VALUES (:s,:p,:d, COALESCE((SELECT is_user_defined FROM "
            "datasource_priority WHERE source=:s), TRUE), NOW()) "
            "ON CONFLICT (source) DO UPDATE SET "
            "priority=EXCLUDED.priority, "
            "description=COALESCE(EXCLUDED.description, datasource_priority.description), "
            "updated_at=NOW()"
        ), {"s": source, "p": int(body.priority), "d": body.description})
        _refresh_unified_view_safe(sess)
    return {"source": source, "priority": int(body.priority),
            "created": not existed}


@app.post("/api/v1/datasources/{source}/objects", tags=["数据源管理"],
          summary="新增 / 修改自定义源中的一行（与 Space-Track 同字段）",
          description=(
              "向指定 ``source`` 写入或更新一条目标记录（字段与 Space-Track / "
              "catalog_objects 一致）；按 ``(source, norad_cat_id)`` 主键 upsert。\n\n"
              "如果 ``source`` 尚未在 datasource_priority 中登记，会自动以默认\n"
              "优先级 100 插入（之后可通过 PUT priority 调整）。"
          ))
def upsert_user_object(
    source: str = Path(...),
    body:   UserDataObjectIn = ...,
):
    from database.db import session_scope, init_db
    init_db()
    from sqlalchemy import text
    import datetime as _dt
    import json as _json

    cols = ["source", "norad_cat_id"] + list(_USER_OBJECT_FIELDS) + ["extra", "updated_at"]
    placeholders = ", ".join(f":{c}" for c in cols)
    update_set = ", ".join(
        f"{c}=EXCLUDED.{c}"
        for c in cols if c not in ("source", "norad_cat_id")
    )
    payload = body.model_dump()
    rec = {
        "source": source,
        "norad_cat_id": int(body.norad_cat_id),
        "extra": _json.dumps(payload.get("extra")) if payload.get("extra") else None,
        "updated_at": _dt.datetime.utcnow(),
    }
    for f in _USER_OBJECT_FIELDS:
        rec[f] = payload.get(f)

    with session_scope() as sess:
        sess.execute(text(
            "INSERT INTO datasource_priority (source, priority, description, "
            "is_user_defined, updated_at) VALUES (:s, 100, NULL, TRUE, NOW()) "
            "ON CONFLICT (source) DO NOTHING"
        ), {"s": source})
        sess.execute(text(
            f"INSERT INTO user_data_objects ({', '.join(cols)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (source, norad_cat_id) DO UPDATE SET {update_set}"
        ), rec)
        _refresh_unified_view_safe(sess)
    return {"source": source, "norad_cat_id": int(body.norad_cat_id), "ok": True}


@app.post("/api/v1/datasources/{source}/objects/bulk", tags=["数据源管理"],
          summary="批量写入自定义源",
          description="一次最多 500 行的批量 upsert（同 POST /objects）。")
def bulk_upsert_user_objects(
    source: str,
    rows: List[UserDataObjectIn],
):
    if not rows:
        return {"source": source, "rows_written": 0}
    if len(rows) > 500:
        raise HTTPException(status_code=413, detail="单次最多 500 行")
    from database.db import session_scope, init_db
    init_db()
    from sqlalchemy import text
    import datetime as _dt
    import json as _json

    cols = ["source", "norad_cat_id"] + list(_USER_OBJECT_FIELDS) + ["extra", "updated_at"]
    placeholders = ", ".join(f":{c}" for c in cols)
    update_set = ", ".join(
        f"{c}=EXCLUDED.{c}"
        for c in cols if c not in ("source", "norad_cat_id")
    )
    payloads = []
    for body in rows:
        payload = body.model_dump()
        rec = {
            "source": source,
            "norad_cat_id": int(body.norad_cat_id),
            "extra": _json.dumps(payload.get("extra")) if payload.get("extra") else None,
            "updated_at": _dt.datetime.utcnow(),
        }
        for f in _USER_OBJECT_FIELDS:
            rec[f] = payload.get(f)
        payloads.append(rec)

    with session_scope() as sess:
        sess.execute(text(
            "INSERT INTO datasource_priority (source, priority, description, "
            "is_user_defined, updated_at) VALUES (:s, 100, NULL, TRUE, NOW()) "
            "ON CONFLICT (source) DO NOTHING"
        ), {"s": source})
        sess.execute(
            text(
                f"INSERT INTO user_data_objects ({', '.join(cols)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT (source, norad_cat_id) DO UPDATE SET {update_set}"
            ),
            payloads,
        )
        _refresh_unified_view_safe(sess)
    return {"source": source, "rows_written": len(payloads)}


@app.delete("/api/v1/datasources/{source}/objects/{norad_cat_id}",
            tags=["数据源管理"],
            summary="删除自定义源中的某一行",
            description="按 ``(source, norad_cat_id)`` 删除自定义源的一行。")
def delete_user_object(source: str, norad_cat_id: int):
    from database.db import session_scope, init_db
    init_db()
    from sqlalchemy import text
    with session_scope() as sess:
        n = sess.execute(text(
            "DELETE FROM user_data_objects "
            "WHERE source=:s AND norad_cat_id=:n"
        ), {"s": source, "n": int(norad_cat_id)}).rowcount
        _refresh_unified_view_safe(sess)
    if not n:
        raise HTTPException(status_code=404, detail="row not found")
    return {"source": source, "norad_cat_id": int(norad_cat_id), "deleted": n}


@app.delete("/api/v1/datasources/{source}", tags=["数据源管理"],
            summary="删除整个自定义数据源",
            description=(
                "删除 ``source`` 在 user_data_objects 中的全部行，并从 "
                "datasource_priority 中摘除该源。\n\n"
                "**只允许删除 is_user_defined=true 的数据源**；试图删除"
                "Space-Track / UCS / ESA-DISCOS 等内置源会返回 403。"
            ))
def delete_datasource(source: str):
    from database.db import session_scope, init_db
    init_db()
    from sqlalchemy import text
    with session_scope() as sess:
        row = sess.execute(text(
            "SELECT is_user_defined FROM datasource_priority WHERE source=:s"
        ), {"s": source}).first()
        if row is None:
            raise HTTPException(status_code=404, detail="source not found")
        if not bool(row[0]):
            raise HTTPException(status_code=403,
                                detail="cannot delete built-in datasource")
        n = sess.execute(text(
            "DELETE FROM user_data_objects WHERE source=:s"
        ), {"s": source}).rowcount
        sess.execute(text(
            "DELETE FROM datasource_priority WHERE source=:s"
        ), {"s": source})
        _refresh_unified_view_safe(sess)
    return {"source": source, "rows_deleted": n}


# ── 太空事件管理（Space-Events） ─────────────────────────────────────────────


class SpaceEventIn(BaseModel):
    """User-facing space-event input — minimal subset, free-form ``raw``."""
    event_type:        str = Field(..., description="FRAGMENTATION/COLLISION/REENTRY/MANEUVER/CDM/OTHER")
    epoch:             datetime
    name:              Optional[str] = None
    description:       Optional[str] = None
    parent_norad:      Optional[int] = None
    secondary_norad:   Optional[int] = None
    altitude_km:       Optional[float] = None
    inclination_deg:   Optional[float] = None
    energy_j:          Optional[float] = None
    energy_to_mass:    Optional[float] = None
    mass_parent_kg:    Optional[float] = None
    mass_target_kg:    Optional[float] = None
    miss_distance_km:  Optional[float] = None
    probability:       Optional[float] = None
    n_fragments_obs:   Optional[int] = None
    source:            str = "manual"
    source_id:         Optional[str] = ""
    raw:               Optional[dict] = None


@app.get("/api/v1/events", tags=["space-events"])
def api_list_events(event_type: Optional[str] = None,
                    source: Optional[str] = None,
                    limit: int = 200):
    from events.types import EventType as _ET
    from events.crud import list_events
    et = None
    if event_type:
        try: et = _ET(event_type.upper())
        except ValueError:
            raise HTTPException(400, f"未知事件类型 {event_type!r}")
    rows = list_events(event_type=et, source=source, limit=int(limit))
    return {"count": len(rows),
            "events": [_event_to_dict(e) for e in rows]}


@app.get("/api/v1/events/{event_id}", tags=["space-events"])
def api_get_event(event_id: int):
    from events.crud import get_event
    evt = get_event(int(event_id))
    if evt is None:
        raise HTTPException(404, "事件不存在")
    return _event_to_dict(evt)


@app.post("/api/v1/events", tags=["space-events"])
def api_create_event(payload: SpaceEventIn):
    from events.types import EventType as _ET, SpaceEvent
    from events.crud import insert_event
    try: et = _ET(payload.event_type.upper())
    except ValueError:
        raise HTTPException(400, f"未知事件类型 {payload.event_type!r}")
    evt = SpaceEvent(
        event_type=et, epoch=payload.epoch,
        **{k: getattr(payload, k) for k in
           ("name","description","parent_norad","secondary_norad",
            "altitude_km","inclination_deg","energy_j","energy_to_mass",
            "mass_parent_kg","mass_target_kg","miss_distance_km","probability",
            "n_fragments_obs","source","source_id","raw") if getattr(payload, k) is not None}
    )
    new_id = insert_event(evt)
    return {"id": new_id, "ok": True}


@app.delete("/api/v1/events/{event_id}", tags=["space-events"])
def api_delete_event(event_id: int):
    from events.crud import delete_event
    if not delete_event(int(event_id)):
        raise HTTPException(404, "事件不存在")
    return {"id": event_id, "deleted": True}


@app.post("/api/v1/events/{event_id}/breakup-simulate", tags=["space-events"])
def api_breakup_simulate(event_id: int,
                          lc_min_m: float = 0.01,
                          lc_max_m: float = 1.0,
                          max_fragments: int = 2000,
                          seed: int = 42):
    from events.crud import get_event
    from events.nasa_sbm import simulate_breakup
    evt = get_event(int(event_id))
    if evt is None:
        raise HTTPException(404, "事件不存在")
    res = simulate_breakup(evt, lc_min_m=lc_min_m, lc_max_m=lc_max_m,
                            max_fragments=int(max_fragments), seed=int(seed))
    return {
        "event_id": event_id,
        "n_total": res.n_total,
        "n_tracked_ge_10cm": res.n_tracked_ge_10cm,
        "n_lethal_ge_1cm":   res.n_lethal_ge_1cm,
        "catastrophic":      res.catastrophic,
        "notes": res.notes,
        "fragments": [
            {"lc_m": fr.lc_m, "mass_kg": fr.mass_kg,
             "am_m2_per_kg": fr.am_m2_per_kg,
             "delta_v_kms": fr.delta_v_kms.tolist(),
             "r_eci_km":   fr.r_eci_km.tolist(),
             "v_eci_km_s": fr.v_eci_km_s.tolist(),
             "is_lethal":  fr.is_lethal,
             "is_tracked": fr.is_tracked}
            for fr in res.fragments[:500]
        ],
    }


@app.get("/api/v1/events/{event_id}/export", tags=["space-events"])
def api_export_event(event_id: int, format: str = "cdm"):
    """Export an event as CCSDS NDM (CDM/OPM/OCM/RDM)."""
    from events.crud import get_event
    from events import write_cdm, write_opm, write_rdm, write_ocm
    fmt = format.lower()
    evt = get_event(int(event_id))
    if evt is None:
        raise HTTPException(404, "事件不存在")
    if fmt == "cdm":   text = write_cdm(evt)
    elif fmt == "opm": text = write_opm(evt)
    elif fmt == "rdm": text = write_rdm(evt)
    elif fmt == "ocm": text = write_ocm(evt)
    else:
        raise HTTPException(400, "format ∈ cdm / opm / ocm / rdm")
    from fastapi.responses import Response
    return Response(content=text, media_type="text/plain",
                    headers={"Content-Disposition":
                             f"attachment; filename=event_{event_id}.{fmt}"})


@app.post("/api/v1/events/import", tags=["space-events"])
def api_import_event(payload: dict):
    """Import a CCSDS NDM (CDM/OPM/OEM/OCM/RDM) raw text body.

    Body: ``{"text": "<CCSDS-KVN body>"}``  (also accepts ``{"file_text": "..."}``)
    """
    from events import parse_ccsds_message
    from events.crud import upsert_event
    text = payload.get("text") or payload.get("file_text") or ""
    if not text.strip():
        raise HTTPException(400, "请求体需要 'text' 字段：CCSDS NDM KVN 文本")
    try:
        evt = parse_ccsds_message(text)
    except Exception as exc:
        raise HTTPException(400, f"CCSDS 解析失败: {exc}")
    new_id = upsert_event(evt)
    return {"id": new_id, "event_type": evt.event_type.value,
            "epoch": evt.epoch.isoformat(), "ok": True}


def _event_to_dict(evt) -> dict:
    return {
        "id":               evt.id,
        "event_type":       evt.event_type.value,
        "epoch":            evt.epoch.isoformat(),
        "name":             evt.name,
        "description":      evt.description,
        "parent_norad":     evt.parent_norad,
        "secondary_norad":  evt.secondary_norad,
        "altitude_km":      evt.altitude_km,
        "inclination_deg":  evt.inclination_deg,
        "energy_j":         evt.energy_j,
        "energy_to_mass":   evt.energy_to_mass,
        "mass_parent_kg":   evt.mass_parent_kg,
        "mass_target_kg":   evt.mass_target_kg,
        "miss_distance_km": evt.miss_distance_km,
        "probability":      evt.probability,
        "n_fragments_obs":  evt.n_fragments_obs,
        "source":           evt.source,
        "source_id":        evt.source_id,
        "raw":              evt.raw,
    }


# ── Documentation page ───────────────────────────────────────────────────────

_DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs_static")


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def docs_index():
    with open(os.path.join(_DOCS_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/docs/{path:path}", include_in_schema=False)
def docs_page(path: str):
    from fastapi.responses import Response
    fp = os.path.join(_DOCS_DIR, path)
    if os.path.isfile(fp):
        mime = "text/css" if fp.endswith(".css") else \
               "application/javascript" if fp.endswith(".js") else \
               "image/svg+xml" if fp.endswith(".svg") else "text/html"
        with open(fp, encoding="utf-8") as f:
            return Response(content=f.read(), media_type=mime)
    if os.path.isfile(fp + ".html"):
        with open(fp + ".html", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>404 - 页面不存在</h1>", status_code=404)


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")
