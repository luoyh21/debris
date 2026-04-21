"""FastAPI service exposing space-debris analysis tools as REST endpoints.

Run standalone:  uvicorn api.main:app --host 0.0.0.0 --port 8502
In Docker:       integrated via run.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional

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
         description="返回数据库中各类目标的总数、数据源统计等概况信息。")
def system_stats():
    from database.db import session_scope
    from sqlalchemy import text
    try:
        with session_scope() as sess:
            total = sess.execute(text("SELECT COUNT(*) FROM catalog_objects")).scalar() or 0
            types = sess.execute(text(
                "SELECT COALESCE(object_type,'UNKNOWN'), COUNT(*) "
                "FROM catalog_objects GROUP BY 1 ORDER BY 2 DESC"
            )).fetchall()
            gp_count = sess.execute(text("SELECT COUNT(*) FROM gp_elements")).scalar() or 0
            traj_count = sess.execute(text(
                "SELECT COUNT(*) FROM trajectory_segments"
            )).scalar() or 0
        return {
            "catalog_total": total,
            "by_type": {r[0]: r[1] for r in types},
            "gp_elements_count": gp_count,
            "trajectory_segments_count": traj_count,
        }
    except Exception as exc:
        return {"error": str(exc)}


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
