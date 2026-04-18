"""LLM agent that answers questions about space debris using the PostGIS database.

System prompt gives the model schema awareness + query tools.
MCP tools (query_debris_in_region, predict_launch_collision_risk) are also
registered here as OpenAI function-call tools so they work inside the
Streamlit chat interface without a separate MCP transport layer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any

from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI
from sqlalchemy import text

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
)
from database.db import session_scope

log = logging.getLogger(__name__)

_client = OpenAI(
    api_key=OPENAI_API_KEY or "missing-key",
    base_url=OPENAI_BASE_URL,
    timeout=OPENAI_TIMEOUT,
    max_retries=OPENAI_MAX_RETRIES,
)

# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_debris_count",
            "description": (
                "Count space debris objects in a geographic region or altitude band. "
                "Returns count and sample object names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat_min":  {"type": "number", "description": "Min latitude  (degrees, -90..90)"},
                    "lat_max":  {"type": "number", "description": "Max latitude  (degrees, -90..90)"},
                    "lon_min":  {"type": "number", "description": "Min longitude (degrees, -180..180)"},
                    "lon_max":  {"type": "number", "description": "Max longitude (degrees, -180..180)"},
                    "alt_min_km": {"type": "number", "description": "Min altitude (km), optional"},
                    "alt_max_km": {"type": "number", "description": "Max altitude (km), optional"},
                    "object_type": {"type": "string",
                                    "description": "Filter: DEBRIS / PAYLOAD / ROCKET BODY / all"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_high_risk_conjunctions",
            "description": "Return the top conjunction (close-approach) events by collision probability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_probability": {"type": "number",
                                        "description": "Minimum Pc threshold (default 1e-6)"},
                    "limit": {"type": "integer", "description": "Max rows to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_catalog_stats",
            "description": "Return aggregate statistics about the catalog: total objects, breakdown by type/country.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a read-only SQL query against the space-debris PostGIS database. "
                "Tables: catalog_objects, gp_elements, trajectory_segments, "
                "launch_windows, collision_risks. "
                "Views: v_debris_density, v_high_risk_events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A SELECT query (no writes allowed)"},
                },
                "required": ["sql"],
            },
        },
    },
    # ── MCP Tool 1 ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "query_debris_in_region",
            "description": (
                "【MCP工具】在指定地理区域和高度范围内查找在轨空间碎片、载荷及火箭级。"
                "使用 PostGIS trajectory_segments 空间索引进行快速检索，"
                "返回目标列表（NORAD ID、名称、类型、轨道高度、倾角、近似距离）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat_deg":    {"type": "number", "description": "中心纬度（°，-90~90）"},
                    "lon_deg":    {"type": "number", "description": "中心经度（°，-180~180）"},
                    "radius_km":  {"type": "number", "description": "搜索半径（km），默认 500"},
                    "alt_min_km": {"type": "number", "description": "最低轨道高度（km），默认 0"},
                    "alt_max_km": {"type": "number", "description": "最高轨道高度（km），默认 2000"},
                    "object_type": {
                        "type": "string",
                        "enum": ["DEBRIS", "PAYLOAD", "ROCKET BODY", "ALL"],
                        "description": "目标类型过滤，默认 ALL",
                    },
                    "t_start_utc": {
                        "type": "string",
                        "description": "时间窗口起点 ISO-8601 UTC，默认为当前时间",
                    },
                    "hours":  {"type": "number", "description": "时间窗口长度（小时），默认 6"},
                    "limit":  {"type": "integer", "description": "返回最多 N 个目标，默认 50"},
                },
                "required": ["lat_deg", "lon_deg"],
            },
        },
    },
    # ── MCP Tool 2 ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "predict_launch_collision_risk",
            "description": (
                "【MCP工具】对指定发射任务进行碰撞概率预测。"
                "内部执行：6-DOF 重力转弯轨迹仿真 → 发射阶段检测 → "
                "PostGIS 候选筛选 → SGP4 碎片传播 → TCA 求解 → Foster 2-D Pc 积分。"
                "支持参数自动估算（如不提供方位角则默认正东）。"
                "若参数不完整，工具将在 assumed_params 字段说明所做假设。"
                "返回：每阶段风险摘要、最高风险合取事件、整体风险等级及中文建议。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle": {
                        "type": "string",
                        "description": (
                            "运载火箭名称或描述，可用预设：CZ-5B / Falcon9 / Ariane6。"
                            "输入其他名称时系统自动匹配最近预设。"
                        ),
                    },
                    "launch_lat_deg": {
                        "type": "number",
                        "description": "发射场纬度（°），常用：文昌=19.61，拜科努尔=45.92",
                    },
                    "launch_lon_deg": {
                        "type": "number",
                        "description": "发射场经度（°），常用：文昌=110.95，拜科努尔=63.34",
                    },
                    "launch_az_deg": {
                        "type": "number",
                        "description": "发射方位角（°，0=北，90=东）。省略时默认 90°（正东）",
                    },
                    "launch_utc": {
                        "type": "string",
                        "description": (
                            "发射时刻 ISO-8601 UTC，如 '2026-04-11T06:00:00Z'。"
                            "省略时默认明日 06:00 UTC。"
                            "注意：碎片 DB 覆盖约未来 3 天，超出范围候选数为 0。"
                        ),
                    },
                    "t_max_s": {
                        "type": "number",
                        "description": "仿真时长（秒，600~7200），默认 3600s（1 小时）",
                    },
                    "include_demo_threats": {
                        "type": "boolean",
                        "description": "是否注入演示合取事件（默认 True）",
                    },
                },
                "required": [],
            },
        },
    },
    # ── MCP Tool 3 ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_debris_reentry_forecast",
            "description": (
                "【MCP工具】预报即将再入大气层的空间目标。"
                "查询 catalog_objects 中已有 decay_date 或近地点过低的目标，"
                "返回预计再入时间、距今天数、轨道高度和目标类型。"
                "用于「未来30天哪些碎片会再入？」「LEO低轨哪些目标快衰减了？」"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead":  {"type": "number", "description": "预报窗口（天），默认 30"},
                    "alt_max_km":  {"type": "number", "description": "无确认再入日期时，近地点 ≤ 此高度视为待再入，默认 300 km"},
                    "object_type": {"type": "string", "enum": ["DEBRIS", "PAYLOAD", "ROCKET BODY", "ALL"], "description": "目标类型过滤，默认 ALL"},
                    "limit":       {"type": "integer", "description": "返回最多 N 个目标，默认 50"},
                },
                "required": [],
            },
        },
    },
    # ── MCP Tool 4 ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_object_tle",
            "description": (
                "【MCP工具】获取指定 NORAD 编号目标的最新 TLE 轨道根数。"
                "返回 TLE Line1/Line2 字符串、轨道历元、倾角、偏心率、平均运动、"
                "升交点赤经、近地点幅角、平近点角、B* 阻力系数。"
                "用于：「给我 ISS 的 TLE」「NORAD 12345 的轨道根数是什么？」"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "norad_cat_id": {"type": "integer", "description": "目标 NORAD 编号，如 25544（ISS）"},
                },
                "required": ["norad_cat_id"],
            },
        },
    },
    # ── MCP Tool 5 ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "query_debris_by_rcs",
            "description": (
                "【MCP工具】按雷达截面积（RCS）大小类别筛选空间目标。"
                "RCS 分三档：SMALL（<0.1 m²）、MEDIUM（0.1–1 m²）、LARGE（>1 m²）。"
                "可同时按高度和目标类型过滤。"
                "用于：「LEO 中有多少大型碎片？」「筛选 LARGE 级别目标用于载人任务避撞」"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rcs_sizes":   {
                        "type": "array",
                        "items": {"type": "string", "enum": ["SMALL", "MEDIUM", "LARGE"]},
                        "description": "RCS 大小类别列表，默认 ['SMALL','MEDIUM','LARGE']",
                    },
                    "alt_min_km":  {"type": "number", "description": "最低轨道高度（km），默认 0"},
                    "alt_max_km":  {"type": "number", "description": "最高轨道高度（km），默认 2000"},
                    "object_type": {"type": "string", "enum": ["DEBRIS", "PAYLOAD", "ROCKET BODY", "ALL"], "description": "目标类型，默认 ALL"},
                    "limit":       {"type": "integer", "description": "返回最多 N 个目标，默认 50"},
                },
                "required": [],
            },
        },
    },
    # ── MCP Tool 6 ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "forecast_conjunction_risk",
            "description": (
                "【MCP工具】基于 NASA ORDEM 3.1 通量模型 + 泊松蒙特卡洛，"
                "评估卫星在指定轨道上整个任务寿命期内的长期碰撞风险。"
                "返回：聚合碰撞概率 Pc_agg、年碰撞率、期望交会次数（均值/P95）、"
                "最近逼近距离（中位/P95）、ORDEM 碎片通量以及建议规避燃料量。"
                "用于：「5年内会有多少次小于2km的接近？」"
                "「800km太阳同步轨道的10年碰撞概率有多高？」"
                "「计划在2000km轨道运营5年，需要准备多少规避燃料？」"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "alt_km": {
                        "type": "number",
                        "description": "目标轨道高度（km），如 800 / 1200 / 2000",
                    },
                    "inc_deg": {
                        "type": "number",
                        "description": "轨道倾角（°），如 53（Starlink）/ 98（SSO）/ 28",
                    },
                    "mission_years": {
                        "type": "number",
                        "description": "任务寿命年数，默认 5 年",
                    },
                    "conjunction_km": {
                        "type": "number",
                        "description": "交会距离阈值（km），默认 2 km（小于此距离视为一次接近）",
                    },
                    "sat_area_m2": {
                        "type": "number",
                        "description": "卫星碰撞截面积（m²），默认 10 m²",
                    },
                    "band_km": {
                        "type": "number",
                        "description": "高度搜索带宽（km），默认 200 km",
                    },
                },
                "required": [],
            },
        },
    },
]

# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def _query_debris_count(
    lat_min: float = -90, lat_max: float = 90,
    lon_min: float = -180, lon_max: float = 180,
    alt_min_km: float | None = None,
    alt_max_km: float | None = None,
    object_type: str = "all",
) -> dict:
    """Count catalog objects matching altitude / type filters.

    NOTE: catalog_objects don't have a single ground-track position; the
    lat/lon arguments are accepted for API compatibility but are not applied
    (orbital objects are in continuous motion).  Altitude bounds filter via
    perigee_km / apogee_km columns.
    """
    conditions: list[str] = []
    params: dict[str, Any] = {}
    if alt_min_km is not None:
        conditions.append("co.perigee_km >= :alt_min_km")
        params["alt_min_km"] = float(alt_min_km)
    if alt_max_km is not None:
        conditions.append("co.apogee_km <= :alt_max_km")
        params["alt_max_km"] = float(alt_max_km)
    if object_type.lower() not in ("all", ""):
        # Sanitise: strip quotes, upper-case, compare exactly
        ot = object_type.upper().replace("'", "")
        conditions.append("co.object_type = :object_type")
        params["object_type"] = ot

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    try:
        with session_scope() as sess:
            cnt = sess.execute(text(
                f"SELECT COUNT(*) FROM v_unified_objects co {where}"
            ), params).scalar() or 0

            sample_rows = sess.execute(text(
                f"SELECT name FROM v_unified_objects co {where} "
                f"ORDER BY norad_cat_id LIMIT 5"
            ), params).fetchall()

        return {
            "count":   int(cnt),
            "samples": [r[0] for r in sample_rows if r[0]],
        }
    except Exception as exc:
        return {"error": str(exc), "count": 0, "samples": []}


def _query_high_risk(min_probability: float = 1e-6, limit: int = 10) -> list:
    sql = text("""
        SELECT launch_name, object_name, object_type,
               tca, miss_distance_km, probability, phase
        FROM v_high_risk_events
        WHERE probability >= :min_pc
        LIMIT :lim
    """)
    with session_scope() as sess:
        rows = sess.execute(sql, {"min_pc": min_probability, "lim": limit}).fetchall()
    return [dict(r._mapping) for r in rows]


def _query_catalog_stats() -> dict:
    sql = text("""
        SELECT
            object_type,
            COUNT(*) AS total,
            COUNT(CASE WHEN decay_date IS NOT NULL THEN 1 END) AS decayed
        FROM v_unified_objects
        GROUP BY object_type
        ORDER BY total DESC
    """)
    with session_scope() as sess:
        rows = sess.execute(sql).fetchall()
    stats = [dict(r._mapping) for r in rows]
    total = sum(r["total"] for r in stats)
    return {"total_objects": total, "by_type": stats}


_DANGEROUS_SQL_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COMMENT|VACUUM|ANALYZE|COPY|CALL|DO|SET|RESET|SHOW|BEGIN|START|"
    r"COMMIT|ROLLBACK|MERGE|EXECUTE|PREPARE|DEALLOCATE|LOCK|REINDEX|"
    r"CLUSTER|REFRESH|NOTIFY|LISTEN|UNLISTEN"
    r")\b",
    re.IGNORECASE,
)


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments to avoid keyword obfuscation."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n\r]*", " ", sql)
    return sql


def _validate_readonly_sql(sql: str) -> str | None:
    """
    Return error message if SQL is unsafe/non-readonly; otherwise None.

    Security policy:
    - Only single-statement SELECT / WITH query is allowed.
    - Reject all known write/DDL/transaction/session-control keywords.
    """
    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        return "SQL 不能为空"

    # Disallow multi-statements and block trailing injected command chains.
    if ";" in cleaned.rstrip(";"):
        return "只允许单条只读查询，不允许多语句"

    upper = cleaned.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return "只允许 SELECT / WITH 只读查询"

    m = _DANGEROUS_SQL_RE.search(cleaned)
    if m:
        return f"检测到禁止关键字: {m.group(1).upper()}"

    return None


def _run_sql(sql: str) -> list:
    err = _validate_readonly_sql(sql)
    if err:
        return [{"error": err}]

    with session_scope() as sess:
        try:
            # Transaction-level readonly: even if validator is bypassed,
            # PostgreSQL will still reject write/DDL operations.
            sess.execute(text("SET TRANSACTION READ ONLY"))
            rows = sess.execute(text(sql)).fetchmany(100)
            return [dict(r._mapping) for r in rows]
        except Exception as exc:
            return [{"error": str(exc)}]


def _call_mcp_query_debris_in_region(args: dict) -> dict:
    try:
        from ingestion.tools import query_debris_in_region
        return query_debris_in_region(**args)
    except Exception as exc:
        return {"error": str(exc), "objects": [], "count": 0}


def _call_mcp_predict_launch_risk(args: dict) -> dict:
    try:
        from ingestion.tools import predict_launch_collision_risk
        return predict_launch_collision_risk(**args)
    except Exception as exc:
        return {"error": f"预测工具调用失败: {exc}", "overall_risk": "UNKNOWN"}


def _call_mcp_reentry_forecast(args: dict) -> dict:
    try:
        from ingestion.tools import get_debris_reentry_forecast
        return get_debris_reentry_forecast(**args)
    except Exception as exc:
        return {"error": str(exc), "objects": [], "count": 0}


def _call_mcp_get_object_tle(args: dict) -> dict:
    try:
        from ingestion.tools import get_object_tle
        return get_object_tle(**args)
    except Exception as exc:
        return {"error": str(exc)}


def _call_mcp_query_by_rcs(args: dict) -> dict:
    try:
        from ingestion.tools import query_debris_by_rcs
        return query_debris_by_rcs(**args)
    except Exception as exc:
        return {"error": str(exc), "objects": [], "count": 0}


def _call_forecast_conjunction_risk(args: dict) -> dict:
    try:
        from mission_risk.mc_risk import fetch_debris_environment, run_monte_carlo
        alt_km        = float(args.get("alt_km", 800.0))
        inc_deg       = float(args.get("inc_deg", 53.0))
        mission_years = float(args.get("mission_years", 5.0))
        conjunction_km = float(args.get("conjunction_km", 2.0))
        sat_area_m2   = float(args.get("sat_area_m2", 10.0))
        band_km       = float(args.get("band_km", 200.0))

        env = fetch_debris_environment(alt_km=alt_km, inc_deg=inc_deg, band_km=band_km)
        report = run_monte_carlo(
            env=env,
            inc_deg=inc_deg,
            mission_years=mission_years,
            hbr_km=0.02,
            conjunction_km=conjunction_km,
            sigma_km=0.1,
            n_mc=2000,
            sat_area_m2=sat_area_m2,
        )
        # Rule-of-thumb: ~15 kg hydrazine per avoidance maneuver
        fuel_est_kg = round(report.n_conj_p95 * 15.0, 1)
        risk_label = (
            "极高（建议重新规划轨道）" if report.agg_pc_mean > 1e-3 else
            "高（需充足规避燃料储备）" if report.agg_pc_mean > 1e-4 else
            "中（建议常规碰撞监测）"   if report.agg_pc_mean > 1e-5 else
            "低（正常运营）"
        )
        return {
            "altitude_km": alt_km,
            "inclination_deg": inc_deg,
            "mission_years": mission_years,
            "conjunction_threshold_km": conjunction_km,
            "risk_level": risk_label,
            "ordem_flux_10cm_per_m2_yr": env.flux_10cm,
            "ordem_flux_1cm_per_m2_yr": env.flux_1cm,
            "pc_orbit_10cm": report.pc_orbit_10cm,
            "pc_orbit_1cm": report.pc_orbit_1cm,
            "agg_pc_mean": report.agg_pc_mean,
            "n_conj_mean": round(report.n_conj_mean, 2),
            "n_conj_p95": round(report.n_conj_p95, 1),
            "min_miss_p50_km": round(report.min_miss_p50_km, 3),
            "min_miss_p95_km": round(report.min_miss_p95_km, 3),
            "collision_rate_per_year": report.collision_rate_per_year,
            "lambda_conj_per_year": round(report.lambda_conj_per_year, 2),
            "estimated_avoidance_fuel_kg": fuel_est_kg,
            "db_objects_in_band": env.n_objects,
            "model": "NASA ORDEM 3.1 + Poisson Monte Carlo (n=2000)",
        }
    except Exception as exc:
        return {"error": str(exc)}


_TOOL_DISPATCH = {
    "query_debris_count":            lambda a: _query_debris_count(**a),
    "query_high_risk_conjunctions":  lambda a: _query_high_risk(**a),
    "query_catalog_stats":           lambda a: _query_catalog_stats(),
    "run_sql":                        lambda a: _run_sql(**a),
    "query_debris_in_region":        lambda a: _call_mcp_query_debris_in_region(a),
    "predict_launch_collision_risk": lambda a: _call_mcp_predict_launch_risk(a),
    "get_debris_reentry_forecast":   lambda a: _call_mcp_reentry_forecast(a),
    "get_object_tle":                lambda a: _call_mcp_get_object_tle(a),
    "query_debris_by_rcs":           lambda a: _call_mcp_query_by_rcs(a),
    "forecast_conjunction_risk":     lambda a: _call_forecast_conjunction_risk(a),
}

# ------------------------------------------------------------------
# Conversation
# ------------------------------------------------------------------

SYSTEM_PROMPT = """你是 SpaceDebrisBot，空间态势感知（SSA）领域的专业助手。

## 数据库（PostgreSQL 15 + PostGIS）
表与关键列：
- **v_unified_objects** — 多源融合物化视图(Space-Track+UCS+ESA DISCOS, 68K+目标): norad_cat_id, name, object_type('DEBRIS'|'PAYLOAD'|'ROCKET BODY'), country_code, launch_date, decay_date, inclination, perigee_km, apogee_km, period_min, rcs_size, ucs_purpose, ucs_users, esa_mass_kg, esa_cross_section_m2, primary_source
- **catalog_objects** — Space-Track原始目录(29K): norad_cat_id(PK), name, object_type, country_code, launch_date, decay_date, inclination, period_min, apogee_km, perigee_km, rcs_size
- **gp_elements** — TLE/GP均根数: norad_cat_id(FK), epoch, mean_motion, eccentricity, inclination, ra_of_asc_node, arg_of_pericenter, mean_anomaly, bstar, tle_line1, tle_line2
- **trajectory_segments** — SGP4传播轨迹: norad_cat_id(FK), t_start, t_end, geom_eci(LINESTRINGZ), geom_geo(LINESTRINGZ srid=4326)
- **launch_windows** — 发射窗口: name, launch_site, launch_time, vehicle_type, payload, orbit_target_alt_km, orbit_target_inc_deg
- **collision_risks** — 碰撞风险: launch_id(FK), norad_cat_id, tca, miss_distance_km, probability(Pc), phase
- **视图 v_debris_density** — 碎片密度聚合
- **视图 v_high_risk_events** — 高风险合取事件

## SQL 编写规范（重要）
- 数据库为 **PostgreSQL 15**；**禁止**在聚合函数内部使用 LIMIT（如 `array_agg(x ORDER BY y LIMIT 5)` 语法错误）
- 正确取前 N 条：`ARRAY(SELECT name FROM ... ORDER BY norad_cat_id LIMIT 5)` 或分步查询
- `object_type` 值全大写：`'DEBRIS'`、`'PAYLOAD'`、`'ROCKET BODY'`
- 轨道高度用 `perigee_km`（近地点）和 `apogee_km`（远地点）筛选

## MCP 工具
**query_debris_in_region**（空间筛选）
- 参数：lat_deg, lon_deg, radius_km, alt_min/max_km, object_type, t_start_utc, hours, limit
- 用于：「文昌上空有哪些碎片？」「搜索 LEO 低倾角区域的目标」

**predict_launch_collision_risk**（发射碰撞风险预测）
- 参数：vehicle（CZ-5B/Falcon9/Ariane6），launch_lat/lon_deg，launch_az_deg，launch_utc，t_max_s，include_demo_threats
- 缺省参数自动估算（方位角默认 90°，时刻默认明日 06:00 UTC）
- 用于：「预测明天从文昌发射长五B的碰撞风险」

**get_debris_reentry_forecast**（再入预报）
- 参数：days_ahead（天数），alt_max_km（近地点阈值），object_type，limit
- 用于：「未来30天哪些碎片会再入大气层？」「低轨哪些目标即将衰减？」

**get_object_tle**（获取 TLE 轨道根数）
- 参数：norad_cat_id（必填）
- 返回：TLE Line1/Line2、轨道历元、六根数（倾角/偏心率/平均运动/升交点赤经/近地点幅角/平近点角）
- 用于：「ISS 的 TLE 是什么？」「获取 NORAD 12345 的轨道根数」

**query_debris_by_rcs**（按 RCS 大小筛选）
- 参数：rcs_sizes（SMALL/MEDIUM/LARGE 列表），alt_min/max_km，object_type，limit
- 用于：「LEO 中有多少大型碎片？」「筛选 LARGE 级目标评估载人任务风险」

**forecast_conjunction_risk**（长期任务碰撞风险预测）⚡ 【优先使用】
- 参数：alt_km（轨道高度，默认800）、inc_deg（倾角，默认53）、mission_years（寿命年数，默认5）、conjunction_km（交会阈值 km，默认2）、sat_area_m2（卫星面积 m²，默认10）、band_km（高度带宽，默认200）
- **所有参数均可选，用户给出任何参数即填入，其余使用默认值**
- 基于 NASA ORDEM 3.1 通量模型 + 泊松蒙特卡洛（n=2000 次试验）
- 返回：聚合碰撞概率 Pc_agg、年碰撞率、交会次数（均值/P95）、最近逼近距离、建议规避燃料量 kg
- **凡用户询问以下任一问题，必须立即调用此工具，禁止猜测或拒绝：**
  - 「X年内会有多少次小于Ykm的接近」
  - 「XXXkm轨道碰撞概率」
  - 「任务寿命期内的碰撞风险」
  - 「需要多少规避燃料」
  - 「长期风险」「聚合概率」「Pc_agg」

## 出错处理（重要）
- 若工具返回含 `"error"` 字段的结果，**必须分析原因、修正后重新调用**，不要把原始错误直接告知用户
- 常见错误修正：SQL 语法错误 → 参照上方规范改写并用 run_sql 重试；字段不存在 → 核对表结构；数据库为空 → 告知用户并建议同步

## 回答规范
- 引用数字时附带单位；风险等级用 🔴🟠🟡🟢 表示
- 若数据库为空，明确说明；不得捏造数据

## 强制工具调用规则（最高优先级）
- **只要用户问到以下任何一种场景，必须调用 `forecast_conjunction_risk` 工具，不得以"缺少参数"为由拒绝——所有参数均有默认值**：
  - 长期碰撞风险 / 任务寿命期碰撞概率
  - 多少次接近 / 交会次数 / 进入 X km 范围的次数
  - 聚合碰撞概率 Pc_agg
  - 需要多少规避燃料
- 从用户描述中提取：高度→alt_km，倾角→inc_deg，年数→mission_years，距离阈值→conjunction_km，面积→sat_area_m2；未提供的用默认值
- 禁止说"缺少必要参数"，直接调用工具
"""


def chat(history: List[Dict], user_message: str) -> str:
    """Single-turn function: append user message, call model, handle tool use, return reply."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(8):      # max tool-call rounds (allows 2-3 retry cycles)
        try:
            response = _client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        except APITimeoutError as exc:
            log.warning("OpenAI request timed out: %s", exc)
            return (
                "请求 LLM 超时（APITimeout）。可能原因：\n"
                "1）模型或网关响应慢——可在 .env 增大 OPENAI_TIMEOUT（当前服务端为 "
                f"{OPENAI_TIMEOUT:g}s）；\n"
                "2）应用跑在 Docker/Colima 内，而 OPENAI_BASE_URL 是公司内网域名——"
                "容器里往往访问不到内网，需在宿主机直连 VPN/内网，或把网关换成容器可访问的地址。"
            )
        except APIConnectionError as exc:
            log.warning("OpenAI connection failed: %s", exc)
            return (
                "无法连接到 LLM 服务（连接错误）。请检查 OPENAI_BASE_URL 是否可从运行环境访问："
                "若在 Docker 中运行，内网网关常需额外路由或 host 网络；并确认本机 VPN/代理已连通。"
                f"\n详情：{exc}"
            )
        except APIStatusError as exc:
            log.warning("OpenAI API error: %s", exc)
            if getattr(exc, "status_code", None) == 404:
                return (
                    "LLM 网关返回 404：常见原因是 OPENAI_BASE_URL 缺少 **/v1**。"
                    "兼容 OpenAI 的网关应配置为 `https://你的域名/v1`（程序也会自动补全 `/v1`）。"
                    f"\n当前使用的 base_url：`{OPENAI_BASE_URL}`"
                )
            return f"LLM 接口错误（HTTP {getattr(exc, 'status_code', '?')}）：{exc}"

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        # Execute tool calls
        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            try:
                result = _TOOL_DISPATCH[fn](args)
            except Exception as exc:
                result = {"error": str(exc)}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return "（达到最大工具调用轮次，请简化问题或直接用 run_sql 工具查询）"
