"""Pure-Python tool implementations for space-debris analysis.

These functions are extracted from mcp/server.py so they can be imported
without triggering the fastmcp / mcp namespace collision that occurs when
agent/debris_agent.py does ``from mcp.server import …``.

Both functions have identical signatures and return values to the
@mcp.tool()-decorated versions in mcp/server.py; that module now delegates
to these functions.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 · query_debris_in_region
# ─────────────────────────────────────────────────────────────────────────────

def query_debris_in_region(
    lat_deg: float,
    lon_deg: float,
    radius_km: float = 500.0,
    alt_min_km: float = 0.0,
    alt_max_km: float = 2000.0,
    object_type: str = "ALL",
    t_start_utc: Optional[str] = None,
    hours: float = 6.0,
    limit: int = 50,
) -> dict:
    """
    Find tracked space objects (debris, payloads, rocket bodies) whose ground
    track passes within `radius_km` of the given point during a time window.

    Uses PostGIS trajectory_segments spatial index for fast lookup, then
    enriches results with orbital parameters from the catalog.

    Returns a structured dict with:
    - region: search parameters
    - count: total matches found (may exceed `limit`)
    - objects: list of matching objects with name, type, altitude, inclination
    - summary: human-readable one-line summary
    """
    from database.db import session_scope
    from sqlalchemy import text

    # Parse time window
    if t_start_utc:
        try:
            t_start = datetime.fromisoformat(t_start_utc.replace("Z", "+00:00"))
        except ValueError:
            t_start = datetime.now(timezone.utc)
    else:
        t_start = datetime.now(timezone.utc)

    t_end = t_start + timedelta(hours=hours)
    deg_r = radius_km / 111.0     # rough degrees

    ot = object_type.upper()
    type_filter = "" if ot == "ALL" \
                  else f" AND UPPER(co.object_type) = '{ot}'"
    alt_filter = (
        f" AND co.perigee_km >= {alt_min_km}"
        f" AND co.apogee_km  <= {alt_max_km + 500}"   # apogee ceiling with buffer
    )

    # Single query: window function returns total count alongside each row,
    # avoiding a second round-trip for the separate count_sql.
    sql = text(f"""
        SELECT
            co.norad_cat_id,
            co.name,
            co.object_type,
            co.country_code,
            co.perigee_km,
            co.apogee_km,
            ge.inclination,
            ge.eccentricity,
            ROUND(CAST(
                ST_Distance(
                    ST_MakePoint(:lon, :lat)::geography,
                    ST_Centroid(ts.geom_geo)::geography
                ) / 1000.0 AS numeric), 1)            AS approx_dist_km,
            COUNT(*) OVER ()                           AS total_count
        FROM (
            SELECT DISTINCT ON (ts.norad_cat_id)
                ts.norad_cat_id,
                ts.geom_geo
            FROM trajectory_segments ts
            JOIN catalog_objects co ON co.norad_cat_id = ts.norad_cat_id
            WHERE ts.t_start <= :t_end
              AND ts.t_end   >= :t_start
              AND ts.geom_geo && ST_Expand(
                    ST_MakePoint(:lon, :lat)::geography::geometry, :deg_r)
            {type_filter}
            {alt_filter}
        ) ts
        JOIN catalog_objects co ON co.norad_cat_id = ts.norad_cat_id
        JOIN gp_elements ge      ON ge.norad_cat_id = co.norad_cat_id
        ORDER BY approx_dist_km
        LIMIT :lim
    """)

    params = {
        "lat": lat_deg, "lon": lon_deg, "deg_r": deg_r,
        "t_start": t_start, "t_end": t_end, "lim": min(limit, 200),
    }

    try:
        with session_scope() as sess:
            rows = sess.execute(sql, params).fetchall()
    except Exception as exc:
        return {"error": str(exc), "objects": [], "count": 0}

    total = int(rows[0][9]) if rows else 0
    type_counts: dict = {}
    objects = []
    for r in rows:
        ot = r[2] or "UNKNOWN"
        type_counts[ot] = type_counts.get(ot, 0) + 1
        objects.append({
            "norad_cat_id": r[0],
            "name":         r[1] or f"NORAD-{r[0]}",
            "object_type":  ot,
            "country":      r[3] or "?",
            "perigee_km":   round(float(r[4]), 1) if r[4] is not None else None,
            "apogee_km":    round(float(r[5]), 1) if r[5] is not None else None,
            "inclination_deg": round(float(r[6]), 2) if r[6] is not None else None,
            "eccentricity": round(float(r[7]), 4)    if r[7] is not None else None,
            "approx_dist_km": float(r[8]) if r[8] is not None else None,
        })

    type_str = ", ".join(f"{v} {k}" for k, v in type_counts.items())
    summary = (
        f"在 ({lat_deg:.2f}°, {lon_deg:.2f}°) 半径 {radius_km:.0f} km、"
        f"高度 {alt_min_km:.0f}–{alt_max_km:.0f} km 范围内，"
        f"共检索到 {total} 个空间对象（{type_str}），"
        f"时间窗口 {t_start.strftime('%Y-%m-%d %H:%M')} – "
        f"{t_end.strftime('%H:%M')} UTC。"
    )

    return {
        "region": {
            "center_lat": lat_deg, "center_lon": lon_deg,
            "radius_km": radius_km,
            "alt_range_km": [alt_min_km, alt_max_km],
            "t_start_utc": t_start.isoformat(),
            "t_end_utc":   t_end.isoformat(),
        },
        "count":   total,
        "showing": len(objects),
        "objects": objects,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 · predict_launch_collision_risk
# ─────────────────────────────────────────────────────────────────────────────

_PRESET_VEHICLES = ["CZ-5B", "Falcon9", "Ariane6"]


def predict_launch_collision_risk(
    vehicle: str = "CZ-5B",
    launch_lat_deg: float = 19.61,
    launch_lon_deg: float = 110.95,
    launch_az_deg: Optional[float] = None,
    launch_utc: Optional[str] = None,
    t_max_s: float = 3600.0,
    include_demo_threats: bool = True,
) -> dict:
    """
    Simulate a rocket launch with 6-DOF physics and assess collision probability
    against the tracked debris catalog using the Foster (1992) algorithm.

    The tool:
    1. Selects or approximates the vehicle from presets.
    2. Runs a gravity-turn 6-DOF trajectory (J2 + USSA-76 atmosphere).
    3. Detects launch phases (ASCENT, PARKING_ORBIT, TRANSFER_BURN, POST_SEPARATION).
    4. For each phase: PostGIS spatial pre-filter -> SGP4 debris propagation ->
       TCA (Time of Closest Approach) via cubic-spline interpolation ->
       2-D Foster Pc integral on encounter plane.
    5. Returns structured per-phase and per-event results plus a natural-language
       risk assessment.

    Output fields:
    - assumed_params: what the tool inferred or defaulted for missing inputs
    - vehicle_used: preset name actually used
    - trajectory: altitude/velocity/mass at key milestones
    - phases[]: per-phase summary (duration, candidates, max_Pc, risk_level)
    - top_events[]: up to 10 highest-Pc conjunction events
    - overall_risk: GREEN/YELLOW/AMBER/RED + Pc value
    - recommendation: concise action text in Chinese
    - caveats: list of assumptions or limitations
    """
    from trajectory.rocketpy_sim import SimConfig, simulate, PRESETS
    from trajectory.launch_phases import detect_phases
    from lcola.fly_through import assess_launch_phases

    assumed: list[str] = []
    caveats: list[str] = []

    # ── vehicle selection ────────────────────────────────────────────────────
    vehicle_used = vehicle.strip()
    if vehicle_used not in PRESETS:
        # Fuzzy match
        for preset in _PRESET_VEHICLES:
            if preset.lower() in vehicle_used.lower() or vehicle_used.lower() in preset.lower():
                vehicle_used = preset
                assumed.append(f"运载火箭名称 '{vehicle}' 未在预设中，已自动匹配为 '{preset}'")
                break
        else:
            vehicle_used = "CZ-5B"
            assumed.append(
                f"运载火箭 '{vehicle}' 无法识别，已回退至默认预设 CZ-5B。"
                f"可用预设：{_PRESET_VEHICLES}"
            )

    # ── launch time ──────────────────────────────────────────────────────────
    if launch_utc:
        try:
            t0 = datetime.fromisoformat(launch_utc.replace("Z", "+00:00"))
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
        except ValueError:
            t0 = datetime.now(timezone.utc).replace(
                hour=6, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            assumed.append(f"发射时刻 '{launch_utc}' 解析失败，已使用默认值 {t0.isoformat()}")
    else:
        t0 = datetime.now(timezone.utc).replace(
            hour=6, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        assumed.append(f"未指定发射时刻，默认使用明日 06:00 UTC ({t0.strftime('%Y-%m-%d')})")

    # ── azimuth ──────────────────────────────────────────────────────────────
    az = launch_az_deg if launch_az_deg is not None else 90.0
    if launch_az_deg is None:
        inc_approx = abs(launch_lat_deg)
        assumed.append(
            f"未指定发射方位角，默认 90°（正东），"
            f"对应轨道倾角约 {inc_approx:.1f}°"
        )

    # ── clamp t_max ──────────────────────────────────────────────────────────
    t_max = max(600.0, min(float(t_max_s), 7200.0))
    if t_max != float(t_max_s):
        assumed.append(f"仿真时长限制在 600–7200s 内，已调整为 {t_max:.0f}s")

    # ── run 6-DOF simulation ─────────────────────────────────────────────────
    try:
        cfg = SimConfig(
            vehicle_name=vehicle_used,
            launch_lat_deg=float(launch_lat_deg),
            launch_lon_deg=float(launch_lon_deg),
            launch_az_deg=float(az),
            launch_utc=t0,
            t_max_s=t_max,
            dt_out_s=30.0,
            run_mc=False,   # skip MC for speed
        )
        result = simulate(cfg)
    except Exception as exc:
        return {
            "error": f"轨迹仿真失败: {exc}",
            "vehicle_used": vehicle_used,
            "assumed_params": assumed,
        }

    # ── detect phases ────────────────────────────────────────────────────────
    phases = detect_phases(
        result.nominal,
        t_meco1=result.t_meco1,
        t_meco2=result.t_meco2,
        t_payload_sep=result.t_payload_sep,
    )

    # ── trajectory milestones ────────────────────────────────────────────────
    nom = result.nominal
    import numpy as np
    traj_summary = {
        "liftoff_mass_t":   round(nom[0].mass_kg / 1000, 1),
        "final_alt_km":     round(nom[-1].alt_km, 1),
        "final_vel_kms":    round(float(np.linalg.norm(nom[-1].vel_eci)), 3),
        "max_alt_km":       round(max(p.alt_km for p in nom), 1),
        "sim_duration_s":   round(nom[-1].t_met_s, 1),
        "t_meco1_s":        round(result.t_meco1, 1) if result.t_meco1 else None,
        "t_meco2_s":        round(result.t_meco2, 1) if result.t_meco2 else None,
        "t_payload_sep_s":  round(result.t_payload_sep, 1) if result.t_payload_sep else None,
    }

    # ── assess collision risk ────────────────────────────────────────────────
    try:
        summaries = assess_launch_phases(
            phases, t0,
            hbr_km=0.02,
            default_sigma_km=1.5,
            fine_km=50.0,
            inject_demo=include_demo_threats,
        )
    except Exception as exc:
        caveats.append(f"碰撞风险评估遇到错误（{exc}），仅返回轨迹数据")
        summaries = []

    # ── phase results ────────────────────────────────────────────────────────
    _phase_cn = {
        "ASCENT":          "上升段",
        "PARKING_ORBIT":   "停泊轨道段",
        "TRANSFER_BURN":   "变轨推进段",
        "POST_SEPARATION": "分离后段",
    }
    _risk_cn = {
        "RED":    "🔴 红色（立即关注）",
        "AMBER":  "🟠 琥珀（载人任务关注）",
        "YELLOW": "🟡 黄色（持续监视）",
        "GREEN":  "🟢 绿色（安全）",
    }

    phase_results = []
    all_events = []
    for s in summaries:
        phase_results.append({
            "phase":         s.phase_name,
            "phase_cn":      _phase_cn.get(s.phase_name, s.phase_name),
            "t_start_met_s": round(s.t_start_met, 1),
            "t_end_met_s":   round(s.t_end_met, 1),
            "duration_s":    round(s.t_end_met - s.t_start_met, 1),
            "n_candidates":  s.n_candidates,
            "n_evaluated":   s.n_evaluated,
            "n_events":      len(s.events),
            "max_pc":        s.max_pc,
            "max_pc_sci":    f"{s.max_pc:.3e}" if s.max_pc > 0 else "0",
            "risk_level":    s.risk_level,
            "risk_level_cn": _risk_cn.get(s.risk_level, s.risk_level),
        })
        for ev in s.events:
            all_events.append({
                "phase":           s.phase_name,
                "phase_cn":        _phase_cn.get(s.phase_name, s.phase_name),
                "norad_cat_id":    ev.norad_cat_id,
                "object_name":     ev.object_name,
                "tca_utc":         ev.tca.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "miss_distance_km": round(ev.miss_distance_km, 3),
                "foster_pc":       ev.probability,
                "foster_pc_sci":   f"{ev.probability:.4e}",
                "v_rel_kms":       round(ev.v_rel_kms, 3),
                "risk_level":      ev.risk_level,
                "is_blackout":     ev.is_blackout,
                "is_demo":         ev.norad_cat_id < 0,
            })

    all_events.sort(key=lambda e: e["foster_pc"], reverse=True)
    top_events = all_events[:10]

    overall_max_pc = max((s.max_pc for s in summaries), default=0.0)
    if overall_max_pc >= 1e-5:
        overall_risk = "RED"
    elif overall_max_pc >= 1e-6:
        overall_risk = "AMBER"
    elif overall_max_pc >= 1e-7:
        overall_risk = "YELLOW"
    else:
        overall_risk = "GREEN"

    blackout_count = sum(1 for e in all_events if e["is_blackout"])

    # ── natural language recommendation ─────────────────────────────────────
    if overall_risk == "GREEN":
        recommendation = (
            "✅ 当前发射方案安全。"
            f"各阶段最大碰撞概率 Pc = {overall_max_pc:.2e}，"
            f"低于无人载荷门限（1e-5）和载人任务门限（1e-6）。"
            "无需采取规避措施。"
        )
    elif overall_risk == "YELLOW":
        recommendation = (
            "🟡 需持续监视。"
            f"最高 Pc = {overall_max_pc:.2e}（介于 1e-7 至 1e-6 之间）。"
            "建议在发射前 24 小时更新碎片目录数据并重新评估。"
        )
    elif overall_risk == "AMBER":
        recommendation = (
            "🟠 载人任务注意：最高 Pc = {:.2e}（≥ 1e-6，超过载人任务门限）。"
            "建议：① 调整发射时刻（±15 分钟）重新评估；"
            "② 针对最高风险目标考虑轨道规避机动（Δv < 0.5 m/s 通常足够）。"
        ).format(overall_max_pc)
    else:  # RED
        worst = top_events[0] if top_events else None
        obj_desc = f"目标 {worst['object_name']}（NORAD {worst['norad_cat_id']}）" if worst else "高风险目标"
        recommendation = (
            f"🔴 警告：{obj_desc} 在 {_phase_cn.get(top_events[0]['phase'] if top_events else '', '某阶段')} "
            f"造成 Pc = {overall_max_pc:.2e}（超过无人载荷门限 1e-5）。"
            f"该发射时刻存在 {blackout_count} 个禁发合取事件。"
            "建议：① 推迟发射至最近的安全窗口；"
            "② 或实施 LCOLA 飞越筛选以寻找最优发射时刻（±60 分钟扫描）。"
        )

    if include_demo_threats:
        caveats.append(
            "结果包含 🧪 DEMO 合成演示对象（norad_cat_id < 0），"
            "非真实碎片数据，仅供功能演示。设置 include_demo_threats=False 可关闭。"
        )

    return {
        "vehicle_used":    vehicle_used,
        "assumed_params":  assumed,
        "launch_config": {
            "site_lat":    launch_lat_deg,
            "site_lon":    launch_lon_deg,
            "azimuth_deg": az,
            "launch_utc":  t0.isoformat(),
        },
        "trajectory":      traj_summary,
        "phases":          phase_results,
        "top_events":      top_events,
        "overall_risk":    overall_risk,
        "overall_risk_cn": _risk_cn.get(overall_risk, overall_risk),
        "max_pc":          overall_max_pc,
        "max_pc_sci":      f"{overall_max_pc:.3e}",
        "blackout_count":  blackout_count,
        "recommendation":  recommendation,
        "caveats":         caveats,
    }
