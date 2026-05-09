"""Streamlit page: 太空事件管理 / Space Event Manager.

Top-level tabs
--------------
1. **事件列表** — table view + filters + delete + export buttons
2. **手动添加** — form to insert a new event (carbon-copy of API model)
3. **NASA SBM 解体模拟** — pick a fragmentation/collision event and run
   the Standard Breakup Model; view debris cloud in 3D + size/AM hist
4. **CCSDS 导入** — paste / upload CDM, OPM, OEM, OCM or RDM to ingest
5. **数据源拉取** — 一键 DISCOS / Space-Track CDM-Decay / SOCRATES / GCAT
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .nav_icons import title_row, section_title
except Exception:                                                                 # pragma: no cover
    from streamlit_app.nav_icons import title_row, section_title

from events import (
    SpaceEvent, EventType, simulate_breakup,
    parse_ccsds_message, write_cdm, write_opm, write_rdm, write_ocm,
)
from events.crud import (
    insert_event, list_events, get_event, delete_event, count_events,
    upsert_event,
)


_TYPES_CN = {
    "FRAGMENTATION": "解体（爆炸）",
    "COLLISION":     "碰撞",
    "REENTRY":       "再入",
    "MANEUVER":      "机动",
    "CDM":           "合取预警",
    "OTHER":         "其他",
}


# ─── helpers ────────────────────────────────────────────────────────────────

def _events_to_df(events: list[SpaceEvent]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append({
            "ID":          e.id,
            "类型":        _TYPES_CN.get(e.event_type.value, e.event_type.value),
            "epoch (UTC)": e.epoch.strftime("%Y-%m-%d %H:%M:%S"),
            "名称":        e.name,
            "主体 NORAD":  e.parent_norad,
            "次体 NORAD":  e.secondary_norad,
            "高度 (km)":   None if e.altitude_km is None else round(e.altitude_km, 1),
            "Pc":          None if e.probability is None
                            else f"{e.probability:.2e}",
            "错失 (km)":   None if e.miss_distance_km is None
                            else round(e.miss_distance_km, 3),
            "观测碎片":    e.n_fragments_obs,
            "源":          e.source,
        })
    return pd.DataFrame(rows)


def _add_earth(fig: go.Figure, R: float = 6378.137, opacity: float = 0.18) -> None:
    u, v = np.linspace(0, 2*np.pi, 36), np.linspace(0, np.pi, 18)
    uu, vv = np.meshgrid(u, v)
    fig.add_trace(go.Surface(
        x=R*np.cos(uu)*np.sin(vv),
        y=R*np.sin(uu)*np.sin(vv),
        z=R*np.cos(vv),
        showscale=False, opacity=opacity, hoverinfo="skip",
        colorscale=[[0, "#0e2a47"], [1, "#0e2a47"]],
    ))


def _debris_cloud_fig(result, propagate_seconds: float = 600.0) -> go.Figure:
    """3D debris cloud after `propagate_seconds` of two-body propagation."""
    MU = 398600.4418
    fragments = result.fragments
    if not fragments:
        return go.Figure()

    # Simple Eulerian step ≈ TwoBody (good for short windows)
    pts = []
    for fr in fragments:
        r = fr.r_eci_km + fr.v_eci_km_s * propagate_seconds
        pts.append(r)
    P = np.asarray(pts)

    sizes = np.array([fr.lc_m for fr in fragments])
    is_lethal = np.array([fr.is_lethal for fr in fragments])
    is_tracked = np.array([fr.is_tracked for fr in fragments])

    fig = go.Figure()
    _add_earth(fig)
    fig.add_trace(go.Scatter3d(
        x=P[:,0], y=P[:,1], z=P[:,2],
        mode="markers",
        marker=dict(
            size=np.clip(2 + np.log10(sizes + 1e-3) + 4, 1, 6),
            color=np.where(is_tracked, "#ef4444",
                   np.where(is_lethal, "#f59e0b", "#94a3b8")),
            opacity=0.75,
        ),
        text=[f"Lc={fr.lc_m*100:.1f} cm  m={fr.mass_kg:.2f} kg  A/M={fr.am_m2_per_kg:.2f}"
              for fr in fragments],
        hoverinfo="text",
        name=f"碎片 ({len(fragments)} 个)",
    ))
    # Mark parent body
    fig.add_trace(go.Scatter3d(
        x=[fragments[0].r_eci_km[0]], y=[fragments[0].r_eci_km[1]],
        z=[fragments[0].r_eci_km[2]],
        mode="markers+text", marker=dict(size=10, color="#22d3ee"),
        text=["母体"], textposition="top center",
    ))
    fig.update_layout(
        title=f"NASA SBM 碎片云  (传播 {propagate_seconds:.0f} s 后)",
        height=560, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis_title="X ECI (km)", yaxis_title="Y ECI (km)",
                    zaxis_title="Z ECI (km)", aspectmode="data",
                    bgcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.05),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _size_distribution_fig(result) -> go.Figure:
    fig = go.Figure()
    if not result.fragments:
        return fig
    lc_cm = np.array([fr.lc_m for fr in result.fragments]) * 100
    fig.add_trace(go.Histogram(
        x=np.log10(lc_cm),
        nbinsx=40,
        marker_color="#22d3ee",
        name="抽样碎片",
    ))
    fig.update_layout(
        title="碎片特征长度 Lc 分布",
        xaxis_title="log₁₀(Lc / cm)",
        yaxis_title="计数",
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── main entry ─────────────────────────────────────────────────────────────

def render_events_page() -> None:
    st.markdown(title_row("events", "太空事件管理"), unsafe_allow_html=True)
    st.caption(
        "管理在轨事件（解体 / 碰撞 / 再入 / 机动 / CDM 预警），融合 ESA DISCOS、Space-Track 公开事件与本地手填记录。"
        "支持 **NASA Standard Breakup Model** 解体仿真和 **CCSDS CDM / OPM / OEM / OCM / RDM** 标准导入导出。"
    )

    n_events = count_events()
    st.info(f"💾 当前数据库已收录 **{n_events:,}** 条空间事件。")

    tab_list, tab_add, tab_sbm, tab_io, tab_ingest = st.tabs([
        "事件列表", "手动添加", "NASA SBM 模拟", "CCSDS 导入 / 导出", "数据源拉取",
    ])

    # ── Tab 1 ──────────────────────────────────────────────────────────────
    with tab_list:
        with st.expander("筛选", expanded=True):
            c1, c2, c3 = st.columns(3)
            type_pick = c1.selectbox("类型",
                ["全部"] + list(_TYPES_CN.keys()), key="ev_type")
            src_pick  = c2.selectbox("来源",
                ["全部", "manual", "DISCOS", "SPACETRACK", "CCSDS-IMPORT", "SBM"],
                key="ev_src")
            limit     = c3.number_input("最多显示", value=200, min_value=10,
                                         max_value=2000, step=50, key="ev_lim")

        et = None if type_pick == "全部" else EventType(type_pick)
        src = None if src_pick == "全部" else src_pick
        events = list_events(event_type=et, source=src, limit=int(limit))

        df = _events_to_df(events)
        st.dataframe(df, use_container_width=True, height=420, hide_index=True)

        if events:
            colL, colR = st.columns([3, 1])
            sel_id = colL.selectbox(
                "选定事件 ID（用于详情 / 导出 / 删除）",
                [e.id for e in events], key="ev_sel",
                format_func=lambda x: f"#{x}  {next((e.name for e in events if e.id==x),'')}",
            )
            if colR.button("🗑 删除选中", use_container_width=True, key="ev_del"):
                if delete_event(int(sel_id)):
                    st.success(f"已删除事件 #{sel_id}")
                    st.rerun()
                else:
                    st.error("删除失败")
            evt = next((e for e in events if e.id == sel_id), None)
            if evt:
                with st.expander("事件详情 + CCSDS 导出", expanded=False):
                    st.json({
                        "id": evt.id,
                        "type": evt.event_type.value,
                        "epoch": evt.epoch.isoformat(),
                        "parent_norad": evt.parent_norad,
                        "secondary_norad": evt.secondary_norad,
                        "altitude_km": evt.altitude_km,
                        "energy_to_mass": evt.energy_to_mass,
                        "miss_distance_km": evt.miss_distance_km,
                        "probability": evt.probability,
                        "source": evt.source,
                        "raw": evt.raw,
                    })
                    fmt = st.radio("导出格式", ["CDM", "OPM", "OCM", "RDM"],
                                    horizontal=True, key="ev_exp_fmt")
                    writers = {"CDM": write_cdm, "OPM": write_opm,
                               "OCM": write_ocm, "RDM": write_rdm}
                    text = writers[fmt](evt)
                    st.code(text[:1500] + ("\n..." if len(text) > 1500 else ""),
                             language="text")
                    st.download_button(
                        f"⬇ 下载 event_{evt.id}.{fmt.lower()}",
                        data=text, file_name=f"event_{evt.id}.{fmt.lower()}",
                        mime="text/plain", use_container_width=True,
                        key="ev_dl",
                    )

    # ── Tab 2 ──────────────────────────────────────────────────────────────
    with tab_add:
        with st.form("ev_add_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            etype = c1.selectbox("类型", list(_TYPES_CN.keys()),
                                  format_func=lambda v: _TYPES_CN[v])
            d = c2.date_input("日期 (UTC)", value=datetime.utcnow().date())
            tm = c3.text_input("时间 HH:MM:SS", value="00:00:00")
            name = st.text_input("事件名称", value="")
            desc = st.text_area("说明 / 描述", value="", height=80)
            cn1, cn2 = st.columns(2)
            n1 = cn1.number_input("主体 NORAD", value=0, step=1)
            n2 = cn2.number_input("次体 NORAD（碰撞 / CDM）", value=0, step=1)
            ca, cb = st.columns(2)
            alt = ca.number_input("高度 (km)", value=600.0, step=10.0)
            inc = cb.number_input("倾角 (°)", value=53.0, step=1.0)
            cm1, cm2 = st.columns(2)
            mp = cm1.number_input("主体质量 (kg)", value=0.0, step=100.0)
            mt = cm2.number_input("次体质量 (kg)", value=0.0, step=10.0)
            ce, cm = st.columns(2)
            E  = ce.number_input("撞击能量 (J)", value=0.0, step=1e7, format="%.3e")
            EM = cm.number_input("能量/质量比 J/g (>40 ⇒ 灾难性)", value=0.0,
                                  step=10.0)
            cmm, cpc = st.columns(2)
            miss = cmm.number_input("最近距离 (km, CDM)", value=0.0, step=0.1)
            pc   = cpc.number_input("Pc (CDM)", value=0.0, step=1e-6,
                                     format="%.3e")
            nfo  = st.number_input("观测到的碎片数", value=0, step=1)
            if st.form_submit_button("➕ 添加事件"):
                try:
                    h, mi, s = (int(x) for x in tm.split(":"))
                    epoch = datetime(d.year, d.month, d.day, h, mi, s,
                                      tzinfo=timezone.utc)
                except Exception:
                    st.error("时间格式错误，请使用 HH:MM:SS")
                    st.stop()
                evt = SpaceEvent(
                    event_type=EventType(etype), epoch=epoch,
                    name=name or _TYPES_CN[etype],
                    description=desc,
                    parent_norad   =int(n1) or None,
                    secondary_norad=int(n2) or None,
                    altitude_km    =float(alt),
                    inclination_deg=float(inc),
                    energy_j       =float(E)  or None,
                    energy_to_mass =float(EM) or None,
                    mass_parent_kg =float(mp) or None,
                    mass_target_kg =float(mt) or None,
                    miss_distance_km=float(miss) or None,
                    probability    =float(pc) or None,
                    n_fragments_obs=int(nfo)  or None,
                    source         ="manual",
                )
                new_id = insert_event(evt)
                st.success(f"已添加事件 #{new_id}")

    # ── Tab 3 — NASA SBM ───────────────────────────────────────────────────
    with tab_sbm:
        candidates = list_events(limit=500)
        candidates = [e for e in candidates if e.event_type in
                       (EventType.FRAGMENTATION, EventType.COLLISION)]
        if not candidates:
            st.info("当前数据库无解体 / 碰撞事件，先在「手动添加」或「数据源拉取」中创建。")
        else:
            sel = st.selectbox(
                "选定事件",
                candidates,
                format_func=lambda e: (
                    f"#{e.id}  {_TYPES_CN[e.event_type.value]}  "
                    f"{e.epoch.strftime('%Y-%m-%d')}  {e.name}"
                ),
                key="sbm_pick",
            )
            c1, c2, c3, c4 = st.columns(4)
            lc_min_cm = c1.number_input("Lc 最小 (cm)", value=1.0, min_value=0.01,
                                          max_value=100.0, step=1.0)
            lc_max_m  = c2.number_input("Lc 最大 (m)",  value=1.0, min_value=0.05,
                                          max_value=10.0, step=0.5)
            n_cap     = c3.number_input("抽样上限", value=2000, min_value=100,
                                          max_value=20000, step=500)
            prop_s    = c4.number_input("传播 (s)", value=600, min_value=0,
                                          max_value=7200, step=60)
            if st.button("⚙️ 运行 NASA SBM", use_container_width=True):
                with st.spinner("生成碎片中..."):
                    res = simulate_breakup(
                        sel,
                        lc_min_m=lc_min_cm / 100.0,
                        lc_max_m=lc_max_m,
                        max_fragments=int(n_cap),
                        seed=42,
                    )
                st.session_state["_sbm_res"] = res

            res = st.session_state.get("_sbm_res")
            if res is not None and res.event.id == sel.id:
                colA, colB, colC, colD = st.columns(4)
                colA.metric("总碎片", f"{res.n_total:,}")
                colB.metric("≥10 cm 可追踪", f"{res.n_tracked_ge_10cm:,}")
                colC.metric("≥1 cm 致命", f"{res.n_lethal_ge_1cm:,}")
                colD.metric("是否灾难性", "✅ 是" if res.catastrophic else "❌ 否")
                for n in res.notes:
                    st.markdown(f"- {n}")

                st.plotly_chart(_debris_cloud_fig(res, prop_s),
                                 use_container_width=True, key="sbm_3d")
                st.plotly_chart(_size_distribution_fig(res),
                                 use_container_width=True, key="sbm_hist")

                with st.expander("导出为 CCSDS OCM（含碎片清单）", expanded=False):
                    text = write_ocm(sel, fragments=res.fragments)
                    st.download_button(
                        f"⬇ 下载 sbm_{sel.id}.ocm",
                        data=text, file_name=f"sbm_{sel.id}.ocm",
                        mime="text/plain", use_container_width=True,
                        key="sbm_dl_ocm",
                    )

    # ── Tab 4 — CCSDS import / export ─────────────────────────────────────
    with tab_io:
        st.markdown("**导入 CCSDS NDM**（支持 CDM / OPM / OEM / OCM / RDM 自动识别）")
        up = st.file_uploader("选择 CCSDS 文件",
                               type=["cdm","opm","oem","ocm","rdm","txt"],
                               key="ev_upload")
        text_blob = st.text_area("或直接粘贴 KVN 文本", height=180, key="ev_paste")
        if st.button("📥 解析并入库", use_container_width=True, key="ev_imp"):
            blob = ""
            if up is not None:
                blob = up.read().decode("utf-8", errors="replace")
            elif text_blob.strip():
                blob = text_blob
            if not blob.strip():
                st.warning("请先上传文件或粘贴 KVN 文本。")
            else:
                try:
                    evt = parse_ccsds_message(blob)
                    new_id = upsert_event(evt)
                    st.success(f"导入成功 ✓  事件 #{new_id}  类型 {evt.event_type.value}")
                except Exception as exc:
                    st.error(f"解析失败：{exc}")

    # ── Tab 5 — external source fetch ─────────────────────────────────────
    with tab_ingest:
        st.markdown(
            "**多源太空事件拉取**（已实现 5 个权威来源）：\n"
            "| 数据源 | 事件类型 | 凭据 |\n"
            "|---|---|---|\n"
            "| ESA DISCOSweb `/fragmentations` | 解体 / 碰撞 | `ESA_DISCOS_TOKEN` |\n"
            "| Space-Track `cdm_public`        | CDM 合取预警 | `SPACETRACK_USERNAME/PASSWORD` |\n"
            "| Space-Track `decay`             | TIP / 再入 | `SPACETRACK_USERNAME/PASSWORD` |\n"
            "| CelesTrak SOCRATES              | 公开实时合取 | – |\n"
            "| Jonathan McDowell GCAT `ecat`   | 历史再入 / GRP 解体 | – |\n\n"
            "**离线 / 增量调度**（与碎片增量同套架构）：\n"
            "```bash\n"
            "# 全量一次性拉取\n"
            "docker exec debris-api-1 python scripts/ingest_events.py --all --max 1000\n\n"
            "# 增量 + 打包 zip\n"
            "docker exec debris-api-1 python scripts/ingest_events_incremental.py \\\n"
            "    --since 2026-04-01 --max 1000\n\n"
            "# 30 天循环 + SMTP 邮件投递\n"
            "docker exec -d debris-api-1 python scripts/events_scheduler.py\n\n"
            "# 在另一台机器上回放压缩包\n"
            "python scripts/apply_events_package.py data/event_packages/events_*.zip\n"
            "```"
        )

        c1, c2 = st.columns(2)
        srcs_default = ["discos", "cdm", "decay", "socrates", "gcat"]
        with c1:
            chosen = st.multiselect("选择数据源", srcs_default,
                                    default=srcs_default, key="ev_srcs")
        with c2:
            max_rows = st.number_input("每个源最多记录数", 50, 5000, 500,
                                        step=50, key="ev_max")
        since_str = st.text_input("增量起点 (留空=不限)", "", key="ev_since",
                                   placeholder="例: 2026-04-01")

        if st.button("🚀 立即拉取（写入数据库）", use_container_width=True,
                      key="ev_fetch"):
            try:
                from scripts.ingest_events import run_ingest, _parse_iso
                since_dt = _parse_iso(since_str) if since_str.strip() else None
                with st.spinner("正在拉取多源事件，请稍候..."):
                    _, counts = run_ingest(sources=chosen,
                                            max_rows=int(max_rows),
                                            since=since_dt)
                st.success(
                    "✓ 拉取完成；明细：" +
                    ", ".join(f"{k}={v}" for k, v in counts.items())
                )
            except Exception as exc:
                st.error(f"拉取失败：{exc}")
