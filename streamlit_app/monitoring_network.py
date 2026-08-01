# -*- coding: utf-8 -*-
"""数据库查询及导出 —— 三大要素库检索 / 批量导出。

  1. 全球天/地基空间物体监测设备数据库  external_ssa_sensors
  2. 全球天/地基空间天气监测设备数据库  external_space_weather_sensors
  3. 全球测控站数据库                    external_ttc_stations

性能要点：
  · 仅加载当前选中的一张表（不预拉三库）
  · 排除 PostGIS geom，列裁剪 + read_sql
  · 检索走 form，避免每次击键全页重跑
  · 地图默认折叠，按需渲染
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from database.db import session_scope

# ── 数据模型说明（供界面标注；对应 migrations/002_monitoring_network.sql）────
DB_SPECS: dict[str, dict[str, Any]] = {
    "ssa": {
        "title": "全球天/地基空间物体监测设备数据库",
        "table": "external_ssa_sensors",
        "model": "SSA Sensor（空间监视传感器）",
        "schema": "database/migrations/002_monitoring_network.sql → external_ssa_sensors",
        "id_col": "sensor_id",
        "columns": [
            "sensor_id", "name", "name_cn", "sensor_class", "network", "operator",
            "country", "lat", "lon", "alt_m", "frequency_band", "capability",
            "status", "notes", "source",
        ],
        "fields": [
            ("sensor_id", "TEXT UNIQUE", "传感器唯一 ID"),
            ("name / name_cn", "TEXT", "英文 / 中文名称"),
            ("sensor_class", "TEXT", "spaceborne / ground_radar / ground_optical / network_node"),
            ("network", "TEXT", "所属网络：SSN / TraCSS / ISON / ESA-SST / SuperDARN …"),
            ("operator / country", "TEXT", "运营方 / 国家（中文）"),
            ("lat, lon, alt_m", "FLOAT", "WGS84 坐标与海拔 (m)；天基可为占位 0,0"),
            ("geom", "geometry(Point,4326)", "PostGIS 点（查询界面不加载）"),
            ("frequency_band", "TEXT", "UHF / L / S / X / optical …"),
            ("capability", "TEXT", "监测能力描述"),
            ("status / source", "TEXT", "状态与数据来源渠道"),
        ],
        "sources": "GEODSS（9台）· ISON · USAFA Falcon · KASI OWL-Net · ILRS · MPC 候选站 · ESA-SST · Space-Track；ExoAnalytic/Slingshot 仅聚合记录",
    },
    "swx": {
        "title": "全球天/地基空间天气监测设备数据库",
        "table": "external_space_weather_sensors",
        "model": "Space Weather Sensor（空间天气传感器）",
        "schema": "database/migrations/002_monitoring_network.sql → external_space_weather_sensors",
        "id_col": "sensor_id",
        "columns": [
            "sensor_id", "name", "name_cn", "sensor_class", "network", "operator",
            "country", "lat", "lon", "alt_m", "observables", "data_format",
            "status", "notes", "source",
        ],
        "fields": [
            ("sensor_id", "TEXT UNIQUE", "传感器唯一 ID"),
            ("name / name_cn", "TEXT", "英文 / 中文名称"),
            ("sensor_class", "TEXT", "spaceborne / magnetometer / ionosonde / neutron_monitor / isr / optical_uv / network"),
            ("network", "TEXT", "SPASE-SMWG / WMO OSCAR/Space / SuperMAG / INTERMAGNET / NMDB / GIRO / SuperDARN …"),
            ("observables", "TEXT", "TEC / F10.7 / Kp·Ap / 高能粒子 / 极光 …"),
            ("data_format", "TEXT", "NetCDF / HDF5 / CSV / IAGA-2002 / REST …"),
            ("lat, lon, alt_m / geom", "FLOAT / Point", "地基坐标；天基占位"),
            ("status / source", "TEXT", "状态与数据来源渠道"),
        ],
        "sources": "NASA HPDE/SPASE-SMWG · WMO OSCAR/Space · SuperMAG（与 INTERMAGNET 去重）· NMDB · GIRO/DIDBase · SuperDARN · Space-Track",
    },
    "ttc": {
        "title": "全球测控站数据库",
        "table": "external_ttc_stations",
        "model": "TT&C / GSaaS Station（测控与地面站）",
        "schema": "database/migrations/002_monitoring_network.sql → external_ttc_stations",
        "id_col": "station_id",
        "columns": [
            "station_id", "name", "name_cn", "network", "operator", "country",
            "lat", "lon", "alt_m", "antenna_diam_m", "bands", "station_type",
            "status", "notes", "source",
        ],
        "fields": [
            ("station_id", "TEXT UNIQUE", "测控站唯一 ID"),
            ("name / name_cn", "TEXT", "英文 / 中文名称"),
            ("network", "TEXT", "ESTRACK / DSN / KSAT / AWS / SatNOGS / Starlink / Brahe …"),
            ("operator / country", "TEXT", "运营方 / 国家（中文）"),
            ("lat, lon, alt_m / geom", "FLOAT / Point", "WGS84 天线坐标（必填 lat/lon）"),
            ("antenna_diam_m", "FLOAT", "天线口径 (m)"),
            ("bands", "TEXT", "S / X / Ka / UHF / VHF …"),
            ("station_type", "TEXT", "deep_space / near_earth / polar / commercial_gsaas / crowdsourced / launch_site"),
            ("status / source", "TEXT", "状态与数据来源渠道"),
        ],
        "sources": "SatNOGS · Starlink 社区 · Brahe GS · ESTRACK/DSN/KSAT/USGS/AWS 策展 · DISCOS 发射场镜像",
    },
}


@st.cache_data(ttl=600, show_spinner=False)
def _load_table(table: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """按列加载（无 geom），进程内缓存 10 分钟。"""
    cols_sql = ", ".join(f'"{c}"' if c != c.lower() else c for c in columns)
    sql = f"SELECT {cols_sql} FROM {table}"
    with session_scope() as s:
        df = pd.read_sql(text(sql), s.connection())
    return df


@st.cache_data(ttl=600, show_spinner=False)
def _table_count(table: str) -> int:
    with session_scope() as s:
        n = s.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    return int(n or 0)


def _filter_df(
    df: pd.DataFrame,
    *,
    name_q: str,
    network: list,
    country_q: str,
    class_or_type: list,
    class_col: str | None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    m = pd.Series(True, index=df.index)
    if name_q.strip():
        q = name_q.strip()
        name_cols = [c for c in ("name", "name_cn", "sensor_id", "station_id") if c in df.columns]
        hit = pd.Series(False, index=df.index)
        for c in name_cols:
            hit = hit | df[c].astype("string").str.contains(q, case=False, na=False)
        m &= hit
    if network and "network" in df.columns:
        m &= df["network"].isin(network)
    if country_q.strip() and "country" in df.columns:
        m &= df["country"].astype("string").str.contains(country_q.strip(), case=False, na=False)
    if class_or_type and class_col and class_col in df.columns:
        m &= df[class_col].isin(class_or_type)
    return df.loc[m].reset_index(drop=True)


def _to_xlsx_bytes(df: pd.DataFrame, sheet: str) -> bytes:
    buf = io.BytesIO()
    export = df.copy()
    for c in export.columns:
        if pd.api.types.is_datetime64_any_dtype(export[c]):
            export[c] = export[c].astype("string")
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        export.to_excel(w, index=False, sheet_name=(sheet or "data")[:31])
    return buf.getvalue()


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def render_monitoring_network():
    st.subheader("数据库查询及导出")
    st.caption(
        "三大要素库检索与批量导出。"
        "存储引擎：**PostgreSQL + PostGIS**（`geometry(Point,4326)` + GiST）；"
        "关系模型见下方「数据模型」。"
    )

    key = st.radio(
        "选择数据库",
        options=list(DB_SPECS.keys()),
        format_func=lambda k: DB_SPECS[k]["title"],
        horizontal=False,
        key="dbq_which",
    )
    spec = DB_SPECS[key]
    table = spec["table"]

    # ── 数据模型标注 ─────────────────────────────────────────────────────────
    with st.expander(f"数据模型 · {spec['model']}", expanded=False):
        st.markdown(
            f"**逻辑模型**：`{spec['model']}`  \n"
            f"**物理表**：`{table}`  \n"
            f"**DDL**：`{spec['schema']}`  \n"
            f"**数据渠道**：{spec['sources']}  \n"
            f"**库内行数**：{_table_count(table):,}"
        )
        st.dataframe(
            pd.DataFrame(spec["fields"], columns=["字段", "类型", "说明"]),
            use_container_width=True,
            hide_index=True,
            height=min(56 + 28 * len(spec["fields"]), 320),
        )

    try:
        with st.spinner("加载库表…"):
            df = _load_table(table, tuple(spec["columns"]))
    except Exception as exc:
        st.error(f"加载失败：{exc}")
        st.info("请先运行：`python scripts/ingest_monitoring_network.py`")
        return

    if df.empty:
        st.warning(f"`{table}` 为空。请运行：`python scripts/ingest_monitoring_network.py`")
        return

    class_col = "sensor_class" if "sensor_class" in df.columns else (
        "station_type" if "station_type" in df.columns else None
    )
    class_label = "传感器类别" if class_col == "sensor_class" else "站型"

    # ── 检索（form：提交才过滤，避免击键重跑）───────────────────────────────
    with st.form(f"dbq_form_{key}"):
        c1, c2, c3, c4 = st.columns(4)
        name_q = c1.text_input("名称 / ID 关键词", "")
        nets = sorted([x for x in df["network"].dropna().unique()]) if "network" in df.columns else []
        network = c2.multiselect("网络", nets, default=[])
        country_q = c3.text_input("国家 / 地区（中文）", "", placeholder="如 美国 / 中国")
        class_opts = (
            sorted([x for x in df[class_col].dropna().unique()]) if class_col else []
        )
        class_sel = c4.multiselect(class_label, class_opts, default=[]) if class_col else []
        submitted = st.form_submit_button("检索", use_container_width=True, type="primary")

    state_key = f"dbq_result_{key}"
    if submitted or state_key not in st.session_state:
        out = _filter_df(
            df,
            name_q=name_q if submitted else "",
            network=network if submitted else [],
            country_q=country_q if submitted else "",
            class_or_type=class_sel if submitted else [],
            class_col=class_col,
        )
        st.session_state[state_key] = out
        st.session_state[f"dbq_crit_{key}"] = {
            "name_q": name_q if submitted else "",
            "network": network if submitted else [],
            "country_q": country_q if submitted else "",
            "class": class_sel if submitted else [],
        }

    out: pd.DataFrame = st.session_state.get(state_key, df)
    crit = st.session_state.get(f"dbq_crit_{key}", {})
    st.success(f"命中 **{len(out):,}** 条（库内 {len(df):,}）· `{table}`")

    st.dataframe(out, use_container_width=True, height=420, hide_index=True)

    # ── 批量导出 ─────────────────────────────────────────────────────────────
    st.markdown("##### 批量导出")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{table}_{ts}"
    has = out is not None and not out.empty
    xlsx_b = _to_xlsx_bytes(out, key) if has else b""
    csv_b = _to_csv_bytes(out) if has else b""
    e1, e2, e3 = st.columns([0.28, 0.28, 0.44])
    with e1:
        st.download_button(
            "⬇ 导出 XLSX",
            data=xlsx_b,
            file_name=f"{stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not has,
            use_container_width=True,
            key=f"dl_xlsx_{key}",
        )
    with e2:
        st.download_button(
            "⬇ 导出 CSV",
            data=csv_b,
            file_name=f"{stem}.csv",
            mime="text/csv",
            disabled=not has,
            use_container_width=True,
            key=f"dl_csv_{key}",
        )
    with e3:
        st.caption(
            f"导出当前检索结果（{len(out):,} 行）。"
            + (f" 筛选：{crit}" if any(crit.values()) else " 当前为全库。")
        )

    # ── 地图（按需，默认不渲染）────────────────────────────────────────────
    with st.expander("地基站点地图（可选，较慢）", expanded=False):
        _map_points(out, spec["title"])


def _map_points(df: pd.DataFrame, title: str):
    if df is None or df.empty or "lat" not in df.columns:
        st.caption("无坐标字段")
        return
    pts = df.dropna(subset=["lat", "lon"]).copy()
    if "alt_m" in pts.columns:
        pts = pts[pts["alt_m"].fillna(0).abs() < 100_000]
    pts = pts[(pts["lat"].between(-90, 90)) & (pts["lon"].between(-180, 180))]
    # 天基占位 (0,0) 过滤
    pts = pts[~((pts["lat"].abs() < 1e-6) & (pts["lon"].abs() < 1e-6))]
    if pts.empty:
        st.caption("当前结果无适合地图展示的地基坐标")
        return
    try:
        import pydeck as pdk
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position="[lon, lat]",
            get_radius=80000,
            get_fill_color=[44, 132, 188, 180],
            pickable=True,
        )
        view = pdk.ViewState(latitude=20, longitude=20, zoom=1.2)
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip={"text": "{name}\n{network}\n{country}"},
            )
        )
        st.caption(f"{title} · 地基分布示意（{len(pts):,} 点）")
    except Exception as exc:
        st.caption(f"地图跳过：{exc}")
