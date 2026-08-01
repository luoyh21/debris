# -*- coding: utf-8 -*-
"""目标目录 · 搜索与导出 页面。

设置筛选条件 → 点击「搜索」查看结果 → 始终可见的「导出 xlsx」按钮一键下载，
并在服务器本地 exports/ 目录另存一份。纯数据逻辑见 analytics/catalog_search.py。
"""
import os
from datetime import datetime, timezone

import streamlit as st

from analytics import catalog_search as cs

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORT_DIR = os.path.join(_APP_ROOT, "exports")


@st.cache_data(ttl=300, show_spinner=False)
def _load_dataset():
    from database.db import session_scope
    with session_scope() as s:
        return cs.load_dataset(s)


def render_search_export():
    st.subheader("目标目录 · 搜索与导出")
    st.caption(
        "按多维条件检索在轨目标（关联 SATCAT + 最新 GP + UCS 运营库 + ESA DISCOS，"
        "自动计算高度 / 速度 / 太阳同步 / 受控 / 质量 / 尺寸），点「搜索」查看结果，"
        "再点「导出 xlsx」一键下载并在服务器本地另存一份。"
    )

    try:
        df = _load_dataset()
    except Exception as exc:
        st.error(f"加载数据失败：{exc}")
        return
    if df is None or df.empty:
        st.warning("数据库暂无可用对象，请先完成数据摄入。")
        return

    # ── 筛选条件（表单，提交才搜索） ─────────────────────────────────────────
    with st.form("search_form"):
        c1, c2, c3 = st.columns(3)
        name_query = c1.text_input("名称 / NORAD / COSPAR 关键词", value="")
        object_types = c2.multiselect("对象类型", cs.OBJECT_TYPES, default=["PAYLOAD"])
        country_query = c3.text_input("国家 / 地区（中文关键词）", value="",
                                      placeholder="如 中国 / 美国 / 俄罗斯，留空=全部")

        c4, c5, c6 = st.columns(3)
        orbit = c4.selectbox("轨道类型", cs.ORBIT_CHOICES, index=0)
        controlled = c5.selectbox("是否受控（推定）", cs.CONTROL_CHOICES, index=0)
        rcs_sizes = c6.multiselect("RCS 尺寸等级", cs.RCS_SIZES, default=[])

        c7, c8 = st.columns(2)
        alt_min = c7.number_input("最低平均高度 (km)", value=0.0, min_value=0.0,
                                  max_value=50000.0, step=50.0)
        alt_max = c8.number_input("最高平均高度 (km)", value=2000.0, min_value=0.0,
                                  max_value=50000.0, step=50.0)

        incl_min, incl_max = st.slider("倾角范围 (°)", 0.0, 180.0, (0.0, 180.0), step=0.5)
        ecc_max = st.slider("偏心率上限", 0.0, 1.0, 1.0, step=0.01)

        c9, c10 = st.columns(2)
        mass_min = c9.number_input("最小质量 (kg, 0=不限)", value=0.0, min_value=0.0, step=10.0)
        mass_max = c10.number_input("最大质量 (kg, 0=不限)", value=0.0, min_value=0.0, step=10.0)

        submitted = st.form_submit_button("🔍 搜索", use_container_width=True, type="primary")

    if submitted:
        criteria = {
            "name_query": name_query,
            "object_types": object_types,
            "country_query": country_query,
            "orbit": orbit,
            "controlled": controlled,
            "rcs_sizes": rcs_sizes,
            "alt_min": alt_min,
            "alt_max": alt_max,
            "incl_min": incl_min,
            "incl_max": incl_max,
            "ecc_max": ecc_max if ecc_max < 1.0 else None,
            "mass_min": mass_min if mass_min > 0 else None,
            "mass_max": mass_max if mass_max > 0 else None,
            "limit": None,  # 不限制：返回全部匹配
        }
        with st.spinner("检索中…"):
            result = cs.filter_dataset(df, criteria)
            export_df = cs.build_export(result)
        st.session_state["se_export_df"] = export_df
        st.session_state["se_criteria"] = criteria

    export_df = st.session_state.get("se_export_df")
    criteria = st.session_state.get("se_criteria", {})

    # ── 结果展示 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    if export_df is None:
        st.info("设置上方条件后点击「🔍 搜索」，结果将在此显示；下方导出按钮始终可用。")
    elif export_df.empty:
        st.warning("没有符合条件的目标，请放宽筛选条件。")
    else:
        st.success(f"命中 **{len(export_df):,}** 个目标")
        st.dataframe(export_df, use_container_width=True, height=460, hide_index=True)

    # ── 始终存在的导出按钮 ───────────────────────────────────────────────────
    has_rows = export_df is not None and not export_df.empty
    xlsx_bytes = b""
    if has_rows:
        try:
            xlsx_bytes = cs.to_xlsx_bytes(export_df, criteria)
        except Exception as exc:
            st.error(f"生成 xlsx 失败：{exc}")
    fname = f"catalog_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    cdl, cinfo = st.columns([0.35, 0.65])
    with cdl:
        clicked = st.download_button(
            "⬇ 导出 xlsx",
            data=xlsx_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
            disabled=not has_rows,
            key="se_download",
            help="导出当前搜索结果为 Excel，并在服务器本地 exports/ 目录另存一份",
        )
    with cinfo:
        if not has_rows:
            st.caption("（先搜索出结果后即可导出）")

    # 点击导出时：在服务器本地另存一份
    if clicked and has_rows:
        try:
            os.makedirs(_EXPORT_DIR, exist_ok=True)
            saved = os.path.join(_EXPORT_DIR, fname)
            with open(saved, "wb") as f:
                f.write(xlsx_bytes)
            st.success(f"已下载，并在服务器本地另存：{saved}")
        except Exception as exc:
            st.warning(f"已下载；服务器本地另存失败（可忽略）：{exc}")
