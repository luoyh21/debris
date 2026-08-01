# -*- coding: utf-8 -*-
"""目标目录·搜索与导出 —— 纯数据逻辑（不依赖 streamlit）。

被前端页面 (streamlit_app/search_export.py) 与 REST API (api/main.py) 共用：
  * load_dataset(session)  -> 关联 catalog/最新GP/UCS/DISCOS 并计算轨道/速度/SSO 的 DataFrame
  * filter_dataset(df, criteria) -> 按条件筛选
  * build_export(df) -> 规范中文列的导出表
  * to_xlsx_bytes(df) -> xlsx 字节
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analytics.country_labels import COUNTRY_OR_ORG_LABELS

# 物理常数
MU = 398600.4418            # km^3/s^2
RE = 6378.137              # km
J2 = 1.08262668e-3
SSO_RATE = 360.0 / 365.2421897   # 太阳同步所需升交点进动率 deg/day (~0.98565)

# SATCAT 国家代码 → 中文
_CC = COUNTRY_OR_ORG_LABELS
# UCS 英文国名 → 中文
_EN = {
    "china": "中国", "usa": "美国", "united states": "美国", "russia": "俄罗斯",
    "russian federation": "俄罗斯", "japan": "日本", "france": "法国", "germany": "德国",
    "united kingdom": "英国", "uk": "英国", "canada": "加拿大", "india": "印度", "italy": "意大利",
    "spain": "西班牙", "south korea": "韩国", "korea, south": "韩国", "saudi arabia": "沙特",
    "netherlands": "荷兰", "luxembourg": "卢森堡", "norway": "挪威", "sweden": "瑞典", "finland": "芬兰",
    "argentina": "阿根廷", "brazil": "巴西", "turkey": "土耳其", "israel": "以色列", "australia": "澳大利亚",
    "thailand": "泰国", "south africa": "南非", "united arab emirates": "阿联酋", "uae": "阿联酋",
    "egypt": "埃及", "poland": "波兰", "czech republic": "捷克", "denmark": "丹麦", "switzerland": "瑞士",
    "belgium": "比利时", "austria": "奥地利", "greece": "希腊", "portugal": "葡萄牙", "vietnam": "越南",
    "indonesia": "印尼", "malaysia": "马来西亚", "philippines": "菲律宾", "taiwan": "中国台湾",
    "iran": "伊朗", "mexico": "墨西哥", "chile": "智利", "new zealand": "新西兰", "singapore": "新加坡",
    "kazakhstan": "哈萨克斯坦", "ukraine": "乌克兰", "algeria": "阿尔及利亚", "nigeria": "尼日利亚",
    "esa": "欧空局", "european space agency": "欧空局", "eumetsat": "EUMETSAT", "multinational": "国际/多国",
}

OBJECT_TYPES = ["PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"]
RCS_SIZES = ["SMALL", "MEDIUM", "LARGE"]
ORBIT_CHOICES = ["全部", "仅太阳同步(SSO)", "排除SSO", "LEO(<2000km)", "MEO(2000-35586km)", "GEO(~35786km)"]
CONTROL_CHOICES = ["全部", "仅受控(推定)", "仅存疑"]

_SQL = """
WITH g AS (
  SELECT DISTINCT ON (norad_cat_id) norad_cat_id, epoch, mean_motion, eccentricity,
         inclination AS gp_incl, ra_of_asc_node, bstar
  FROM gp_elements WHERE norad_cat_id IS NOT NULL ORDER BY norad_cat_id, epoch DESC),
u AS (
  SELECT DISTINCT ON (norad_cat_id) norad_cat_id, country, operator, users, purpose,
         detailed_purpose, orbit_class, orbit_type, launch_mass_kg, expected_lifetime_yr
  FROM external_ucs_satellites WHERE norad_cat_id IS NOT NULL ORDER BY norad_cat_id),
d AS (
  SELECT DISTINCT ON (satno) satno, mass, shape, "xSectMax" AS xsect_max,
         "xSectMin" AS xsect_min, "xSectAvg" AS xsect_avg, "objectClass" AS object_class, active
  FROM external_esa_discos WHERE satno IS NOT NULL ORDER BY satno)
SELECT c.norad_cat_id, c.name AS satcat_name, c.object_type, c.country_code, c.launch_date,
       c.decay_date, c.rcs_size, c.object_id, c.apogee_km AS cat_apogee_km,
       c.perigee_km AS cat_perigee_km, c.inclination AS cat_incl, c.period_min,
       g.epoch AS gp_epoch, g.mean_motion, g.eccentricity, g.gp_incl, g.ra_of_asc_node,
       u.country AS ucs_country, u.operator AS ucs_operator, u.users AS ucs_users,
       u.purpose AS ucs_purpose, u.detailed_purpose AS ucs_detailed_purpose,
       u.orbit_class AS ucs_orbit_class, u.orbit_type AS ucs_orbit_type,
       u.launch_mass_kg AS ucs_mass_kg, u.expected_lifetime_yr,
       d.mass AS discos_mass_kg, d.shape AS discos_shape, d.xsect_max AS discos_xsect_max_m2,
       d.xsect_min AS discos_xsect_min_m2, d.xsect_avg AS discos_xsect_avg_m2,
       d.object_class AS discos_class, d.active AS discos_active
FROM catalog_objects c
JOIN g ON g.norad_cat_id = c.norad_cat_id
LEFT JOIN u ON u.norad_cat_id = c.norad_cat_id
LEFT JOIN d ON d.satno = c.norad_cat_id
WHERE c.decay_date IS NULL
"""


def _norm_country(ucs_country, country_code) -> str:
    if isinstance(ucs_country, str) and ucs_country.strip():
        return _EN.get(ucs_country.strip().lower(), ucs_country.strip())
    cc = str(country_code or "").strip()
    return _CC.get(cc, cc) if cc else "未知"


def load_dataset(session) -> pd.DataFrame:
    """关联多源并计算轨道/速度/SSO/受控/质量/尺寸，返回 DataFrame（仅在轨对象）。"""
    from sqlalchemy import text
    rows = session.execute(text(_SQL)).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r._mapping) for r in rows])

    mm = pd.to_numeric(df["mean_motion"], errors="coerce")          # rev/day
    ecc = pd.to_numeric(df["eccentricity"], errors="coerce").fillna(0.0)
    inc = pd.to_numeric(df["gp_incl"], errors="coerce")             # deg
    n_rad = mm * 2 * np.pi / 86400.0                                # rad/s
    a = (MU / n_rad ** 2) ** (1.0 / 3.0)                            # km

    df["mean_alt_km"] = a - RE
    df["peri_alt_km"] = a * (1 - ecc) - RE
    df["apo_alt_km"] = a * (1 + ecc) - RE
    df["period_min_calc"] = (2 * np.pi / n_rad) / 60.0
    df["v_mean"] = np.sqrt(MU / a)
    df["v_peri"] = np.sqrt(MU * (2 / (a * (1 - ecc)) - 1 / a))
    df["v_apo"] = np.sqrt(MU * (2 / (a * (1 + ecc)) - 1 / a))
    df["omega_rad_s"] = n_rad
    df["omega_deg_s"] = np.degrees(n_rad)
    df["inc"] = inc
    df["ecc"] = ecc
    df["mm"] = mm

    # J2 升交点进动率 + 太阳同步判定
    i_rad = np.radians(inc)
    p = a * (1 - ecc ** 2)
    node_rate = -1.5 * n_rad * J2 * (RE / p) ** 2 * np.cos(i_rad)
    df["node_rate_deg_day"] = node_rate * 86400.0 * 180.0 / np.pi
    k = -1.5 * n_rad * J2 * (RE / p) ** 2 * 86400.0 * 180.0 / np.pi
    cos_i_req = np.where(k != 0, SSO_RATE / k, np.nan)
    df["sso_req_inc"] = np.degrees(np.arccos(np.clip(cos_i_req, -1, 1)))
    df["is_sso"] = (
        (df["node_rate_deg_day"].sub(SSO_RATE).abs() < 0.07)
        & (df["ecc"] < 0.05) & df["inc"].between(95, 102)
    )

    # 受控（推定）
    in_ucs = df["ucs_country"].notna()
    discos_active = df["discos_active"].astype("string").str.lower().isin(["true", "t", "1"])
    df["in_ucs"] = in_ucs
    df["discos_active_b"] = discos_active
    df["controlled"] = in_ucs | discos_active

    # 国家、质量、尺寸
    df["country"] = [
        _norm_country(uc, cc) for uc, cc in zip(df["ucs_country"], df["country_code"])
    ]
    mass = pd.to_numeric(df["discos_mass_kg"], errors="coerce")
    mass = mass.fillna(pd.to_numeric(df["ucs_mass_kg"], errors="coerce"))
    df["mass_kg"] = mass
    df["mass_src"] = np.where(
        pd.to_numeric(df["discos_mass_kg"], errors="coerce").notna(), "DISCOS",
        np.where(pd.to_numeric(df["ucs_mass_kg"], errors="coerce").notna(), "UCS", ""))
    xa = pd.to_numeric(df["discos_xsect_avg_m2"], errors="coerce")
    df["xsect_avg"] = xa
    df["equiv_size_m"] = 2 * np.sqrt(xa / np.pi)
    return df


def filter_dataset(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """按条件筛选。criteria 见各键默认值；空/None 表示不限制。"""
    if df.empty:
        return df
    c = criteria or {}
    m = pd.Series(True, index=df.index)

    q = (c.get("name_query") or "").strip()
    if q:
        by_name = df["satcat_name"].astype("string").str.contains(q, case=False, na=False)
        by_norad = df["norad_cat_id"].astype("string").str.contains(q, na=False)
        by_cospar = df["object_id"].astype("string").str.contains(q, case=False, na=False)
        m &= (by_name | by_norad | by_cospar)

    ots = c.get("object_types") or []
    if ots:
        m &= df["object_type"].astype("string").str.upper().isin([o.upper() for o in ots])

    cq = (c.get("country_query") or "").strip()
    if cq:
        m &= df["country"].astype("string").str.contains(cq, case=False, na=False)
    countries = c.get("countries") or []
    if countries:
        m &= df["country"].isin(countries)

    if c.get("alt_min") is not None:
        m &= df["mean_alt_km"] >= float(c["alt_min"])
    if c.get("alt_max") is not None:
        m &= df["mean_alt_km"] <= float(c["alt_max"])
    if c.get("incl_min") is not None:
        m &= df["inc"] >= float(c["incl_min"])
    if c.get("incl_max") is not None:
        m &= df["inc"] <= float(c["incl_max"])
    if c.get("ecc_max") is not None:
        m &= df["ecc"] <= float(c["ecc_max"])
    if c.get("mass_min") is not None:
        m &= df["mass_kg"] >= float(c["mass_min"])
    if c.get("mass_max") is not None:
        m &= df["mass_kg"] <= float(c["mass_max"])

    rcs = c.get("rcs_sizes") or []
    if rcs:
        m &= df["rcs_size"].astype("string").str.upper().isin([r.upper() for r in rcs])

    orbit = c.get("orbit") or "全部"
    if orbit == "仅太阳同步(SSO)":
        m &= df["is_sso"]
    elif orbit == "排除SSO":
        m &= ~df["is_sso"]
    elif orbit.startswith("LEO"):
        m &= df["mean_alt_km"] < 2000
    elif orbit.startswith("MEO"):
        m &= df["mean_alt_km"].between(2000, 35586)
    elif orbit.startswith("GEO"):
        m &= df["mean_alt_km"].between(35586, 35986)

    ctrl = c.get("controlled") or "全部"
    if ctrl == "仅受控(推定)":
        m &= df["controlled"]
    elif ctrl == "仅存疑":
        m &= ~df["controlled"]

    out = df[m].sort_values(["mean_alt_km", "norad_cat_id"])
    lim = c.get("limit")
    if lim:
        out = out.head(int(lim))
    return out.reset_index(drop=True)


def build_export(df: pd.DataFrame) -> pd.DataFrame:
    """把筛选结果整理成规范中文列（用于表格展示与 xlsx 导出）。"""
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "NORAD": df["norad_cat_id"],
        "国际编号(COSPAR)": df["object_id"],
        "名称": df["satcat_name"],
        "国家/地区": df["country"],
        "运营方": df["ucs_operator"],
        "用途": df["ucs_purpose"],
        "对象类型": df["object_type"],
        "平均高度(km)": df["mean_alt_km"].round(1),
        "近地点高度(km)": df["peri_alt_km"].round(1),
        "远地点高度(km)": df["apo_alt_km"].round(1),
        "倾角(°)": df["inc"].round(3),
        "偏心率": df["ecc"].round(5),
        "轨道周期(min)": df["period_min_calc"].round(2),
        "太阳同步(SSO)": np.where(df["is_sso"], "是", "否"),
        "升交点进动(°/day)": df["node_rate_deg_day"].round(4),
        "该高度SSO理论倾角(°)": df["sso_req_inc"].round(3),
        "线速度·平均(km/s)": df["v_mean"].round(4),
        "线速度·近地点(km/s)": df["v_peri"].round(4),
        "线速度·远地点(km/s)": df["v_apo"].round(4),
        "角速度(°/s)": df["omega_deg_s"].round(5),
        "角速度(rad/s)": df["omega_rad_s"].round(7),
        "平均运动(rev/day)": df["mm"].round(6),
        "质量(kg)": df["mass_kg"].round(1),
        "质量来源": df["mass_src"],
        "形状": df["discos_shape"],
        "平均截面积(m²)": df["xsect_avg"].round(4),
        "等效尺寸(m)": df["equiv_size_m"].round(3),
        "RCS尺寸等级": df["rcs_size"],
        "可控(推定)": np.where(df["controlled"], "是", "存疑"),
        "在UCS运营库": np.where(df["in_ucs"], "是", "否"),
        "DISCOS仍活跃": np.where(df["discos_active_b"], "是", "否"),
        # 转字符串：避免 Excel 不支持带时区 datetime，并规避 NaT/日期类型问题
        "发射日期": df["launch_date"].astype("string"),
        "GP历元": pd.to_datetime(df["gp_epoch"], utc=True, errors="coerce")
                    .dt.strftime("%Y-%m-%d %H:%M UTC").astype("string"),
    })


def _notes_df(n_rows: int, criteria: dict) -> pd.DataFrame:
    crit = "; ".join(f"{k}={v}" for k, v in (criteria or {}).items() if v not in (None, "", [], "全部"))
    return pd.DataFrame({"项目": [
        "数据来源", "筛选条件", "高度/半长轴", "太阳同步判定", "线速度", "角速度",
        "质量", "尺寸", "国家/地区", "可控(推定)", "记录数", "导出时间(UTC)"],
        "说明": [
        "本地库：catalog_objects(SATCAT)+最新GP+UCS运营卫星库+ESA DISCOS（仅在轨对象）",
        crit or "(无附加筛选)",
        "由最新平均运动 n 反推 a=(μ/n²)^(1/3)，μ=398600.4418，Rᴇ=6378.137；平均高度=a−Rᴇ",
        "J2 进动率 Ω̇=−1.5·n·J2·(Rᴇ/p)²·cos i，判据 |Ω̇−0.98565°/day|<0.07 且 e<0.05 且 95°<i<102°",
        "平均=√(μ/a)；近/远地点用活力公式 v=√(μ(2/r−1/a))",
        "轨道平均角速率=平均运动 n，给出 °/s 与 rad/s",
        "优先 ESA DISCOS 实测质量，缺失回退 UCS 发射质量",
        "ESA DISCOS 平均截面积+形状；等效尺寸=2√(A/π)；另附 SATCAT RCS 等级",
        "优先 UCS 国家全称，否则按 SATCAT 国家代码映射",
        "载荷且(在 UCS 运营库 或 DISCOS 活跃)→『是』，否则『存疑』(库中无姿态可控字段，为最佳代理)",
        str(n_rows),
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")]})


def to_xlsx_bytes(export_df: pd.DataFrame, criteria: dict | None = None,
                  sheet_name: str = "搜索结果") -> bytes:
    """把导出表写成 xlsx 字节（含『字段与口径说明』sheet、列宽自适应）。"""
    bio = io.BytesIO()
    notes = _notes_df(len(export_df), criteria or {})
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        (export_df if not export_df.empty else pd.DataFrame({"提示": ["无符合条件的结果"]})
         ).to_excel(w, sheet_name=sheet_name, index=False)
        notes.to_excel(w, sheet_name="字段与口径说明", index=False)
        for sh in w.sheets.values():
            for col in sh.columns:
                letter = col[0].column_letter
                width = max((len(str(c.value)) if c.value is not None else 0) for c in col)
                sh.column_dimensions[letter].width = min(max(width + 2, 8), 50)
    return bio.getvalue()
