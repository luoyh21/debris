#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多渠道聚合摄入：监测与测控网络三库 + DISCOS 发射场/组织（合并去重，统计渠道贡献）。

三库：
  1. external_ssa_sensors           — 天/地基空间物体监测设备
  2. external_space_weather_sensors — 天/地基空间天气监测设备
  3. external_ttc_stations          — 全球测控站

DISCOS 扩展：
  4. external_discos_launch_sites   — DISCOSweb launch-sites
  5. external_discos_organisations  — DISCOSweb organisations

公开渠道（自动拉取）：
  · SatNOGS Network API          https://network.satnogs.org/api/stations/
  · INTERMAGNET Observatory List https://imag-data.bgs.ac.uk/...
  · SuperMAG station_info        全球地磁台站（与 INTERMAGNET 按 IAGA code 去重）
  · NMDB station registry        全球中子监测站
  · GIRO / DIDBase Ionosonde     https://lgdc.uml.edu/common/DIDBFastStationList
  · SuperDARN Dartmouth sites    https://superdarn.thayer.dartmouth.edu/
  · ILRS active stations         在役卫星激光测距站
  · MPC Observatory Codes        名称明确的空间监视/巡天候选站
  · NASA HPDE SPASE-SMWG         天基日地物理仪器注册表
  · WMO OSCAR/Space              空间天气专用仪器类型
  · Space-Track SATCAT           （.env 凭证）天基 SWX / SSA 载荷
  · ESA DISCOSweb                /api/launch-sites · /api/organisations（ESA_DISCOS_TOKEN）
  · Brahe 社区地面站 GeoJSON     duncaneddy/brahe data/groundstations（AWS/KSAT/DSN/…）
  · Starlink 社区网关表          juliensimon/starlink-viz（Satellitemap 类公开站表）

手工/文档策展渠道（公开坐标，文档 docs/空间监测数据库构建.md + 官方页）：
  · SSN / GEODSS / 公开雷达站
  · ESTRACK / NASA DSN / USGS Landsat / KSAT / NEN 代表站
  · TraCSS / LookUpSpace / TMDS / ISON / ThumbNet 枢纽
  · NOAA SWPC / 子午工程枢纽
  · Azure Orbital（已退役；保留历史公开近似坐标）

用法：
  python scripts/ingest_monitoring_network.py
  python scripts/ingest_monitoring_network.py --offline-cache /path/to/cache
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text

from database.db import get_engine, init_db

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CACHE_DIR = Path(os.path.dirname(__file__), "..", "data", "monitoring_network")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 渠道贡献计数（写入后打印）
CHANNEL_STATS: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


# ── HTML table parser ────────────────────────────────────────────────────────
class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._t = self._r = None
        self._cell = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._t = []
            self.tables.append(self._t)
        elif tag == "tr" and self._t is not None:
            self._r = []
            self._t.append(self._r)
        elif tag in ("td", "th") and self._r is not None:
            self._buf = []
            self._cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell:
            self._r.append("".join(self._buf).strip())
            self._cell = False
        elif tag == "tr":
            self._r = None
        elif tag == "table":
            self._t = None

    def handle_data(self, data):
        if self._cell:
            self._buf.append(data)


def _get(url: str, timeout: int = 120, **kw) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "debris-monitor/1.0"}, **kw)
    r.raise_for_status()
    return r


def _norm_lon(lon: float) -> float:
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def _dedup_key(lat, lon, name: str) -> str:
    try:
        return f"{float(lat):.3f}|{_norm_lon(float(lon)):.3f}|{(name or '').strip().lower()}"
    except Exception:
        return f"name|{(name or '').strip().lower()}"


def _add(bucket: dict, key: str, row: dict, channel: str, table: str):
    if key in bucket:
        CHANNEL_STATS[table][f"{channel}(dup_skip)"] += 1
        return
    row["source"] = channel
    bucket[key] = row
    CHANNEL_STATS[table][channel] += 1


# ── Fetchers ─────────────────────────────────────────────────────────────────
def fetch_satnogs(cache: Path) -> list[dict]:
    fp = cache / "satnogs_stations.json"
    if fp.exists() and fp.stat().st_size > 1000:
        print(f"  [cache] SatNOGS {fp}")
        return json.loads(fp.read_text())
    url = "https://network.satnogs.org/api/stations/"
    params = {"format": "json", "limit": 100}
    allr, page = [], 1
    while url:
        r = requests.get(url, params=params if page == 1 else None, timeout=120,
                         headers={"User-Agent": "debris-monitor/1.0"})
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            allr.extend(data)
            break
        allr.extend(data.get("results") or [])
        url = data.get("next")
        params = None
        page += 1
        print(f"  SatNOGS page {page-1}: {len(allr)}", flush=True)
        time.sleep(0.25)
    fp.write_text(json.dumps(allr))
    return allr


def fetch_intermagnet(cache: Path) -> list[dict]:
    fp = cache / "intermagnet.json"
    if fp.exists():
        return json.loads(fp.read_text())
    html = _get("https://imag-data.bgs.ac.uk/GIN_V1/GINForms2?request=ShowObservatoryDetails").text
    p = TableParser(); p.feed(html)
    rows = []
    for t in p.tables:
        for row in t:
            if len(row) >= 5 and re.match(r"^[A-Z]{3}$", row[0]):
                try:
                    elev = float(row[5]) if len(row) > 5 and re.match(r"^-?\d", row[5] or "") else None
                    rows.append({
                        "code": row[0], "name": row[1], "country_code": row[2],
                        "lat": float(row[3]), "lon": _norm_lon(float(row[4])), "alt_m": elev,
                    })
                except Exception:
                    pass
    fp.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def fetch_giro(cache: Path) -> list[dict]:
    fp = cache / "giro.json"
    if fp.exists():
        return json.loads(fp.read_text())
    html = _get("https://lgdc.uml.edu/common/DIDBFastStationList?orderByField=lon").text
    p = TableParser(); p.feed(html)
    rows = []
    for t in p.tables:
        for row in t:
            if len(row) >= 5 and re.match(r"^\d+$", row[0]):
                try:
                    rows.append({
                        "ursi": row[1], "name": row[2],
                        "lat": float(row[3]), "lon": _norm_lon(float(row[4])),
                    })
                except Exception:
                    pass
    fp.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def fetch_superdarn(cache: Path) -> list[dict]:
    fp = cache / "superdarn.json"
    if fp.exists():
        return json.loads(fp.read_text())
    html = _get("https://superdarn.thayer.dartmouth.edu/").text
    p = TableParser(); p.feed(html)
    rows = []
    for t in p.tables:
        for row in t:
            floats = []
            for c in row:
                try:
                    floats.append(float(c))
                except Exception:
                    continue
            if len(floats) >= 2 and abs(floats[0]) <= 90 and abs(floats[1]) <= 360:
                name = row[0] if row else ""
                if not name or "geo" in name.lower() or name == "Site":
                    continue
                rows.append({"name": name, "lat": floats[0], "lon": _norm_lon(floats[1])})
    fp.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def fetch_spacetrack_payloads(cache: Path) -> list[dict]:
    fp = cache / "spacetrack_payloads.json"
    if fp.exists():
        return json.loads(fp.read_text())
    user = os.getenv("SPACETRACK_USERNAME", "")
    pwd = os.getenv("SPACETRACK_PASSWORD", "")
    if not user or not pwd:
        print("  [skip] Space-Track：.env 缺少 SPACETRACK_USERNAME/PASSWORD")
        return []
    s = requests.Session()
    r = s.post("https://www.space-track.org/ajaxauth/login",
               data={"identity": user, "password": pwd}, timeout=60)
    r.raise_for_status()
    # 精确 NORAD / 名称（避免短词误匹配）
    specs = [
        ("GOES", "OBJECT_NAME/~~GOES%/OBJECT_TYPE/PAYLOAD"),
        ("DSCOVR", "NORAD_CAT_ID/40390"),
        ("SOHO", "NORAD_CAT_ID/23726"),
        ("ACE", "NORAD_CAT_ID/24912"),  # Advanced Composition Explorer often 24912? check
        ("WIND", "NORAD_CAT_ID/23333"),
        ("AURA", "NORAD_CAT_ID/28376"),
        ("INTEGRAL", "NORAD_CAT_ID/27540"),
        ("STEREO-A", "NORAD_CAT_ID/29510"),
        ("STEREO-B", "NORAD_CAT_ID/29511"),
        ("SDO", "NORAD_CAT_ID/36395"),
        ("TIMED", "NORAD_CAT_ID/26998"),
        ("METOP", "OBJECT_NAME/~~METOP/OBJECT_TYPE/PAYLOAD"),
        ("HIMAWARI", "OBJECT_NAME/~~HIMAWARI/OBJECT_TYPE/PAYLOAD"),
        ("FENGYUN", "OBJECT_NAME/~~FENGYUN/OBJECT_TYPE/PAYLOAD"),
        ("NOAA", "OBJECT_NAME/~~NOAA/OBJECT_TYPE/PAYLOAD"),
        ("SWARM", "OBJECT_NAME/~~SWARM/OBJECT_TYPE/PAYLOAD"),
        ("CLUSTER", "OBJECT_NAME/~~CLUSTER%20II/OBJECT_TYPE/PAYLOAD"),
        ("SBSS", "OBJECT_NAME/~~SBSS/OBJECT_TYPE/PAYLOAD"),
        ("GSSAP", "OBJECT_NAME/~~GSSAP/OBJECT_TYPE/PAYLOAD"),
        ("ORS-5", "OBJECT_NAME/~~ORS%205/OBJECT_TYPE/PAYLOAD"),
        ("SAPPHIRE", "OBJECT_NAME/~~SAPPHIRE/OBJECT_TYPE/PAYLOAD"),
        ("NEOSSAT", "OBJECT_NAME/~~NEOSSAT/OBJECT_TYPE/PAYLOAD"),
        ("ARASE", "OBJECT_NAME/~~ARASE/OBJECT_TYPE/PAYLOAD"),
    ]
    # ACE NORAD verify via search
    out, seen = [], set()
    for label, pred in specs:
        q = (f"https://www.space-track.org/basicspacedata/query/class/satcat/"
             f"{pred}/DECAY/null-val/format/json")
        try:
            rr = s.get(q, timeout=90)
            data = rr.json() if rr.ok else []
        except Exception as e:
            print(f"  Space-Track {label}: {e}")
            continue
        n = 0
        for x in data:
            if x.get("OBJECT_TYPE") and x["OBJECT_TYPE"] != "PAYLOAD":
                continue
            name = (x.get("OBJECT_NAME") or "").upper()
            if any(k in name for k in (" DEB", " R/B", "AKM", "PKM", "FREGAT")):
                continue
            # 过滤误匹配：ACE/WIND 短词
            if label == "NOAA" and not re.match(r"^NOAA\s*\d", x.get("OBJECT_NAME") or "", re.I):
                continue
            if label == "GOES" and "GOES" not in name:
                continue
            nid = x.get("NORAD_CAT_ID")
            if nid in seen:
                continue
            seen.add(nid)
            x["_channel_label"] = label
            out.append(x)
            n += 1
        print(f"  Space-Track {label}: +{n}")
        time.sleep(0.3)
    # ACE 单独用名称精确查
    for exact in ("ACE",):
        q = ("https://www.space-track.org/basicspacedata/query/class/satcat/"
             f"OBJECT_NAME/={exact}/OBJECT_TYPE/PAYLOAD/DECAY/null-val/format/json")
        rr = s.get(q, timeout=60)
        for x in (rr.json() if rr.ok else []):
            nid = x.get("NORAD_CAT_ID")
            if nid not in seen:
                seen.add(nid)
                x["_channel_label"] = exact
                out.append(x)
                print(f"  Space-Track exact {exact}: +1 ({nid})")
    fp.write_text(json.dumps(out))
    return out


def fetch_supermag(cache: Path) -> list[dict]:
    """SuperMAG 静态站表（IAGA/地理坐标/运营网络）。

    官方页面提供 ASCII 下载；这里使用其公开镜像，字段与官方 station_info.txt
    一致。与 INTERMAGNET 按 IAGA code 在 build 阶段去重。
    """
    fp = cache / "supermag_stations.txt"
    if not fp.exists() or fp.stat().st_size < 1000:
        url = "https://raw.githubusercontent.com/spacecataz/supermag/master/station_info.txt"
        try:
            fp.write_bytes(_get(url, timeout=60).content)
        except Exception as exc:
            print(f"  SuperMAG: {exc}")
            return []
    rows = []
    pattern = re.compile(
        r'^([A-Z0-9]{3})\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+'
        r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"([^"]+)"\s+'
        r'(\d+)\s*(.*)$'
    )
    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line)
        if not m:
            continue
        code, lon, lat, mlon, mlat, name, _, operators = m.groups()
        ops = re.findall(r'"([^"]+)"', operators)
        rows.append({
            "code": code, "name": name,
            "lat": float(lat), "lon": _norm_lon(float(lon)),
            "mlat": float(mlat), "mlon": _norm_lon(float(mlon)),
            "operators": "/".join(ops) or "SuperMAG",
        })
    print(f"  SuperMAG: {len(rows)}")
    return rows


def fetch_nmdb(cache: Path) -> list[dict]:
    """NMDB 中子监测站元数据（逐站公开页面，带坐标/海拔/截止刚度）。"""
    fp = cache / "nmdb_stations.json"
    if fp.exists() and fp.stat().st_size > 100:
        return json.loads(fp.read_text(encoding="utf-8"))
    base = "https://www.nmdb.eu/station/"
    try:
        root = _get(base, timeout=60).text
    except Exception as exc:
        print(f"  NMDB index: {exc}")
        return []
    # 前 21 个链接是国家聚合页；仅保留大写站代码条目。
    links = []
    for code, label in re.findall(r'href=/station/([^/]+)/>(.*?)</a>', root, re.I | re.S):
        label = re.sub(r"<[^>]+>", "", label).strip()
        if re.fullmatch(r"[A-Z0-9]+(?:,\s*[A-Z0-9]+)*", label) and code not in links:
            links.append(code)

    def one(code: str) -> dict | None:
        try:
            page = _get(base + code + "/", timeout=45).text
        except Exception:
            return None
        plain = html.unescape(re.sub(r"<[^>]+>", " ", page))
        plain = re.sub(r"\s+", " ", plain)
        title = re.sub(r"\s*\|\s*NMDB.*$", "", plain.strip())
        title = title.split(" Search Cosmic Rays", 1)[0].strip()
        lat_m = re.search(r"Geographic latitude\s+([0-9.]+).*?\s([NS])", plain, re.I)
        lon_m = re.search(r"Geographic longitude\s+([0-9.]+).*?\s([EW])", plain, re.I)
        alt_m = re.search(r"Altitude\s+([0-9.]+)\s*m", plain, re.I)
        rig_m = re.search(r"(?:cutoff rigidity).*?([0-9.]+)\s*GV", plain, re.I)
        if not lat_m:
            return None
        lat = float(lat_m.group(1)) * (-1 if lat_m.group(2).upper() == "S" else 1)
        # 极点站可能公开 longitude=N/A。
        lon = 0.0
        if lon_m:
            lon = float(lon_m.group(1)) * (-1 if lon_m.group(2).upper() == "W" else 1)
        return {
            "code": code.upper(), "name": title or code.upper(),
            "lat": lat, "lon": _norm_lon(lon),
            "alt_m": float(alt_m.group(1)) if alt_m else None,
            "cutoff_gv": float(rig_m.group(1)) if rig_m else None,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, code): code for code in links}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)
    rows.sort(key=lambda x: x["code"])
    fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  NMDB: {len(rows)}")
    return rows


def fetch_ilrs_active(cache: Path) -> list[dict]:
    """ILRS 在役卫星激光测距站（官方活动站表 + 各站坐标页）。"""
    fp = cache / "ilrs_active_stations.json"
    if fp.exists() and fp.stat().st_size > 100:
        return json.loads(fp.read_text(encoding="utf-8"))
    base = "https://ilrs.gsfc.nasa.gov"
    index_url = base + "/network/stations/active/"
    try:
        index = _get(index_url, timeout=60).text
    except Exception as exc:
        print(f"  ILRS index: {exc}")
        return []
    codes = []
    for code in re.findall(
        r"/network/stations/active/([A-Z0-9]+)_station_info\.html", index
    ):
        if code not in codes:
            codes.append(code)

    def one(code: str) -> dict | None:
        url = f"{base}/network/stations/active/{code}_station_info.html"
        try:
            page = _get(url, timeout=45).text
        except Exception:
            return None
        plain = html.unescape(re.sub(r"<[^>]+>", " ", page))
        plain = re.sub(r"\s+", " ", plain)
        lat_m = re.search(r"Latitude\s*\[deg\]\s*([0-9.]+)\s*([NS])", plain, re.I)
        lon_m = re.search(r"Longitude\s*\[deg\]\s*([0-9.]+)\s*([EW])", plain, re.I)
        alt_m = re.search(r"Elevation\s*\[m\]\s*([0-9.]+)", plain, re.I)
        if not (lat_m and lon_m):
            return None
        lat = float(lat_m.group(1)) * (-1 if lat_m.group(2).upper() == "S" else 1)
        lon = float(lon_m.group(1)) * (-1 if lon_m.group(2).upper() == "W" else 1)
        # 页面 title 通常为 “CODE - Location | ILRS” 或含站名；无则用代码。
        title_m = re.search(r"<title>(.*?)</title>", page, re.I | re.S)
        name = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else code
        if not name or name.lower().startswith("international laser"):
            name = f"ILRS {code}"
        return {
            "code": code, "name": name, "lat": lat, "lon": _norm_lon(lon),
            "alt_m": float(alt_m.group(1)) if alt_m else None,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, code): code for code in codes}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)
    rows.sort(key=lambda x: x["code"])
    fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  ILRS active: {len(rows)}")
    return rows


def _geocentric_to_geodetic(lon_deg: float, rho_cos: float, rho_sin: float) -> tuple[float, float]:
    """MPC 视差常数近似转 WGS84 纬经度（足够用于站点地图）。"""
    lat = __import__("math").degrees(__import__("math").atan2(rho_sin, rho_cos))
    return lat, _norm_lon(lon_deg)


def fetch_mpc_ssa_candidates(cache: Path) -> list[dict]:
    """MPC 中名称明确指向 Spaceguard/Space Surveillance/Tracking 的地基站。

    不把 2700+ 个普通小行星观测站全部冒充 SSA 设备；只收录名称明确的空间监视
    节点，并在 notes 中标注其属于“公开光学天体测量候选能力”。
    """
    fp = cache / "mpc_ssa_candidates.json"
    if fp.exists() and fp.stat().st_size > 100:
        return json.loads(fp.read_text(encoding="utf-8"))
    try:
        payload = _get(
            "https://data.minorplanetcenter.net/api/obscodes",
            timeout=90, json={},
        ).json()
    except Exception as exc:
        print(f"  MPC obscodes: {exc}")
        return []
    # 排除明确天基望远镜；只保留名称直接声明 surveillance/spaceguard/tracking。
    include = re.compile(
        r"(space\s*surveillance|spaceguard|space\s*tracking|sstac|spacewatch|survey)",
        re.I,
    )
    exclude = re.compile(
        r"(space telescope|spacecraft|geocentric|occultation|neo surveyor)",
        re.I,
    )
    rows = []
    for code, x in (payload.items() if isinstance(payload, dict) else []):
        name = str(x.get("name") or "")
        if not include.search(name) or exclude.search(name):
            continue
        try:
            lon = float(x["longitude"])
            lat, lon = _geocentric_to_geodetic(
                lon, float(x["rhocosphi"]), float(x["rhosinphi"])
            )
        except Exception:
            continue
        rows.append({
            "code": code, "name": name.strip(), "lat": lat, "lon": lon,
            "lastdate": x.get("lastdate"), "observations_type": x.get("observations_type"),
        })
    fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  MPC explicit SSA candidates: {len(rows)}")
    return rows


def fetch_oscar_space_weather(cache: Path) -> list[dict]:
    """WMO OSCAR/Space 中明确的空间天气仪器类型子集。"""
    fp = cache / "oscar_space_weather_instruments.json"
    if fp.exists() and fp.stat().st_size > 100:
        return json.loads(fp.read_text(encoding="utf-8"))
    params = {
        "iDisplayLength": 5000, "iDisplayStart": 0,
        "iSortCol_0": 0, "sSortDir_0": "asc", "draw": 1,
    }
    try:
        r = requests.get(
            "https://space.oscar.wmo.int/instruments",
            params=params, timeout=120, verify=False,
            headers={
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": "debris-monitor/1.0",
            },
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        print(f"  OSCAR/Space: {exc}")
        return []
    relevant_types = {
        "Energetic particle spectrometer",
        "Solar activity monitor",
        "Field or radiowave sensor",
        "Solar irradiance monitor",
    }

    def clean(v) -> str:
        return html.unescape(re.sub(r"<[^>]+>", " ", str(v or ""))).strip()

    rows = []
    for raw in payload.get("aaData") or []:
        if len(raw) < 7:
            continue
        itype = clean(raw[3])
        if itype not in relevant_types:
            continue
        href = re.search(r'href="([^"]+)"', str(raw[0]))
        slug = (href.group(1).rstrip("/").split("/")[-1] if href else clean(raw[0]))
        rows.append({
            "slug": slug, "name": clean(raw[0]), "full_name": clean(raw[1]),
            "agency": clean(raw[2]), "instrument_type": itype,
            "satellites": clean(raw[4]), "usage_from": clean(raw[5]), "usage_to": clean(raw[6]),
        })
    fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  OSCAR/Space SWx subset: {len(rows)}")
    return rows


def fetch_spase_space_instruments(cache: Path) -> list[dict]:
    """NASA/HPDE SPASE-SMWG 仪器注册表中的天基日地物理仪器。

    以 ObservatoryRegion 是否包含 Earth.Surface 区分地基/天基；排除
    InstrumentType=Platform（平台本身不是传感器）。
    """
    out_fp = cache / "spase_space_instruments.json"
    if out_fp.exists() and out_fp.stat().st_size > 100:
        return json.loads(out_fp.read_text(encoding="utf-8"))
    archive = cache / "spase_smwg_master.tar.gz"
    if not archive.exists() or archive.stat().st_size < 100_000:
        try:
            archive.write_bytes(_get(
                "https://codeload.github.com/hpde/SMWG/tar.gz/refs/heads/master",
                timeout=180,
            ).content)
        except Exception as exc:
            print(f"  SPASE archive: {exc}")
            return []

    def local(el) -> str:
        return el.tag.split("}")[-1]

    def first(root, tag: str) -> str | None:
        for el in root.iter():
            if local(el) == tag and el.text and el.text.strip():
                return el.text.strip()
        return None

    obs_regions: dict[str, set[str]] = {}
    rows = []
    try:
        with tarfile.open(archive, "r:gz") as tf:
            names = tf.getnames()
            for name in names:
                if "/Observatory/" not in name or not name.endswith(".xml"):
                    continue
                try:
                    root = ET.fromstring(tf.extractfile(name).read())
                except Exception:
                    continue
                rid = first(root, "ResourceID")
                if not rid:
                    continue
                obs_regions[rid] = {
                    el.text.strip() for el in root.iter()
                    if local(el) == "ObservatoryRegion" and el.text
                }
            for name in names:
                if "/Instrument/" not in name or not name.endswith(".xml"):
                    continue
                try:
                    root = ET.fromstring(tf.extractfile(name).read())
                except Exception:
                    continue
                rid = first(root, "ResourceID")
                oid = first(root, "ObservatoryID")
                itypes = [
                    el.text.strip() for el in root.iter()
                    if local(el) == "InstrumentType" and el.text
                ]
                if not rid or not oid or "Platform" in itypes:
                    continue
                regions = obs_regions.get(oid, set())
                if "Earth.Surface" in regions:
                    continue
                # ResourceHeader 下的 ResourceName 是设备名。
                resource_name = None
                for el in root.iter():
                    if local(el) == "ResourceHeader":
                        resource_name = first(el, "ResourceName")
                        break
                rows.append({
                    "resource_id": rid, "name": resource_name or rid.rsplit("/", 1)[-1],
                    "observatory_id": oid, "instrument_types": "/".join(itypes) or "Unspecified",
                    "regions": "/".join(sorted(regions)),
                })
    except Exception as exc:
        print(f"  SPASE parse: {exc}")
        return []
    out_fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  SPASE space instruments: {len(rows)}")
    return rows


def _discos_headers() -> dict[str, str] | None:
    token = (os.environ.get("ESA_DISCOS_TOKEN") or "").strip()
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "DiscosWeb-Api-Version": "2",
        "Accept": "application/vnd.api+json",
        "User-Agent": "debris-monitor/1.0",
    }


def _discos_paginate(path: str, cache_fp: Path, *, page_size: int = 100) -> list[dict]:
    """拉取 DISCOSweb JSON:API 列表；有缓存则优先读本地（可离线重放）。"""
    if cache_fp.exists() and cache_fp.stat().st_size > 50:
        print(f"  [cache] DISCOS {cache_fp.name}")
        return json.loads(cache_fp.read_text(encoding="utf-8"))
    headers = _discos_headers()
    if not headers:
        print(f"  DISCOS {path}: 无 ESA_DISCOS_TOKEN，且无缓存 — 跳过")
        return []
    base = "https://discosweb.esoc.esa.int/api"
    url = f"{base}/{path}"
    params: dict[str, Any] = {"page[size]": page_size, "page[number]": 1}
    rows: list[dict] = []
    while url:
        r = requests.get(url, headers=headers, params=params, timeout=90)
        if r.status_code == 401:
            print(f"  DISCOS {path}: 401 Unauthorized — 检查 ESA_DISCOS_TOKEN")
            return []
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("data") or []:
            attrs = item.get("attributes") or {}
            rows.append({"discos_id": str(item.get("id")), **attrs})
        print(f"  DISCOS {path}: page {params.get('page[number]', '?')} → {len(rows)}", flush=True)
        nxt = (payload.get("links") or {}).get("next")
        if not nxt:
            break
        url = nxt if nxt.startswith("http") else f"https://discosweb.esoc.esa.int{nxt}"
        params = {}
        time.sleep(0.35)
    # 规范化缓存字段（camelCase → snake 便于离线复用）
    normed = []
    for x in rows:
        normed.append({
            "discos_id": x.get("discos_id"),
            "name": x.get("name"),
            "lat": x.get("latitude") if "latitude" in x else x.get("lat"),
            "lon": x.get("longitude") if "longitude" in x else x.get("lon"),
            "alt_m": x.get("altitude") if "altitude" in x else x.get("alt_m"),
            "pads": json.dumps(x["pads"], ensure_ascii=False) if isinstance(x.get("pads"), (list, dict)) else x.get("pads"),
            "azimuths": json.dumps(x["azimuths"], ensure_ascii=False) if isinstance(x.get("azimuths"), (list, dict)) else x.get("azimuths"),
            "constraints": json.dumps(x["constraints"], ensure_ascii=False) if isinstance(x.get("constraints"), (list, dict)) else x.get("constraints"),
            "date_range": json.dumps(x["dateRange"], ensure_ascii=False) if isinstance(x.get("dateRange"), (list, dict)) else (
                x.get("dateRange") or x.get("date_range")
            ),
        })
    cache_fp.write_text(json.dumps(normed, ensure_ascii=False), encoding="utf-8")
    return normed


def fetch_discos_launch_sites(cache: Path) -> list[dict]:
    return _discos_paginate("launch-sites", cache / "discos_launch_sites.json")


def fetch_discos_organisations(cache: Path) -> list[dict]:
    return _discos_paginate("organisations", cache / "discos_organisations.json")


def fetch_brahe_groundstations(cache: Path) -> list[dict]:
    """Brahe 社区地面站（AWS / KSAT / DSN / Atlas / Leaf / NEN / SSC / Viasat）。"""
    out_dir = cache / "brahe"
    out_dir.mkdir(parents=True, exist_ok=True)
    providers = ("atlas", "aws", "dsn", "ksat", "leaf", "nen", "ssc", "viasat")
    features: list[dict] = []
    for name in providers:
        fp = out_dir / f"{name}.json"
        if not fp.exists() or fp.stat().st_size < 50:
            url = (f"https://raw.githubusercontent.com/duncaneddy/brahe/main/"
                   f"data/groundstations/{name}.json")
            try:
                r = _get(url, timeout=40)
                fp.write_bytes(r.content)
            except Exception as e:
                print(f"  Brahe {name}: {e}")
                continue
        try:
            gj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Brahe {name} parse: {e}")
            continue
        feats = gj.get("features") if isinstance(gj, dict) else gj
        n = 0
        for f in feats or []:
            if not isinstance(f, dict):
                continue
            geom = f.get("geometry") or {}
            props = f.get("properties") or {}
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            alt = float(coords[2]) if len(coords) > 2 else None
            bands = props.get("frequency_bands") or []
            if isinstance(bands, list):
                bands_s = "/".join(str(b) for b in bands)
            else:
                bands_s = str(bands) if bands else None
            provider = props.get("provider") or name
            st_name = props.get("name") or f"{provider}-{lat:.2f}"
            features.append(dict(
                provider=str(provider),
                name=str(st_name),
                lat=lat, lon=_norm_lon(lon), alt_m=alt,
                bands=bands_s,
                source_file=name,
            ))
            n += 1
        print(f"  Brahe {name}: +{n}")
    return features


def fetch_starlink_ground_stations(cache: Path) -> list[dict]:
    """Satellitemap 类社区公开站表（Starlink POP/gateway，juliensimon/starlink-viz）。"""
    fp = cache / "starlink_ground_stations.json"
    if not fp.exists() or fp.stat().st_size < 100:
        url = ("https://raw.githubusercontent.com/juliensimon/starlink-viz/"
               "master/data/ground-stations.json")
        try:
            r = _get(url, timeout=60)
            fp.write_bytes(r.content)
        except Exception as e:
            print(f"  Starlink GS: {e}")
            return []
    else:
        print(f"  [cache] Starlink {fp.name}")
    try:
        payload = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  Starlink parse: {e}")
        return []
    stations = payload.get("stations") if isinstance(payload, dict) else payload
    out = []
    for s in stations or []:
        try:
            lat, lon = float(s["lat"]), float(s["lon"])
        except Exception:
            continue
        out.append(dict(
            name=s.get("name") or f"Starlink-{lat:.3f},{lon:.3f}",
            lat=lat, lon=_norm_lon(lon),
            status=s.get("status") or "operational",
            stype=s.get("type") or "gateway",
            last_updated=(payload.get("lastUpdated") if isinstance(payload, dict) else None),
        ))
    print(f"  Starlink community GS: {len(out)}")
    return out


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^\w]+", "-", (s or "").strip(), flags=re.UNICODE)
    return (s.strip("-") or "x")[:n]


# ── Curated public catalogs ──────────────────────────────────────────────────
# 国家代码粗映射
_CC = {
    "US": "美国", "USA": "美国", "PRC": "中国", "CN": "中国", "CIS": "俄罗斯", "RU": "俄罗斯",
    "UK": "英国", "GB": "英国", "FR": "法国", "JPN": "日本", "JP": "日本", "IND": "印度",
    "IN": "印度", "ESA": "欧空局", "EU": "欧空局", "GER": "德国", "DE": "德国", "IT": "意大利",
    "CA": "加拿大", "AU": "澳大利亚", "AS": "澳大利亚", "NO": "挪威", "SW": "瑞典", "SE": "瑞典",
    "ES": "西班牙", "AR": "阿根廷", "ZA": "南非", "BR": "巴西", "KR": "韩国", "TW": "中国台湾",
    "ET": "埃塞俄比亚", "UA": "乌克兰", "WS": "萨摩亚", "NZ": "新西兰", "CL": "智利",
    "GR": "希腊", "PT": "葡萄牙", "AE": "阿联酋", "SG": "新加坡", "MU": "毛里求斯",
    "GL": "格陵兰", "AQ": "南极", "BIOT": "英属印度洋领地",
}


def _cc(code: str) -> str:
    return _CC.get((code or "").upper(), code or "未知")


CURATED_SSA = [
    # SSN / GEODSS / 公开雷达（公开事实页 / 文献）
    dict(sensor_id="SSN-GEODSS-SOC-CAM1", name="GEODSS Socorro CAM1", name_cn="GEODSS 索科罗 1号镜",
         sensor_class="ground_optical", network="SSN/GEODSS", operator="USSF 15 SPSS",
         country="美国", lat=33.817, lon=-106.660, alt_m=1450, frequency_band="optical",
         capability="1.2m 深空光学监视；MEO/GEO/HEO", channel="curated:USSF-GEODSS"),
    dict(sensor_id="SSN-GEODSS-MAU-CAM1", name="GEODSS Maui CAM1 (Haleakala)", name_cn="GEODSS 毛伊 1号镜",
         sensor_class="ground_optical", network="SSN/GEODSS", operator="USSF 15 SPSS",
         country="美国", lat=20.708, lon=-156.257, alt_m=3058, frequency_band="optical",
         capability="1.2m 深空光学监视；MEO/GEO/HEO", channel="curated:USSF-GEODSS"),
    dict(sensor_id="SSN-GEODSS-DGO-CAM1", name="GEODSS Diego Garcia CAM1", name_cn="GEODSS 迪戈加西亚 1号镜",
         sensor_class="ground_optical", network="SSN/GEODSS", operator="USSF 15 SPSS",
         country="英属印度洋领地", lat=-7.411, lon=72.452, alt_m=5, frequency_band="optical",
         capability="1.2m 深空光学监视；MEO/GEO/HEO", channel="curated:USSF-GEODSS"),
    dict(sensor_id="SSN-Eglin-ANFPS85", name="Eglin AN/FPS-85", name_cn="埃格林 AN/FPS-85",
         sensor_class="ground_radar", network="SSN", operator="USSF",
         country="美国", lat=30.572, lon=-86.215, alt_m=30, frequency_band="UHF",
         capability="LEO/MEO 相控阵跟踪", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-Millstone", name="Millstone Hill Radar", name_cn="米尔石山雷达",
         sensor_class="ground_radar", network="SSN", operator="MIT LL",
         country="美国", lat=42.619, lon=-71.491, alt_m=150, frequency_band="L/UHF",
         capability="精密定轨", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-Haystack", name="Haystack / HAX", name_cn="干草堆/HAX 雷达",
         sensor_class="ground_radar", network="SSN", operator="MIT LL",
         country="美国", lat=42.623, lon=-71.488, alt_m=150, frequency_band="X",
         capability="小尺寸碎片统计", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-Globus-II", name="Globus II Vardo", name_cn="Globus II 瓦尔德",
         sensor_class="ground_radar", network="SSN", operator="NOR/USSF",
         country="挪威", lat=70.367, lon=31.127, alt_m=300, frequency_band="X",
         capability="高纬深空监视", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-Cobalt", name="Cobra Dane Shemya", name_cn="Cobra Dane 谢米亚",
         sensor_class="ground_radar", network="SSN", operator="USSF",
         country="美国", lat=52.737, lon=174.092, alt_m=100, frequency_band="L",
         capability="相控阵空间监视", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-Clear-SSPARS", name="Clear SSPARs", name_cn="克利尔固态相控阵",
         sensor_class="ground_radar", network="SSN", operator="USSF",
         country="美国", lat=64.290, lon=-149.190, alt_m=200, frequency_band="UHF",
         capability="导弹预警兼空间监视", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-CapeCod-SSPARS", name="Cape Cod SSPARs", name_cn="科德角固态相控阵",
         sensor_class="ground_radar", network="SSN", operator="USSF",
         country="美国", lat=41.752, lon=-70.538, alt_m=50, frequency_band="UHF",
         capability="导弹预警兼空间监视", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-Thule-SSPARS", name="Thule/Pituffik SSPARs", name_cn="图勒固态相控阵",
         sensor_class="ground_radar", network="SSN", operator="USSF",
         country="格陵兰", lat=76.570, lon=-68.300, alt_m=200, frequency_band="UHF",
         capability="导弹预警兼空间监视", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-Fylingdales", name="RAF Fylingdales", name_cn="法林代尔斯",
         sensor_class="ground_radar", network="SSN", operator="UK RAF/USSF",
         country="英国", lat=54.358, lon=-0.670, alt_m=250, frequency_band="UHF",
         capability="导弹预警兼空间监视", channel="curated:SSN-radar"),
    dict(sensor_id="SSN-MOTIF", name="Maui Space Surveillance Complex", name_cn="毛伊空间监视综合体",
         sensor_class="ground_optical", network="SSN/MSSC", operator="USSF",
         country="美国", lat=20.708, lon=-156.258, alt_m=3058, frequency_band="optical",
         capability="自适应光学/成像", channel="curated:SSN-optical"),
    dict(sensor_id="SSN-18SDS", name="USSF 18th SDS SSN Hub", name_cn="第18太空防御中队枢纽",
         sensor_class="network_node", network="SSN", operator="USSF 18 SDS",
         country="美国", lat=38.880, lon=-104.760, alt_m=1800, frequency_band="multi",
         capability="编目与 Space-Track 发布", channel="curated:docs"),
    dict(sensor_id="TRACSS-OASIS", name="TraCSS-OASIS", name_cn="TraCSS OASIS 数据库",
         sensor_class="network_node", network="TraCSS", operator="NOAA OSC",
         country="美国", lat=38.990, lon=-77.030, alt_m=50, frequency_band="multi",
         capability="民用空间交通协调", channel="curated:docs"),
    dict(sensor_id="ISON-HUB", name="ISON Network Hub", name_cn="ISON 光学网络枢纽",
         sensor_class="ground_optical", network="ISON", operator="ISON Consortium",
         country="国际/多国", lat=55.750, lon=37.620, alt_m=150, frequency_band="optical",
         capability="中高轨光学监测", channel="curated:docs"),
    dict(sensor_id="TMDS-TW", name="Taiwan Meteor Detection System", name_cn="台湾流星观测系统",
         sensor_class="ground_optical", network="TMDS", operator="学术网络",
         country="中国台湾", lat=23.700, lon=120.960, alt_m=100, frequency_band="optical",
         capability="多站三角测量 LEO 定轨", channel="curated:docs"),
    dict(sensor_id="LOOKUP-SORASYS", name="LookUpSpace SORASYS", name_cn="LookUpSpace SORASYS 雷达",
         sensor_class="ground_radar", network="LookUpSpace", operator="LookUpSpace",
         country="商业/多国", lat=48.860, lon=2.350, alt_m=50, frequency_band="HF",
         capability="高频 LEO 碎片跟踪", channel="curated:docs"),
    dict(sensor_id="ESA-SST", name="ESA SST Coordination", name_cn="ESA SST 协调节点",
         sensor_class="network_node", network="ESA-SST", operator="ESA",
         country="欧空局", lat=49.870, lon=8.620, alt_m=100, frequency_band="multi",
         capability="欧洲 SST 数据协调", channel="curated:docs"),
    dict(sensor_id="CN-CAS-SSN", name="CAS Space Surveillance Node", name_cn="中科院空间目标监视节点",
         sensor_class="network_node", network="CAS-SSN", operator="中科院",
         country="中国", lat=40.000, lon=116.380, alt_m=50, frequency_band="multi",
         capability="天/地基监视协调", channel="curated:docs"),
    # 欧/日/加/俄/澳等公开 SSA 站（ESA SST / Spaceguard / 文献坐标）
    dict(sensor_id="ESA-SST-OGS", name="ESA Optical Ground Station Tenerife", name_cn="ESA OGS 特内里费",
         sensor_class="ground_optical", network="ESA-SST", operator="ESA", country="西班牙",
         lat=28.300, lon=-16.510, alt_m=2390, frequency_band="optical",
         capability="空间碎片光学观测", channel="curated:public-SSA"),
    dict(sensor_id="ESA-SST-ZIM", name="Zimmerwald Observatory", name_cn="齐默尔瓦尔德天文台",
         sensor_class="ground_optical", network="ESA-SST", operator="AIUB", country="瑞士",
         lat=46.877, lon=7.465, alt_m=907, frequency_band="optical",
         capability="空间碎片/卫星激光测距", channel="curated:public-SSA"),
    dict(sensor_id="ESA-SST-GRAZ", name="Graz Lustbuehel", name_cn="格拉茨",
         sensor_class="ground_optical", network="ESA-SST", operator="IWF", country="奥地利",
         lat=47.067, lon=15.493, alt_m=540, frequency_band="optical",
         capability="激光测距/光学", channel="curated:public-SSA"),
    dict(sensor_id="SSN-STARFIRE", name="Starfire Optical Range", name_cn="星火光学靶场",
         sensor_class="ground_optical", network="SSN", operator="USAF AFRL", country="美国",
         lat=34.965, lon=-106.462, alt_m=1877, frequency_band="optical",
         capability="自适应光学", channel="curated:public-SSA"),
    dict(sensor_id="CNES-TAROT-CHI", name="TAROT Chile", name_cn="TAROT 智利",
         sensor_class="ground_optical", network="ESA-SST/CNES", operator="CNES", country="智利",
         lat=-29.261, lon=-70.739, alt_m=2347, frequency_band="optical",
         capability="快速光学巡天", channel="curated:public-SSA"),
    dict(sensor_id="CNES-TAROT-CAL", name="TAROT Calern", name_cn="TAROT 卡勒恩",
         sensor_class="ground_optical", network="ESA-SST/CNES", operator="CNES", country="法国",
         lat=43.752, lon=6.923, alt_m=1270, frequency_band="optical",
         capability="快速光学巡天", channel="curated:public-SSA"),
    dict(sensor_id="CNES-TAROT-REU", name="TAROT Reunion", name_cn="TAROT 留尼汪",
         sensor_class="ground_optical", network="ESA-SST/CNES", operator="CNES", country="法国",
         lat=-21.199, lon=55.410, alt_m=1000, frequency_band="optical",
         capability="快速光学巡天", channel="curated:public-SSA"),
    dict(sensor_id="JP-BSDC", name="Bisei Spaceguard Center", name_cn="美星空间卫士中心",
         sensor_class="ground_optical", network="Japan Spaceguard", operator="JSGA", country="日本",
         lat=34.672, lon=133.545, alt_m=420, frequency_band="optical",
         capability="近地天体/碎片光学", channel="curated:public-SSA"),
    dict(sensor_id="JP-KSGC", name="Kamisaibara Spaceguard Center", name_cn="上斋原空间卫士中心",
         sensor_class="ground_radar", network="Japan Spaceguard", operator="JSGA", country="日本",
         lat=35.175, lon=133.916, alt_m=450, frequency_band="UHF",
         capability="空间监视雷达", channel="curated:public-SSA"),
    dict(sensor_id="CA-SAPPHIRE", name="Sapphire SSA Satellite", name_cn="Sapphire 天基监视",
         sensor_class="spaceborne", network="Canadian SSA", operator="DND/CAF", country="加拿大",
         lat=0.0, lon=0.0, alt_m=None, frequency_band="optical",
         capability="天基光学 SSA", channel="curated:public-SSA"),
    dict(sensor_id="CA-NEOSSat", name="NEOSSat", name_cn="NEOSSat",
         sensor_class="spaceborne", network="CSA", operator="CSA", country="加拿大",
         lat=0.0, lon=0.0, alt_m=None, frequency_band="optical",
         capability="近地天体/SSA", channel="curated:public-SSA"),
    dict(sensor_id="RU-OKNO", name="Okno Space Surveillance", name_cn="窗口光学电子系统",
         sensor_class="ground_optical", network="Russian SSN", operator="VKO", country="俄罗斯",
         lat=38.558, lon=69.375, alt_m=2200, frequency_band="optical",
         capability="光学电子监视", channel="curated:public-SSA"),
    dict(sensor_id="RU-KRONA", name="Krona Radar-Optical", name_cn="克朗纳雷达光学",
         sensor_class="ground_radar", network="Russian SSN", operator="VKO", country="俄罗斯",
         lat=43.824, lon=41.345, alt_m=2100, frequency_band="multi",
         capability="雷达/光学综合", channel="curated:public-SSA"),
    dict(sensor_id="AU-EOS", name="EOS Space Systems Mt Stromlo", name_cn="EOS 斯特罗姆洛山",
         sensor_class="ground_optical", network="Australian SSA", operator="EOS", country="澳大利亚",
         lat=-35.320, lon=149.007, alt_m=770, frequency_band="optical",
         capability="激光测距/光学 SSA", channel="curated:public-SSA"),
    dict(sensor_id="DE-TIRA", name="TIRA Tracking Radar", name_cn="TIRA 跟踪雷达",
         sensor_class="ground_radar", network="Fraunhofer FHR", operator="FHR", country="德国",
         lat=50.616, lon=7.130, alt_m=300, frequency_band="L/Ku",
         capability="空间目标雷达", channel="curated:public-SSA"),
    dict(sensor_id="DE-GESTRA", name="GESTRA Experimental Radar", name_cn="GESTRA",
         sensor_class="ground_radar", network="DLR/FHR", operator="DLR", country="德国",
         lat=50.616, lon=7.130, alt_m=300, frequency_band="UHF",
         capability="实验空间监视雷达", channel="curated:public-SSA"),
    dict(sensor_id="UK-Starbrook", name="Starbrook Optical", name_cn="Starbrook 光学",
         sensor_class="ground_optical", network="UK SST", operator="UKSA", country="英国",
         lat=51.150, lon=-1.830, alt_m=100, frequency_band="optical",
         capability="英国 SST 光学", channel="curated:public-SSA"),
    dict(sensor_id="IT-MLRO", name="Matera MLRO", name_cn="马泰拉激光测距",
         sensor_class="ground_optical", network="ASI", operator="ASI", country="意大利",
         lat=40.649, lon=16.704, alt_m=536, frequency_band="optical",
         capability="卫星激光测距", channel="curated:public-SSA"),
    dict(sensor_id="FR-MeO", name="MeO Laser Ranging Calern", name_cn="MeO 激光测距",
         sensor_class="ground_optical", network="CNES/OCA", operator="OCA", country="法国",
         lat=43.755, lon=6.921, alt_m=1270, frequency_band="optical",
         capability="月球/卫星激光测距", channel="curated:public-SSA"),
    dict(sensor_id="ES-ROA", name="ROA San Fernando", name_cn="圣费尔南多天文台",
         sensor_class="ground_optical", network="ESA-SST", operator="ROA", country="西班牙",
         lat=36.464, lon=-6.206, alt_m=30, frequency_band="optical",
         capability="光学观测", channel="curated:public-SSA"),
]


def _optical_sensor(
    sensor_id: str, name: str, network: str, operator: str, country: str,
    lat: float, lon: float, alt_m: float | None, capability: str, channel: str,
) -> dict:
    return dict(
        sensor_id=sensor_id, name=name, name_cn=None,
        sensor_class="ground_optical", network=network, operator=operator,
        country=country, lat=lat, lon=lon, alt_m=alt_m,
        frequency_band="optical", capability=capability, channel=channel,
    )


# 逐台可公开定位的光学 SSA 增量。商业网络只保留聚合节点，
# 不将“350+ / 200+”宣传总量伪造为数百个同坐标设备。
CURATED_SSA_OPTICAL_EXPANSION = [
    # GEODSS：USSF 官方明确 3 个站、每站 3 台 1.2 m 望远镜。
    *[
        _optical_sensor(
            f"SSN-GEODSS-{site}-CAM{cam}", f"GEODSS {name} CAM{cam}",
            "SSN/GEODSS", "USSF 15 SPSS", country, lat, lon, alt,
            "1.2m 深空光学监视；MEO/GEO/HEO", "curated:USSF-GEODSS",
        )
        for site, name, country, lat, lon, alt in (
            ("SOC", "Socorro", "美国", 33.817, -106.660, 1450),
            ("MAU", "Maui", "美国", 20.708, -156.257, 3058),
            ("DGO", "Diego Garcia", "英属印度洋领地", -7.411, 72.452, 5),
        )
        for cam in (2, 3)
    ],
    # Falcon Telescope Network：USAFA 官方站表，0.5 m 遥控望远镜。
    *[
        _optical_sensor(
            f"FTN-{code}", f"Falcon Telescope Network {name}",
            "Falcon Telescope Network", "USAFA CSSAR", country, lat, lon, alt,
            "0.5m 机器人望远镜；卫星光度/光谱表征；LEO-MEO-GEO",
            "curated:USAFA-FTN",
        )
        for code, name, country, lat, lon, alt in (
            ("USAFA", "USAFA Campus", "美国", 39.010, -104.880, 2212),
            ("FRH", "Farish", "美国", 39.010, -104.990, 2790),
            ("YDR", "Yoder", "美国", 38.890, -104.200, 1961),
            ("CMU", "Grand Junction", "美国", 39.960, -108.240, 1380),
            ("FLC", "Durango", "美国", 37.270, -107.870, 1880),
            ("NJC", "Sterling", "美国", 40.650, -103.200, 1177),
            ("OJC", "La Junta", "美国", 37.970, -103.540, 1221),
            ("PSU", "State College", "美国", 40.860, -77.830, 317),
            ("MMO", "Vicuna", "智利", -29.990, -70.680, 1139),
            ("CBR", "Canberra", "澳大利亚", -35.280, 149.170, 600),
            ("GDC", "Gingin", "澳大利亚", -31.360, 115.710, 18),
            ("TUBS", "Braunschweig", "德国", 52.270, 10.540, 73),
        )
    ],
    # KASI OWL-Net：五个相同的 0.5 m 机器人站。
    *[
        _optical_sensor(
            f"OWL-{code}", f"OWL-Net {name}", "KASI OWL-Net", "KASI",
            country, lat, lon, alt, "0.5m/1.1° FOV；LEO/GEO 光学跟踪",
            "curated:KASI-OWL",
        )
        for code, name, country, lat, lon, alt in (
            ("KOR", "Daejeon Testbed", "韩国", 36.397635, 127.375679, 139),
            ("MNG", "Songino", "蒙古", 47.886126, 106.334762, 1674),
            ("MAR", "Oukaimeden", "摩洛哥", 31.206472, -7.866500, 2725),
            ("ISR", "Wise Observatory", "以色列", 30.595833, 34.763333, 875),
            ("USA", "Mt Lemmon", "美国", 32.443333, -110.788056, 2790),
        )
    ],
    _optical_sensor(
        "CNES-TAROT-GIN", "TAROT / Zadko Gingin", "ESA-SST/CNES",
        "UWA/CNES", "澳大利亚", -31.3567, 115.7136, 50,
        "1m Zadko；快速光学巡天/卫星跟踪", "curated:public-SSA",
    ),
    # ISON：论文明确给出设备型号，坐标由 MPC 站码或机构站址交叉校准。
    *[
        _optical_sensor(
            f"ISON-{code}", f"ISON {instrument} @ {site}", "ISON",
            "KIAM/partner observatory", country, lat, lon, alt,
            f"{aperture}；GEO/HEO/MEO 空间碎片巡天/跟踪", "curated:ISON-public",
        )
        for code, instrument, site, country, lat, lon, alt, aperture in (
            ("NAU-ZTSH", "ZTSh", "Nauchnyi", "乌克兰", 44.727, 34.016, 600, "2.6m"),
            ("NAU-AT64", "AT-64", "Nauchnyi", "乌克兰", 44.727, 34.016, 600, "0.64m"),
            ("SIM-Z1000", "Zeiss-1000", "Simeiz", "乌克兰", 44.406, 33.997, 360, "1.0m"),
            ("ABA-AS32", "AS-32", "Abastumani", "格鲁吉亚", 41.755, 42.820, 1650, "0.7m"),
            ("MAY-RK800", "RK-800", "Odesa-Mayaki", "乌克兰", 46.397, 30.272, 40, "0.8m"),
            ("MAI-AZT22", "AZT-22", "Maidanak", "乌兹别克斯坦", 38.673, 66.896, 2593, "1.5m"),
            ("MAI-Z600", "Zeiss-600", "Maidanak", "乌兹别克斯坦", 38.673, 66.896, 2593, "0.6m"),
            ("ROZ-Z2000", "Zeiss-2000", "Rozhen", "保加利亚", 41.693, 24.738, 1759, "2.0m"),
            ("ROZ-Z600", "Zeiss-600", "Rozhen", "保加利亚", 41.693, 24.738, 1759, "0.6m"),
            ("TSH-Z1000", "Zeiss-1000", "Tien-Shan", "哈萨克斯坦", 43.058, 76.972, 2735, "1.0m"),
            ("MAYH-S400", "SANTEL-400A", "Mayhill", "美国", 32.903, -105.528, 2225, "0.4m"),
            ("SSO-ASA16", "ASA 16-inch", "Siding Spring", "澳大利亚", -31.273, 149.069, 1165, "0.4m"),
            ("CAS-22", "ORI/22cm", "Castelgrande", "意大利", 40.817, 15.463, 1250, "0.22m"),
            ("USS-22", "ORI/22cm", "Ussuriysk", "俄罗斯", 43.700, 132.166, 200, "0.22m"),
            ("KHU-ORI40", "ORI-40", "Khureltogoot", "蒙古", 47.865, 107.053, 1620, "0.4m"),
            ("KHU-VT78", "VT-78", "Khureltogoot", "蒙古", 47.865, 107.053, 1620, "0.192m"),
            ("URU-N1M", "Nanshan 1m Wide-field", "Urumqi", "中国", 43.467, 87.167, 2080, "1.0m"),
            ("KOT-OSTS", "OSTS", "Kottamia", "埃及", 29.933, 31.882, 480, "光学跟踪站"),
        )
    ],
    # 商业网络仅作聚合记录；精确站表和逐台坐标未公开。
    dict(
        sensor_id="EXOANALYTIC-EGTN-AGG", name="ExoAnalytic Global Telescope Network",
        name_cn="ExoAnalytic 全球望远镜网络", sensor_class="network_node",
        network="ExoAnalytic EGTN", operator="ExoAnalytic/Anduril",
        country="国际/多国", lat=0.0, lon=0.0, alt_m=None, frequency_band="optical",
        capability="350+ 自主望远镜、约35站；MEO/GEO/HEO；逐台坐标非公开",
        status="aggregate_only", channel="curated:commercial-public-count",
    ),
    dict(
        sensor_id="SLINGSHOT-SGSN-AGG", name="Slingshot Global Sensor Network",
        name_cn="Slingshot 全球传感器网络", sensor_class="network_node",
        network="Slingshot SGSN", operator="Slingshot Aerospace",
        country="国际/多国", lat=0.0, lon=0.0, alt_m=None, frequency_band="visible/SWIR/NIR",
        capability="200+ 传感器、20+站；Varda/Horus/Argus；逐台坐标非公开",
        status="aggregate_only", channel="curated:commercial-public-count",
    ),
]


CURATED_SWX_GROUND = [
    dict(sensor_id="NOAA-SWPC", name="NOAA SWPC", name_cn="NOAA 空间天气预测中心",
         sensor_class="network", network="SWPC", operator="NOAA", country="美国",
         lat=40.010, lon=-105.270, alt_m=1655, observables="F10.7/Kp/Ap/警报",
         data_format="JSON/NetCDF", channel="curated:docs"),
    dict(sensor_id="TROMSO-ISR", name="Tromsø Ionospheric Suite", name_cn="特罗姆瑟电离层设备",
         sensor_class="isr", network="EISCAT/Tromsø", operator="UiT", country="挪威",
         lat=69.660, lon=18.940, alt_m=100, observables="TEC/ISR",
         data_format="HDF5", channel="curated:docs"),
    dict(sensor_id="EISCAT-Tromso", name="EISCAT Tromsø", name_cn="EISCAT 特罗姆瑟",
         sensor_class="isr", network="EISCAT", operator="EISCAT", country="挪威",
         lat=69.583, lon=19.227, alt_m=100, observables="Ne/Ti/漂移",
         data_format="HDF5", channel="curated:EISCAT"),
    dict(sensor_id="EISCAT-Svalbard", name="EISCAT Svalbard", name_cn="EISCAT 斯瓦尔巴",
         sensor_class="isr", network="EISCAT", operator="EISCAT", country="挪威",
         lat=78.153, lon=16.029, alt_m=400, observables="极区电离层",
         data_format="HDF5", channel="curated:EISCAT"),
    dict(sensor_id="APIS-HST", name="APIS / HST FUV Aurora", name_cn="APIS 行星极光库",
         sensor_class="optical_uv", network="APIS", operator="学术联盟", country="国际/多国",
         lat=28.470, lon=-80.580, alt_m=10, observables="行星极光 FUV 元数据",
         data_format="REST/FITS", channel="curated:docs"),
    dict(sensor_id="CN-Meridian", name="Chinese Meridian Project", name_cn="中国子午工程",
         sensor_class="network", network="Meridian", operator="NSSC/CAS", country="中国",
         lat=40.000, lon=116.380, alt_m=50, observables="地磁/电离层/中高层大气",
         data_format="HDF5/NetCDF", channel="curated:docs"),
]

CURATED_TTC = [
    # ESTRACK
    dict(station_id="ESTRACK-NNO-1", name="New Norcia NNO-1 (DSA-1)", name_cn="新诺西亚 NNO-1",
         network="ESTRACK", operator="ESA", country="澳大利亚",
         lat=-31.0482, lon=116.1915, alt_m=252, antenna_diam_m=35, bands="S/X",
         station_type="deep_space", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-NNO-2", name="New Norcia NNO-2", name_cn="新诺西亚 NNO-2",
         network="ESTRACK", operator="ESA", country="澳大利亚",
         lat=-31.0485, lon=116.1910, alt_m=252, antenna_diam_m=4.5, bands="S/X",
         station_type="deep_space", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-NNO-3", name="New Norcia NNO-3 (DSA-4)", name_cn="新诺西亚 DSA-4",
         network="ESTRACK", operator="ESA", country="澳大利亚",
         lat=-31.0478, lon=116.1920, alt_m=252, antenna_diam_m=35, bands="S/X/Ka",
         station_type="deep_space", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-CEB", name="Cebreros DSA-2", name_cn="塞夫雷罗斯 DSA-2",
         network="ESTRACK", operator="ESA", country="西班牙",
         lat=40.4527, lon=-4.3675, alt_m=794, antenna_diam_m=35, bands="X/Ka",
         station_type="deep_space", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-MLG", name="Malargüe DSA-3", name_cn="马拉圭 DSA-3",
         network="ESTRACK", operator="ESA", country="阿根廷",
         lat=-35.7759, lon=-69.3981, alt_m=1550, antenna_diam_m=35, bands="X/Ka",
         station_type="deep_space", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-KIR", name="Kiruna", name_cn="基律纳",
         network="ESTRACK", operator="ESA", country="瑞典",
         lat=67.8571, lon=20.9644, alt_m=400, antenna_diam_m=15, bands="S/X",
         station_type="near_earth", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-RED", name="Redu", name_cn="雷杜",
         network="ESTRACK", operator="ESA", country="比利时",
         lat=50.0015, lon=5.1453, alt_m=400, antenna_diam_m=15, bands="S/X/Ka",
         station_type="near_earth", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-KRU", name="Kourou", name_cn="库鲁",
         network="ESTRACK", operator="ESA", country="法属圭亚那",
         lat=5.2514, lon=-52.8047, alt_m=20, antenna_diam_m=15, bands="S/X",
         station_type="near_earth", channel="curated:ESTRACK"),
    dict(station_id="ESTRACK-SMA", name="Santa Maria", name_cn="圣玛丽亚",
         network="ESTRACK", operator="ESA", country="葡萄牙",
         lat=36.997, lon=-25.136, alt_m=200, antenna_diam_m=15, bands="S/X",
         station_type="near_earth", channel="curated:ESTRACK"),
    # DSN complexes + major DSS
    dict(station_id="DSN-GOLDSTONE", name="Goldstone DSCC", name_cn="戈尔德斯通",
         network="DSN", operator="NASA/JPL", country="美国",
         lat=35.4267, lon=-116.8900, alt_m=1000, antenna_diam_m=70, bands="S/X/Ka",
         station_type="deep_space", channel="curated:DSN"),
    dict(station_id="DSN-CANBERRA", name="Canberra DSCC", name_cn="堪培拉",
         network="DSN", operator="NASA/JPL", country="澳大利亚",
         lat=-35.4014, lon=148.9813, alt_m=650, antenna_diam_m=70, bands="S/X/Ka",
         station_type="deep_space", channel="curated:DSN"),
    dict(station_id="DSN-MADRID", name="Madrid DSCC", name_cn="马德里",
         network="DSN", operator="NASA/JPL", country="西班牙",
         lat=40.4277, lon=-4.2497, alt_m=720, antenna_diam_m=70, bands="S/X/Ka",
         station_type="deep_space", channel="curated:DSN"),
    # USGS Landsat
    dict(station_id="USGS-SGS", name="Sioux Falls EROS", name_cn="苏福尔斯",
         network="USGS-Landsat", operator="USGS", country="美国",
         lat=43.736, lon=-96.625, alt_m=450, antenna_diam_m=11, bands="X",
         station_type="near_earth", channel="curated:USGS-Landsat"),
    dict(station_id="USGS-ASA", name="Alice Springs", name_cn="阿利斯普林斯",
         network="USGS-Landsat", operator="USGS/GA", country="澳大利亚",
         lat=-23.758, lon=133.881, alt_m=550, antenna_diam_m=10, bands="X",
         station_type="near_earth", channel="curated:USGS-Landsat"),
    dict(station_id="USGS-NSG", name="Neustrelitz", name_cn="诺伊斯特里利茨",
         network="USGS-Landsat", operator="DLR/USGS", country="德国",
         lat=53.330, lon=13.070, alt_m=50, antenna_diam_m=7.3, bands="X",
         station_type="near_earth", channel="curated:USGS-Landsat"),
    dict(station_id="USGS-GIL", name="Gilmore Creek", name_cn="吉尔摩克里克",
         network="USGS-Landsat", operator="NOAA/USGS", country="美国",
         lat=64.978, lon=-147.518, alt_m=200, antenna_diam_m=13, bands="X/L",
         station_type="polar", channel="curated:USGS-Landsat"),
    dict(station_id="USGS-SVAL", name="Svalbard Landsat", name_cn="斯瓦尔巴 Landsat",
         network="USGS-Landsat", operator="KSAT/USGS", country="挪威",
         lat=78.230, lon=15.390, alt_m=450, antenna_diam_m=11, bands="X/S",
         station_type="polar", channel="curated:USGS-Landsat"),
    # KSAT sites (Wikipedia / KSAT public)
    dict(station_id="KSAT-SVALBARD", name="KSAT SvalSat", name_cn="KSAT 斯瓦尔巴",
         network="KSAT", operator="KSAT", country="挪威",
         lat=78.2307, lon=15.3890, alt_m=450, antenna_diam_m=13, bands="S/X/Ka",
         station_type="polar", channel="curated:KSAT"),
    dict(station_id="KSAT-TROLL", name="KSAT TrollSat", name_cn="KSAT Troll",
         network="KSAT", operator="KSAT", country="南极/挪威",
         lat=-72.011, lon=2.535, alt_m=1300, antenna_diam_m=7.3, bands="S/X",
         station_type="polar", channel="curated:KSAT"),
    dict(station_id="KSAT-TROMSO", name="KSAT Tromsø", name_cn="KSAT 特罗姆瑟",
         network="KSAT", operator="KSAT", country="挪威",
         lat=69.662, lon=18.940, alt_m=100, antenna_diam_m=11, bands="S/X",
         station_type="polar", channel="curated:KSAT"),
    dict(station_id="KSAT-INUVIK", name="KSAT Inuvik", name_cn="KSAT 伊努维克",
         network="KSAT", operator="KSAT", country="加拿大",
         lat=68.319, lon=-133.549, alt_m=50, antenna_diam_m=7.3, bands="S/X",
         station_type="polar", channel="curated:KSAT"),
    dict(station_id="KSAT-PUNTA", name="KSAT Punta Arenas", name_cn="KSAT 蓬塔阿雷纳斯",
         network="KSAT", operator="KSAT", country="智利",
         lat=-53.106, lon=-70.877, alt_m=30, antenna_diam_m=7.3, bands="S/X",
         station_type="polar", channel="curated:KSAT"),
    dict(station_id="KSAT-HAWAII", name="KSAT Hawaii", name_cn="KSAT 夏威夷",
         network="KSAT", operator="KSAT", country="美国",
         lat=19.014, lon=-155.663, alt_m=50, antenna_diam_m=5, bands="S/X",
         station_type="commercial_gsaas", channel="curated:KSAT"),
    dict(station_id="KSAT-SINGAPORE", name="KSAT Singapore", name_cn="KSAT 新加坡",
         network="KSAT", operator="KSAT", country="新加坡",
         lat=1.352, lon=103.820, alt_m=20, antenna_diam_m=5, bands="S/X",
         station_type="commercial_gsaas", channel="curated:KSAT"),
    dict(station_id="KSAT-DUBAI", name="KSAT Dubai", name_cn="KSAT 迪拜",
         network="KSAT", operator="KSAT", country="阿联酋",
         lat=25.204, lon=55.270, alt_m=20, antenna_diam_m=5, bands="S/X",
         station_type="commercial_gsaas", channel="curated:KSAT"),
    dict(station_id="KSAT-HARTE", name="KSAT Hartebeesthoek", name_cn="KSAT 哈特比斯胡克",
         network="KSAT", operator="KSAT", country="南非",
         lat=-25.887, lon=27.685, alt_m=1400, antenna_diam_m=7.3, bands="S/X",
         station_type="commercial_gsaas", channel="curated:KSAT"),
    dict(station_id="KSAT-AWARUA", name="KSAT Awarua", name_cn="KSAT 阿瓦鲁阿",
         network="KSAT", operator="KSAT", country="新西兰",
         lat=-46.528, lon=168.377, alt_m=20, antenna_diam_m=5, bands="S/X",
         station_type="commercial_gsaas", channel="curated:KSAT"),
    dict(station_id="KSAT-FAIRBANKS", name="KSAT Fairbanks", name_cn="KSAT 费尔班克斯",
         network="KSAT", operator="KSAT", country="美国",
         lat=64.978, lon=-147.518, alt_m=200, antenna_diam_m=7.3, bands="S/X",
         station_type="polar", channel="curated:KSAT"),
    dict(station_id="KSAT-NUUK", name="KSAT Nuuk", name_cn="KSAT 努克",
         network="KSAT", operator="KSAT", country="格陵兰",
         lat=64.183, lon=-51.721, alt_m=50, antenna_diam_m=5, bands="S/X",
         station_type="polar", channel="curated:KSAT"),
    # GSaaS hubs
    # AWS：精确坐标以 Brahe aws.json 为准（此处仅保留旧近似点作兜底；同名去重靠 station_id）
    dict(station_id="AWS-GS-OHIO", name="AWS Ground Station Ohio", name_cn="AWS GS 俄亥俄",
         network="AWS Ground Station", operator="AWS", country="美国",
         lat=40.10, lon=-83.00, alt_m=250, antenna_diam_m=8, bands="S/X",
         station_type="commercial_gsaas", channel="curated:GSaaS"),
    dict(station_id="AWS-GS-OREGON", name="AWS Ground Station Oregon", name_cn="AWS GS 俄勒冈",
         network="AWS Ground Station", operator="AWS", country="美国",
         lat=45.80, lon=-119.50, alt_m=200, antenna_diam_m=8, bands="S/X",
         station_type="commercial_gsaas", channel="curated:GSaaS"),
    dict(station_id="AWS-GS-IRELAND", name="AWS Ground Station Ireland", name_cn="AWS GS 爱尔兰",
         network="AWS Ground Station", operator="AWS", country="爱尔兰",
         lat=53.35, lon=-6.26, alt_m=50, antenna_diam_m=8, bands="S/X",
         station_type="commercial_gsaas", channel="curated:GSaaS"),
    # Azure Orbital（Microsoft 已宣布退役；公开近似坐标，供历史链路研究）
    dict(station_id="AZURE-ORB-QUINCY", name="Azure Orbital Quincy (retired)", name_cn="Azure Orbital 昆西",
         network="Azure Orbital", operator="Microsoft", country="美国",
         lat=47.234, lon=-119.852, alt_m=400, antenna_diam_m=7.3, bands="S/X",
         station_type="commercial_gsaas", channel="curated:AzureOrbital",
         notes="Azure Orbital 地面站服务已退役；坐标为公开近似值"),
    dict(station_id="AZURE-ORB-BOARDMAN", name="Azure Orbital Boardman (retired)", name_cn="Azure Orbital 博德曼",
         network="Azure Orbital", operator="Microsoft", country="美国",
         lat=45.840, lon=-119.700, alt_m=150, antenna_diam_m=7.3, bands="S/X",
         station_type="commercial_gsaas", channel="curated:AzureOrbital",
         notes="Azure Orbital 地面站服务已退役；坐标为公开近似值"),
    dict(station_id="AZURE-ORB-CHANDLER", name="Azure Orbital Chandler (retired)", name_cn="Azure Orbital 钱德勒",
         network="Azure Orbital", operator="Microsoft", country="美国",
         lat=33.306, lon=-111.841, alt_m=370, antenna_diam_m=7.3, bands="S/X",
         station_type="commercial_gsaas", channel="curated:AzureOrbital",
         notes="Azure Orbital 地面站服务已退役；坐标为公开近似值"),
    dict(station_id="ATLAS-FREEDOM", name="ATLAS Freedom Hub", name_cn="ATLAS Freedom",
         network="ATLAS Freedom", operator="ATLAS", country="美国",
         lat=42.70, lon=-84.50, alt_m=250, antenna_diam_m=5, bands="UHF/S/X",
         station_type="commercial_gsaas", channel="curated:GSaaS"),
    dict(station_id="LEAF-MILANO", name="Leaf Space", name_cn="Leaf Space",
         network="Leaf Space", operator="Leaf Space", country="意大利",
         lat=45.46, lon=9.19, alt_m=120, antenna_diam_m=3.7, bands="UHF/S/X",
         station_type="commercial_gsaas", channel="curated:GSaaS"),
    dict(station_id="INFOSTELLAR-JP", name="Infostellar Node", name_cn="Infostellar",
         network="Infostellar", operator="Infostellar", country="日本",
         lat=35.68, lon=139.69, alt_m=40, antenna_diam_m=3, bands="UHF/S",
         station_type="commercial_gsaas", channel="curated:GSaaS"),
    dict(station_id="RBC-SIGNALS", name="RBC Signals Hub", name_cn="RBC Signals",
         network="RBC Signals", operator="RBC Signals", country="美国",
         lat=32.78, lon=-96.80, alt_m=130, antenna_diam_m=5, bands="UHF/S/X",
         station_type="commercial_gsaas", channel="curated:GSaaS"),
    dict(station_id="THUMBNET-HUB", name="ThumbNet Hub", name_cn="ThumbNet 枢纽",
         network="ThumbNet", operator="ThumbSat", country="国际/多国",
         lat=37.33, lon=-121.89, alt_m=20, antenna_diam_m=0, bands="UHF/VHF",
         station_type="crowdsourced", channel="curated:docs",
         notes="公开页未暴露全量节点坐标；文档称全球>249节点"),
    dict(station_id="CN-XIAN", name="Xi'an SCC Node", name_cn="西安卫星测控中心",
         network="中国测控网", operator="XSCC", country="中国",
         lat=34.34, lon=108.94, alt_m=400, antenna_diam_m=15, bands="S/X/Ka",
         station_type="near_earth", channel="curated:CN-TTC"),
    dict(station_id="CN-KASHI", name="Kashgar GS", name_cn="喀什测控站",
         network="中国测控网", operator="XSCC", country="中国",
         lat=39.47, lon=75.99, alt_m=1300, antenna_diam_m=12, bands="S/X",
         station_type="near_earth", channel="curated:CN-TTC"),
    dict(station_id="CN-SANYA", name="Sanya GS", name_cn="三亚测控站",
         network="中国测控网", operator="XSCC", country="中国",
         lat=18.25, lon=109.50, alt_m=20, antenna_diam_m=12, bands="S/X",
         station_type="near_earth", channel="curated:CN-TTC"),
]


# ── Build catalogs ───────────────────────────────────────────────────────────
def build_catalogs(cache: Path) -> tuple[list, list, list, list, list]:
    ssa: dict[str, dict] = {}
    swx: dict[str, dict] = {}
    ttc: dict[str, dict] = {}
    launch_sites: dict[str, dict] = {}
    organisations: dict[str, dict] = {}

    print("→ SatNOGS …")
    for st in fetch_satnogs(cache):
        try:
            lat, lon = float(st["lat"]), float(st["lng"])
        except Exception:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        sid = f"SATNOGS-{st['id']}"
        status = {1: "operational", 2: "testing", 0: "offline"}.get(st.get("status"), str(st.get("status")))
        row = dict(
            station_id=sid, name=st.get("name") or sid, name_cn=None,
            network="SatNOGS", operator=st.get("owner") or "SatNOGS",
            country="国际/众包", lat=lat, lon=_norm_lon(lon),
            alt_m=st.get("altitude"), antenna_diam_m=None, bands="UHF/VHF/S",
            station_type="crowdsourced", status=status,
            notes=(st.get("description") or "")[:500] or None,
        )
        _add(ttc, sid, row, "SatNOGS-API", "ttc")

    print("→ INTERMAGNET …")
    for o in fetch_intermagnet(cache):
        sid = f"INTERMAGNET-{o['code']}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="magnetometer", network="INTERMAGNET",
            operator="INTERMAGNET", country=_cc(o.get("country_code")),
            lat=o["lat"], lon=o["lon"], alt_m=o.get("alt_m"),
            observables="地磁矢量/K指数贡献", data_format="IAGA-2002/CDF",
            status="operational",
        )
        _add(swx, sid, row, "INTERMAGNET", "swx")

    print("→ SuperMAG …")
    for o in fetch_supermag(cache):
        # INTERMAGNET 是 SuperMAG 的重要上游之一；同 IAGA code 只保留一条物理设备。
        if f"INTERMAGNET-{o['code']}" in swx:
            CHANNEL_STATS["swx"]["SuperMAG/INTERMAGNET(dup_skip)"] += 1
            continue
        sid = f"SUPERMAG-{o['code']}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="magnetometer", network="SuperMAG",
            operator=o.get("operators") or "SuperMAG", country="国际/多国",
            lat=o["lat"], lon=o["lon"], alt_m=None,
            observables=f"地磁矢量；AACGM={o.get('mlat')},{o.get('mlon')}",
            data_format="SuperMAG ASCII/Web Service", status="registered",
            notes=f"IAGA/SuperMAG code={o['code']}",
        )
        _add(swx, sid, row, "SuperMAG-station_info", "swx")

    print("→ NMDB …")
    for o in fetch_nmdb(cache):
        sid = f"NMDB-{o['code']}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="neutron_monitor", network="NMDB",
            operator="NMDB/data provider", country="国际/多国",
            lat=o["lat"], lon=o["lon"], alt_m=o.get("alt_m"),
            observables=f"宇宙线中子计数/地面增强事件；cutoff={o.get('cutoff_gv')} GV",
            data_format="NMDB NEST/ASCII", status="registered",
            notes=f"NMDB station code={o['code']}",
        )
        _add(swx, sid, row, "NMDB", "swx")

    print("→ GIRO/DIDBase …")
    for o in fetch_giro(cache):
        sid = f"GIRO-{o['ursi']}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="ionosonde", network="GIRO/DIDBase",
            operator="UML GIRO", country="国际/多国",
            lat=o["lat"], lon=o["lon"], alt_m=None,
            observables="foF2/hmF2/电离层层析", data_format="SAO/CDF",
            status="operational",
        )
        _add(swx, sid, row, "GIRO-DIDBase", "swx")

    print("→ SuperDARN …")
    for o in fetch_superdarn(cache):
        code = re.sub(r"\W+", "-", o["name"])[:40]
        sid = f"SUPERDARN-{code}"
        base = dict(
            name=o["name"], name_cn=None, network="SuperDARN",
            operator="SuperDARN Consortium", country="国际/多国",
            lat=o["lat"], lon=o["lon"], alt_m=None, status="operational",
        )
        # SuperDARN 测量电离层等离子体对流，不直接承担人工空间目标编目；
        # 因此只归 SWx，不能为满足 SSA 数量目标而重复计入。
        _add(swx, sid + "-SWX", {**base, "sensor_id": sid + "-SWX",
                                 "sensor_class": "isr", "observables": "电离层对流/电场",
                                 "data_format": "FitACF/HDF5"},
             "SuperDARN", "swx")

    print("→ Space-Track payloads …")
    for x in fetch_spacetrack_payloads(cache):
        nid = x.get("NORAD_CAT_ID")
        name = x.get("OBJECT_NAME") or f"NORAD-{nid}"
        country = _cc(x.get("COUNTRY") or "")
        label = x.get("_channel_label") or "Space-Track"
        # 粗分：SSA vs SWX
        is_ssa = any(k in name.upper() for k in (
            "SBSS", "GSSAP", "ORS-5", "ORS 5", "SAPPHIRE", "NEOSSAT",
            "SILENT BARKER",
        ))
        sid = f"ST-{nid}"
        # 天基无固定经纬：用 0,0 + 高高度标记（前端地图会过滤）
        row_common = dict(
            name=name, name_cn=None, operator=x.get("SITE") or "Space-Track",
            country=country or "未知", lat=0.0, lon=0.0, alt_m=None,
            status="operational",
            notes=f"NORAD {nid}; COSPAR {x.get('INTLDES')}; launch {x.get('LAUNCH')}",
        )
        if is_ssa:
            _add(ssa, sid, {**row_common, "sensor_id": sid, "sensor_class": "spaceborne",
                            "network": "Space-Track/SSA", "frequency_band": "optical/EO",
                            "capability": "天基空间目标监视载荷"},
                 f"Space-Track:{label}", "ssa")
        else:
            _add(swx, sid, {**row_common, "sensor_id": sid, "sensor_class": "spaceborne",
                            "network": "Space-Track/SWX", "observables": "空间天气/气象遥感",
                            "data_format": "NetCDF/HDF5"},
                 f"Space-Track:{label}", "swx")

    print("→ ILRS active SLR stations …")
    for o in fetch_ilrs_active(cache):
        sid = f"ILRS-{o['code']}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="laser", network="ILRS",
            operator="ILRS station operator", country="国际/多国",
            lat=o["lat"], lon=o["lon"], alt_m=o.get("alt_m"),
            frequency_band="optical/laser",
            capability="在役卫星激光测距（SLR）；精密轨道与空间目标测距",
            status="operational", notes=f"ILRS active station code={o['code']}",
        )
        _add(ssa, sid, row, "ILRS-active", "ssa")

    print("→ MPC explicit SSA/optical survey candidates …")
    for o in fetch_mpc_ssa_candidates(cache):
        sid = f"MPC-{o['code']}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="ground_optical", network="MPC Observatory Codes",
            operator="MPC-registered observatory", country="国际/多国",
            lat=o["lat"], lon=o["lon"], alt_m=None,
            frequency_band="optical",
            capability="公开光学天体测量/巡天候选能力；不等同于已认证常态SST资产",
            status="registered",
            notes=(
                f"MPC code={o['code']}; type={o.get('observations_type')}; "
                "按名称筛选 Space Surveillance/Spaceguard/Tracking/Survey"
            ),
        )
        _add(ssa, sid, row, "MPC-ObsCodes:candidate", "ssa")

    print("→ SPASE space instruments …")
    for o in fetch_spase_space_instruments(cache):
        digest = hashlib.sha1(o["resource_id"].encode("utf-8")).hexdigest()[:16]
        sid = f"SPASE-{digest}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="spaceborne", network="NASA HPDE/SPASE-SMWG",
            operator=o["observatory_id"].rsplit("/", 1)[-1], country="国际/多国",
            lat=0.0, lon=0.0, alt_m=None,
            observables=o["instrument_types"], data_format="SPASE XML",
            status="registered",
            notes=(
                f"{o['resource_id']}; observatory={o['observatory_id']}; "
                f"regions={o.get('regions')}"
            ),
        )
        _add(swx, sid, row, "SPASE-SMWG:space", "swx")

    print("→ WMO OSCAR/Space SWx instruments …")
    # 名称+搭载平台双键去重；OSCAR 与 SPASE 可有同名但不同平台的真实独立载荷。
    seen_oscar = set()
    for o in fetch_oscar_space_weather(cache):
        natural = re.sub(r"\W+", "", f"{o['name']}|{o['satellites']}").lower()
        if natural in seen_oscar:
            continue
        seen_oscar.add(natural)
        sid = f"OSCARSPACE-{_slug(o['slug'], 42)}"
        row = dict(
            sensor_id=sid, name=o["name"], name_cn=None,
            sensor_class="spaceborne", network="WMO OSCAR/Space",
            operator=o.get("agency") or "WMO member agency", country="国际/多国",
            lat=0.0, lon=0.0, alt_m=None,
            observables=o["instrument_type"], data_format="OSCAR/Space JSON",
            status="registered",
            notes=(
                f"{o.get('full_name')}; satellites={o.get('satellites')}; "
                f"usage={o.get('usage_from')}..{o.get('usage_to')}"
            ),
        )
        _add(swx, sid, row, "WMO-OSCAR-Space", "swx")

    print("→ Curated SSA/SWX/TTC …")
    for r in CURATED_SSA + CURATED_SSA_OPTICAL_EXPANSION:
        row = dict(r)
        ch = row.pop("channel")
        _add(ssa, row["sensor_id"], row, ch, "ssa")
    for r in CURATED_SWX_GROUND:
        row = dict(r)
        ch = row.pop("channel")
        _add(swx, row["sensor_id"], row, ch, "swx")
    for r in CURATED_TTC:
        row = dict(r)
        ch = row.pop("channel")
        _add(ttc, row["station_id"], row, ch, "ttc")

    print("→ Brahe community groundstations …")
    for g in fetch_brahe_groundstations(cache):
        prov = g["provider"]
        sid = f"BRAHE-{_slug(prov, 16)}-{_slug(g['name'], 32)}"
        net = {
            "aws": "AWS Ground Station", "Aws": "AWS Ground Station",
            "ksat": "KSAT", "KSAT": "KSAT",
            "NASA DSN": "DSN", "dsn": "DSN",
            "Atlas": "ATLAS Freedom", "atlas": "ATLAS Freedom",
            "Leaf Space": "Leaf Space", "leaf": "Leaf Space",
            "NASA NEN": "NASA NEN", "nen": "NASA NEN",
            "ssc": "SSC", "SSC": "SSC",
            "Viasat": "Viasat", "viasat": "Viasat",
        }.get(prov, prov)
        row = dict(
            station_id=sid, name=g["name"], name_cn=None,
            network=net, operator=prov, country="国际/多国",
            lat=g["lat"], lon=g["lon"], alt_m=g.get("alt_m"),
            antenna_diam_m=None, bands=g.get("bands"),
            station_type="commercial_gsaas", status="operational",
            notes=f"Brahe/{g.get('source_file')}.json 社区公开坐标",
        )
        _add(ttc, sid, row, f"Brahe:{g.get('source_file')}", "ttc")

    print("→ Starlink community ground stations …")
    for i, g in enumerate(fetch_starlink_ground_stations(cache)):
        sid = f"STARLINK-{_slug(g['name'], 36)}-{i}"
        row = dict(
            station_id=sid, name=g["name"], name_cn=None,
            network="Starlink", operator="SpaceX/community",
            country="国际/多国", lat=g["lat"], lon=g["lon"], alt_m=None,
            antenna_diam_m=None, bands="Ku/Ka",
            station_type="commercial_gsaas",
            status=g.get("status") or "operational",
            notes=f"type={g.get('stype')}; Satellitemap-class community list",
        )
        _add(ttc, sid, row, "Starlink-community", "ttc")

    print("→ DISCOS launch-sites …")
    for x in fetch_discos_launch_sites(cache):
        did = str(x.get("discos_id") or "")
        if not did:
            continue
        lat = lon = alt = None
        try:
            if x.get("lat") is not None and str(x.get("lat")).strip() != "":
                lat = float(x["lat"])
            if x.get("lon") is not None and str(x.get("lon")).strip() != "":
                lon = float(x["lon"])
            if x.get("alt_m") is not None and str(x.get("alt_m")).strip() != "":
                alt = float(x["alt_m"])
        except Exception:
            pass
        if lon is not None:
            lon = _norm_lon(lon)
        row = dict(
            discos_id=did, name=x.get("name") or f"DISCOS-LS-{did}",
            lat=lat, lon=lon, alt_m=alt,
            pads=x.get("pads"), azimuths=x.get("azimuths"),
            constraints=x.get("constraints"),
            source="DISCOSweb:/api/launch-sites",
        )
        _add(launch_sites, did, row, "DISCOSweb:launch-sites", "discos_ls")
        # 有坐标的发射场同步进测控/地面基础设施视图（便于地图统一浏览）
        if lat is not None and lon is not None and abs(lat) <= 90:
            sid = f"DISCOS-LS-{did}"
            _add(ttc, sid, dict(
                station_id=sid, name=row["name"], name_cn=None,
                network="DISCOS Launch Site", operator="ESA DISCOS",
                country="国际/多国", lat=lat, lon=lon, alt_m=alt,
                antenna_diam_m=None, bands=None,
                station_type="launch_site", status="operational",
                notes=f"pads={row.get('pads')}",
            ), "DISCOSweb:launch-sites→ttc", "ttc")

    print("→ DISCOS organisations …")
    for x in fetch_discos_organisations(cache):
        did = str(x.get("discos_id") or "")
        if not did:
            continue
        row = dict(
            discos_id=did,
            name=x.get("name") or f"DISCOS-ORG-{did}",
            date_range=x.get("date_range"),
            source="DISCOSweb:/api/organisations",
        )
        _add(organisations, did, row, "DISCOSweb:organisations", "discos_org")

    return (
        list(ssa.values()), list(swx.values()), list(ttc.values()),
        list(launch_sites.values()), list(organisations.values()),
    )


# ── DB write ─────────────────────────────────────────────────────────────────
def _apply_migration(engine):
    mig_dir = Path(__file__).resolve().parent.parent / "database" / "migrations"
    for name in ("002_monitoring_network.sql", "003_discos_network.sql"):
        mig = mig_dir / name
        if not mig.exists():
            continue
        with open(mig, encoding="utf-8") as f:
            sql = f.read()
        with engine.begin() as conn:
            conn.exec_driver_sql(sql)


def _write_table(engine, table: str, rows: list[dict]) -> int:
    if not rows:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {table} RESTART IDENTITY CASCADE"))
        return 0
    df = pd.DataFrame(rows)
    # 对齐列
    with engine.connect() as conn:
        cols = [r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name='{table}' AND column_name NOT IN ('id','geom','updated_at') "
            "ORDER BY ordinal_position"
        ))]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {table} RESTART IDENTITY CASCADE"))
    df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=500)
    with engine.begin() as conn:
        has_geom = conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            f"WHERE table_name='{table}' AND column_name='geom'"
        )).scalar()
        if has_geom:
            conn.execute(text(f"""
                UPDATE {table}
                SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326)
                WHERE lat IS NOT NULL AND lon IS NOT NULL
            """))
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    return int(n or 0)


def write_channel_report(path: Path):
    lines = ["# 监测与测控网络 · 渠道贡献统计\n"]
    for table in ("ssa", "swx", "ttc", "discos_ls", "discos_org"):
        lines.append(f"\n## {table}\n")
        lines.append("| 渠道 | 新增行数 |\n|---|---:|\n")
        for ch, n in sorted(CHANNEL_STATS[table].items(), key=lambda x: -x[1]):
            lines.append(f"| {ch} | {n} |\n")
        total = sum(v for k, v in CHANNEL_STATS[table].items() if not k.endswith("(dup_skip)"))
        dups = sum(v for k, v in CHANNEL_STATS[table].items() if k.endswith("(dup_skip)"))
        lines.append(f"\n**入库合计**: {total}　**去重跳过**: {dups}\n")
    path.write_text("".join(lines), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


def ingest(cache: Path | None = None, *, refresh_discos: bool = False) -> dict[str, int]:
    cache = cache or CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    if refresh_discos:
        for name in ("discos_launch_sites.json", "discos_organisations.json"):
            fp = cache / name
            if fp.exists():
                fp.unlink()
                print(f"  refresh: removed {fp.name}")
    init_db()
    engine = get_engine()
    _apply_migration(engine)
    ssa, swx, ttc, ls, orgs = build_catalogs(cache)
    counts = {
        "external_ssa_sensors": _write_table(engine, "external_ssa_sensors", ssa),
        "external_space_weather_sensors": _write_table(engine, "external_space_weather_sensors", swx),
        "external_ttc_stations": _write_table(engine, "external_ttc_stations", ttc),
        "external_discos_launch_sites": _write_table(engine, "external_discos_launch_sites", ls),
        "external_discos_organisations": _write_table(engine, "external_discos_organisations", orgs),
    }
    report = cache / "CHANNEL_STATS.md"
    write_channel_report(report)
    (cache / "channel_stats.json").write_text(
        json.dumps({k: dict(v) for k, v in CHANNEL_STATS.items()}, ensure_ascii=False, indent=2))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-cache", type=Path, default=None,
                    help="使用已有缓存目录（跳过部分网络请求若文件存在）")
    ap.add_argument("--refresh-discos", action="store_true",
                    help="强制重新拉取 DISCOS launch-sites / organisations")
    args = ap.parse_args()
    cache = args.offline_cache or CACHE_DIR
    counts = ingest(cache, refresh_discos=args.refresh_discos)
    print("\n=== 入库完成 ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
