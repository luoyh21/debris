# -*- coding: utf-8 -*-
"""DISCOSweb API v2 公共客户端。

踩坑点：必须显式声明版本头 ``DiscosWeb-Api-Version: 2``，
并使用 ``Accept: application/vnd.api+json``。
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterator, Optional

import requests

BASE_URL = "https://discosweb.esoc.esa.int/api"


def discos_headers(token: Optional[str] = None) -> dict[str, str]:
    tok = (token if token is not None else os.environ.get("ESA_DISCOS_TOKEN") or "").strip()
    if not tok:
        raise RuntimeError("ESA_DISCOS_TOKEN 未配置（DISCOSweb 个人资料页 Personal Access Token）")
    return {
        "Authorization": f"Bearer {tok}",
        "DiscosWeb-Api-Version": "2",
        "Accept": "application/vnd.api+json",
        "User-Agent": "debris-monitor/1.0",
    }


def discos_paginate(
    path: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    page_size: int = 100,
    max_pages: Optional[int] = None,
    sleep_s: float = 0.35,
    timeout: int = 90,
) -> Iterator[dict[str, Any]]:
    """逐页 yield JSON:API 整页响应（含 data / included / meta / links）。"""
    hdrs = headers or discos_headers()
    url = path if path.startswith("http") else f"{BASE_URL}/{path.lstrip('/')}"
    q: dict[str, Any] = dict(params or {})
    q.setdefault("page[size]", page_size)
    q.setdefault("page[number]", 1)
    page_i = 0
    while url:
        page_i += 1
        if max_pages is not None and page_i > max_pages:
            break
        r = None
        for attempt in range(6):
            try:
                r = requests.get(url, headers=hdrs, params=q, timeout=timeout)
                if r.status_code == 429:
                    time.sleep(30)
                    continue
                break
            except requests.RequestException:
                time.sleep(2 ** attempt)
        if r is None:
            raise RuntimeError(f"DISCOS {path}: 无响应")
        if r.status_code != 200:
            raise RuntimeError(f"DISCOS {path}: HTTP {r.status_code}: {r.text[:300]}")
        payload = r.json()
        yield payload
        nxt = (payload.get("links") or {}).get("next")
        if not nxt:
            break
        url = nxt if str(nxt).startswith("http") else f"https://discosweb.esoc.esa.int{nxt}"
        q = {}  # next 链接已含分页参数
        time.sleep(sleep_s)
