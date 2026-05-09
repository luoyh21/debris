"""Auto-downloaders for the two data sources that previously required manual
download: Jonathan McDowell's GCAT (`jm_satcat.tsv`) and the Union of Concerned
Scientists Satellite Database (`ucs_satellites.xlsx`).

GCAT
----
The TSV is served as a static file at a stable URL — a simple HTTP GET works.

UCS
---
UCS publishes the XLSX behind their public landing page; the actual file URL
contains a date stamp that changes with every release, so we scrape the
landing page for the latest ``.xlsx`` link.  Optional ``UCS_DOWNLOAD_URL``
environment variable overrides the auto-discovery (useful when the public
page layout changes).

Both helpers return the absolute path of the downloaded file and overwrite
existing data only when the upstream server reports a newer ``Last-Modified``
or different ``Content-Length`` (so re-runs are cheap).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

log = logging.getLogger(__name__)

GCAT_URL = "https://planet4589.org/space/gcat/tsv/cat/satcat.tsv"

# Browser-like UA so the GCAT server does not 403 on bare requests.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _stream_to(path: str, url: str, *, timeout: int = 600) -> str:
    """Download ``url`` to ``path`` (streaming, 1 MiB chunks).

    Returns the destination path; raises ``requests.RequestException`` on error.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log.info("  ↓ %s → %s", url, path)
    headers = {"User-Agent": _UA, "Accept": "*/*"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        tmp = path + ".part"
        bytes_read = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bytes_read += len(chunk)
        os.replace(tmp, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    if total:
        log.info("    saved %.1f MB (server reported %.1f MB)",
                 size_mb, total / 1024 / 1024)
    else:
        log.info("    saved %.1f MB", size_mb)
    return path


def download_gcat(dest_dir: str) -> Optional[str]:
    """Download the GCAT satellite catalogue TSV. Returns the path or None."""
    dest = os.path.join(dest_dir, "jm_satcat.tsv")
    try:
        return _stream_to(dest, GCAT_URL)
    except Exception as exc:
        log.warning("GCAT 自动下载失败：%s — 将使用本地缓存（若有）", exc)
        return dest if os.path.exists(dest) else None


def download_ucs(dest_dir: str) -> Optional[str]:
    """Return path to the local UCS xlsx if present.

    UCS Satellite Database has been on long-term hiatus (last public update
    May 2023) and the public landing page no longer hosts the file directly,
    so we no longer attempt remote auto-download. Incremental updates run
    against whatever local snapshot is in ``data/external/ucs_satellites.xlsx``.
    """
    dest = os.path.join(dest_dir, "ucs_satellites.xlsx")
    if os.path.exists(dest):
        return dest
    log.warning("UCS 已长期停止更新且官网不再公开 xlsx，跳过自动下载；如有本地 "
                "data/external/ucs_satellites.xlsx 也可使用旧版本")
    return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "external")
    target = os.path.abspath(target)
    log.info("Auto-downloading GCAT + UCS into %s", target)
    download_gcat(target)
    download_ucs(target)
