"""STK 交叉验证结果的持久化（JSON 文件 + 简单读写）。

为方便文档页 (`api/docs_static/modules/stk_validation.html`) / API (`/api/v1/stk-validation`)
共享同一份"最近一次验证"快照，把所有 :class:`ValidationReport` 写到一个聚合 JSON 文件中。

文件结构（``data/validation/stk_validation.json``）::

    {
      "schema_version": 1,
      "updated_at_utc": "2026-...Z",
      "platform": {...},                # 写入时主机的 STK 可用性快照
      "history": [ <ValidationReport>, ... ]   # 倒序，最新在最前
    }
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .availability import detect_stk_availability
from .comparison import ValidationReport


# 项目根：本文件位于 stk_validation/ 下
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = _PROJECT_ROOT / "data" / "validation" / "stk_validation.json"

# 单文件读写互斥
_LOCK = threading.Lock()
_SCHEMA_VERSION = 1
_HISTORY_LIMIT = 30


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _empty_doc() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "updated_at_utc": _utcnow_iso(),
        "platform": detect_stk_availability().to_dict(),
        "history": [],
    }


def load_latest_report(
    path: Optional[Path] = None,
    *,
    label: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """读取最近一次验证结果。

    Parameters
    ----------
    label
        若指定，则只返回与之匹配的最近一条（如 ``'sgp4_vs_stk'``）；否则返回 history[0]。
    """
    p = Path(path) if path else DEFAULT_REPORT_PATH
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return None
    history: List[dict[str, Any]] = list(doc.get("history") or [])
    if not history:
        return None
    if label:
        for item in history:
            if item.get("label") == label:
                return item
        return None
    return history[0]


def load_full_doc(path: Optional[Path] = None) -> dict[str, Any]:
    """读取整份验证文档（首次或损坏时返回空文档）。"""
    p = Path(path) if path else DEFAULT_REPORT_PATH
    if not p.exists():
        return _empty_doc()
    try:
        with p.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or "history" not in doc:
            return _empty_doc()
        return doc
    except Exception:
        return _empty_doc()


def save_report(
    report: ValidationReport | Iterable[ValidationReport],
    *,
    path: Optional[Path] = None,
) -> Path:
    """把一条或多条 :class:`ValidationReport` 追加到聚合 JSON 文件中。

    历史记录上限 ``_HISTORY_LIMIT``（默认 30）。同一 ``label`` 的旧记录不会被立即删除，
    用户可在 UI 中比较多次运行的趋势。
    """
    if isinstance(report, ValidationReport):
        items = [report]
    else:
        items = list(report)
    if not items:
        raise ValueError("save_report: 至少需要一条 ValidationReport")

    p = Path(path) if path else DEFAULT_REPORT_PATH
    _ensure_dir(p)

    with _LOCK:
        doc = load_full_doc(p)
        history: List[dict[str, Any]] = list(doc.get("history") or [])

        for r in items:
            entry = r.to_dict() if isinstance(r, ValidationReport) else dict(r)
            history.insert(0, entry)

        if len(history) > _HISTORY_LIMIT:
            history = history[:_HISTORY_LIMIT]

        doc["schema_version"] = _SCHEMA_VERSION
        doc["updated_at_utc"] = _utcnow_iso()
        doc["platform"] = detect_stk_availability().to_dict()
        doc["history"] = history

        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    return p
