"""Persistent visit counter for the Streamlit dashboard.

Schema: a single-row KV table ``app_visit_stats`` (auto-created on first use,
no Alembic migration required).

* ``total_sessions``  — count of distinct Streamlit sessions that visited
* ``total_actions``   — every sidebar render / rerun bump (page nav clicks, etc.)
* ``last_visit_utc``  — timestamp of the most recent bump

The helper API is intentionally fault-tolerant: any DB error is swallowed and
returns an empty/no-op result so the UI never breaks because of counter glitches.
"""
from __future__ import annotations

from sqlalchemy import text

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS app_visit_stats (
    id              INT PRIMARY KEY,
    total_sessions  BIGINT      NOT NULL DEFAULT 0,
    total_actions   BIGINT      NOT NULL DEFAULT 0,
    last_visit_utc  TIMESTAMPTZ
)
"""

_DDL_SEED = (
    "INSERT INTO app_visit_stats (id, total_sessions, total_actions) "
    "VALUES (1, 0, 0) ON CONFLICT (id) DO NOTHING"
)

_ensured = False


def _ensure_table() -> bool:
    """Run CREATE TABLE IF NOT EXISTS + seed row exactly once per process."""
    global _ensured
    if _ensured:
        return True
    try:
        from database.db import session_scope

        with session_scope() as s:
            s.execute(text(_DDL_TABLE))
            s.execute(text(_DDL_SEED))
        _ensured = True
        return True
    except Exception:
        return False


def record_visit(new_session: bool = False) -> None:
    """Increment counters.

    Called once per Streamlit sidebar render (= every rerun).
    Pass ``new_session=True`` exactly once per Streamlit session so the
    ``total_sessions`` counter only bumps on the very first rerun.
    """
    if not _ensure_table():
        return
    try:
        from database.db import session_scope

        with session_scope() as s:
            s.execute(
                text(
                    "UPDATE app_visit_stats "
                    "SET total_sessions = total_sessions + :ns, "
                    "    total_actions  = total_actions  + 1, "
                    "    last_visit_utc = NOW() "
                    "WHERE id = 1"
                ),
                {"ns": 1 if new_session else 0},
            )
    except Exception:
        pass


def get_visit_stats() -> dict:
    """Snapshot of the counters.  Returns ``{}`` on any DB error."""
    if not _ensure_table():
        return {}
    try:
        from database.db import session_scope

        with session_scope() as s:
            row = s.execute(
                text(
                    "SELECT total_sessions, total_actions, last_visit_utc "
                    "FROM app_visit_stats WHERE id = 1"
                )
            ).first()
        if not row:
            return {}
        return {
            "total_sessions": int(row[0] or 0),
            "total_actions": int(row[1] or 0),
            "last_visit_utc": row[2].isoformat() if row[2] else None,
        }
    except Exception:
        return {}
