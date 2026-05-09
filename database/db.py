"""Database connection factory and session helpers.

Resilient to transient network disruptions (e.g. VPN switches):
- pool_pre_ping=True     → detect stale connections before handing them out
- pool_recycle (env)     → recycle TCP periodically (default ~280 s)
- connect_timeout (env)→ fail fast when DB host unreachable (default 10 s)
- pool_timeout (env)    → do not block forever waiting for a pooled connection
- TCP keepalives         → detect dropped VPN / NAT sooner
- session_scope retries up to 3 times on OperationalError

See ``DB_*`` env vars in ``config/settings.py``.
"""
import os
import time
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

from config.settings import (
    DB_DSN,
    DB_CONNECT_TIMEOUT,
    DB_POOL_TIMEOUT,
    DB_POOL_RECYCLE,
    DB_STATEMENT_TIMEOUT_SEC,
)
from database.models import Base

_log = logging.getLogger(__name__)
_engine = None
_SessionLocal = None

_MAX_RETRIES = 3
_RETRY_DELAYS = (1, 3, 6)       # seconds between retries


def _reset_pool():
    """Dispose existing pool so next call creates fresh TCP connections."""
    global _engine, _SessionLocal
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {
            "connect_timeout": DB_CONNECT_TIMEOUT,
            # 强制客户端用 UTF-8 编码，避免 Windows 中文系统（GBK）下
            # psycopg2 解码 PostgreSQL 错误消息时抛出 UnicodeDecodeError
            "client_encoding": "utf8",
            # VPN / 网络闪断时更快发现死连接
            "keepalives": 1,
            "keepalives_idle": int(os.getenv("DB_KEEPALIVES_IDLE", "30")),
            "keepalives_interval": int(os.getenv("DB_KEEPALIVES_INTERVAL", "10")),
            "keepalives_count": int(os.getenv("DB_KEEPALIVES_COUNT", "5")),
        }
        if DB_STATEMENT_TIMEOUT_SEC and DB_STATEMENT_TIMEOUT_SEC > 0:
            connect_args["options"] = (
                f"-c statement_timeout={int(DB_STATEMENT_TIMEOUT_SEC)}s"
            )
        _engine = create_engine(
            DB_DSN,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=DB_POOL_RECYCLE,
            pool_timeout=DB_POOL_TIMEOUT,
            connect_args=connect_args,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Yield a transactional session; auto-retry on connection failures."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        factory = get_session_factory()
        sess = factory()
        try:
            yield sess
            sess.commit()
            return
        except OperationalError as exc:
            sess.rollback()
            last_exc = exc
            _log.warning("DB connection error (attempt %d/%d): %s",
                         attempt + 1, _MAX_RETRIES, exc)
            sess.close()
            _reset_pool()
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])
        except Exception:
            sess.rollback()
            raise
        finally:
            try:
                sess.close()
            except Exception:
                pass
    raise last_exc


def init_db():
    """Create all tables (does NOT run raw SQL migrations)."""
    engine = get_engine()
    # Ensure PostGIS exists before creating tables
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        conn.commit()
    Base.metadata.create_all(engine)
    _seed_default_priorities()


# Default source priorities (lower number = wins on conflict).  These match
# the order historically used in v_unified_objects (Space-Track > UCS > ESA).
DEFAULT_SOURCE_PRIORITIES = [
    ("Space-Track", 10, "Space-Track GP catalogue (NORAD primary)", False),
    ("UCS",         20, "Union of Concerned Scientists Satellite Database", False),
    ("ESA-DISCOS",  30, "ESA DISCOSweb in-orbit object registry", False),
    ("GCAT",        40, "Jonathan McDowell GCAT (history aggregates)", False),
    ("UNOOSA",      50, "UN OOSA registered objects", False),
    ("Asterank",    60, "Asterank asteroid / NEO catalogue", False),
    ("TechPort",    70, "NASA TechPort projects", False),
]


def _seed_default_priorities() -> None:
    """Insert default rows into datasource_priority if missing.

    User-edited priorities are preserved (ON CONFLICT DO NOTHING).
    """
    try:
        with get_engine().begin() as conn:
            for source, prio, desc, user in DEFAULT_SOURCE_PRIORITIES:
                conn.execute(
                    text(
                        "INSERT INTO datasource_priority "
                        "(source, priority, description, is_user_defined, updated_at) "
                        "VALUES (:s, :p, :d, :u, NOW()) "
                        "ON CONFLICT (source) DO NOTHING"
                    ),
                    {"s": source, "p": prio, "d": desc, "u": user},
                )
    except Exception as exc:
        _log.warning("seeding datasource_priority failed: %s", exc)
