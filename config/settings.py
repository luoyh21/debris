"""Central configuration loaded from .env."""
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _normalize_openai_base_url(url: str) -> str:
    """OpenAI SDK 会在 base_url 后拼接 /chat/completions 等路径，因此 base 必须以 /v1 结尾。"""
    u = (url or "").strip().rstrip("/")
    if not u:
        return "https://api.openai.com/v1"
    if u.endswith("/v1"):
        return u
    return f"{u}/v1"


# Space-Track credentials
SPACETRACK_USERNAME = os.getenv("SPACETRACK_USERNAME", "")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD", "")

# OpenAI / LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _normalize_openai_base_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
# 单次 HTTP 请求超时（秒）；工具多轮调用会发多次请求，总耗时会叠加
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "300"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

# Database（strip 避免 .env 行尾空格/换行导致认证失败）
DB_HOST = (os.getenv("DB_HOST", "localhost") or "localhost").strip()
DB_PORT = int((os.getenv("DB_PORT", "5432") or "5432").strip())
DB_NAME = (os.getenv("DB_NAME", "space_debris") or "space_debris").strip()
DB_USER = (os.getenv("DB_USER", "postgres") or "postgres").strip()
DB_PASSWORD = (os.getenv("DB_PASSWORD", "postgres") or "postgres").strip()

# TCP / 连接池：VPN 断开或数据库不可达时尽快失败，避免界面长时间「卡在开始」
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "20"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "280"))
# 单次 SQL 最大执行时间（秒）；不设环境变量则表示不限制（大型摄入脚本勿随意开启）
_db_stmt = os.getenv("DB_STATEMENT_TIMEOUT_SEC", "").strip()
DB_STATEMENT_TIMEOUT_SEC = int(_db_stmt) if _db_stmt else None

DB_DSN = (
    f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{quote_plus(DB_NAME)}"
    f"?client_encoding=utf8"
)

# Propagation
SEGMENT_MINUTES = int(os.getenv("SEGMENT_MINUTES", "10"))
PROPAGATION_HORIZON_DAYS = int(os.getenv("PROPAGATION_HORIZON_DAYS", "3"))
