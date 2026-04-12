"""Central configuration loaded from .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Space-Track credentials
SPACETRACK_USERNAME = os.getenv("SPACETRACK_USERNAME", "")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD", "")

# OpenAI / LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "space_debris")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Propagation
SEGMENT_MINUTES = int(os.getenv("SEGMENT_MINUTES", "10"))
PROPAGATION_HORIZON_DAYS = int(os.getenv("PROPAGATION_HORIZON_DAYS", "3"))
