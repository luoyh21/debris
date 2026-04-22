#!/usr/bin/env bash
# 将数据库中 postgres 用户密码改为与项目根目录 .env 中 DB_PASSWORD 一致。
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
exec python3 scripts/sync_db_password.py
