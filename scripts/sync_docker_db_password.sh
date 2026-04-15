#!/usr/bin/env bash
# 若 pgdata 卷是以前用「别的密码」初始化的，容器内本地连接通常仍可进库，执行本脚本把 postgres 用户密码改回与 docker-compose 一致。
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
docker compose exec -T db psql -U postgres -d postgres -c "ALTER USER postgres WITH PASSWORD 'lq3525926';"
echo "已执行：ALTER USER postgres PASSWORD 'lq3525926'（与 compose 中 POSTGRES_PASSWORD 一致）"
