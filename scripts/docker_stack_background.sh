#!/usr/bin/env bash
# Docker 任选其一：Colima（推荐 CLI）或 Docker Desktop。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
if [[ -d "/Applications/Docker.app/Contents/Resources/bin" ]]; then
  export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
fi

COLIMA_PROFILE="${COLIMA_PROFILE:-default}"
COLIMA_SOCK="${HOME}/.colima/${COLIMA_PROFILE}/docker.sock"

ensure_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  # Docker Desktop 常用 sock
  if [[ -S "${HOME}/.docker/run/docker.sock" ]]; then
    export DOCKER_HOST="unix://${HOME}/.docker/run/docker.sock"
    docker info >/dev/null 2>&1 && return 0
  fi
  # Colima
  if [[ -S "${COLIMA_SOCK}" ]]; then
    export DOCKER_HOST="unix://${COLIMA_SOCK}"
    docker info >/dev/null 2>&1 && return 0
  fi
  if command -v colima >/dev/null 2>&1; then
    if ! colima status 2>/dev/null | grep -qiE 'is running|Running'; then
      echo ">>> 启动 Colima（profile: ${COLIMA_PROFILE}）…"
      colima start --profile "${COLIMA_PROFILE}"
    fi
    export DOCKER_HOST="unix://${COLIMA_SOCK}"
    docker info >/dev/null 2>&1 && return 0
  fi
  return 1
}

echo ">>> 连接 Docker（Colima 或 Docker Desktop）…"
if ! ensure_docker; then
  echo "未检测到可用的 Docker。请先执行: colima start"
  echo "（若使用自定义 profile: COLIMA_PROFILE=xxx $0）"
  exit 1
fi
echo ">>> Docker 已就绪 (${DOCKER_HOST:-default context})"

echo ">>> 启动 PostGIS"
docker compose up -d db

echo ">>> 等待数据库就绪"
until docker compose exec -T db pg_isready -U postgres -d space_debris >/dev/null 2>&1; do
  sleep 2
done

echo ">>> 初始化表结构 + 迁移（v_debris_density / v_high_risk_events 自动建立；可重复执行）"
docker compose run --rm app python run.py init-db

echo ">>> 后台启动全量 ingest（容器名 debris-ingest）"
echo "    顺序：Space-Track → GCAT/UNOOSA/UCS/ESA → Asterank → 刷新 v_unified_objects"
docker rm -f debris-ingest 2>/dev/null || true
docker compose run -d --name debris-ingest app sh -c '
  set -e
  echo "=== [1/4] Space-Track ingest ==="
  python run.py ingest
  echo "=== [2/4] External sources (GCAT + UNOOSA + UCS + ESA DISCOS) ==="
  python scripts/ingest_external.py
  echo "=== [3/4] Asterank (asteroids / NEO) ==="
  python scripts/ingest_asterank.py
  echo "=== [4/4] Refresh v_unified_objects ==="
  python run.py refresh-views
  echo "=== ALL DONE ==="
'

echo ">>> 构建并后台启动 Streamlit + FastAPI（端口 8501 / 8502）"
docker compose up -d --build app api

echo ""
echo "完成。"
echo "  仪表盘:    http://localhost:8501"
echo "  API 文档:  http://localhost:8502/docs"
echo "  查看 ingest 日志: docker logs -f debris-ingest"
echo "  如需按样本量小跑: docker compose run --rm app sh -c \"python run.py ingest --limit 500 && python scripts/ingest_external.py --limit 500 && python scripts/ingest_asterank.py --limit 500 && python run.py refresh-views\""
