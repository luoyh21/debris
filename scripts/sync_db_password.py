#!/usr/bin/env python3
"""
将 PostgreSQL 中 DB_USER 的登录密码改为与项目根目录 .env 中 DB_PASSWORD 一致。

适用场景：你在 .env 里改了 DB_PASSWORD，需要数据库角色密码同步更新。
Docker：在 db 容器内以本地连接执行 ALTER USER，无需旧密码。
本机直连：若数据库里仍是旧密码，可任选其一：
  - 设置环境变量 PGPASSWORD 为旧密码；或
  - 在 .env 中临时增加一行 DB_PASSWORD_OLD=旧密码（脚本仅用其连接，改完后可删除）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from psycopg2.extensions import adapt  # noqa: E402
import psycopg2  # noqa: E402
from psycopg2.sql import SQL, Identifier, Literal  # noqa: E402


def _assert_safe_ident(user: str) -> None:
    if not user.replace("_", "").isalnum():
        raise ValueError("DB_USER 含非法字符")


def _sql_alter_password(user: str, password: str) -> str:
    _assert_safe_ident(user)
    pw_lit = adapt(password).getquoted().decode("utf-8")
    return f"ALTER USER {user} WITH PASSWORD {pw_lit};"


def _docker_db_running() -> bool:
    try:
        r = subprocess.run(
            ["docker", "compose", "ps", "-q", "db"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool((r.stdout or "").strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _sync_via_docker(user: str, password: str) -> None:
    _assert_safe_ident(user)
    sql = _sql_alter_password(user, password)
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "db",
        "psql",
        "-U",
        user,
        "-d",
        "postgres",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    print("通过 Docker 服务 db 执行 ALTER USER …")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _find_psql_windows() -> str | None:
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "PostgreSQL"
    if not base.is_dir():
        return None
    for p in sorted(base.glob("*/bin/psql.exe"), reverse=True):
        return str(p)
    return None


def _sync_via_psql_exe(user: str, new_password: str, connect_pw: str) -> None:
    """使用 psql 执行 ALTER（部分 Windows 环境下 psycopg2 会因 libpq 报文编码报错）。"""
    psql = shutil.which("psql") or _find_psql_windows()
    if not psql:
        raise FileNotFoundError("psql")

    host = (os.getenv("DB_HOST", "localhost") or "localhost").strip()
    port = (os.getenv("DB_PORT", "5432") or "5432").strip()
    sql = _sql_alter_password(user, new_password)
    env = os.environ.copy()
    env["PGPASSWORD"] = connect_pw
    env.setdefault("PGCLIENTENCODING", "UTF8")
    cmd = [
        psql,
        "-h", host,
        "-p", port,
        "-U", user,
        "-d", "postgres",
        "-v", "ON_ERROR_STOP=1",
        "-c", sql,
    ]
    print(f"通过 psql ({psql}) 执行 ALTER USER …")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _connect_psycopg2_local(
    host: str, port: int, user: str, connect_pw: str,
) -> "psycopg2.extensions.connection":
    """连接本机 Postgres；在部分中文 Windows 环境下 libpq 可能因服务端报错文本编码触发 UnicodeDecodeError，尝试多种 PGCLIENTENCODING。"""
    kw = dict(
        host=host,
        port=port,
        user=user,
        password=connect_pw,
        dbname="postgres",
        connect_timeout=15,
        options="-c client_encoding=UTF8",
    )
    last_decode: UnicodeDecodeError | None = None
    for enc in ("UTF8", "LATIN1", "GBK"):
        os.environ["PGCLIENTENCODING"] = enc
        try:
            return psycopg2.connect(**kw)
        except UnicodeDecodeError as e:
            last_decode = e
            continue
        except psycopg2.OperationalError:
            raise
    if last_decode:
        raise last_decode
    raise RuntimeError("无法建立数据库连接")


def _sync_via_psycopg2(user: str, new_password: str) -> None:
    """本机直连：用 psycopg2 执行 ALTER USER（无需安装 psql 客户端）。"""
    _assert_safe_ident(user)
    host = (os.getenv("DB_HOST", "localhost") or "localhost").strip()
    port = int((os.getenv("DB_PORT", "5432") or "5432").strip())
    # 连接用密码：PGPASSWORD > DB_PASSWORD_OLD（.env）> 假定库中已是新密码
    connect_pw = (
        (os.environ.get("PGPASSWORD") or "").strip()
        or (os.getenv("DB_PASSWORD_OLD") or "").strip()
        or new_password
    )

    print("通过本机 psycopg2 连接 Postgres 执行 ALTER USER …")
    try:
        conn = _connect_psycopg2_local(host, port, user, connect_pw)
    except UnicodeDecodeError:
        print("psycopg2 因编码异常无法连接，改用 psql …", file=sys.stderr)
        _sync_via_psql_exe(user, new_password, connect_pw)
        return
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                SQL("ALTER USER {} WITH PASSWORD {}").format(
                    Identifier(user), Literal(new_password)
                )
            )
    finally:
        conn.close()


def main() -> int:
    user = (os.getenv("DB_USER", "postgres") or "postgres").strip()
    password = (os.getenv("DB_PASSWORD", "") or "").strip()
    if not password:
        print("错误：.env 中 DB_PASSWORD 为空。", file=sys.stderr)
        return 1

    if _docker_db_running():
        _sync_via_docker(user, password)
        print("已同步：数据库角色密码已与 .env 中 DB_PASSWORD 一致（Docker db）。")
        return 0

    try:
        _sync_via_psycopg2(user, password)
    except FileNotFoundError:
        print(
            "未找到 psql.exe，且 psycopg2 无法连接。请安装 PostgreSQL 客户端或检查 PATH。",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError:
        print(
            "执行失败。若数据库仍使用旧密码，请先：\n"
            "  $env:PGPASSWORD = '<旧密码>'\n"
            "然后再次运行本脚本。",
            file=sys.stderr,
        )
        return 1
    except psycopg2.OperationalError as exc:
        print(f"连接失败：{exc}", file=sys.stderr)
        print(
            "若本机 Postgres 仍使用旧密码，请先：\n"
            "  $env:PGPASSWORD = '<旧密码>'\n"
            "然后再次运行本脚本。",
            file=sys.stderr,
        )
        return 1
    print("已同步：数据库角色密码已与 .env 中 DB_PASSWORD 一致（本机直连）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
