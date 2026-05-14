"""Periodic (default 30-day) space-event incremental + email scheduler.

Mirrors ``scripts/incremental_scheduler.py`` but for the space-events
subsystem.  Each cycle:

1. Calls :mod:`scripts.ingest_events_incremental` with a rolling ``--since``
   cutoff.
2. Captures the resulting zip in ``data/event_packages/``.
3. Emails it via SMTP (credentials from ``.env``).
4. Persists ``last_since`` / ``last_run_at`` to ``state.json`` so restarts
   resume correctly.

CLI mirrors the parent scheduler::

    python3 scripts/events_scheduler.py --once --since 2026-04-01
    python3 scripts/events_scheduler.py --interval-days 30
    python3 scripts/events_scheduler.py --once --no-email   # dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
except Exception:
    pass

from scripts.ingest_events_incremental import (
    main as run_events_incremental,
    PKG_DIR,
    parse_since,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

STATE_PATH = os.path.join(PKG_DIR, "state.json")


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    os.makedirs(PKG_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _send_email(zip_path: str, since: dt.datetime, run_at: dt.datetime) -> bool:
    host = os.environ.get("SMTP_HOST", "").strip()
    if not host:
        log.warning("SMTP_HOST 未配置，跳过邮件发送"); return False
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "").strip()
    pwd  = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SMTP_FROM", "").strip() or user
    rcpt   = os.environ.get("SMTP_TO", "").strip()
    if not (user and pwd and rcpt):
        log.warning("SMTP_USER/SMTP_PASSWORD/SMTP_TO 未全部配置，跳过邮件发送")
        return False
    use_tls = os.environ.get("SMTP_USE_TLS", "1").strip() not in ("0","false","no")
    rcpts = [r.strip() for r in rcpt.split(",") if r.strip()]

    msg = EmailMessage()
    msg["Subject"] = (f"[空间碎片监测] 太空事件增量包 "
                      f"{since:%Y-%m-%d} → {run_at:%Y-%m-%d}")
    msg["From"] = sender
    msg["To"]   = ", ".join(rcpts)
    msg.set_content(
        f"自动定时增量更新已生成太空事件快照压缩包，请见附件。\n\n"
        f"  数据起点 (since): {since.isoformat()}\n"
        f"  生成时间        : {run_at.isoformat()}\n"
        f"  压缩包          : {os.path.basename(zip_path)}\n"
        f"  大小            : {os.path.getsize(zip_path)/1024.0:.2f} KB\n\n"
        f"应用该压缩包到任意数据库的命令：\n"
        f"  python3 scripts/apply_events_package.py "
        f"{os.path.basename(zip_path)}\n",
        subtype="plain", charset="utf-8")

    with open(zip_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application",
                           subtype="zip", filename=os.path.basename(zip_path))

    log.info("发送邮件 → %s（SMTP %s:%d, TLS=%s）", rcpts, host, port, use_tls)
    try:
        if use_tls:
            ctx = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=120) as s:
                s.ehlo(); s.starttls(context=ctx); s.ehlo()
                s.login(user, pwd); s.send_message(msg)
        else:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=120) as s:
                s.login(user, pwd); s.send_message(msg)
    except Exception as exc:
        log.error("邮件发送失败：%s", exc); return False
    log.info("邮件已发送")
    return True


def _do_one_cycle(since: dt.datetime, *, send_email: bool,
                  max_rows: int, sources: str) -> str | None:
    saved = sys.argv[:]
    sys.argv = [
        "scripts/ingest_events_incremental.py",
        "--since", since.strftime("%Y-%m-%dT%H:%M:%S"),
        "--max",   str(max_rows),
        "--sources", sources,
    ]
    try:
        zip_path = run_events_incremental()
    finally:
        sys.argv = saved

    if not zip_path:
        log.info("本次未产生压缩包"); return None
    log.info("生成压缩包: %s", zip_path)
    if send_email:
        _send_email(zip_path, since, dt.datetime.now(dt.timezone.utc))
    return zip_path


def main():
    ap = argparse.ArgumentParser(description="太空事件 30 天增量调度器")
    ap.add_argument("--interval-days", type=int, default=30)
    ap.add_argument("--since", default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--sources", default="all")
    args = ap.parse_args()

    interval = max(1, int(args.interval_days))
    state = _load_state()
    if args.since:
        next_since = parse_since(args.since)
    elif "last_since" in state:
        next_since = parse_since(state["last_since"])
    else:
        next_since = (dt.datetime.now(dt.timezone.utc)
                      - dt.timedelta(days=interval))

    log.info("调度器启动: interval=%d 天, since=%s, once=%s, email=%s",
             interval, next_since.isoformat(), args.once, not args.no_email)

    while True:
        run_at = dt.datetime.now(dt.timezone.utc)
        try:
            _do_one_cycle(next_since,
                          send_email=not args.no_email,
                          max_rows=args.max,
                          sources=args.sources)
            state.update({"last_since": next_since.isoformat(),
                          "last_run_at": run_at.isoformat()})
            _save_state(state)
        except Exception as exc:
            log.exception("本次增量执行失败: %s", exc)
        if args.once:
            log.info("--once 已执行完成，退出"); return
        next_since = run_at
        sleep_s = interval * 86400
        log.info("下一次运行时间: %s（%.1f 天后）",
                 (run_at + dt.timedelta(seconds=sleep_s)).isoformat(), interval)
        try: time.sleep(sleep_s)
        except KeyboardInterrupt:
            log.info("收到中断信号，退出"); return


if __name__ == "__main__":
    main()
