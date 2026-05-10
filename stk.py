"""STK 集成的顶层初始化 / 诊断入口。

用法
====

直接命令行运行（推荐，不需要 import 项目其它模块）::

    .\\venv\\Scripts\\python.exe stk.py            # 完整端到端自检
    .\\venv\\Scripts\\python.exe stk.py --check    # 只查可用性，不拉起 STK
    .\\venv\\Scripts\\python.exe stk.py --json     # 以 JSON 输出（便于脚本消费）
    .\\venv\\Scripts\\python.exe stk.py --tle iss  # 用 ISS TLE 跑一次完整 SGP4 验证

也可在交互式 / Jupyter 中 ``import stk`` 后调 :func:`stk.init`、:func:`stk.test`。

设计
----
* 不依赖项目其它模块的运行时配置，启动时不连接数据库 / 不读取 .env。
* 所有 STK SDK 调用都走 :mod:`stk_validation.stk_adapter`，保持单一职责。
* 自检失败时给出明确的修复建议（缺包 / 缺 STK / OS 不支持 / ProgID 不匹配等）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_utf8_stdout() -> None:
    """Windows 中文 GBK 终端兜底：用 UTF-8 重新包装 stdout，避免 emoji 报错。"""
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass


def _ensure_project_root_on_syspath() -> None:
    """允许从任意目录运行 ``python stk.py``。"""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ─── 公开 API ────────────────────────────────────────────────────────────────────

def check() -> Dict[str, Any]:
    """只做可用性探测（不真正拉起 STK）。返回 :class:`StkAvailability` 字典。"""
    _ensure_project_root_on_syspath()
    from stk_validation import detect_stk_availability
    return detect_stk_availability().to_dict()


def init(*, verbose: bool = True) -> Optional[object]:
    """打开 STK 会话并返回内部 session 句柄。失败返回 None。

    ``session`` 可以传给 :func:`stk_validation.stk_adapter.stk_propagate_sgp4`
    / ``stk_propagate_hpop`` 复用，避免反复启动 STK。
    """
    _ensure_project_root_on_syspath()
    from stk_validation.stk_adapter import _open_stk

    avail = check()
    if not avail.get("os_supported"):
        if verbose:
            print(f"⚠ 当前操作系统 `{avail.get('os_name')}` 不支持 Ansys STK Engine。")
        return None
    if not avail.get("available"):
        if verbose:
            print("⚠ 未检测到任何 STK Python SDK / COM ProgID。")
            print(f"  reason   = {avail.get('reason')}")
            print(f"  hint     = {avail.get('install_hint')}")
        return None

    session = _open_stk()
    if session is None:
        if verbose:
            print("✗ STK 会话打开失败：")
            print("  · 已检测到 SDK 但无法 Dispatch / new_object_root；")
            print("  · 通常意味着 STK 桌面版未安装、或本进程无 COM 注册权限。")
        return None

    if verbose:
        sdk_kind = session[0]
        print(f"✓ STK 会话已建立（kind = {sdk_kind}）。")
        print("  使用完后请显式调用 stk_validation.stk_adapter.shutdown_stk(session)，")
        print("  否则 STK 进程可能残留。")
    return session


def test(*, json_output: bool = False) -> Dict[str, Any]:
    """端到端自检：打开 STK → 加载 ISS TLE → 读取 5 个点 ECI 状态 → 关闭。"""
    _ensure_project_root_on_syspath()
    from stk_validation.stk_adapter import diagnose

    result = {"availability": check()}
    result["diagnostic"] = diagnose()

    if json_output:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result

    avail = result["availability"]
    diag = result["diagnostic"]

    print("=" * 60)
    print("STK 端到端自检")
    print("=" * 60)
    print(f"OS          : {avail.get('os_name')} (supported={avail.get('os_supported')})")
    print(f"Available   : {avail.get('available')}")
    print(f"SDK         : {avail.get('sdk')}  ver={avail.get('sdk_version')}")
    print(f"InstallDir  : {avail.get('install_dir') or '—'}")
    print("-" * 60)
    print(f"Step        : {diag.get('step')}")
    print(f"Session     : {diag.get('session')}")
    if diag.get("ok"):
        s = diag.get("sample") or {}
        print("Result      : ✓ STK SGP4 推演成功！")
        print(f"  样本数     = {s.get('n')}")
        print(f"  first pos = {s.get('first_pos_km')} km")
        print(f"  first vel = {s.get('first_vel_kms')} km/s")
        print(f"  last  pos = {s.get('last_pos_km')} km")
    else:
        print(f"Result      : ✗ STK 自检失败")
        print(f"  error     = {diag.get('error')}")
        print()
        print("修复建议：")
        if diag.get("step") == "open_stk":
            print("  · 检查 STK 桌面版是否已正确安装并启动过一次（首启会注册 COM 类）。")
            print("  · 用管理员命令行重新运行 `regsvr32` 注册 STK COM 组件。")
            print("  · 或者改装 PySTK：`pip install ansys-stk-core` 走更现代的接口。")
        elif diag.get("step") == "stk_propagate_sgp4":
            print("  · 已成功打开 STK 会话，但 ImportTLEFile / DataProvider 出错。")
            print("  · 请确保 STK 安装包含 SGP4 模块（默认含），并具有有效 license。")
            print("  · 重启 STK 桌面版一次后再试；偶发的 -2147220988 错误通常与场景缓存相关。")
        else:
            print("  · 详见 error 字段，必要时把上面 step + error 回报给开发者。")

    return result


def run_full_validation(line1: Optional[str] = None,
                        line2: Optional[str] = None,
                        *,
                        norad_id: int = 25544,
                        duration_s: float = 86400.0,
                        step_s: float = 600.0,
                        threshold_km: float = 0.005) -> Dict[str, Any]:
    """跑一次完整 SGP4 vs STK 交叉验证（结果会写入聚合 JSON）。"""
    _ensure_project_root_on_syspath()
    from stk_validation import run_sgp4_validation

    if line1 is None or line2 is None:
        line1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990"
        line2 = "2 25544  51.6400 130.0000 0001000   0.0000   0.0000 15.50000000000010"
    epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    print(f"运行 SGP4 vs STK 交叉验证 ── NORAD={norad_id}, "
          f"duration={duration_s/3600:.1f} h, step={step_s/60:.1f} min")
    rep = run_sgp4_validation(
        line1, line2,
        norad_id=int(norad_id),
        duration_s=float(duration_s),
        step_s=float(step_s),
        threshold_km=float(threshold_km),
        epoch_utc=epoch,
    )
    print()
    print(rep.short_summary())
    print(f"reference = {rep.reference}")
    print(f"位置 RMS  = {rep.pos_rms_km*1000:.4f} m")
    print(f"In-track  = {rep.in_track_rms_km*1000:.4f} m")
    print(f"Cross     = {rep.cross_track_rms_km*1000:.4f} m")
    print(f"Radial    = {rep.radial_rms_km*1000:.4f} m")
    print(f"passed    = {rep.passed}")
    if rep.notes:
        print("诊断:")
        for n in rep.notes:
            print(f"  - {n}")
    return rep.to_dict()


# ─── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="STK 集成初始化 / 端到端诊断工具",
    )
    parser.add_argument("--check", action="store_true",
                        help="只查 STK 可用性（不拉起进程）")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 输出（便于脚本消费）")
    parser.add_argument("--tle", choices=["iss"], default=None,
                        help="使用预设 TLE 跑一次完整 SGP4 vs STK 交叉验证")
    parser.add_argument("--duration-h", type=float, default=24.0,
                        help="完整验证的时长（小时），默认 24")
    parser.add_argument("--step-min", type=float, default=10.0,
                        help="完整验证的步长（分钟），默认 10")
    args = parser.parse_args()

    _ensure_utf8_stdout()

    if args.check:
        avail = check()
        if args.json:
            print(json.dumps(avail, ensure_ascii=False, indent=2))
        else:
            print("STK 可用性探测结果：")
            for k, v in avail.items():
                print(f"  {k:14s}= {v}")
        return 0 if avail.get("available") else 1

    if args.tle == "iss":
        run_full_validation(
            duration_s=args.duration_h * 3600.0,
            step_s=args.step_min * 60.0,
        )
        return 0

    test(json_output=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
