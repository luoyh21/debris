"""STK 验证 smoke-test：使用 ISS TLE 跑一次 SGP4 vs 参考实现对照。

无需 STK；当前主机若未安装 STK SDK，会自动回退到独立 ``sgp4`` 库做参考真值。
"""
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

# 允许 `python scripts/smoke_test_stk.py` 直接调用（脚本目录之外的项目根）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Windows GBK 控制台兜底：用 UTF-8 重新包装 stdout，避免 emoji 报错
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from stk_validation import detect_stk_availability, run_sgp4_validation  # noqa: E402
from stk_validation.report import load_full_doc, DEFAULT_REPORT_PATH  # noqa: E402


# ISS (ZARYA) — Celestrak 上 2024 公开示例 TLE
ISS_L1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990"
ISS_L2 = "2 25544  51.6400 130.0000 0001000   0.0000   0.0000 15.50000000000010"


def main() -> None:
    avail = detect_stk_availability()
    print("== availability ==")
    print(f"  available={avail.available}  os={avail.os_name}  sdk={avail.sdk}")

    print("\n== run_sgp4_validation (1 day, 10-min step, threshold 1 m) ==")
    rep = run_sgp4_validation(
        ISS_L1, ISS_L2,
        norad_id=25544,
        duration_s=86400.0, step_s=600.0,
        threshold_km=0.001,
        epoch_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    print("  ", rep.short_summary())
    print(f"  reference={rep.reference}")
    print(f"  pos_rms={rep.pos_rms_km*1000:.4f} m  in_track_rms={rep.in_track_rms_km*1000:.4f} m  "
          f"cross={rep.cross_track_rms_km*1000:.4f} m  radial={rep.radial_rms_km*1000:.4f} m")
    print(f"  passed={rep.passed}  notes={rep.notes}")

    doc = load_full_doc()
    print(f"\n== persisted to {DEFAULT_REPORT_PATH} ==")
    print(f"  history len = {len(doc.get('history', []))}")
    if doc.get("history"):
        print(f"  latest label = {doc['history'][0].get('label')}")


if __name__ == "__main__":
    main()
