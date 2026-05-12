"""大气密度模型：NRLMSISE-00（首选）与 USSA-76（fallback）。

设计目标
--------
* 提供一个统一函数 :func:`density_kg_m3`，按场景自动选择更合适的模型。
* 当本机已安装 ``nrlmsise00`` 库时，使用 NRLMSISE-00 经验大气模型（包含
  太阳活动、地磁指数、年/日变化）；否则回退到 USSA-1976 指数大气
  （与 ``trajectory/six_dof.py::atmo_density`` 同源），保证算法在任何环境
  下都能跑通。
* 太阳活动 / 地磁参数缺省值取 2024 年（太阳第 25 周期峰值）的中位数
  ``F107=165, F107A=165, Ap=10``。可由调用方覆盖。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

try:
    from nrlmsise00 import msise_model      # type: ignore
    _NRLMSISE_OK = True
except Exception:
    _NRLMSISE_OK = False


def is_nrlmsise_available() -> bool:
    """运行时判断本机是否能用 NRLMSISE-00。"""
    return _NRLMSISE_OK


# ─── USSA-1976 指数大气（与 trajectory/six_dof.py 一致） ───────────────────────
_USSA76 = [
    (0,    1.225,      8.44),
    (11,   3.6392e-1,  6.343),
    (20,   8.8035e-2,  6.682),
    (32,   1.3225e-2,  7.554),
    (47,   1.4275e-3,  8.382),
    (51,   8.6160e-4,  7.714),
    (71,   6.4211e-5,  5.442),
    (86,   5.6040e-6,  7.500),
    (150,  2.0700e-9,  37.70),
    (300,  1.9200e-11, 89.80),
    (500,  5.2100e-13, 63.30),
]


def _density_ussa76(alt_km: float) -> float:
    """USSA-1976 指数大气，单位 kg/m³，700 km 以上为 0。"""
    if alt_km > 700.0:
        return 0.0
    for h_b, rho_b, H in reversed(_USSA76):
        if alt_km >= h_b:
            return rho_b * math.exp(-(alt_km - h_b) / H)
    return _USSA76[0][1]


def _density_nrlmsise(
    epoch: datetime, alt_km: float, lat_deg: float, lon_deg: float,
    f107a: float, f107: float, ap: float,
) -> float:
    """调用 nrlmsise00 库，返回总质量密度 [kg/m³]。

    * 输出元组 ``(densities, temperatures)`` 中 ``densities[5]`` 是
      total mass density，单位 g/cm³，需 × 1000 转 kg/m³。
    * 高度上限 1000 km；超出范围回退 0。
    """
    if alt_km < 0 or alt_km > 1000.0:
        return 0.0
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    out = msise_model(epoch, alt_km, lat_deg, lon_deg, f107a, f107, ap)
    rho_g_cm3 = float(out[0][5])
    return rho_g_cm3 * 1000.0


def density_kg_m3(
    alt_km: float,
    *,
    epoch: Optional[datetime] = None,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    use_nrlmsise: bool = False,
    f107a: float = 165.0,
    f107: float = 165.0,
    ap: float = 10.0,
) -> float:
    """统一入口：返回大气密度 [kg/m³]。

    Parameters
    ----------
    alt_km : float
        几何高度（km）。
    epoch : datetime, optional
        UTC 时刻。仅 NRLMSISE-00 用到（年 / 日 / 季节性）。
    lat_deg, lon_deg : float
        地心 / 地理纬度、经度（度）。仅 NRLMSISE-00 用到。
    use_nrlmsise : bool
        ``True`` → 用 NRLMSISE-00（缺库则自动回退 USSA-76）；
        ``False`` → 直接 USSA-76（默认，保证与 trajectory/six_dof.py 一致）。
    f107a, f107, ap : float
        太阳 10.7 cm 通量（81 天平均 / 当日，sfu）和地磁 Ap 指数。
        缺省取 2024 年 Solar Cycle 25 峰值的中位数。
    """
    if use_nrlmsise and _NRLMSISE_OK and epoch is not None:
        try:
            return _density_nrlmsise(epoch, alt_km, lat_deg, lon_deg, f107a, f107, ap)
        except Exception:
            pass
    return _density_ussa76(alt_km)
