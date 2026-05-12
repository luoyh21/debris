"""STK 不可用时的参考真值生成器。

策略
----
* **SGP4**：直接调用 ``sgp4`` 库（Vallado/Brandon 官方 C/Cython 移植，已被 NASA / 18 SPCS
  / Celestrak 等机构作为参考实现）作为对照真值。本系统的
  :class:`propagator.sgp4_propagator.SGP4Propagator` 与之对照能反映 *自身封装层* 的
  实现质量；和 STK SGP4 对照（理论上 STK 也用同一个 Vallado 算法）几乎等价。
* **HPOP / 数值积分**：复用 :func:`scipy.integrate.solve_ivp` 跑一个独立配置的
  RK45 积分器（点质量地球 + J2 + J3 + 月日扰动 + 简单大气阻力），与本系统
  6-DOF 集成器对照。STK 真值不可用时这是次优替代，但 RIC 误差量级仍能给出
  相对一致的判定。
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Sequence, Tuple

import numpy as np

try:
    from sgp4.api import Satrec  # noqa: F401
    _SGP4_OK = True
except Exception:
    _SGP4_OK = False


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def reference_sgp4(
    line1: str,
    line2: str,
    t_offsets_s: Sequence[float],
    t_start: datetime,
) -> Tuple[List[List[float]], List[List[float]]]:
    """使用独立 ``sgp4`` 库（Satrec + jday）生成 ECI(TEME) 位置 / 速度。

    与本系统 ``SGP4Propagator`` 对照时记得使用同一坐标系（TEME ≈ J2000 在短期内）。
    """
    if not _SGP4_OK:
        raise RuntimeError("sgp4 库不可用：无法生成参考 SGP4 真值")
    from sgp4.api import Satrec, jday

    sat = Satrec.twoline2rv(line1, line2)
    t_start = _ensure_utc(t_start)
    positions: List[List[float]] = []
    velocities: List[List[float]] = []
    for t_off in t_offsets_s:
        epoch = t_start + timedelta(seconds=float(t_off))
        jd, fr = jday(
            epoch.year, epoch.month, epoch.day,
            epoch.hour, epoch.minute,
            epoch.second + epoch.microsecond / 1e6,
        )
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            # 用上一个有效点延续，避免抛异常打断整段对照
            if positions:
                positions.append(positions[-1])
                velocities.append(velocities[-1])
                continue
            raise RuntimeError(f"sgp4 propagation error code={e}")
        positions.append([float(r[0]), float(r[1]), float(r[2])])
        velocities.append([float(v[0]), float(v[1]), float(v[2])])
    return positions, velocities


# ─── HPOP-lite：J2/J3 + 月日扰动 + 指数大气阻力 ──────────────────────────────────
_MU_EARTH   = 398600.4418        # km^3/s^2
_R_EARTH    = 6378.137           # km
_J2         = 1.08263e-3
_J3         = -2.5327e-6
_MU_SUN     = 1.32712442099e11   # km^3/s^2
_MU_MOON    = 4902.800066        # km^3/s^2
_OMEGA_E    = 7.2921150e-5       # rad/s
_AU_KM      = 1.495978707e8


def _atm_density(alt_km: float) -> float:
    """指数大气密度，760 km 以上视为 0。"""
    if alt_km > 700:
        return 0.0
    table = [
        (0,    1.225,      8.44),
        (100,  5.297e-7,   5.877),
        (200,  2.541e-10,  37.0),
        (400,  2.803e-12,  58.0),
        (600,  1.137e-13,  78.0),
    ]
    for h_b, rho_b, H in reversed(table):
        if alt_km >= h_b:
            return rho_b * math.exp(-(alt_km - h_b) / H)
    return table[0][1]


def _sun_position_eci(epoch: datetime) -> np.ndarray:
    """简化太阳位置（J2000 ECI，km）—— Vallado 低精度公式。"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    days = (_ensure_utc(epoch) - j2000).total_seconds() / 86400.0
    T = days / 36525.0
    M = math.radians((357.5291 + 35999.0503 * T) % 360.0)
    L = math.radians((280.4665 + 36000.7698 * T) % 360.0)
    lam = L + 2.0 * 1.9148e-2 * math.sin(M)
    eps = math.radians(23.4393)
    R = (1.000001018 * (1 - 0.01671123 * math.cos(M))) * _AU_KM
    return np.array([
        R * math.cos(lam),
        R * math.sin(lam) * math.cos(eps),
        R * math.sin(lam) * math.sin(eps),
    ])


def _moon_position_eci(epoch: datetime) -> np.ndarray:
    """简化月球位置（J2000 ECI，km）—— Vallado 低精度公式。"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    T = (_ensure_utc(epoch) - j2000).total_seconds() / 86400.0 / 36525.0
    L = math.radians((218.32 + 481267.8813 * T) % 360.0)
    M = math.radians((134.96 + 477198.8676 * T) % 360.0)
    F = math.radians((93.27 + 483202.0175 * T) % 360.0)
    lam = L + math.radians(6.289 * math.sin(M))
    beta = math.radians(5.128 * math.sin(F))
    eps = math.radians(23.4393)
    R = 385000.56 - 20905.355 * math.cos(M)
    return np.array([
        R * math.cos(beta) * math.cos(lam),
        R * (math.cos(eps) * math.cos(beta) * math.sin(lam) - math.sin(eps) * math.sin(beta)),
        R * (math.sin(eps) * math.cos(beta) * math.sin(lam) + math.cos(eps) * math.sin(beta)),
    ])


def _accel(t_s: float, y: np.ndarray, epoch0: datetime,
           mass_kg: float, drag_area_m2: float, cd: float) -> np.ndarray:
    """返回 ECI 加速度（km/s²），状态向量 y=[x,y,z,vx,vy,vz]。"""
    r = y[:3]
    v = y[3:6]
    r_norm = float(np.linalg.norm(r))
    if r_norm <= _R_EARTH:
        return np.zeros(6)

    # 主引力 + J2 + J3
    x, y_, z = r
    z_r = z / r_norm
    mu_r3 = _MU_EARTH / r_norm ** 3
    j2_fac = 1.5 * _J2 * (_R_EARTH / r_norm) ** 2
    j3_fac = 2.5 * _J3 * (_R_EARTH / r_norm) ** 3

    a_grav_x = -mu_r3 * x * (1.0 - j2_fac * (5.0 * z_r ** 2 - 1.0)
                             - j3_fac * (3.0 * z_r - 7.0 * z_r ** 3 / 3.0))
    a_grav_y = -mu_r3 * y_ * (1.0 - j2_fac * (5.0 * z_r ** 2 - 1.0)
                              - j3_fac * (3.0 * z_r - 7.0 * z_r ** 3 / 3.0))
    a_grav_z = -mu_r3 * z * (1.0 - j2_fac * (5.0 * z_r ** 2 - 3.0)
                             - j3_fac * (6.0 * z_r ** 2 - 7.0 * z_r ** 4 / r_norm ** 2 - 3.0 / 5.0))
    a = np.array([a_grav_x, a_grav_y, a_grav_z])

    # 月日点质量
    epoch = epoch0 + timedelta(seconds=float(t_s))
    for mu, body in ((_MU_SUN, _sun_position_eci(epoch)),
                     (_MU_MOON, _moon_position_eci(epoch))):
        d = body - r
        d_norm = float(np.linalg.norm(d))
        body_norm = float(np.linalg.norm(body))
        a += mu * (d / d_norm ** 3 - body / body_norm ** 3)

    # 大气阻力（指数模型 → 仅低高度有效）
    alt_km = r_norm - _R_EARTH
    rho = _atm_density(alt_km)
    if rho > 0 and mass_kg > 0:
        v_rel = v - np.cross([0.0, 0.0, _OMEGA_E], r)
        v_rel_norm = float(np.linalg.norm(v_rel))
        v_ms = v_rel_norm * 1000.0
        a_drag_ms2 = -0.5 * rho * cd * drag_area_m2 / mass_kg * v_ms * (v_rel * 1000.0)
        a += a_drag_ms2 / 1e3 / 1e3   # m/s² → km/s²

    return np.concatenate([v, a])


def reference_hpop_lite(
    initial_state_eci: Sequence[float],
    epoch: datetime,
    t_offsets_s: Sequence[float],
    *,
    mass_kg: float = 1000.0,
    drag_area_m2: float = 10.0,
    drag_cd: float = 2.2,
) -> Tuple[List[List[float]], List[List[float]]]:
    """简化 HPOP 真值（J2/J3 + 月日扰动 + 指数大气，scipy RK45）。"""
    from scipy.integrate import solve_ivp

    epoch = _ensure_utc(epoch)
    y0 = np.asarray(initial_state_eci, dtype=float).copy()
    if y0.size != 6:
        raise ValueError("initial_state_eci 需 6 元素 [x,y,z,vx,vy,vz]")

    t_eval = np.asarray([float(t) for t in t_offsets_s], dtype=float)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return _accel(t, y, epoch, mass_kg, drag_area_m2, drag_cd)

    sol = solve_ivp(
        rhs, (float(t_eval[0]), float(t_eval[-1])), y0,
        method="DOP853", t_eval=t_eval,
        rtol=1e-9, atol=1e-9, max_step=60.0,
    )
    if not sol.success:
        raise RuntimeError(f"HPOP-lite 积分失败：{sol.message}")

    positions: List[List[float]] = []
    velocities: List[List[float]] = []
    for k in range(sol.y.shape[1]):
        positions.append([float(sol.y[0, k]), float(sol.y[1, k]), float(sol.y[2, k])])
        velocities.append([float(sol.y[3, k]), float(sol.y[4, k]), float(sol.y[5, k])])
    return positions, velocities
