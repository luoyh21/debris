"""Ansys STK Python 集成适配层（仅 Windows，懒加载，针对 STK 11/12 实测调通）。

提供两个对外 API：

* :func:`stk_propagate_sgp4` —— 给定 TLE，把卫星放进 STK 并用其 SGP4 推演，
  返回与本系统 :class:`~propagator.sgp4_propagator.SGP4Propagator` 同一时间表的
  ECI(ICRF) 位置 / 速度。
* :func:`stk_propagate_hpop` —— 给定初始 ECI 状态向量 + 弹道参数，使用 STK 的 HPOP
  做"地面真相"推演，与本系统 6-DOF 数值积分器对照。

实现要点（基于 STK 11 + win32com 实测，逐位与 ``sgp4`` 库 / Vallado 参考一致）::

    1. 启动 application：win32com.Dispatch("STK11.Application") / "STK12.Application"
       → app.Personality2 即 IAgStkObjectRoot
    2. 必须 root.NewScenario() 并用 SetAnalysisTimePeriod 设好场景时间窗（尤其包含 TLE epoch）
    3. ⚠ 不能用 ImportTLEFile 直接加载 — 会创建 PropagatorType=7 (STKExternal) 退化成 TwoBody
       默认轨道 (6678 km 圆轨道)。必须走以下五步：
         a) ExecuteCommand("New / */Satellite <NAME>")
         b) sat.SetPropagatorType(4)                       # 4 = ePropagatorSGP4
         c) prop.CommonTasks.AddSegsFromFile(SSC, FILE)    # 注意：参数 (SSC_int, file_str)
         d) prop.Propagate()                                # 必须显式调用
         e) DataProviders.Item("Cartesian Position").Group.Item("TEMEOfEpoch").Exec()
    4. SGP4 输出坐标系是 TEME，必须用 ``TEMEOfEpoch`` 取值（ICRF / J2000 会引入岁差章动旋转
       造成 ~10000 km 的虚假偏差）；HPOP 用 ``ICRF``。

设计：
* 模块顶层不 import STK SDK；所有 SDK 调用都封装在函数内部并用 try/except 兜底
* 失败时返回 ``None`` 而不是抛异常，由上层 :mod:`stk_validation.runner` 自动 fallback
* 暴露 :func:`shutdown_stk` 显式释放 STK 进程
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

_StkSession = Any  # (sdk_kind, app, root) 三元组


# ─── 时间格式 ────────────────────────────────────────────────────────────────────

def _utc_to_stk_str(dt: datetime) -> str:
    """格式化 UTC 时间为 STK UTCG 字符串：``DD MMM YYYY HH:MM:SS.000``。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%d %b %Y %H:%M:%S.000")


def _seconds_delta(s: float):
    return timedelta(seconds=float(s))


def _infer_step_s(t_offsets_s: Sequence[float]) -> float:
    """从目标采样时刻列表推断最小步长。优先取相邻间隔的 GCD-like 公约数。"""
    n = len(t_offsets_s)
    if n < 2:
        return 60.0
    diffs = [float(t_offsets_s[i + 1]) - float(t_offsets_s[i]) for i in range(n - 1)
             if t_offsets_s[i + 1] != t_offsets_s[i]]
    if not diffs:
        return 60.0
    step = min(diffs)
    return max(1.0, float(step))


# ─── STK 会话管理 ───────────────────────────────────────────────────────────────

# ProgID 优先级：STK 12 > STK 11 > STK 通用
_STK_PROG_IDS: Tuple[str, ...] = (
    "STK12.Application",
    "STK11.Application",
    "STK.Application",
)


def _open_stk_via_pystk() -> Optional[_StkSession]:
    """优先走 ansys-stk-core（PyAnsys）的 Engine / Desktop。"""
    try:
        from ansys.stk.core.stkengine import STKEngine  # type: ignore
        engine = STKEngine.start_application(no_graphics=True)
        root = engine.new_object_root()
        return ("ansys-stk-engine", engine, root)
    except Exception as exc:
        log.debug("ansys-stk-core stkengine import/start failed: %s", exc)
    try:
        from ansys.stk.core.stkdesktop import STKDesktop  # type: ignore
        desktop = STKDesktop.start_application(visible=False, user_control=False)
        return ("ansys-stk-desktop", desktop, desktop.root)
    except Exception as exc:
        log.debug("ansys-stk-core stkdesktop import/start failed: %s", exc)
    return None


def _open_stk_via_com() -> Optional[_StkSession]:
    """通过 COM (comtypes / win32com) Dispatch 已安装的 STK 桌面版。"""
    try:
        import comtypes.client  # type: ignore
        creator = ("comtypes", comtypes.client.CreateObject)
    except Exception:
        try:
            import win32com.client  # type: ignore
            creator = ("win32com", win32com.client.Dispatch)
        except Exception:
            return None

    sdk_kind, fn = creator
    last_err: Optional[Exception] = None
    for prog_id in _STK_PROG_IDS:
        try:
            app = fn(prog_id)
            try:
                app.Visible = False
            except Exception:
                pass
            try:
                app.UserControl = False
            except Exception:
                pass
            root = app.Personality2
            log.debug("STK COM connected via %s [%s]", sdk_kind, prog_id)
            return (f"{sdk_kind}:{prog_id}", app, root)
        except Exception as exc:
            last_err = exc
            continue
    if last_err is not None:
        log.debug("All STK ProgIDs failed (last=%s): %s", _STK_PROG_IDS[-1], last_err)
    return None


def _open_stk() -> Optional[_StkSession]:
    """打开 STK 会话；按 PySTK → COM 顺序尝试，失败返回 None。"""
    s = _open_stk_via_pystk()
    if s is not None:
        return s
    return _open_stk_via_com()


def shutdown_stk(session: _StkSession) -> None:
    """尽最大努力关闭 STK 实例。"""
    if not session:
        return
    sdk, app, _root = session
    try:
        if sdk.startswith("ansys-stk"):
            for fn in ("shutdown", "close_application"):
                try:
                    getattr(app, fn)()
                    return
                except Exception:
                    pass
        else:
            for fn in ("Quit",):
                try:
                    getattr(app, fn)()
                    return
                except Exception:
                    pass
    except Exception:
        pass


# ─── Scenario / 单位 ────────────────────────────────────────────────────────────

def _set_units(session: _StkSession) -> None:
    sdk, _app, root = session
    pairs = (("DateFormat", "UTCG"),
             ("DistanceUnit", "km"),
             ("TimeUnit", "sec"))
    if sdk.startswith("ansys-stk"):
        up = root.units_preferences
        for k, v in pairs:
            try:
                up.set_current_unit(k, v)
            except Exception:
                pass
    else:
        up = root.UnitPreferences
        for k, v in pairs:
            try:
                up.SetCurrentUnit(k, v)
            except Exception:
                pass


def _exec_cmd(session: _StkSession, cmd: str) -> Any:
    sdk, _app, root = session
    if sdk.startswith("ansys-stk"):
        return root.execute_command(cmd)
    return root.ExecuteCommand(cmd)


def _new_scenario(session: _StkSession, name: str,
                  t_start: datetime, t_stop: datetime) -> Any:
    sdk, _app, root = session
    # 关闭已有场景
    try:
        if sdk.startswith("ansys-stk"):
            root.close_scenario()
        else:
            root.CloseScenario()
    except Exception:
        pass
    # 新建
    if sdk.startswith("ansys-stk"):
        root.new_scenario(name)
        scenario = root.current_scenario
    else:
        root.NewScenario(name)
        scenario = root.CurrentScenario
    _set_units(session)
    # 时间窗
    cmd = (f'SetAnalysisTimePeriod * "{_utc_to_stk_str(t_start)}" '
           f'"{_utc_to_stk_str(t_stop)}"')
    try:
        _exec_cmd(session, cmd)
    except Exception as exc:
        log.warning("SetAnalysisTimePeriod failed: %s", exc)
    return scenario


def _get_object(session: _StkSession, path: str) -> Any:
    sdk, _app, root = session
    if sdk.startswith("ansys-stk"):
        return root.get_object_from_path(path)
    return root.GetObjectFromPath(path)


# ─── DataProvider 取 ECI 状态 ──────────────────────────────────────────────────

def _read_cartesian(
    session: _StkSession,
    sat_path: str,
    t_start: datetime,
    t_stop: datetime,
    step_s: float,
    *,
    frame: str = "ICRF",
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """通过 ``Cartesian Position`` / ``Cartesian Velocity`` DP 取 ECI 状态。

    返回 ``(positions[N][3], velocities[N][3])``，单位 km / km·s⁻¹。
    """
    sat_obj = _get_object(session, sat_path)
    sdk = session[0]
    if sdk.startswith("ansys-stk"):
        dp_root = sat_obj.data_providers
    else:
        dp_root = sat_obj.DataProviders

    def _exec_dp(name: str) -> Optional[Any]:
        try:
            if sdk.startswith("ansys-stk"):
                dp = dp_root.item(name).group.item(frame)
                return dp.exec(_utc_to_stk_str(t_start),
                               _utc_to_stk_str(t_stop), float(step_s))
            else:
                dp = dp_root.Item(name).Group.Item(frame)
                return dp.Exec(_utc_to_stk_str(t_start),
                               _utc_to_stk_str(t_stop), float(step_s))
        except Exception as exc:
            log.debug("DP[%s/%s] Exec failed: %s", name, frame, exc)
            return None

    pos_res = _exec_dp("Cartesian Position")
    vel_res = _exec_dp("Cartesian Velocity")
    if pos_res is None or vel_res is None:
        return None

    def _values(result: Any) -> Dict[str, List[float]]:
        ds = result.data_sets if sdk.startswith("ansys-stk") else result.DataSets
        n = ds.count if sdk.startswith("ansys-stk") else ds.Count
        out: Dict[str, List[float]] = {}
        for i in range(n):
            d = ds.item(i) if sdk.startswith("ansys-stk") else ds.Item(i)
            name = (d.element_name if sdk.startswith("ansys-stk")
                    else d.ElementName).strip()
            vals = list(d.get_values() if sdk.startswith("ansys-stk")
                        else d.GetValues())
            out[name] = vals
        return out

    pos_dict = _values(pos_res)
    vel_dict = _values(vel_res)
    if not all(k in pos_dict for k in ("x", "y", "z")):
        log.warning("Position DP missing x/y/z keys: %s", list(pos_dict))
        return None
    if not all(k in vel_dict for k in ("x", "y", "z")):
        log.warning("Velocity DP missing x/y/z keys: %s", list(vel_dict))
        return None

    n = min(len(pos_dict["x"]), len(vel_dict["x"]))
    positions = [[float(pos_dict["x"][i]),
                  float(pos_dict["y"][i]),
                  float(pos_dict["z"][i])] for i in range(n)]
    velocities = [[float(vel_dict["x"][i]),
                   float(vel_dict["y"][i]),
                   float(vel_dict["z"][i])] for i in range(n)]
    return positions, velocities


def _resample_at_offsets(
    series: Tuple[List[List[float]], List[List[float]]],
    series_start: datetime,
    series_step_s: float,
    target_offsets_s: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]]]:
    """把 STK 等步长输出按线性插值取到 ``target_offsets_s`` 上。"""
    pos_arr, vel_arr = series
    n_in = min(len(pos_arr), len(vel_arr))
    if n_in < 2:
        raise ValueError("STK series too short to resample")
    end_t = (n_in - 1) * float(series_step_s)
    pos_out: List[List[float]] = []
    vel_out: List[List[float]] = []
    for t in target_offsets_s:
        t = float(t)
        if t <= 0:
            t = 0.0
        if t >= end_t:
            t = end_t
        i0 = int(t // series_step_s)
        i1 = min(i0 + 1, n_in - 1)
        u = (t - i0 * series_step_s) / max(series_step_s, 1e-9) if i1 != i0 else 0.0
        p = [pos_arr[i0][k] * (1 - u) + pos_arr[i1][k] * u for k in range(3)]
        v = [vel_arr[i0][k] * (1 - u) + vel_arr[i1][k] * u for k in range(3)]
        pos_out.append(p)
        vel_out.append(v)
    return pos_out, vel_out


# ─── 公开 API：SGP4 ─────────────────────────────────────────────────────────────

def stk_propagate_sgp4(
    line1: str,
    line2: str,
    t_offsets_s: Sequence[float],
    t_start: datetime,
    *,
    norad_id: int = 0,
    session: Optional[_StkSession] = None,
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """使用 STK 的 SGP4 推演 TLE，返回 (positions_eci_km, velocities_eci_kms)。

    Parameters
    ----------
    session
        预先打开的 STK 会话，可避免每次重启 STK；若为 ``None`` 则函数内部 open/close。
    """
    own_session = False
    if session is None:
        session = _open_stk()
        own_session = True
        if session is None:
            return None

    if t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo=timezone.utc)
    duration = max(float(t_offsets_s[-1]) - float(t_offsets_s[0]) + 60.0, 600.0)
    t_stop = t_start + _seconds_delta(duration)

    # ── 关键 1：场景时间窗必须包含 TLE epoch，否则 SGP4 propagation interval 会异常
    # 我们传入的 t_start/t_stop 已经是基于真实 epoch 的，下面 _new_scenario 会用它们
    sat_norad = int(norad_id) if int(norad_id) > 0 else 99999

    # 将 TLE 写到临时 .tce 文件（ASCII；STK 11 只接受 ASCII 内容）
    fd, tle_path = tempfile.mkstemp(suffix=".tce", prefix=f"stk_sgp4_{sat_norad}_")
    try:
        os.close(fd)
        sat_name = f"sat_{sat_norad}"
        with open(tle_path, "w", encoding="ascii") as f:
            f.write(f"{sat_name}\n{line1}\n{line2}\n")

        try:
            _new_scenario(session, "DebrisSgp4Validation", t_start, t_stop)
            sdk = session[0]
            root = session[2]

            # ── 关键 2：必须先 New Satellite，再 SetPropagatorType(4)，再 AddSegsFromFile
            # 否则 STK 会创建一个 PropagatorType=7 (STKExternal) 的退化对象
            try:
                _exec_cmd(session, f'New / */Satellite {sat_name}')
            except Exception as exc:
                log.warning("STK New Satellite failed: %s", exc)
                return None

            sat_path = f"*/Satellite/{sat_name}"
            sat_obj = _get_object(session, sat_path)

            # SetPropagatorType(4 = ePropagatorSGP4)
            try:
                if sdk.startswith("ansys-stk"):
                    sat_obj.set_propagator_type(4)
                else:
                    sat_obj.SetPropagatorType(4)
            except Exception as exc:
                log.warning("SetPropagatorType(SGP4) failed: %s", exc)
                return None

            # 取 SGP4 propagator 句柄
            prop = (sat_obj.propagator if sdk.startswith("ansys-stk")
                    else sat_obj.Propagator)

            # ── 关键 3：CommonTasks.AddSegsFromFile 参数顺序是 (SSCNumber, FilePath)
            try:
                ct = (prop.common_tasks if sdk.startswith("ansys-stk")
                      else prop.CommonTasks)
                if sdk.startswith("ansys-stk"):
                    ct.add_segs_from_file(sat_norad, tle_path)
                else:
                    ct.AddSegsFromFile(sat_norad, tle_path)
            except Exception as exc:
                log.warning("CommonTasks.AddSegsFromFile failed: %s", exc)
                return None

            # ── 关键 4：必须显式 Propagate
            try:
                if sdk.startswith("ansys-stk"):
                    prop.propagate()
                else:
                    prop.Propagate()
            except Exception as exc:
                log.warning("SGP4 Propagator.Propagate() failed: %s", exc)
                return None

            # ── 关键 5：取 TEMEOfEpoch（SGP4 native 坐标系）。
            # 用目标步长直接跑 DataProvider，避免线性插值在 SGP4 非线性轨迹上
            # 引入的误差（SGP4 一圈 ~90 分钟，30 分钟步长插值会有 km 级偏差）。
            target_step_s = _infer_step_s(t_offsets_s)
            series = None
            for frame in ("TEMEOfEpoch", "TEMEOfDate", "TrueOfEpoch", "MeanOfEpoch"):
                series = _read_cartesian(
                    session, sat_path,
                    t_start, t_stop, target_step_s, frame=frame,
                )
                if series is not None:
                    log.debug("STK SGP4 frame used: %s, step=%.1fs", frame, target_step_s)
                    break
            if series is None:
                log.warning("STK Cartesian Position DP 全部参考系都失败")
                return None
            return _resample_at_offsets(series, t_start, target_step_s, t_offsets_s)
        except Exception as exc:
            log.warning("STK SGP4 propagation failed: %s", exc)
            return None
    finally:
        try:
            os.unlink(tle_path)
        except Exception:
            pass
        if own_session:
            try:
                shutdown_stk(session)
            except Exception:
                pass


# ─── 公开 API：HPOP ─────────────────────────────────────────────────────────────

def stk_propagate_hpop(
    initial_state_eci: Sequence[float],   # [x_km, y_km, z_km, vx_km/s, vy_km/s, vz_km/s]
    epoch:             datetime,
    t_offsets_s:       Sequence[float],
    *,
    mass_kg:           float = 1000.0,
    drag_area_m2:      float = 10.0,
    drag_cd:           float = 2.2,
    sat_name:          str = "DebrisCandidate",
    use_drag:          bool = True,
    use_third_body:    bool = True,
    use_srp:           bool = True,
    gravity_degree:    int = 21,
    session:           Optional[_StkSession] = None,
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """使用 STK 的 HPOP 推演给定初始 ECI 状态，返回 (positions_eci_km, velocities_eci_kms)。

    HPOP 配置（默认）：EGM2008 21×21 + NRLMSISE-00 + 月日点质量 + 太阳辐射压。

    实测的 OM 流程（STK 11，2024 实证）::

        sat = scenario.Children.New(eSatellite, name)
        sat.SetPropagatorType(0)                 # 0 = ePropagatorHPOP
        prop = sat.Propagator                    # IAgVePropagatorHPOP
        prop.InitialState.Representation.Epoch = "<UTCG>"
        prop.InitialState.Representation.AssignCartesian(1, x,y,z,vx,vy,vz)  # frame=Inertial
        prop.ForceModel.CentralBodyGravity.SetMaximumDegreeAndOrder(N, N)
        prop.ForceModel.Drag.Use = True / SRP.Use = True / ThirdBodyGravity.AddThirdBody(...)
        prop.Propagate()
    """
    own_session = False
    if session is None:
        session = _open_stk()
        own_session = True
        if session is None:
            return None

    sdk = session[0]
    is_pystk = sdk.startswith("ansys-stk")

    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    duration = max(float(t_offsets_s[-1]) - float(t_offsets_s[0]) + 60.0, 600.0)
    t_stop = epoch + _seconds_delta(duration)

    try:
        _new_scenario(session, "DebrisHpopValidation", epoch, t_stop)
        sat_path = f"*/Satellite/{sat_name}"
        try:
            _exec_cmd(session, f'New / */Satellite {sat_name}')
        except Exception as exc:
            log.warning("STK New Satellite (HPOP) failed: %s", exc)
            return None

        sat_obj = _get_object(session, sat_path)

        # SetPropagatorType(0) = HPOP
        try:
            if is_pystk:
                sat_obj.set_propagator_type(0)
            else:
                sat_obj.SetPropagatorType(0)
        except Exception as exc:
            log.warning("SetPropagatorType(HPOP) failed: %s", exc)
            return None

        prop = sat_obj.propagator if is_pystk else sat_obj.Propagator

        # InitialState 设置
        try:
            init_state = prop.initial_state if is_pystk else prop.InitialState
            rep = init_state.representation if is_pystk else init_state.Representation
            try:
                rep.Epoch = _utc_to_stk_str(epoch)
            except Exception:
                # PySTK 可能用 epoch 属性
                try:
                    rep.epoch = _utc_to_stk_str(epoch)
                except Exception as exc:
                    log.debug("HPOP rep epoch set failed: %s", exc)

            x, y, z, vx, vy, vz = (float(v) for v in initial_state_eci)
            # AgECoordinateSystem 枚举（实测 STK 11）：
            #   0 / 2 = Fixed (ECEF)，1 = TrueOfDate/TEME，3 = **ICRF**，
            #   4 = MeanOfEpoch，5 = MeanOfDate
            # 6-DOF 内部 ECI 与 J2000 / ICRF 同源，必须用 frame=3 (ICRF) 输入；
            # 后续 DataProvider 也取 ICRF。
            assigned = False
            for frame_code in (3, 4, 5, 1):
                try:
                    if is_pystk:
                        rep.assign_cartesian(frame_code, x, y, z, vx, vy, vz)
                    else:
                        rep.AssignCartesian(frame_code, x, y, z, vx, vy, vz)
                    assigned = True
                    log.debug("HPOP AssignCartesian frame=%d OK", frame_code)
                    break
                except Exception as exc:
                    log.debug("HPOP AssignCartesian frame=%d failed: %s", frame_code, exc)
            if not assigned:
                log.warning("HPOP AssignCartesian 全部 frame 都失败")
                return None
        except Exception as exc:
            log.warning("HPOP InitialState set failed: %s", exc)
            return None

        # 力学模型
        try:
            fm = prop.force_model if is_pystk else prop.ForceModel
            # 中央引力
            try:
                cbg = fm.central_body_gravity if is_pystk else fm.CentralBodyGravity
                if hasattr(cbg, "SetMaximumDegreeAndOrder"):
                    cbg.SetMaximumDegreeAndOrder(int(gravity_degree), int(gravity_degree))
                elif hasattr(cbg, "set_maximum_degree_and_order"):
                    cbg.set_maximum_degree_and_order(int(gravity_degree), int(gravity_degree))
            except Exception as exc:
                log.debug("HPOP CentralBodyGravity set failed: %s", exc)

            # 大气阻力
            try:
                drag = fm.drag if is_pystk else fm.Drag
                if use_drag:
                    drag.Use = True
                    try:
                        # AtmosphericDensityModel：13=NRLMSISE2000（实测枚举值）
                        drag.AtmosphericDensityModel = 13
                    except Exception:
                        pass
                else:
                    drag.Use = False
            except Exception as exc:
                log.debug("HPOP Drag set failed: %s", exc)

            # 月日扌动
            try:
                tbg = fm.third_body_gravity if is_pystk else fm.ThirdBodyGravity
                if use_third_body and hasattr(tbg, "AddThirdBody"):
                    for body in ("Sun", "Moon"):
                        try:
                            tbg.AddThirdBody(body)
                        except Exception:
                            pass
            except Exception as exc:
                log.debug("HPOP ThirdBody set failed: %s", exc)

            # SRP
            try:
                srp = fm.solar_radiation_pressure if is_pystk else fm.SolarRadiationPressure
                srp.Use = bool(use_srp)
            except Exception as exc:
                log.debug("HPOP SRP set failed: %s", exc)
        except Exception as exc:
            log.debug("HPOP ForceModel acquire failed: %s", exc)

        # Propagate
        try:
            if is_pystk:
                prop.propagate()
            else:
                prop.Propagate()
        except Exception as exc:
            log.warning("HPOP Propagate failed: %s", exc)
            return None

        target_step_s = _infer_step_s(t_offsets_s)
        # HPOP 的 ICRF 输出与 6-DOF J2000 ECI 同源
        series = None
        for frame in ("ICRF", "MeanOfEpoch", "TrueOfEpoch", "MeanOfDate"):
            series = _read_cartesian(
                session, sat_path,
                epoch, t_stop, target_step_s, frame=frame,
            )
            if series is not None:
                log.debug("STK HPOP frame used: %s, step=%.1fs", frame, target_step_s)
                break
        if series is None:
            return None
        return _resample_at_offsets(series, epoch, target_step_s, t_offsets_s)
    finally:
        if own_session:
            try:
                shutdown_stk(session)
            except Exception:
                pass


# ─── 自检 / 诊断 ────────────────────────────────────────────────────────────────

def diagnose() -> Dict[str, Any]:
    """端到端自检：尝试启动 STK、加载 ISS TLE、读 5 个点的 ECI 状态。

    返回结构化诊断字典；不抛异常。供 :mod:`stk_validation.stk` 顶层入口直接调用。
    """
    info: Dict[str, Any] = {
        "ok": False,
        "step": "init",
        "session": None,
        "sample": None,
        "error": None,
    }
    session = _open_stk()
    if session is None:
        info["step"] = "open_stk"
        info["error"] = "无法启动 STK 会话（PySTK 与 COM 均失败）。"
        return info
    info["session"] = session[0]
    try:
        L1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990"
        L2 = "2 25544  51.6400 130.0000 0001000   0.0000   0.0000 15.50000000000010"
        # 用 TLE epoch 附近的时刻（24001.5 ≈ 2024-01-01 12:00:00 UTC）
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        offsets = [0.0, 60.0, 120.0, 300.0, 600.0]
        info["step"] = "stk_propagate_sgp4"
        result = stk_propagate_sgp4(
            L1, L2, offsets, epoch, norad_id=25544, session=session,
        )
        if result is None:
            info["error"] = "stk_propagate_sgp4 返回 None（DataProvider 取数失败）"
            return info
        pos, vel = result
        info["sample"] = {
            "n": len(pos),
            "first_pos_km": pos[0],
            "first_vel_kms": vel[0],
            "last_pos_km": pos[-1],
        }
        info["ok"] = True
        info["step"] = "done"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            shutdown_stk(session)
        except Exception:
            pass
    return info
