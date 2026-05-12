"""检测 STK Python 集成在当前主机上的可用性。

策略
----
* **OS**：仅 Windows 支持本地 STK Engine；其它平台直接返回 ``available=False``，
  ``reason='os_unsupported'``，由 Streamlit 侧负责禁用按钮 / 隐藏界面。
* **Pythonic SDK**：优先尝试 Ansys 官方 ``ansys.stk.core``（PySTK，STK 12.1+），
  其次回退到 ``comtypes.client`` / ``win32com.client`` 的 COM 互操作。
* **license / desktop**：模块级别只做 *import 探测*，不真正 ``CreateObject`` —— 实例化
  会拉起完整的 STK 进程，应当由用户主动点击「运行 STK 验证」时再触发。
"""
from __future__ import annotations

import importlib
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class StkAvailability:
    """STK 在本机能否被调用的探测结果。"""

    available: bool                       # True 表示至少有一种 SDK 能 import
    os_name: str                          # platform.system()
    os_supported: bool                    # 当前 OS 是否原生支持 STK Engine
    sdk: Optional[str] = None             # 'ansys-stk-core' / 'comtypes' / 'win32com' / None
    sdk_version: Optional[str] = None
    install_dir: Optional[str] = None     # %STK12_INSTALL_DIR% 等环境变量提示
    reason: Optional[str] = None          # 不可用时给前端的原因
    install_hint: str = ""                # 安装 / 配置提示，便于在 UI 上提示

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_import(modname: str) -> tuple[bool, Optional[str]]:
    """try-import 一个模块；返回 (是否成功, 模块版本)。"""
    try:
        m = importlib.import_module(modname)
    except Exception:
        return False, None
    ver = getattr(m, "__version__", None)
    return True, str(ver) if ver else None


def _stk_install_dir_hint() -> Optional[str]:
    """从常见环境变量 + Program Files 默认路径推测本机 STK 安装目录。"""
    for key in (
        "STK12_INSTALL_DIR", "STK11_INSTALL_DIR", "STK_INSTALL_DIR",
        "STKHOME", "STKHOME12", "STKHOME11",
    ):
        val = os.environ.get(key)
        if val and os.path.isdir(val):
            return val
    # Program Files 默认安装路径
    candidates = []
    for base in (
        r"C:\Program Files\AGI",
        r"C:\Program Files (x86)\AGI",
        r"C:\Program Files\Ansys",
    ):
        if os.path.isdir(base):
            try:
                for sub in os.listdir(base):
                    full = os.path.join(base, sub)
                    if os.path.isdir(full) and sub.lower().startswith("stk"):
                        candidates.append(full)
            except OSError:
                pass
    if candidates:
        # 取版本号最大的
        candidates.sort(reverse=True)
        return candidates[0]
    return None


def _detect_stk_com_progid() -> Optional[str]:
    """通过 win32com / comtypes 查询本机注册表中 STK COM ProgID 是否真的可用。

    只做 ``Dispatch`` 看是否抛 ProgID 解析错误（不真的拉起 STK 完整流程）；
    Dispatch 成功后立刻 ``Quit()`` 释放，对系统几乎无影响。
    """
    try:
        import win32com.client as w  # type: ignore
        creator = w.Dispatch
    except Exception:
        try:
            import comtypes.client  # type: ignore
            creator = comtypes.client.CreateObject
        except Exception:
            return None

    for prog_id in ("STK12.Application", "STK11.Application", "STK.Application"):
        try:
            app = creator(prog_id)
            try:
                app.Visible = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                app.UserControl = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                app.Quit()  # type: ignore[attr-defined]
            except Exception:
                pass
            return prog_id
        except Exception:
            continue
    return None


_INSTALL_HINT = (
    "在 Windows 上：1) 安装 Ansys STK 12.1+ 桌面版（含命令行许可）；"
    "2) `pip install ansys-stk-core`（若已订阅 PyAnsys，仅安装即可）。"
    "若仅想本地试用，可改用 STK Engine for Engineers 免费版本，并将 STK 安装目录写入 "
    "`STK12_INSTALL_DIR` 环境变量。"
)


def detect_stk_availability() -> StkAvailability:
    """探测当前主机能否调用 STK Python 接口（不会拉起 STK 进程）。"""
    os_name = platform.system()
    os_supported = os_name.lower() == "windows"
    install_dir = _stk_install_dir_hint() if os_supported else None

    if not os_supported:
        return StkAvailability(
            available=False,
            os_name=os_name,
            os_supported=False,
            reason="os_unsupported",
            install_hint=(
                "Ansys STK Engine 仅在 Windows 上原生支持，当前操作系统 "
                f"`{os_name}` 不可用。建议在 Windows 主机上运行 STK 交叉验证；"
                "若需在 Linux 上对照，只能通过 PyAnsys 远程连接已部署的 STK "
                "Cloud / 自建 gRPC 服务（不在本系统默认配置内）。"
            ),
        )

    # Windows：依次尝试 PySTK → COM
    ok_pystk, ver_pystk = _safe_import("ansys.stk.core")
    if ok_pystk:
        return StkAvailability(
            available=True,
            os_name=os_name,
            os_supported=True,
            sdk="ansys-stk-core",
            sdk_version=ver_pystk,
            install_dir=install_dir,
        )

    # 关键修复：仅 import comtypes/win32com 不足以确认 STK 可用 —— 必须 Dispatch
    # 一次 STK ProgID 才能确定 STK 桌面版已被注册。否则 sdk='win32com' 但无 STK
    # 安装时整条链路会在 _open_stk 阶段失败。
    ok_comtypes, ver_comtypes = _safe_import("comtypes.client")
    ok_win32, ver_win32 = _safe_import("win32com.client")

    progid = None
    if ok_win32 or ok_comtypes:
        progid = _detect_stk_com_progid()

    if progid is not None:
        sdk_label = "comtypes" if ok_comtypes else "win32com"
        ver = ver_comtypes if ok_comtypes else ver_win32
        return StkAvailability(
            available=True,
            os_name=os_name,
            os_supported=True,
            sdk=f"{sdk_label} → {progid}",
            sdk_version=ver,
            install_dir=install_dir,
        )

    # 有 COM 包但 ProgID Dispatch 失败 → 桌面版未安装 / COM 未注册
    if ok_win32 or ok_comtypes:
        return StkAvailability(
            available=False,
            os_name=os_name,
            os_supported=True,
            sdk=("comtypes" if ok_comtypes else "win32com"),
            sdk_version=(ver_comtypes if ok_comtypes else ver_win32),
            install_dir=install_dir,
            reason="stk_desktop_missing",
            install_hint=(
                "已检测到 COM 互操作库（pywin32/comtypes），但本机注册表中"
                "没有 `STK12.Application` / `STK11.Application` / `STK.Application` "
                "ProgID — 通常意味着 Ansys STK 桌面版未安装、或安装后从未启动过一次"
                "（首次启动会注册 COM 类）。\n\n"
                "解决方案：\n"
                "  1) 启动一次 STK 桌面版 GUI（让其完成 COM 注册）；\n"
                "  2) 或以管理员身份在 `<STK_INSTALL>\\bin` 下执行 "
                "`AgUiApplication.exe /RegServer`；\n"
                "  3) 或改装 PyAnsys：`pip install ansys-stk-core` 走 gRPC Engine。"
            ),
        )

    return StkAvailability(
        available=False,
        os_name=os_name,
        os_supported=True,
        reason="sdk_missing",
        install_dir=install_dir,
        install_hint=_INSTALL_HINT,
    )
