"""STK 跨算法验证模块。

将本系统的轨道传播 / 6-DOF 仿真结果与 Ansys STK（HPOP / SGP4）做位置-速度
RMS 与 RIC（Radial / In-track / Cross-track）误差对照。

设计目标
--------
* Windows 上检测到 ``ansys.stk.core``（PySTK）或本地 STK Engine（COM）即启用 STK 真值；
* 否则自动降级为 ``高保真参考传播器`` —— 使用独立 ``sgp4`` 库 + 直接 Vallado 检验向量
  作为参照，仍能给出可信的「自洽性」交叉验证结果。
* 模块入口（`runner.py`）统一返回 :class:`ValidationReport`；UI / API / 文档页共用同一份。
"""
from .availability import (  # noqa: F401
    StkAvailability,
    detect_stk_availability,
)
from .comparison import (  # noqa: F401
    PerSampleError,
    ValidationReport,
    compute_rms_errors,
)
from .runner import (  # noqa: F401
    run_sgp4_validation,
    run_six_dof_validation,
)
from .report import (  # noqa: F401
    DEFAULT_REPORT_PATH,
    load_latest_report,
    save_report,
)
