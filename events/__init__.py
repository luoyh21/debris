"""Space-event subsystem: fetching, modelling, CCSDS NDM I/O.

Sub-modules
-----------
* :mod:`events.types`         — Pydantic dataclasses (``SpaceEvent``, fragments).
* :mod:`events.nasa_sbm`      — NASA Standard Breakup Model (Johnson 2001).
* :mod:`events.ccsds`         — KVN parsers + writers for CDM (508.0),
                                 OPM/OEM/OCM (502.0) and RDM (508.1).
* :mod:`events.ingest`        — DISCOS fragmentation + Space-Track CDM/TIP fetchers.
* :mod:`events.crud`          — DB CRUD helpers around the ``space_events`` table.
"""

from .types import (
    SpaceEvent,
    EventType,
    Fragment,
    BreakupRunResult,
)
from .nasa_sbm import simulate_breakup
from .ccsds import (
    parse_ccsds_message,
    write_cdm,
    write_opm,
    write_rdm,
    write_ocm,
    detect_format,
)

__all__ = [
    "SpaceEvent",
    "EventType",
    "Fragment",
    "BreakupRunResult",
    "simulate_breakup",
    "parse_ccsds_message",
    "write_cdm",
    "write_opm",
    "write_rdm",
    "write_ocm",
    "detect_format",
]
