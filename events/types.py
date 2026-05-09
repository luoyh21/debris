"""Pydantic / dataclass types for the space-event subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np


class EventType(str, Enum):
    FRAGMENTATION = "FRAGMENTATION"   # explosion, anti-sat, breakup
    COLLISION     = "COLLISION"       # two bodies hit each other
    REENTRY       = "REENTRY"         # uncontrolled re-entry / TIP
    MANEUVER      = "MANEUVER"        # planned in-orbit maneuver
    CDM           = "CDM"             # conjunction data message
    OTHER         = "OTHER"


@dataclass
class SpaceEvent:
    """In-memory representation of a row in ``space_events``."""
    event_type:        EventType
    epoch:             datetime
    name:              str = ""
    description:       str = ""
    parent_norad:      Optional[int] = None
    secondary_norad:   Optional[int] = None
    altitude_km:       Optional[float] = None
    inclination_deg:   Optional[float] = None
    energy_j:          Optional[float] = None
    energy_to_mass:    Optional[float] = None        # J/g  – >40 ⇒ catastrophic
    mass_parent_kg:    Optional[float] = None
    mass_target_kg:    Optional[float] = None
    miss_distance_km:  Optional[float] = None
    probability:       Optional[float] = None
    n_fragments_obs:   Optional[int] = None
    source:            str = "manual"
    source_id:         str = ""
    raw:               Optional[dict] = None         # original payload
    id:                Optional[int]  = None         # populated after persist


@dataclass
class Fragment:
    """A single SBM-generated debris piece."""
    lc_m:        float       # characteristic length [m]
    am_m2_per_kg: float       # area-to-mass ratio [m²/kg]
    mass_kg:     float
    area_m2:     float
    delta_v_kms: np.ndarray   # (3,) ejection ΔV in body-LVLH frame [km/s]
    r_eci_km:    np.ndarray   # (3,) initial position
    v_eci_km_s:  np.ndarray   # (3,) initial velocity (parent + ΔV rotated to ECI)
    is_lethal:   bool = False  # ≥ 1 cm
    is_tracked:  bool = False  # ≥ 10 cm


@dataclass
class BreakupRunResult:
    """Output of :func:`events.nasa_sbm.simulate_breakup`."""
    event:                SpaceEvent
    fragments:            List[Fragment] = field(default_factory=list)
    n_total:              int = 0
    n_tracked_ge_10cm:    int = 0
    n_lethal_ge_1cm:      int = 0
    catastrophic:         bool = False
    notes:                List[str] = field(default_factory=list)
