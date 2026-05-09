"""Collision avoidance maneuver design.

Sub-modules
-----------
* :mod:`avoidance.bplane`           — analytic impulsive ΔV in the B-plane
                                       (Lagrange multipliers, eigen-direction).
* :mod:`avoidance.low_thrust`       — continuous low-thrust equivalent
                                       (linear SCP / single-shot SOCP-lite).
* :mod:`avoidance.ascent_corridor`  — ascent-phase azimuth / pitch-rate trim
                                       inside a spatio-temporal drivable
                                       corridor (MPC-style first iteration).

Common output type
------------------
:class:`AvoidanceSolution` carries: maneuver vector / profile, predicted
post-maneuver miss distance, post-maneuver collision probability, ΔV cost,
fuel mass cost (if Isp / m provided), and a list of post-maneuver
``TrajectoryPoint`` samples for visualisation.
"""

from .bplane import (
    BPlaneFrame,
    bplane_from_states,
    optimal_impulsive_dv,
)
from .low_thrust import design_low_thrust_burn
from .ascent_corridor import design_ascent_correction
from .common import (
    AvoidanceSolution,
    ConjunctionInputs,
    inputs_from_event,
)

__all__ = [
    "BPlaneFrame",
    "bplane_from_states",
    "optimal_impulsive_dv",
    "design_low_thrust_burn",
    "design_ascent_correction",
    "AvoidanceSolution",
    "ConjunctionInputs",
    "inputs_from_event",
]
