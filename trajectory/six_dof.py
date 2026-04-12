"""6-DOF rocket trajectory integrator (gravity turn, ECEF frame).

State vector: [x, y, z, vx, vy, vz, mass]   (km, km/s, kg) in ECEF
Includes: J2 gravity, atmospheric drag, thrust (gravity-turn pitch program),
          Coriolis + centrifugal acceleration (rotating frame).

Coordinate conventions
----------------------
  ECEF: Earth-Centred Earth-Fixed, Z toward geographic North Pole
  ECI J2000: non-rotating, output via ecef_to_eci()
  Geodetic altitude: WGS-84 ellipsoid
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# ─── physical constants ────────────────────────────────────────────────────────
MU_KM3S2   = 398600.4418       # km³/s²
R_EARTH_KM = 6378.137          # km (WGS-84 equatorial radius)
F_EARTH    = 1 / 298.257223563 # WGS-84 flattening
B_EARTH    = R_EARTH_KM * (1 - F_EARTH)   # km, polar radius
J2         = 1.08263e-3
OMEGA_E    = 7.2921150e-5      # rad/s  (Earth rotation rate)
G0         = 9.80665           # m/s²


# ─── atmosphere model (USSA-1976, exponential segments) ────────────────────────
_ATM_TABLE = [
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

def atmo_density(alt_km: float) -> float:
    """Return air density [kg/m³] at geodetic altitude [km]."""
    if alt_km > 700:
        return 0.0
    for i in range(len(_ATM_TABLE) - 1, -1, -1):
        h_b, rho_b, H = _ATM_TABLE[i]
        if alt_km >= h_b:
            return rho_b * math.exp(-(alt_km - h_b) / H)
    return _ATM_TABLE[0][1]


# ─── coordinate conversions ────────────────────────────────────────────────────

def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """Return (lat_deg, lon_deg, alt_km) from ECEF position [km]."""
    x, y, z = r_ecef
    lon = math.degrees(math.atan2(y, x))
    p = math.hypot(x, y)
    # Bowring iterative
    lat = math.atan2(z, p * (1 - F_EARTH))
    for _ in range(6):
        N = R_EARTH_KM / math.sqrt(1 - (2*F_EARTH - F_EARTH**2)*math.sin(lat)**2)
        lat = math.atan2(z + (2*F_EARTH - F_EARTH**2)*N*math.sin(lat), p)
    N = R_EARTH_KM / math.sqrt(1 - (2*F_EARTH - F_EARTH**2)*math.sin(lat)**2)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-9 else abs(z)/math.sin(lat) - N*(1 - F_EARTH**2)
    return math.degrees(lat), lon, alt


def ecef_to_eci(pos_ecef: np.ndarray, vel_ecef: np.ndarray,
                t_utc_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate ECEF → ECI J2000 (ignores precession/nutation – adequate for launch ops).
    t_utc_s : seconds since J2000 epoch (2000-01-01 12:00:00 UTC)
    """
    theta = OMEGA_E * t_utc_s   # Earth rotation angle (rad)
    c, s  = math.cos(theta), math.sin(theta)
    R = np.array([[ c, s, 0],
                  [-s, c, 0],
                  [ 0, 0, 1]])
    omega_vec = np.array([0.0, 0.0, OMEGA_E])
    pos_eci = R @ pos_ecef
    vel_eci = R @ vel_ecef + np.cross(omega_vec, pos_eci)
    return pos_eci, vel_eci


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = 2*F_EARTH - F_EARTH**2
    N  = R_EARTH_KM / math.sqrt(1 - e2*math.sin(lat)**2)
    x  = (N + alt_km) * math.cos(lat) * math.cos(lon)
    y  = (N + alt_km) * math.cos(lat) * math.sin(lon)
    z  = (N*(1 - e2) + alt_km) * math.sin(lat)
    return np.array([x, y, z])


# ─── data structures ───────────────────────────────────────────────────────────

@dataclass
class RocketStage:
    """One propulsive stage."""
    name:         str
    mass_prop_kg: float          # propellant mass
    mass_dry_kg:  float          # dry mass (structure + engine)
    thrust_vac_N: float          # vacuum thrust [N]
    isp_vac_s:    float          # vacuum Isp [s]
    burn_time_s:  float          # derived from prop/mdot
    cd:           float  = 0.50  # drag coefficient
    area_m2:      float  = 20.0  # reference area [m²]
    ignition_t:   float  = 0.0   # MET of ignition [s]

    def __post_init__(self):
        self.mdot_kgs = self.thrust_vac_N / (self.isp_vac_s * G0)
        # Derive burn_time from propellant mass if provided value is inconsistent.
        # Physical burn time = mass_prop / mdot
        phys_burn = self.mass_prop_kg / self.mdot_kgs
        if self.burn_time_s <= 0 or self.burn_time_s > phys_burn * 1.05:
            self.burn_time_s = phys_burn

    @property
    def cutoff_t(self) -> float:
        return self.ignition_t + self.burn_time_s

    def is_burning(self, t: float) -> bool:
        return self.ignition_t <= t < self.cutoff_t


@dataclass
class LaunchVehicle:
    """Multi-stage launch vehicle."""
    name:          str
    stages:        List[RocketStage]
    payload_kg:    float          = 10_000.0
    fairing_cd:    float          = 0.50
    fairing_area:  float          = 20.0
    pitch_kick_t:  float          = 10.0    # s – begin gravity turn
    pitch_kick_deg: float         = 4.0     # kept for legacy; not used in pitch program
    # Stage-1 programmed pitch profile: FPA(horizontal) goes from 90° to
    # fpa_meco1_deg over the stage-1 burn, following a power-law curve.
    fpa_meco1_deg: float          = 12.0    # flight-path angle at MECO1 [deg from horiz]
    fpa_final_deg: float          =  3.0    # flight-path angle at final-stage cutoff [deg]
    pitch_exp:     float          = 0.55    # power-law exponent for stage-1 pitch-over
    pitch_exp_s2:  float          = 0.0     # stage-2+ exponent (0 = same as pitch_exp)

    @property
    def initial_mass_kg(self) -> float:
        return sum(s.mass_prop_kg + s.mass_dry_kg for s in self.stages) + self.payload_kg

    def current_thrust_N(self, t: float) -> float:
        for s in self.stages:
            if s.is_burning(t):
                return s.thrust_vac_N
        return 0.0

    def current_mdot(self, t: float) -> float:
        for s in self.stages:
            if s.is_burning(t):
                return s.mdot_kgs
        return 0.0

    def current_cd_area(self, t: float) -> Tuple[float, float]:
        # Drop fairings above 120 km (typically)
        # Use first burning stage aero params
        for s in self.stages:
            if s.is_burning(t):
                return s.cd, s.area_m2
        return self.fairing_cd, self.fairing_area


# ─── ODE right-hand side ───────────────────────────────────────────────────────

class _IntegratorState:
    """Mutable state tracked across ODE calls."""
    def __init__(self, vehicle: LaunchVehicle, launch_lat_deg: float, launch_az_deg: float):
        self.vehicle = vehicle
        self.pitch_kick_applied = False
        self.launch_lat = math.radians(launch_lat_deg)
        self.launch_az  = math.radians(launch_az_deg)
        self._last_v_vec: Optional[np.ndarray] = None


def _gravity_j2(r: np.ndarray) -> np.ndarray:
    """J2-perturbed gravitational acceleration [km/s²] in ECEF."""
    x, y, z = r
    r_norm = math.sqrt(x**2 + y**2 + z**2)
    r2 = r_norm**2
    mu_r3 = MU_KM3S2 / r_norm**3
    j2_fac = 1.5 * J2 * (R_EARTH_KM / r_norm)**2
    z_r2   = (z / r_norm)**2

    f_grav_x = -mu_r3 * x * (1 - j2_fac * (5*z_r2 - 1))
    f_grav_y = -mu_r3 * y * (1 - j2_fac * (5*z_r2 - 1))
    f_grav_z = -mu_r3 * z * (1 - j2_fac * (5*z_r2 - 3))
    return np.array([f_grav_x, f_grav_y, f_grav_z])


def _build_ode(vehicle: LaunchVehicle,
               launch_lat_deg: float,
               launch_lon_deg: float,
               launch_az_deg:  float,
               sigma_thrust:   float = 0.0):
    """Return the ODE function for scipy.integrate.solve_ivp.

    Pitch control strategy
    ----------------------
    Stage 1 (atmospheric burn):
        Programmed pitch profile.  Flight-path angle (from horizontal)
        decreases smoothly from 90° at pitch-kick time to
        ``vehicle.fpa_meco1_deg`` at MECO1 following a power-law curve.
        This matches typical launch-vehicle guidance during the atmospheric
        ascent phase and ensures the rocket reaches sufficient altitude
        before tilting horizontal.

    Stage 2+ (vacuum burn):
        Pure gravity turn — thrust along the current velocity vector.
        By MECO1 the velocity vector is already well off-vertical, so the
        gravity turn converges to the orbital horizontal without over-tilting.
    """

    omega = np.array([0.0, 0.0, OMEGA_E])
    thrust_factor = 1.0 + sigma_thrust

    # ── Launch-site ENU vectors in ECEF ─────────────────────────────────────
    lat = math.radians(launch_lat_deg)
    lon = math.radians(launch_lon_deg)
    az  = math.radians(launch_az_deg)

    e_east  = np.array([-math.sin(lon),
                         math.cos(lon),
                         0.0])
    e_north = np.array([-math.sin(lat) * math.cos(lon),
                        -math.sin(lat) * math.sin(lon),
                         math.cos(lat)])
    # Azimuth direction in horizontal plane at launch site
    horiz_launch = math.sin(az) * e_east + math.cos(az) * e_north
    horiz_launch /= math.sqrt(float(np.dot(horiz_launch, horiz_launch)))

    # ── Pitch program control points ────────────────────────────────────────
    # We define FPA (from horizontal) at three MET times:
    #   t_kick  → 90°  (start pitching over)
    #   t_meco1 → fpa_meco1_deg  (FPA at 1st stage cutoff)
    #   t_meco_last → fpa_final_deg  (FPA at last engine cutoff, ~orbital horizontal)
    # Between stages (coast) we interpolate so that stage-2 ignition picks up
    # smoothly.  The profile is only used while thrust > 0 but defined for all t.
    t_kick       = vehicle.pitch_kick_t
    t_meco1      = vehicle.stages[0].cutoff_t
    t_meco_last  = vehicle.stages[-1].cutoff_t
    fpa_meco1    = getattr(vehicle, "fpa_meco1_deg", 12.0)
    fpa_final    = getattr(vehicle, "fpa_final_deg",  3.0)
    pitch_exp    = getattr(vehicle, "pitch_exp",       0.55)
    _pexp_s2_raw = getattr(vehicle, "pitch_exp_s2",    0.0)
    pitch_exp_s2 = _pexp_s2_raw if _pexp_s2_raw > 0 else pitch_exp

    def _programmed_fpa(t: float) -> float:
        """Flight-path angle [deg from horizontal] at time t.

        Stage-1 (t_kick → t_meco1): power-law with pitch_exp
        Stage-2+ (t_meco1 → t_meco_last): power-law with pitch_exp_s2
        A different (often smaller) exponent for stage 2 lets the upper stage
        pitch over more aggressively immediately after stage-1 separation —
        matching vehicles like CZ-5B whose core phase is nearly horizontal
        within ~100 s of booster sep.
        """
        if t <= t_kick:
            return 90.0
        if t <= t_meco1:
            p = (t - t_kick) / max(t_meco1 - t_kick, 1.0)
            return 90.0 - (90.0 - fpa_meco1) * (p ** pitch_exp)
        if t <= t_meco_last:
            p = (t - t_meco1) / max(t_meco_last - t_meco1, 1.0)
            return fpa_meco1 - (fpa_meco1 - fpa_final) * (p ** pitch_exp_s2)
        return fpa_final

    def ode(t: float, y: np.ndarray) -> np.ndarray:
        x, yx_, z, vx, vy, vz, mass = y
        r = np.array([x, yx_, z])
        v = np.array([vx, vy, vz])
        r_norm = math.sqrt(x**2 + yx_**2 + z**2)
        v_norm = math.hypot(vx, math.hypot(vy, vz))

        _, _, alt_km = ecef_to_geodetic(r)

        # Gravity (J2)
        a_grav = _gravity_j2(r)

        # Coriolis + centrifugal (ECEF rotating frame)
        a_cor  = -2.0 * np.cross(omega, v)
        a_cent = -np.cross(omega, np.cross(omega, r))

        # Aerodynamic drag
        cd, area_m2 = vehicle.current_cd_area(t)
        rho      = atmo_density(alt_km)
        v_ms     = v_norm * 1000.0
        q        = 0.5 * rho * v_ms ** 2
        f_drag_N = cd * area_m2 * q
        a_drag   = np.zeros(3)
        if v_norm > 1e-6 and mass > 0:
            a_drag = -(f_drag_N / mass) * (v / v_norm) / 1000.0

        # Thrust + pitch program
        T_N  = vehicle.current_thrust_N(t) * thrust_factor
        mdot = vehicle.current_mdot(t) * thrust_factor
        a_thrust = np.zeros(3)
        if T_N > 0 and mass > 0:
            u_up = r / r_norm     # local vertical (radial outward)

            # ── Pitch program ─────────────────────────────────────────────────
            # If pitch_exp_s2 > 0 (explicitly set): use pure time-based FPA for
            # ALL stages.  This is appropriate for vehicles whose stage-2 T/W
            # is high enough to maintain altitude even at shallow FPA (e.g. CZ-5B).
            #
            # If pitch_exp_s2 == 0 (default): use time-based for stage 1, and
            # velocity-fraction for stage 2+.  Velocity-fraction ensures the
            # rocket holds sufficient FPA to maintain altitude while building
            # orbital speed — critical for low-T/W upper stages (Falcon9 ~0.71).
            is_stage1 = t <= t_meco1
            if is_stage1 or _pexp_s2_raw > 0:
                # Time-based programmed pitch (all stages or stage-1 only)
                fpa_deg: float = _programmed_fpa(t)
            else:
                # Velocity-fraction pitch for stage 2+
                v_circ = math.sqrt(MU_KM3S2 / r_norm)
                v_frac = min(1.0, v_norm / v_circ)
                if v_frac < 0.85:
                    fpa_deg = fpa_meco1
                elif v_frac < 0.97:
                    t_frac  = (v_frac - 0.85) / 0.12
                    fpa_deg = fpa_meco1 - (fpa_meco1 - fpa_final) * (t_frac ** 0.7)
                else:
                    fpa_deg = fpa_final

            fpa_rad = math.radians(fpa_deg)

            # Horizontal direction: velocity projected onto local horizontal plane.
            # Fall back to launch azimuth direction if velocity is still radial.
            v_horiz      = v - float(np.dot(v, u_up)) * u_up
            v_horiz_norm = float(np.linalg.norm(v_horiz))
            u_horiz      = (v_horiz / v_horiz_norm
                            if v_horiz_norm > 0.02
                            else horiz_launch)

            thrust_dir = math.sin(fpa_rad) * u_up + math.cos(fpa_rad) * u_horiz
            a_thrust   = (T_N / mass / 1000.0) * thrust_dir

        a_total = a_grav + a_cor + a_cent + a_drag + a_thrust
        return [vx, vy, vz, a_total[0], a_total[1], a_total[2], -mdot]

    return ode


# ─── public API ────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    t_met_s:   float      # mission elapsed time [s]
    pos_ecef:  np.ndarray # [km] (3,)
    vel_ecef:  np.ndarray # [km/s] (3,)
    pos_eci:   np.ndarray # [km] J2000
    vel_eci:   np.ndarray # [km/s] J2000
    mass_kg:   float
    alt_km:    float
    lat_deg:   float
    lon_deg:   float
    cov_6x6:   Optional[np.ndarray] = None   # position+velocity covariance in ECI


def _make_orbit_insertion_event(vehicle: "LaunchVehicle"):
    """Return a terminal event that fires when a stable orbit is achieved.

    Trigger condition (all must hold simultaneously):
      - All engines off (no thrust)
      - Altitude > 200 km
      - Orbital energy < 0 (gravitationally bound)

    Event function = orbital_energy.
    Positive during burn / suborbital; crosses zero at energy → negative.
    direction = -1 fires on that downward crossing.
    """
    def orbital_insertion(t, y, *_):
        if vehicle.current_thrust_N(t) > 0:
            return 1.0           # still burning → stay positive

        pos = np.array(y[:3])
        vel = np.array(y[3:6])
        _, _, alt = ecef_to_geodetic(pos)
        if alt < 150.0:
            return 1.0           # too low – wait for rocket to clear dense atmosphere

        r      = float(np.linalg.norm(pos))
        energy = float(np.dot(vel, vel)) / 2.0 - MU_KM3S2 / r
        return energy            # zero at orbital boundary; negative when bound

    orbital_insertion.terminal  = True   # type: ignore[attr-defined]
    orbital_insertion.direction = -1     # type: ignore[attr-defined]
    return orbital_insertion


def integrate_trajectory(
    vehicle:          LaunchVehicle,
    launch_lat_deg:   float,
    launch_lon_deg:   float,
    launch_alt_km:    float,
    launch_az_deg:    float,
    t0_j2000_s:       float,          # seconds since J2000 epoch
    t_max_s:          float  = 6000,  # integration end [s MET]
    dt_out_s:         float  = 10.0,  # output cadence
    sigma_thrust:     float  = 0.0,   # thrust noise for Monte Carlo
    sigma_mass:       float  = 0.0,   # mass noise
    events:           list   = (),
    auto_stop_orbit:  bool   = False, # stop as soon as stable orbit detected
) -> List[TrajectoryPoint]:
    """Integrate 6-DOF gravity-turn trajectory with proper multi-stage separation.

    Each stage's dry mass is jettisoned at cutoff+2 s (stage separation).
    Integration is restarted at each separation with the updated mass state so
    the ODE mass variable stays consistent with the physical vehicle.
    """
    pos0_ecef = geodetic_to_ecef(launch_lat_deg, launch_lon_deg, launch_alt_km)
    vel0_ecef = np.zeros(3)
    total_mass = vehicle.initial_mass_kg * (1.0 + sigma_mass)

    y_cur = np.array(list(pos0_ecef) + list(vel0_ecef) + [total_mass])

    ode_fn = _build_ode(vehicle, launch_lat_deg, launch_lon_deg,
                        launch_az_deg, sigma_thrust)

    def ground_impact(t, y, *_):
        _, _, alt = ecef_to_geodetic(np.array(y[:3]))
        return alt - 0.01
    ground_impact.terminal  = True
    ground_impact.direction = -1

    all_events = [ground_impact] + list(events)
    if auto_stop_orbit:
        all_events.append(_make_orbit_insertion_event(vehicle))

    # Stage-separation schedule: (t_sep, dry_mass_to_drop_kg)
    # Drop each stage's dry structure 2 s after its propellant is exhausted.
    sep_schedule: List[tuple] = []
    for stage in vehicle.stages[:-1]:          # all stages except the last
        sep_schedule.append((stage.cutoff_t + 2.0, stage.mass_dry_kg))
    sep_schedule.sort()

    def _collect(sol) -> List[TrajectoryPoint]:
        pts: List[TrajectoryPoint] = []
        for i, t_met in enumerate(sol.t):
            state = sol.y[:, i]
            pos_e = state[0:3]
            vel_e = state[3:6]
            mass  = state[6]
            lat, lon, alt = ecef_to_geodetic(pos_e)
            t_j2000 = t0_j2000_s + t_met
            pos_i, vel_i = ecef_to_eci(pos_e, vel_e, t_j2000)
            pts.append(TrajectoryPoint(
                t_met_s=float(t_met),
                pos_ecef=pos_e.copy(), vel_ecef=vel_e.copy(),
                pos_eci=pos_i.copy(),  vel_eci=vel_i.copy(),
                mass_kg=float(mass),
                alt_km=float(alt), lat_deg=float(lat), lon_deg=float(lon),
            ))
        return pts

    points: List[TrajectoryPoint] = []
    t_cur = 0.0

    # ── Segmented integration with mass drops at stage separations ────────────
    for t_sep, mass_drop in sep_schedule:
        if t_sep >= t_max_s:
            break
        t_eval_seg = np.arange(t_cur, t_sep + 1e-9, dt_out_s)
        t_eval_seg = t_eval_seg[t_eval_seg <= t_sep + 1e-9]
        if len(t_eval_seg) < 2:
            t_eval_seg = np.array([t_cur, t_sep])

        sol = solve_ivp(
            ode_fn, (t_cur, t_sep), y_cur,
            method="RK45", t_eval=t_eval_seg,
            events=all_events, rtol=1e-8, atol=1e-10, dense_output=False,
        )
        points.extend(_collect(sol))

        if sol.status == 1:          # terminal event (ground impact / orbit insertion)
            return points

        # Apply stage separation: drop spent dry structure
        y_cur = sol.y[:, -1].copy()
        min_mass = sum(s.mass_dry_kg for s in vehicle.stages[-1:]) + vehicle.payload_kg
        y_cur[6] = max(float(y_cur[6]) - mass_drop, min_mass)
        t_cur    = float(sol.t[-1])

    # ── Final segment (last stage burn + coast to t_max or orbit) ─────────────
    t_eval_fin = np.arange(t_cur, t_max_s + dt_out_s * 0.5, dt_out_s)
    t_eval_fin = t_eval_fin[t_eval_fin <= t_max_s + 1e-9]
    if len(t_eval_fin) == 0:
        t_eval_fin = np.array([t_cur, t_max_s])

    sol = solve_ivp(
        ode_fn, (t_cur, t_max_s), y_cur,
        method="RK45", t_eval=t_eval_fin,
        events=all_events, rtol=1e-8, atol=1e-10, dense_output=False,
    )
    points.extend(_collect(sol))
    return points


def monte_carlo_covariance(
    vehicle:        LaunchVehicle,
    launch_lat_deg: float,
    launch_lon_deg: float,
    launch_alt_km:  float,
    launch_az_deg:  float,
    t0_j2000_s:     float,
    t_max_s:        float  = 6000,
    dt_out_s:       float  = 30.0,
    n_runs:         int    = 50,
    sigma_thrust:   float  = 0.02,   # ±2 % thrust 1-sigma
    sigma_mass:     float  = 0.005,  # ±0.5 % mass 1-sigma
) -> Tuple[List[TrajectoryPoint], np.ndarray]:
    """
    Run nominal + N Monte Carlo dispersions.
    Returns (nominal_trajectory, covariance_array)
    where covariance_array[i] is 6×6 position+velocity covariance
    at the i-th time step of the nominal trajectory (ECI frame).
    """
    nom = integrate_trajectory(
        vehicle, launch_lat_deg, launch_lon_deg, launch_alt_km, launch_az_deg,
        t0_j2000_s, t_max_s, dt_out_s,
    )
    n_steps = len(nom)

    # Collect perturbed state vectors
    dispersions: List[List[np.ndarray]] = [[] for _ in range(n_steps)]

    rng = np.random.default_rng(42)
    for _ in range(n_runs):
        st = rng.standard_normal()
        sm = rng.standard_normal()
        pert = integrate_trajectory(
            vehicle, launch_lat_deg, launch_lon_deg, launch_alt_km, launch_az_deg,
            t0_j2000_s, t_max_s, dt_out_s,
            sigma_thrust=st * sigma_thrust,
            sigma_mass=sm * sigma_mass,
        )
        for i in range(min(n_steps, len(pert))):
            sv = np.concatenate([pert[i].pos_eci, pert[i].vel_eci])
            dispersions[i].append(sv)

    covs = np.zeros((n_steps, 6, 6))
    for i, pts in enumerate(dispersions):
        if len(pts) < 4:
            covs[i] = np.eye(6) * (1e-3 if i == 0 else 1.0)
            continue
        mat  = np.array(pts)                  # (N, 6)
        mean = mat.mean(axis=0)
        diff = mat - mean                     # (N, 6)
        covs[i] = (diff.T @ diff) / (len(pts) - 1)
        # Enforce positive-definiteness
        covs[i] = 0.5 * (covs[i] + covs[i].T) + np.eye(6) * 1e-12

    # Attach covariances to nominal points
    for i, pt in enumerate(nom):
        pt.cov_6x6 = covs[i]

    return nom, covs
