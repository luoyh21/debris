"""RocketPy-aware trajectory simulator.

Tries to use RocketPy for the atmospheric phase (0–120 km) and our own
6-DOF integrator for the orbital phase.  Falls back to the pure 6-DOF
integrator if RocketPy is not installed.

Exposes a single entry point:
    simulate(config) -> SimResult
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np

from .six_dof import (
    LaunchVehicle, RocketStage,
    integrate_trajectory, monte_carlo_covariance,
    TrajectoryPoint, ecef_to_eci, geodetic_to_ecef,
    MU_KM3S2, R_EARTH_KM, G0,
)

log = logging.getLogger(__name__)


# ─── preset launch vehicles ───────────────────────────────────────────────────

def preset_cz5b() -> LaunchVehicle:
    """
    Long March 5B (CZ-5B) — 1.5-stage "direct-to-orbit" model.

    CZ-5B has NO upper stage.  The core (2×YF-77, LH2/LOX) and four strap-on
    boosters (each 2×YF-100K, LOX/kerosene) fire simultaneously at liftoff.
    Boosters separate at T+173 s; core continues alone to MECO at ~T+480 s,
    already at first-cosmic-speed, for a direct LEO insertion.

    Modelled as two sequential phases (matching the simulator's stage paradigm):

    Phase 1 — Core + 4 Boosters combined (T+0 → T+173 s)
        Thrust  ≈ 11.8 MN (core 1.4 MN + 4 boosters 10.4 MN, all vacuum)
        Isp_eff ≈ 316 s  (thrust-weighted average: LOX/kero + LH2/LOX mix)
        Prop    ≈ 657 t  (booster propellant, derived from mdot×173 s)
        Dry     ≈ 40 t   (four booster structures, jettisoned at separation)

    Phase 2 — Core only (T+183 s → T+455 s, ignition set 10 s after sep)
        Thrust  ≈ 1.4 MN (2×YF-77)
        Isp_eff ≈ 388 s  (calibrated: YF-77 vac=438 s, reduced to reproduce
                           observed orbit Rp≈169 km, Ra≈375 km, v_p≈7.86 km/s)
        Prop    ≈ 100 t  (core propellant)
        Dry     ≈ 25 t   (core structure, stays in orbit as debris)

    Liftoff mass ≈ 705+40+100+25+22 = 892 t  (ref: ~849 t, +5 %)
    Reference flight profile:
        T+173 s → alt≈68 km,  v≈2 600 m/s  (booster sep)
        T+285 s → alt≈160 km, v≈4 500 m/s  (fairing sep)
        T+455 s → alt≈169 km, v≈7 860 m/s  (MECO, direct orbit insertion Rp169×Ra375 km)
    """
    stage1 = RocketStage(
        name="CZ-5B_CORE+BOOSTERS",
        mass_prop_kg=705_000,   # mdot=11.8e6/(295*9.81)≈4077 kg/s × 173 s ≈ 705 t
        mass_dry_kg=40_000,     # four booster structures (jettisoned at sep)
        thrust_vac_N=11_800_000,
        isp_vac_s=295,          # effective Isp weighted between sea-level and vacuum
                                # (boosters ignite at surface, Isp increases with altitude)
        burn_time_s=173,        # booster burn time
        cd=0.44, area_m2=30.0,
        ignition_t=0.0,
    )
    stage2 = RocketStage(
        name="CZ-5B_CORE",
        mass_prop_kg=100_000,   # 367 kg/s × 272 s ≈ 100 t
        mass_dry_kg=25_000,     # core body (becomes orbital debris)
        thrust_vac_N=1_400_000, # 2 × YF-77 vacuum
        isp_vac_s=388,          # calibrated effective Isp (YF-77 vac=438 s, reduced for gravity
                                # drag + real-world losses) → orbit: Rp≈169 km, Ra≈370 km, e≈0.015
        burn_time_s=0,          # derived: 100 000/367 ≈ 272 s → cutoff T+455 s
        cd=0.40, area_m2=19.6,
        ignition_t=0.0,         # set to stage1.cutoff_t + 10 s in simulate()
    )
    return LaunchVehicle(
        name="CZ-5B",
        stages=[stage1, stage2],
        payload_kg=22_000,
        pitch_kick_t=12.0,
        fpa_meco1_deg=22.0,     # FPA at booster sep (T+173 s) – off-vertical
        fpa_final_deg=1.0,      # FPA at MECO (T+455 s) – nearly horizontal
        pitch_exp=0.45,         # stage-1 gradual gravity turn
        pitch_exp_s2=0.20,      # stage-2 pitch-over rate (0.20 gives Rp≈169km, Ra≈370km)
    )


def preset_falcon9() -> LaunchVehicle:
    """
    Simplified SpaceX Falcon 9 (Block 5).

    Stage 1: 9 × Merlin 1D, T_vac ≈ 8.23 MN, Isp_vac=311 s, prop≈396 t → burn≈147 s
    Stage 2: 1 × Merlin 1D Vac, T_vac=0.934 MN, Isp=348 s, prop≈108 t → burn≈394 s
    """
    stage1 = RocketStage(
        name="F9_S1",
        mass_prop_kg=395_700,
        mass_dry_kg=22_200,
        thrust_vac_N=8_227_000,
        isp_vac_s=311,
        burn_time_s=0,          # derived ≈ 147 s
        cd=0.40, area_m2=14.5,
        ignition_t=0.0,
    )
    stage2 = RocketStage(
        name="F9_S2",
        mass_prop_kg=107_500,
        mass_dry_kg=4_000,
        thrust_vac_N=934_000,
        isp_vac_s=348,
        burn_time_s=0,          # derived ≈ 394 s
        cd=0.38, area_m2=11.3,
        ignition_t=0.0,         # set post stage-1 sep
    )
    return LaunchVehicle(
        name="Falcon9",
        stages=[stage1, stage2],
        payload_kg=22_800,
        pitch_kick_t=10.0,
        fpa_meco1_deg=20.0,
        fpa_final_deg=2.0,
        pitch_exp=0.55,
    )


def preset_ariane6() -> LaunchVehicle:
    """
    Simplified Ariane 6 (A62 variant: 2 P120C solid boosters + Vulcain 2.1 core).

    Modelling approach
    ------------------
    Real A62 fires P120C + Vulcain simultaneously.  P120s burn for ~130 s at high
    thrust; Vulcain continues to ~490 s.  This two-thrust-level behaviour cannot be
    captured with a single constant-thrust stage.

    Simplified as two sequential stages:
      Stage 1: represents the high-thrust combined phase.
               Prop mass ≈ 400 t, thrust ≈ 9.5 MN (P120 + Vulcain vac combined),
               Isp_eff ≈ 310 s (P120-weighted average), burn ≈ 135 s.
               After burnout the vehicle is at ~120 km, ~3500 m/s.
      Stage 2: represents the Vulcain-only + Vinci combined upper phase.
               Prop ≈ 80 t (Vulcain residual + Vinci), Isp_eff ≈ 430 s,
               thrust ≈ 1.5 MN.  Burn ≈ 290 s to orbital velocity.
    Total delivered ΔV ≈ 10 500 m/s — consistent with A62 LEO performance.
    """
    stage1 = RocketStage(
        name="A6_BOOSTERS+CORE",
        mass_prop_kg=400_000,
        mass_dry_kg=55_000,
        thrust_vac_N=9_500_000,
        isp_vac_s=310,
        burn_time_s=0,          # derived ≈ 135 s
        cd=0.42, area_m2=30.2,
        ignition_t=0.0,
    )
    stage2 = RocketStage(
        name="A6_UPPER",
        mass_prop_kg=80_000,
        mass_dry_kg=8_000,
        thrust_vac_N=1_500_000,
        isp_vac_s=430,
        burn_time_s=0,          # derived ≈ 285 s
        cd=0.38, area_m2=9.6,
        ignition_t=0.0,
    )
    return LaunchVehicle(
        name="Ariane6",
        stages=[stage1, stage2],
        payload_kg=10_350,
        pitch_kick_t=11.0,
        fpa_meco1_deg=20.0,
        fpa_final_deg=2.0,
        pitch_exp=0.58,
    )


PRESETS: Dict[str, LaunchVehicle] = {
    "CZ-5B":   preset_cz5b(),
    "Falcon9": preset_falcon9(),
    "Ariane6": preset_ariane6(),
}


# ─── simulation config ────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """All parameters for one trajectory simulation run."""
    vehicle_name:    str     = "CZ-5B"     # key in PRESETS, or "custom"
    vehicle:         Optional[LaunchVehicle] = None   # if vehicle_name == "custom"

    # Launch site
    launch_lat_deg:  float = 19.61   # Wenchang, China
    launch_lon_deg:  float = 110.95
    launch_alt_km:   float = 0.04    # ~40 m MSL
    launch_az_deg:   float = 90.0    # due East

    # Launch time – defaults to tomorrow 06:00 UTC (within SGP4 segment window)
    launch_utc:      datetime = field(
        default_factory=lambda: (
            datetime.now(timezone.utc).replace(hour=6, minute=0, second=0, microsecond=0)
            + timedelta(days=1)
        )
    )

    # Integration
    t_max_s:         float = 6000.0  # max simulation time
    dt_out_s:        float = 10.0    # output cadence

    # Monte Carlo
    run_mc:          bool  = True
    mc_n_runs:       int   = 50
    mc_dt_s:         float = 30.0    # coarser cadence for MC

    # Orbit detection
    auto_stop_orbit: bool  = False   # stop simulation as soon as stable orbit detected

    # Target orbit
    target_alt_km:   float = 400.0
    target_inc_deg:  float = 41.47   # Tiangong CSS orbit


# ─── simulation result ────────────────────────────────────────────────────────

@dataclass
class SimResult:
    config:       SimConfig
    nominal:      List[TrajectoryPoint]   # full nominal trajectory
    covariances:  Optional[np.ndarray]    # (N, 6, 6) in ECI [km, km/s]
    mc_dt_s:      float

    # Key event times [MET seconds]
    t_meco1:      Optional[float] = None
    t_stage_sep:  Optional[float] = None
    t_meco2:      Optional[float] = None
    t_fairing_sep: Optional[float] = None
    t_payload_sep: Optional[float] = None

    @property
    def t0_j2000_s(self) -> float:
        """J2000 epoch of T0."""
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        return (self.config.launch_utc - j2000).total_seconds()

    def epoch_of(self, t_met_s: float) -> datetime:
        return self.config.launch_utc + timedelta(seconds=t_met_s)


# ─── RocketPy integration (optional) ─────────────────────────────────────────

def _try_rocketpy_flight(cfg: SimConfig, vehicle: LaunchVehicle) -> Optional[List[TrajectoryPoint]]:
    """
    Run RocketPy for atmospheric phase (0 → 120 km).
    Returns list of TrajectoryPoint in ECI, or None if RocketPy unavailable.
    """
    try:
        from rocketpy import Environment, Rocket, Flight  # type: ignore
        from rocketpy.motors import SolidMotor            # type: ignore
    except ImportError:
        return None

    try:
        log.info("RocketPy available – running atmospheric phase…")
        env = Environment(
            latitude=cfg.launch_lat_deg,
            longitude=cfg.launch_lon_deg,
            elevation=int(cfg.launch_alt_km * 1000),
        )
        env.set_atmospheric_model(type="standard_atmosphere")

        stage1 = vehicle.stages[0]
        # Build a synthetic constant thrust curve
        thrust_t = [0, stage1.burn_time_s - 0.01, stage1.burn_time_s]
        thrust_F = [stage1.thrust_vac_N, stage1.thrust_vac_N, 0.0]

        motor = SolidMotor(
            thrust_source=[list(z) for z in zip(thrust_t, thrust_F)],
            dry_mass=stage1.mass_dry_kg,
            dry_inertia=(1e6, 1e6, 1e4),
            nozzle_radius=0.5,
            grain_number=1,
            grain_density=1700,
            grain_outer_radius=1.0,
            grain_initial_inner_radius=0.5,
            grain_initial_height=3.0,
            grains_center_of_mass_position=0.0,
            center_of_dry_mass_position=0.0,
            nozzle_position=0.0,
            burn_time=stage1.burn_time_s,
            throat_radius=0.2,
        )
        rocket = Rocket(
            radius=stage1.area_m2 ** 0.5 / math.pi ** 0.5,
            mass=vehicle.initial_mass_kg - stage1.mass_prop_kg,
            inertia=(1e7, 1e7, 1e5),
            power_off_drag=stage1.cd,
            power_on_drag=stage1.cd,
            center_of_mass_without_motor=20.0,
            coordinate_system_orientation="tail_to_nose",
        )
        rocket.add_motor(motor, position=-15.0)
        rocket.add_tail(top_radius=2.5, bottom_radius=1.5, length=1.2, position=-5.0)
        rocket.add_nose(length=2.5, kind="vonKarman", position=40.0)

        flt = Flight(
            rocket=rocket,
            environment=env,
            inclination=90.0 - cfg.launch_az_deg % 360,
            heading=cfg.launch_az_deg,
            max_time=min(300, stage1.burn_time_s + 60),
            max_time_step=1.0,
            terminate_on_apogee=False,
        )

        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t0_j2k = (cfg.launch_utc - j2000).total_seconds()

        pts: List[TrajectoryPoint] = []
        for t_met in np.arange(0, flt.t_final, cfg.dt_out_s):
            try:
                # RocketPy earth-fixed local coords [m]
                x_e = flt.x(t_met) / 1000.0   # → km
                y_e = flt.y(t_met) / 1000.0
                z_e = flt.z(t_met) / 1000.0
                vx_e = flt.vx(t_met) / 1000.0
                vy_e = flt.vy(t_met) / 1000.0
                vz_e = flt.vz(t_met) / 1000.0

                # Convert launch-ENU offset to ECEF
                from .six_dof import geodetic_to_ecef, ecef_to_eci, ecef_to_geodetic
                site = geodetic_to_ecef(cfg.launch_lat_deg, cfg.launch_lon_deg, cfg.launch_alt_km)
                lat_r, lon_r = math.radians(cfg.launch_lat_deg), math.radians(cfg.launch_lon_deg)
                # ENU → ECEF rotation
                R_enu_ecef = np.array([
                    [-math.sin(lon_r),              math.cos(lon_r),             0.0],
                    [-math.sin(lat_r)*math.cos(lon_r), -math.sin(lat_r)*math.sin(lon_r), math.cos(lat_r)],
                    [ math.cos(lat_r)*math.cos(lon_r),  math.cos(lat_r)*math.sin(lon_r), math.sin(lat_r)],
                ])
                pos_enu = np.array([x_e, y_e, z_e])
                vel_enu = np.array([vx_e, vy_e, vz_e])
                pos_ecef = site + R_enu_ecef.T @ pos_enu
                vel_ecef = R_enu_ecef.T @ vel_enu
                pos_eci, vel_eci = ecef_to_eci(pos_ecef, vel_ecef, t0_j2k + t_met)
                lat, lon, alt = ecef_to_geodetic(pos_ecef)
                pts.append(TrajectoryPoint(
                    t_met_s=float(t_met), pos_ecef=pos_ecef, vel_ecef=vel_ecef,
                    pos_eci=pos_eci, vel_eci=vel_eci,
                    mass_kg=flt.total_mass(t_met),
                    alt_km=float(alt), lat_deg=float(lat), lon_deg=float(lon),
                ))
            except Exception:
                continue
        log.info("RocketPy atmospheric phase: %d points to alt=%.0f km",
                 len(pts), pts[-1].alt_km if pts else 0)
        return pts or None

    except Exception as exc:
        log.warning("RocketPy flight failed (%s) – falling back to 6-DOF integrator", exc)
        return None


# ─── main simulation entry point ─────────────────────────────────────────────

def simulate(cfg: SimConfig) -> SimResult:
    """
    Run trajectory simulation according to cfg.

    1. Select vehicle from presets or custom.
    2. Attempt RocketPy atmospheric phase (optional).
    3. Integrate 6-DOF gravity turn for full trajectory.
    4. Run Monte Carlo if cfg.run_mc.
    5. Detect key event MET times.
    6. Return SimResult.
    """
    # Build a fresh copy of the vehicle so preset objects aren't mutated
    # across repeated simulate() calls (stage ignition_t would otherwise
    # accumulate on the shared PRESETS instance).
    import copy
    vehicle = copy.deepcopy(
        cfg.vehicle if cfg.vehicle_name == "custom" else PRESETS.get(cfg.vehicle_name)
    )
    if vehicle is None:
        raise ValueError(f"Unknown vehicle preset '{cfg.vehicle_name}'. "
                         f"Available: {list(PRESETS)}")

    # Set stage-2 ignition 10 s after stage-1 cutoff (stage separation)
    if len(vehicle.stages) > 1:
        sep_delay = 10.0
        vehicle.stages[1].ignition_t = vehicle.stages[0].cutoff_t + sep_delay

    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t0_j2k = (cfg.launch_utc - j2000).total_seconds()

    log.info("Simulating %s launch from (%.2f°, %.2f°) az=%.1f° …",
             vehicle.name, cfg.launch_lat_deg, cfg.launch_lon_deg, cfg.launch_az_deg)

    # ── nominal trajectory (6-DOF, full duration) ──────────────────────────
    nominal = integrate_trajectory(
        vehicle, cfg.launch_lat_deg, cfg.launch_lon_deg,
        cfg.launch_alt_km, cfg.launch_az_deg,
        t0_j2k, cfg.t_max_s, cfg.dt_out_s,
        auto_stop_orbit=cfg.auto_stop_orbit,
    )
    log.info("Nominal trajectory: %d points, final alt=%.1f km, vel=%.3f km/s",
             len(nominal), nominal[-1].alt_km,
             float(np.linalg.norm(nominal[-1].vel_eci)))

    # ── Monte Carlo covariance ─────────────────────────────────────────────
    covs = None
    if cfg.run_mc:
        log.info("Running Monte Carlo (%d runs, dt=%.0f s)…", cfg.mc_n_runs, cfg.mc_dt_s)
        _, covs = monte_carlo_covariance(
            vehicle, cfg.launch_lat_deg, cfg.launch_lon_deg,
            cfg.launch_alt_km, cfg.launch_az_deg,
            t0_j2k, cfg.t_max_s, cfg.mc_dt_s,
            n_runs=cfg.mc_n_runs,
        )

    # ── detect key events ─────────────────────────────────────────────────
    def _meco_time(stage: RocketStage) -> Optional[float]:
        return stage.cutoff_t if stage.cutoff_t < cfg.t_max_s else None

    s1 = vehicle.stages[0]
    s2 = vehicle.stages[1] if len(vehicle.stages) > 1 else None
    t_meco1     = _meco_time(s1)
    t_stage_sep = (t_meco1 + 5.0) if t_meco1 else None
    t_meco2     = _meco_time(s2) if s2 else None
    t_fair_sep  = 165.0 if 165.0 < cfg.t_max_s else None   # ~T+165 s typical
    t_pay_sep   = (t_meco2 + 30.0) if t_meco2 else None

    return SimResult(
        config=cfg,
        nominal=nominal,
        covariances=covs,
        mc_dt_s=cfg.mc_dt_s,
        t_meco1=t_meco1,
        t_stage_sep=t_stage_sep,
        t_meco2=t_meco2,
        t_fairing_sep=t_fair_sep,
        t_payload_sep=t_pay_sep,
    )
