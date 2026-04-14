#!/usr/bin/env python3
"""Top-level CLI entry point for space_debris project.

Commands
--------
  init-db      Create PostGIS schema
  ingest       Fetch from Space-Track and propagate trajectories
  simulate     Run 6-DOF rocket trajectory simulation + Monte Carlo
  lcola        Run LCOLA fly-through screening for a mission
  app          Launch Streamlit dashboard
"""
import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def cmd_init_db(args):
    from database.db import init_db
    init_db()
    print("Database initialized.")


def cmd_ingest(args):
    from ingestion.ingest_gp import ingest
    ingest(
        limit=args.limit,
        object_types=args.types if args.types else ["DEBRIS", "PAYLOAD", "ROCKET BODY"],
        propagate=not args.no_propagate,
        horizon_days=args.horizon_days,
        seg_minutes=args.seg_minutes,
    )


def cmd_simulate(args):
    """Run 6-DOF trajectory simulation and write CCSDS OEM file."""
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from datetime import datetime, timezone
    from trajectory.rocketpy_sim import SimConfig, simulate, PRESETS
    from trajectory.launch_phases import detect_phases
    from trajectory.oem_io import sim_result_to_oem_segments, write_oem

    launch_utc = datetime.fromisoformat(args.launch_utc).replace(tzinfo=timezone.utc)

    if args.vehicle not in PRESETS:
        print(f"Unknown vehicle '{args.vehicle}'.  Available: {list(PRESETS)}")
        sys.exit(1)

    cfg = SimConfig(
        vehicle_name=args.vehicle,
        launch_lat_deg=args.lat,
        launch_lon_deg=args.lon,
        launch_alt_km=args.alt_km,
        launch_az_deg=args.azimuth,
        launch_utc=launch_utc,
        t_max_s=args.t_max,
        dt_out_s=args.dt,
        run_mc=not args.no_mc,
        mc_n_runs=args.mc_runs,
    )

    result = simulate(cfg)

    phases = detect_phases(
        result.nominal,
        t_meco1=result.t_meco1,
        t_stage_sep=result.t_stage_sep,
        t_meco2=result.t_meco2,
        t_payload_sep=result.t_payload_sep,
    )

    print("\n── Phase Summary ─────────────────────────────────────────────────")
    for ph in phases:
        a0, a1 = ph.alt_range_km
        print(f"  {ph.name:<20s}  MET {ph.t_start_met:>6.0f}–{ph.t_end_met:<7.0f}s"
              f"  alt {a0:>5.0f}–{a1:<5.0f} km"
              f"  pts={len(ph.points)}")

    oem_segs = sim_result_to_oem_segments(result, phases, mission_id=args.mission_id)
    write_oem(args.output, oem_segs)
    print(f"\nOEM written → {args.output}  ({len(oem_segs)} segments)")

    # Print final state
    last = result.nominal[-1]
    import numpy as np
    print(f"\nFinal state (MET {last.t_met_s:.0f} s):")
    print(f"  alt = {last.alt_km:.1f} km")
    print(f"  |v| = {np.linalg.norm(last.vel_eci):.4f} km/s")
    print(f"  lat = {last.lat_deg:.2f}°,  lon = {last.lon_deg:.2f}°")


def cmd_lcola(args):
    """Run LCOLA fly-through screening."""
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from datetime import datetime, timezone, timedelta
    from trajectory.oem_io import read_oem
    from trajectory.launch_phases import detect_phases, LaunchPhase
    from trajectory.six_dof import TrajectoryPoint
    from lcola.fly_through import FlyThroughScreener, PC_UNCREWED, PC_CREWED
    import numpy as np

    # Load OEM file
    segs = read_oem(args.oem)
    print(f"Loaded {len(segs)} OEM segments from {args.oem}")

    # Reconstruct LaunchPhase objects from OEM segments
    phases = []
    for seg in segs:
        pts = []
        for st in seg.states:
            t_met = (st.epoch - datetime.fromisoformat(args.launch_utc.replace('Z','+00:00'))).total_seconds()
            from trajectory.six_dof import ecef_to_geodetic
            import math
            r = float(np.linalg.norm(st.pos_km))
            R_EARTH = 6378.137
            alt = r - R_EARTH
            pts.append(TrajectoryPoint(
                t_met_s=t_met,
                pos_ecef=st.pos_km.copy(),
                vel_ecef=st.vel_kms.copy(),
                pos_eci=st.pos_km.copy(),
                vel_eci=st.vel_kms.copy(),
                mass_kg=0.0,
                alt_km=alt,
                lat_deg=0.0,
                lon_deg=0.0,
                cov_6x6=st.cov_6x6,
            ))
        phase_name = seg.phase_comment.replace("Launch phase: ", "") if seg.phase_comment else seg.object_name
        if pts:
            phases.append(LaunchPhase(
                name=phase_name,
                t_start_met=pts[0].t_met_s,
                t_end_met=pts[-1].t_met_s,
                points=pts,
            ))

    nominal_launch = datetime.fromisoformat(args.launch_utc.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
    window_open    = nominal_launch + timedelta(minutes=args.window_minus)
    window_close   = nominal_launch + timedelta(minutes=args.window_plus)
    pc_thresh      = PC_CREWED if args.crewed else PC_UNCREWED

    screener = FlyThroughScreener(
        oem_segments=segs,
        phases=phases,
        mission_name=args.mission_id,
        pc_threshold=pc_thresh,
        hbr_km=args.hbr,
    )
    report = screener.screen(
        window_open=window_open,
        window_close=window_close,
        nominal_launch=nominal_launch,
        step_s=args.step,
    )

    # Print report
    print(f"\n══ LCOLA Report: {report.mission_name} ══")
    print(f"  Window: {window_open} → {window_close}")
    print(f"  Step: {report.step_s:.0f} s   Pc threshold: {report.pc_threshold:.0e}")
    print(f"  Total launch times evaluated: {len(report.results)}")
    bo = report.blackout_windows
    print(f"  Blackout windows: {len(bo)}")
    for t0, t1 in bo:
        print(f"    {t0.strftime('%H:%M:%S')} – {t1.strftime('%H:%M:%S')} UTC")

    print(f"\n  Top conjunctions (max Pc):")
    for ev in report.top_events[:10]:
        print(f"    NORAD {ev.norad_cat_id:<7d}  {ev.phase:<20s}  "
              f"miss={ev.miss_distance_km:7.2f} km  Pc={ev.probability:.3e}  [{ev.risk_level}]")

    # Save JSON report
    if args.output:
        import json
        data = {
            "mission": report.mission_name,
            "blackout_windows": [
                [t0.isoformat(), t1.isoformat()] for t0, t1 in bo
            ],
            "top_events": [
                {
                    "norad_id": e.norad_cat_id,
                    "name": e.object_name,
                    "phase": e.phase,
                    "tca": e.tca.isoformat(),
                    "miss_km": round(e.miss_distance_km, 4),
                    "pc": e.probability,
                    "risk": e.risk_level,
                }
                for e in report.top_events
            ],
        }
        with open(args.output, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\n  Report saved → {args.output}")


def cmd_app(args):
    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app", "app.py")
    # Release 行为主要由 .streamlit/config.toml 与 STREAMLIT_* 环境变量控制；
    # 此处仅压低日志级别，避免无效 CLI 选项在不同版本上报错。
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(args.port),
         "--server.address", "0.0.0.0",
         "--server.headless", "true",
         "--logger.level", "warning"],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Space Debris Monitor CLI")
    sub = parser.add_subparsers(dest="command")

    # ── init-db ──────────────────────────────────────────────────────────────
    sub.add_parser("init-db", help="Initialize PostGIS schema")

    # ── ingest ───────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Fetch & propagate from Space-Track")
    p_ingest.add_argument("--limit",        type=int,   default=0,
                          help="Max objects to fetch; 0 = no limit (full catalog, default)")
    p_ingest.add_argument("--types",        nargs="+")
    p_ingest.add_argument("--no-propagate", action="store_true")
    p_ingest.add_argument("--horizon-days", type=int,   default=3)
    p_ingest.add_argument("--seg-minutes",  type=int,   default=10)

    # ── simulate ─────────────────────────────────────────────────────────────
    p_sim = sub.add_parser("simulate", help="Run 6-DOF trajectory simulation → OEM")
    p_sim.add_argument("--vehicle",     default="CZ-5B",
                        choices=["CZ-5B", "Falcon9", "Ariane6"],
                        help="Launch vehicle preset")
    p_sim.add_argument("--launch-utc",  dest="launch_utc",
                        default="2026-04-15T06:00:00",
                        help="Launch UTC ISO-8601")
    p_sim.add_argument("--lat",         type=float, default=19.61,  help="Launch lat [°]")
    p_sim.add_argument("--lon",         type=float, default=110.95, help="Launch lon [°]")
    p_sim.add_argument("--alt-km",      type=float, default=0.04,   help="Launch alt [km]")
    p_sim.add_argument("--azimuth",     type=float, default=90.0,   help="Launch azimuth [°]")
    p_sim.add_argument("--t-max",       type=float, default=6000.0, help="Sim duration [s]")
    p_sim.add_argument("--dt",          type=float, default=10.0,   help="Output cadence [s]")
    p_sim.add_argument("--no-mc",       action="store_true",         help="Skip Monte Carlo")
    p_sim.add_argument("--mc-runs",     type=int,   default=50)
    p_sim.add_argument("--mission-id",  default="2026-001")
    p_sim.add_argument("--output",      default="mission.oem",       help="OEM output path")

    # ── lcola ────────────────────────────────────────────────────────────────
    p_lc = sub.add_parser("lcola", help="Run LCOLA fly-through screening")
    p_lc.add_argument("--oem",          required=True,  help="OEM file path")
    p_lc.add_argument("--launch-utc",   dest="launch_utc", required=True)
    p_lc.add_argument("--mission-id",   default="MISSION")
    p_lc.add_argument("--window-minus", type=float, default=-30.0,  help="Window open [min from T0]")
    p_lc.add_argument("--window-plus",  type=float, default= 30.0,  help="Window close [min from T0]")
    p_lc.add_argument("--step",         type=float, default=60.0,   help="Screening step [s]")
    p_lc.add_argument("--hbr",          type=float, default=0.02,   help="Hard-body radius [km]")
    p_lc.add_argument("--crewed",       action="store_true",         help="Use crewed Pc threshold (1e-6)")
    p_lc.add_argument("--output",       default="lcola_report.json", help="JSON report path")

    # ── app ──────────────────────────────────────────────────────────────────
    p_app = sub.add_parser("app", help="Launch Streamlit dashboard")
    p_app.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()
    dispatch = {
        "init-db":  cmd_init_db,
        "ingest":   cmd_ingest,
        "simulate": cmd_simulate,
        "lcola":    cmd_lcola,
        "app":      cmd_app,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
