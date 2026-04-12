"""SGP4 / SGP4-XP orbit propagator.

Falls back to sgp4 library (python-sgp4) which implements Vallado's SGP4.
SGP4-XP extended perturbations are invoked when available via the
sgp4 library's WGS84 constants and the 'afspc' / 'improved' mode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional

import numpy as np

try:
    from sgp4.api import Satrec, WGS84          # sgp4 >= 2.x
    from sgp4.conveniences import sat_epoch_datetime
    _SGP4_AVAILABLE = True
except ImportError:
    _SGP4_AVAILABLE = False


@dataclass
class StateVector:
    """ECI position (km) and velocity (km/s) at a given epoch."""
    epoch: datetime           # UTC
    x: float                  # km
    y: float                  # km
    z: float                  # km
    vx: float                 # km/s
    vy: float                 # km/s
    vz: float                 # km/s

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def pos_eci_km(self) -> np.ndarray:
        """Alias for fly_through compatibility."""
        return self.pos

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    @property
    def altitude_km(self) -> float:
        R_EARTH = 6371.0
        return math.sqrt(self.x**2 + self.y**2 + self.z**2) - R_EARTH

    def to_ecef(self) -> Tuple[float, float, float]:
        """Rotate ECI → ECEF using Greenwich Sidereal Time (approximate)."""
        # GST in radians (simplified, no nutation/precession)
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt_days = (self.epoch - j2000).total_seconds() / 86400.0
        gst = math.fmod(280.46061837 + 360.98564736629 * dt_days, 360.0)
        gst_rad = math.radians(gst)
        cos_g, sin_g = math.cos(gst_rad), math.sin(gst_rad)
        x_ecef = self.x * cos_g + self.y * sin_g
        y_ecef = -self.x * sin_g + self.y * cos_g
        return x_ecef, y_ecef, self.z

    def to_geodetic(self) -> Tuple[float, float, float]:
        """Convert ECI → geodetic (lat°, lon°, alt km)."""
        xe, ye, ze = self.to_ecef()
        R_EARTH = 6378.137
        e2 = 0.00669437999014
        lon = math.degrees(math.atan2(ye, xe))
        p = math.sqrt(xe**2 + ye**2)
        lat = math.degrees(math.atan2(ze, p * (1 - e2)))
        for _ in range(5):          # iterate to convergence
            lat_r = math.radians(lat)
            N = R_EARTH / math.sqrt(1 - e2 * math.sin(lat_r)**2)
            lat = math.degrees(math.atan2(ze + e2 * N * math.sin(lat_r), p))
        lat_r = math.radians(lat)
        N = R_EARTH / math.sqrt(1 - e2 * math.sin(lat_r)**2)
        alt = p / math.cos(lat_r) - N if abs(lat) < 89.9 else ze / math.sin(lat_r) - N * (1 - e2)
        return lat, lon, alt


@dataclass
class OrbitSegment:
    """A 3-D line segment (LineStringZ) representing a debris trajectory chunk."""
    norad_id: int
    t_start: datetime
    t_end: datetime
    points: List[Tuple[float, float, float]]    # (x_km, y_km, z_km) ECI
    geodetic_points: List[Tuple[float, float, float]] = field(default_factory=list)  # (lat, lon, alt)


class SGP4Propagator:
    """Wraps python-sgp4 for single-object propagation."""

    def __init__(self, gp_record: dict):
        """
        Parameters
        ----------
        gp_record : dict
            A GP record from Space-Track (keys: TLE_LINE1, TLE_LINE2 or
            MEAN_MOTION, ECCENTRICITY, … for GP mean elements).
        """
        if not _SGP4_AVAILABLE:
            raise ImportError("sgp4 library not installed – run: pip install sgp4")

        self.norad_id = int(gp_record.get("NORAD_CAT_ID", 0))
        self._sat = self._build_satrec(gp_record)

    @staticmethod
    def _build_satrec(rec: dict) -> "Satrec":
        line1 = rec.get("TLE_LINE1")
        line2 = rec.get("TLE_LINE2")
        if line1 and line2:
            return Satrec.twoline2rv(line1, line2)

        # Build from GP mean elements directly (Space-Track JSON fields)
        sat = Satrec()
        sat.sgp4init(
            WGS84,
            "i",                                    # improved mode (SGP4-XP)
            int(rec["NORAD_CAT_ID"]),
            float(rec["BSTAR"]),
            0.0,                                    # ndot (unused in SGP4)
            0.0,                                    # nddot
            float(rec["ECCENTRICITY"]),
            math.radians(float(rec["ARG_OF_PERICENTER"])),
            math.radians(float(rec["INCLINATION"])),
            math.radians(float(rec["MEAN_ANOMALY"])),
            float(rec["MEAN_MOTION"]) * 2 * math.pi / 1440.0,  # rad/min
            math.radians(float(rec["RA_OF_ASC_NODE"])),
        )
        return sat

    def propagate(self, epoch: datetime) -> Optional[StateVector]:
        """Propagate to a specific UTC datetime."""
        epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch
        # Compute minutes since TLE epoch (Julian date arithmetic)
        from sgp4.api import jday
        jd, fr = jday(epoch.year, epoch.month, epoch.day,
                      epoch.hour, epoch.minute,
                      epoch.second + epoch.microsecond / 1e6)
        e, r, v = self._sat.sgp4(jd, fr)
        if e != 0:
            return None     # propagation error (decay, etc.)
        return StateVector(epoch=epoch,
                           x=r[0], y=r[1], z=r[2],
                           vx=v[0], vy=v[1], vz=v[2])

    def generate_segments(
        self,
        t_start: datetime,
        t_end: datetime,
        segment_minutes: int = 10,
        points_per_segment: int = 7,
    ) -> List[OrbitSegment]:
        """Discretize the orbit into fixed-width 3D LineStringZ segments."""
        segments: List[OrbitSegment] = []
        seg_delta = timedelta(minutes=segment_minutes)
        pt_delta = seg_delta / (points_per_segment - 1)

        t = t_start
        while t < t_end:
            t_seg_end = min(t + seg_delta, t_end)
            pts_eci, pts_geo = [], []
            for i in range(points_per_segment):
                t_pt = t + pt_delta * i
                if t_pt > t_seg_end:
                    t_pt = t_seg_end
                sv = self.propagate(t_pt)
                if sv is None:
                    break
                pts_eci.append((sv.x, sv.y, sv.z))
                pts_geo.append(sv.to_geodetic())
            if len(pts_eci) >= 2:
                segments.append(OrbitSegment(
                    norad_id=self.norad_id,
                    t_start=t,
                    t_end=t_seg_end,
                    points=pts_eci,
                    geodetic_points=pts_geo,
                ))
            t += seg_delta
        return segments
