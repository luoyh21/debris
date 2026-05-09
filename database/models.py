"""SQLAlchemy models with GeoAlchemy2 geometry columns."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Double, Float,
    ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from geoalchemy2 import Geometry


class Base(DeclarativeBase):
    pass


class CatalogObject(Base):
    __tablename__ = "catalog_objects"

    norad_cat_id  = Column(Integer, primary_key=True)
    name          = Column(String)
    object_type   = Column(String)
    country_code  = Column(String)
    launch_date   = Column(Date)
    launch_site   = Column(String)
    decay_date    = Column(Date)
    inclination   = Column(Double)
    period_min    = Column(Double)
    apogee_km     = Column(Double)
    perigee_km    = Column(Double)
    rcs_size      = Column(String)
    object_id     = Column(String)
    updated_at    = Column(DateTime(timezone=True))

    gp_elements   = relationship("GpElement", back_populates="object",
                                  cascade="all, delete-orphan")
    segments      = relationship("TrajectorySegment", back_populates="object",
                                  cascade="all, delete-orphan")


class GpElement(Base):
    __tablename__ = "gp_elements"

    id                  = Column(BigInteger, primary_key=True, autoincrement=True)
    norad_cat_id        = Column(Integer, ForeignKey("catalog_objects.norad_cat_id",
                                                       ondelete="CASCADE"), nullable=False)
    epoch               = Column(DateTime(timezone=True), nullable=False)
    mean_motion         = Column(Double)
    eccentricity        = Column(Double)
    inclination         = Column(Double)
    ra_of_asc_node      = Column(Double)
    arg_of_pericenter   = Column(Double)
    mean_anomaly        = Column(Double)
    bstar               = Column(Double)
    tle_line1           = Column(Text)
    tle_line2           = Column(Text)
    source              = Column(String, default="spacetrack")
    ingested_at         = Column(DateTime(timezone=True))

    object = relationship("CatalogObject", back_populates="gp_elements")


class TrajectorySegment(Base):
    __tablename__ = "trajectory_segments"

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    norad_cat_id = Column(Integer, ForeignKey("catalog_objects.norad_cat_id",
                                               ondelete="CASCADE"), nullable=False)
    t_start      = Column(DateTime(timezone=True), nullable=False)
    t_end        = Column(DateTime(timezone=True), nullable=False)
    geom_eci     = Column(Geometry("LINESTRINGZ", srid=0), nullable=False)
    geom_geo     = Column(Geometry("LINESTRINGZ", srid=4326))
    gp_epoch     = Column(DateTime(timezone=True))
    created_at   = Column(DateTime(timezone=True))

    object = relationship("CatalogObject", back_populates="segments")


class LaunchWindow(Base):
    __tablename__ = "launch_windows"

    id                      = Column(BigInteger, primary_key=True, autoincrement=True)
    name                    = Column(String, nullable=False)
    launch_site             = Column(String)
    launch_time             = Column(DateTime(timezone=True), nullable=False)
    window_open             = Column(DateTime(timezone=True), nullable=False)
    window_close            = Column(DateTime(timezone=True), nullable=False)
    nominal_trajectory      = Column(Geometry("LINESTRINGZ", srid=4326))
    vehicle_type            = Column(String)
    payload                 = Column(String)
    orbit_target_alt_km     = Column(Double)
    orbit_target_inc_deg    = Column(Double)
    created_at              = Column(DateTime(timezone=True))

    risks = relationship("CollisionRisk", back_populates="launch",
                          cascade="all, delete-orphan")


class DataSourcePriority(Base):
    """Per-source priority used to dedupe ``v_unified_objects``.

    Lower number = higher priority.  When two sources contribute a row for
    the same NORAD ID the source with the lowest ``priority`` wins.  Sources
    that exist in the priority table but contain no rows are still honoured
    when the user later inserts data into them.
    """
    __tablename__ = "datasource_priority"

    source           = Column(String, primary_key=True)
    priority         = Column(Integer, nullable=False)
    description      = Column(String)
    is_user_defined  = Column(Boolean, default=False, nullable=False)
    updated_at       = Column(DateTime(timezone=True))


class UserDataObject(Base):
    """Records inserted via the ``/api/v1/datasources/{source}/objects`` API.

    Schema mirrors ``catalog_objects`` so the rows can participate in
    ``v_unified_objects`` directly.  ``source`` identifies the user-defined
    data source name (must already exist in ``datasource_priority``).
    """
    __tablename__ = "user_data_objects"

    source        = Column(String, primary_key=True)
    norad_cat_id  = Column(Integer, primary_key=True)
    name          = Column(String)
    object_type   = Column(String)
    country_code  = Column(String)
    launch_date   = Column(Date)
    launch_site   = Column(String)
    decay_date    = Column(Date)
    inclination   = Column(Double)
    period_min    = Column(Double)
    apogee_km     = Column(Double)
    perigee_km    = Column(Double)
    rcs_size      = Column(String)
    object_id     = Column(String)
    extra         = Column(Text)
    updated_at    = Column(DateTime(timezone=True))


class SpaceEvent(Base):
    """In-orbit event (fragmentation, collision, re-entry, maneuver, CDM, ...).

    ``raw`` (JSON-encoded text) stores the full source payload (CDM KVN,
    DISCOS fragmentation record, RDM, etc.) so we never lose information
    that the canonical columns can't represent.
    """
    __tablename__ = "space_events"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type      = Column(String, nullable=False)         # FRAGMENTATION/COLLISION/REENTRY/MANEUVER/CDM/OTHER
    epoch           = Column(DateTime(timezone=True), nullable=False)
    parent_norad    = Column(Integer)                         # primary object NORAD ID
    secondary_norad = Column(Integer)                         # for collisions / CDM
    name            = Column(String)                          # short label
    description     = Column(Text)
    altitude_km     = Column(Double)
    inclination_deg = Column(Double)
    energy_j        = Column(Double)                          # collision kinetic energy (J)
    energy_to_mass  = Column(Double)                          # J/g – >40 ⇒ catastrophic
    mass_parent_kg  = Column(Double)
    mass_target_kg  = Column(Double)
    miss_distance_km = Column(Double)                         # for CDM
    probability      = Column(Double)                         # for CDM
    n_fragments_obs  = Column(Integer)                        # observed fragment count (DISCOS)
    source           = Column(String, default="manual")       # DISCOS / SPACETRACK / MANUAL / CCSDS-IMPORT / SBM
    source_id        = Column(String)                         # external identifier
    raw              = Column(Text)                           # JSON of original payload
    created_at       = Column(DateTime(timezone=True))
    updated_at       = Column(DateTime(timezone=True))


class CollisionRisk(Base):
    __tablename__ = "collision_risks"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    launch_id        = Column(BigInteger, ForeignKey("launch_windows.id",
                                                      ondelete="CASCADE"))
    norad_cat_id     = Column(Integer)
    tca              = Column(DateTime(timezone=True))
    miss_distance_km = Column(Double)
    probability      = Column(Double)
    sigma_combined   = Column(Double)
    phase            = Column(String)
    computed_at      = Column(DateTime(timezone=True))

    launch = relationship("LaunchWindow", back_populates="risks")
