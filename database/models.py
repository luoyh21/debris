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
