"""DB CRUD helpers for ``space_events``.

All functions accept either a live SQLAlchemy ``Session`` (unit-testing)
or auto-open one via :func:`database.db.session_scope`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import desc, select

from database.db import session_scope
from database.models import SpaceEvent as SpaceEventORM
from .types import EventType, SpaceEvent


# ─── (de)serialisation ───────────────────────────────────────────────────────

def _to_orm(evt: SpaceEvent) -> SpaceEventORM:
    raw = json.dumps(evt.raw, default=str, ensure_ascii=False) if evt.raw else None
    return SpaceEventORM(
        event_type      = evt.event_type.value,
        epoch           = evt.epoch,
        parent_norad    = evt.parent_norad,
        secondary_norad = evt.secondary_norad,
        name            = evt.name,
        description     = evt.description,
        altitude_km     = evt.altitude_km,
        inclination_deg = evt.inclination_deg,
        energy_j        = evt.energy_j,
        energy_to_mass  = evt.energy_to_mass,
        mass_parent_kg  = evt.mass_parent_kg,
        mass_target_kg  = evt.mass_target_kg,
        miss_distance_km= evt.miss_distance_km,
        probability     = evt.probability,
        n_fragments_obs = evt.n_fragments_obs,
        source          = evt.source,
        source_id       = evt.source_id,
        raw             = raw,
        created_at      = datetime.now(timezone.utc),
        updated_at      = datetime.now(timezone.utc),
    )


def _from_orm(row: SpaceEventORM) -> SpaceEvent:
    raw = None
    if row.raw:
        try: raw = json.loads(row.raw)
        except Exception: raw = {"_raw_text": row.raw}
    try:
        et = EventType(row.event_type)
    except ValueError:
        et = EventType.OTHER
    return SpaceEvent(
        event_type      = et,
        epoch           = row.epoch,
        parent_norad    = row.parent_norad,
        secondary_norad = row.secondary_norad,
        name            = row.name or "",
        description     = row.description or "",
        altitude_km     = row.altitude_km,
        inclination_deg = row.inclination_deg,
        energy_j        = row.energy_j,
        energy_to_mass  = row.energy_to_mass,
        mass_parent_kg  = row.mass_parent_kg,
        mass_target_kg  = row.mass_target_kg,
        miss_distance_km= row.miss_distance_km,
        probability     = row.probability,
        n_fragments_obs = row.n_fragments_obs,
        source          = row.source or "manual",
        source_id       = row.source_id or "",
        raw             = raw,
        id              = row.id,
    )


# ─── public API ─────────────────────────────────────────────────────────────

def insert_event(evt: SpaceEvent) -> int:
    """Persist a new event, return its DB id."""
    with session_scope() as sess:
        row = _to_orm(evt)
        sess.add(row)
        sess.flush()
        sess.refresh(row)
        return int(row.id)


def upsert_event(evt: SpaceEvent) -> int:
    """Insert or update by ``(source, source_id)``; falls back to insert."""
    if not evt.source_id:
        return insert_event(evt)
    with session_scope() as sess:
        row = sess.query(SpaceEventORM).filter(
            SpaceEventORM.source == evt.source,
            SpaceEventORM.source_id == evt.source_id,
        ).one_or_none()
        if row is None:
            row = _to_orm(evt)
            sess.add(row)
            sess.flush(); sess.refresh(row)
            return int(row.id)
        for k in ("event_type","epoch","parent_norad","secondary_norad","name",
                  "description","altitude_km","inclination_deg","energy_j",
                  "energy_to_mass","mass_parent_kg","mass_target_kg",
                  "miss_distance_km","probability","n_fragments_obs"):
            v = getattr(evt, k, None)
            if v is None: continue
            setattr(row, k, v.value if isinstance(v, EventType) else v)
        if evt.raw:
            row.raw = json.dumps(evt.raw, default=str, ensure_ascii=False)
        row.updated_at = datetime.now(timezone.utc)
        return int(row.id)


def list_events(*, event_type: Optional[EventType] = None,
                limit: int = 200,
                source: Optional[str] = None,
                ) -> List[SpaceEvent]:
    with session_scope() as sess:
        q = sess.query(SpaceEventORM).order_by(desc(SpaceEventORM.epoch))
        if event_type is not None:
            q = q.filter(SpaceEventORM.event_type == event_type.value)
        if source:
            q = q.filter(SpaceEventORM.source == source)
        rows = q.limit(int(limit)).all()
        return [_from_orm(r) for r in rows]


def get_event(event_id: int) -> Optional[SpaceEvent]:
    with session_scope() as sess:
        row = sess.get(SpaceEventORM, int(event_id))
        return _from_orm(row) if row else None


def delete_event(event_id: int) -> bool:
    with session_scope() as sess:
        row = sess.get(SpaceEventORM, int(event_id))
        if row is None: return False
        sess.delete(row)
        return True


def count_events() -> int:
    with session_scope() as sess:
        return int(sess.query(SpaceEventORM).count())
