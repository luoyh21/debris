-- Space Debris PostGIS Database Schema
-- Run: psql -U postgres -d space_debris -f 001_init_postgis.sql

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- for text search

-- -------------------------------------------------------
-- 1. Catalog objects (one row per NORAD object)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS catalog_objects (
    norad_cat_id    INTEGER PRIMARY KEY,
    name            TEXT,
    object_type     TEXT,           -- PAYLOAD / DEBRIS / ROCKET BODY / UNKNOWN
    country_code    TEXT,
    launch_date     DATE,
    launch_site     TEXT,
    decay_date      DATE,
    inclination     DOUBLE PRECISION,
    period_min      DOUBLE PRECISION,
    apogee_km       DOUBLE PRECISION,
    perigee_km      DOUBLE PRECISION,
    rcs_size        TEXT,           -- SMALL / MEDIUM / LARGE
    object_id       TEXT,           -- COSPAR / international designator
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalog_object_type ON catalog_objects(object_type);
CREATE INDEX IF NOT EXISTS idx_catalog_country    ON catalog_objects(country_code);
CREATE INDEX IF NOT EXISTS idx_catalog_name_trgm  ON catalog_objects USING gin(name gin_trgm_ops);

-- -------------------------------------------------------
-- 2. Raw GP mean elements (TLE / SP data)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS gp_elements (
    id              BIGSERIAL PRIMARY KEY,
    norad_cat_id    INTEGER REFERENCES catalog_objects(norad_cat_id) ON DELETE CASCADE,
    epoch           TIMESTAMPTZ NOT NULL,
    mean_motion     DOUBLE PRECISION,       -- rev/day
    eccentricity    DOUBLE PRECISION,
    inclination     DOUBLE PRECISION,       -- deg
    ra_of_asc_node  DOUBLE PRECISION,       -- deg
    arg_of_pericenter DOUBLE PRECISION,     -- deg
    mean_anomaly    DOUBLE PRECISION,       -- deg
    bstar           DOUBLE PRECISION,
    tle_line1       TEXT,
    tle_line2       TEXT,
    source          TEXT DEFAULT 'spacetrack',
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gp_norad  ON gp_elements(norad_cat_id);
CREATE INDEX IF NOT EXISTS idx_gp_epoch  ON gp_elements(epoch DESC);

-- Latest GP per object (materialized for speed)
CREATE UNIQUE INDEX IF NOT EXISTS idx_gp_latest
    ON gp_elements(norad_cat_id, epoch DESC);

-- -------------------------------------------------------
-- 3. Propagated trajectory segments (3-D LineStringZ in ECI km)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS trajectory_segments (
    id              BIGSERIAL PRIMARY KEY,
    norad_cat_id    INTEGER NOT NULL,
    t_start         TIMESTAMPTZ NOT NULL,
    t_end           TIMESTAMPTZ NOT NULL,
    -- ECI geometry: SRID=0 (Cartesian, units = km)
    geom_eci        geometry(LineStringZ, 0) NOT NULL,
    -- Geodetic geometry: SRID=4326 (lon/lat/alt-km)
    geom_geo        geometry(LineStringZ, 4326),
    gp_epoch        TIMESTAMPTZ,            -- which GP record was used
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index on ECI geometry (for 3-D proximity queries)
CREATE INDEX IF NOT EXISTS idx_traj_geom_eci  ON trajectory_segments USING GIST(geom_eci);
CREATE INDEX IF NOT EXISTS idx_traj_geom_geo  ON trajectory_segments USING GIST(geom_geo);
-- Time-range index (most critical for LCOLA window filtering)
CREATE INDEX IF NOT EXISTS idx_traj_time      ON trajectory_segments(t_start, t_end);
CREATE INDEX IF NOT EXISTS idx_traj_norad     ON trajectory_segments(norad_cat_id);

-- -------------------------------------------------------
-- 4. Launch window definitions
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS launch_windows (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    launch_site     TEXT,
    launch_time     TIMESTAMPTZ NOT NULL,
    window_open     TIMESTAMPTZ NOT NULL,
    window_close    TIMESTAMPTZ NOT NULL,
    -- Nominal trajectory: SRID=4326
    nominal_trajectory geometry(LineStringZ, 4326),
    vehicle_type    TEXT,
    payload         TEXT,
    orbit_target_alt_km DOUBLE PRECISION,
    orbit_target_inc_deg DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lw_time ON launch_windows(launch_time);
CREATE INDEX IF NOT EXISTS idx_lw_traj ON launch_windows USING GIST(nominal_trajectory);

-- -------------------------------------------------------
-- 5. Collision risk assessments (per launch × debris object)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS collision_risks (
    id              BIGSERIAL PRIMARY KEY,
    launch_id       BIGINT REFERENCES launch_windows(id) ON DELETE CASCADE,
    norad_cat_id    INTEGER,
    tca             TIMESTAMPTZ,            -- Time of Closest Approach
    miss_distance_km DOUBLE PRECISION,
    probability     DOUBLE PRECISION,       -- Pc (collision probability)
    sigma_combined  DOUBLE PRECISION,       -- combined covariance size km
    phase           TEXT,                   -- ASCENT / SEPARATION / COAST
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_launch  ON collision_risks(launch_id);
CREATE INDEX IF NOT EXISTS idx_risk_norad   ON collision_risks(norad_cat_id);
CREATE INDEX IF NOT EXISTS idx_risk_pc      ON collision_risks(probability DESC);

-- -------------------------------------------------------
-- 6. Handy views
-- -------------------------------------------------------
CREATE OR REPLACE VIEW v_debris_density AS
SELECT
    ROUND(ST_Y(pt.geom)::NUMERIC, 0)  AS lat_bin,
    ROUND(ST_X(pt.geom)::NUMERIC, 0)  AS lon_bin,
    COUNT(*)                            AS segment_count
FROM trajectory_segments ts,
     LATERAL ST_DumpPoints(geom_geo) pt(path, geom)
GROUP BY 1, 2;

CREATE OR REPLACE VIEW v_high_risk_events AS
SELECT
    cr.*,
    lw.name AS launch_name,
    co.name AS object_name,
    co.object_type
FROM collision_risks cr
JOIN launch_windows lw ON lw.id = cr.launch_id
JOIN catalog_objects co ON co.norad_cat_id = cr.norad_cat_id
WHERE cr.probability > 1e-5
ORDER BY cr.probability DESC;
