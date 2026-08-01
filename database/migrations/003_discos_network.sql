-- 003_discos_network.sql
-- DISCOSweb 发射场 / 组织机构 + 监测网扩展支撑表（幂等可重复执行）

CREATE EXTENSION IF NOT EXISTS postgis;

-- ── DISCOSweb launch-sites ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS external_discos_launch_sites (
    id              SERIAL PRIMARY KEY,
    discos_id       TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    alt_m           DOUBLE PRECISION,
    pads            TEXT,
    azimuths        TEXT,
    constraints     TEXT,
    source          TEXT DEFAULT 'DISCOSweb:/api/launch-sites',
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    geom            geometry(Point, 4326)
);
CREATE INDEX IF NOT EXISTS idx_discos_ls_geom ON external_discos_launch_sites USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_discos_ls_name ON external_discos_launch_sites (name);

-- ── DISCOSweb organisations ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS external_discos_organisations (
    id              SERIAL PRIMARY KEY,
    discos_id       TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    date_range      TEXT,
    source          TEXT DEFAULT 'DISCOSweb:/api/organisations',
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_discos_org_name ON external_discos_organisations (name);
