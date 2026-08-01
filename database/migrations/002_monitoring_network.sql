-- 002_monitoring_network.sql
-- 全球空间多要素监测与测控网络三库（空间物体监测设备 / 空间天气监测设备 / 测控站）
-- 参见 docs/空间监测数据库构建.md；幂等可重复执行。

CREATE EXTENSION IF NOT EXISTS postgis;

-- ── 1. 全球天/地基空间物体监测设备 ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS external_ssa_sensors (
    id              SERIAL PRIMARY KEY,
    sensor_id       TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    name_cn         TEXT,
    sensor_class    TEXT NOT NULL,          -- spaceborne / ground_radar / ground_optical / network_node
    network         TEXT,                   -- SSN / ISON / TraCSS / LookUpSpace / TMDS / commercial …
    operator        TEXT,
    country         TEXT,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    alt_m           DOUBLE PRECISION,
    frequency_band  TEXT,                   -- UHF / L / S / X / optical …
    capability      TEXT,                   -- LEO tracking / GEO optical / short-arc OD …
    status          TEXT DEFAULT 'operational',
    notes           TEXT,
    source          TEXT DEFAULT 'docs/空间监测数据库构建.md',
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    geom            geometry(Point, 4326)
);
CREATE INDEX IF NOT EXISTS idx_ssa_sensors_geom ON external_ssa_sensors USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_ssa_sensors_network ON external_ssa_sensors (network);
CREATE INDEX IF NOT EXISTS idx_ssa_sensors_class ON external_ssa_sensors (sensor_class);

-- ── 2. 全球天/地基空间天气监测设备 ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS external_space_weather_sensors (
    id              SERIAL PRIMARY KEY,
    sensor_id       TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    name_cn         TEXT,
    sensor_class    TEXT NOT NULL,          -- spaceborne / magnetometer / ionosonde / isr / optical_uv / network
    network         TEXT,                   -- SWPC / GOES-R / APIS / ground networks …
    operator        TEXT,
    country         TEXT,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    alt_m           DOUBLE PRECISION,
    observables     TEXT,                   -- TEC / F10.7 / Kp/Ap / energetic particles / aurora …
    data_format     TEXT,                   -- NetCDF / CSV / HDF5 / REST …
    status          TEXT DEFAULT 'operational',
    notes           TEXT,
    source          TEXT DEFAULT 'docs/空间监测数据库构建.md',
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    geom            geometry(Point, 4326)
);
CREATE INDEX IF NOT EXISTS idx_swx_sensors_geom ON external_space_weather_sensors USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_swx_sensors_network ON external_space_weather_sensors (network);
CREATE INDEX IF NOT EXISTS idx_swx_sensors_class ON external_space_weather_sensors (sensor_class);

-- ── 3. 全球测控站（TT&C / GSaaS） ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS external_ttc_stations (
    id              SERIAL PRIMARY KEY,
    station_id      TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    name_cn         TEXT,
    network         TEXT NOT NULL,          -- ESTRACK / DSN / USGS-Landsat / KSAT / AWS / SatNOGS / ThumbNet …
    operator        TEXT,
    country         TEXT,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    alt_m           DOUBLE PRECISION,
    antenna_diam_m  DOUBLE PRECISION,
    bands           TEXT,                   -- S / X / Ka / UHF …
    station_type    TEXT,                   -- deep_space / near_earth / polar / commercial_gsaas / crowdsourced
    status          TEXT DEFAULT 'operational',
    notes           TEXT,
    source          TEXT DEFAULT 'docs/空间监测数据库构建.md',
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    geom            geometry(Point, 4326)
);
CREATE INDEX IF NOT EXISTS idx_ttc_geom ON external_ttc_stations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_ttc_network ON external_ttc_stations (network);
CREATE INDEX IF NOT EXISTS idx_ttc_country ON external_ttc_stations (country);
