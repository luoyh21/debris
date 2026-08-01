-- 004_discos_esalof_esalog.sql
-- DISCOSweb v2 · EsaLOF（碎片化事件）与 EsaLOG（GEO 带物体 + RCS/质量）
-- 幂等可重复执行。

-- ── EsaLOF: /api/fragmentations?include=objects ─────────────────────────────
CREATE TABLE IF NOT EXISTS external_discos_esalof (
    id                      SERIAL PRIMARY KEY,
    discos_id               TEXT UNIQUE NOT NULL,   -- fragmentation id
    epoch                   DATE,
    event_type              TEXT,                   -- Propulsion / Collision / …
    comment                 TEXT,
    latitude                DOUBLE PRECISION,
    longitude               DOUBLE PRECISION,
    altitude_km             DOUBLE PRECISION,
    object_discos_id        TEXT,
    satno                   INTEGER,
    cospar_id               TEXT,
    object_name             TEXT,
    object_class            TEXT,
    mass_kg                 DOUBLE PRECISION,
    shape                   TEXT,
    xsect_avg_m2            DOUBLE PRECISION,
    xsect_max_m2            DOUBLE PRECISION,
    xsect_min_m2            DOUBLE PRECISION,
    catalogued_fragments    INTEGER,
    onorbit_fragments       INTEGER,
    source                  TEXT DEFAULT 'DISCOSweb:/api/fragmentations',
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_esalof_epoch ON external_discos_esalof (epoch DESC);
CREATE INDEX IF NOT EXISTS idx_esalof_satno ON external_discos_esalof (satno);
CREATE INDEX IF NOT EXISTS idx_esalof_event_type ON external_discos_esalof (event_type);

-- ── EsaLOG: GEO 带 initial-orbits + object（质量 / RCS）────────────────────
CREATE TABLE IF NOT EXISTS external_discos_esalog (
    id                      SERIAL PRIMARY KEY,
    discos_id               TEXT UNIQUE NOT NULL,   -- object id
    satno                   INTEGER,
    cospar_id               TEXT,
    name                    TEXT,
    object_class            TEXT,
    mass_kg                 DOUBLE PRECISION,
    shape                   TEXT,
    xsect_avg_m2            DOUBLE PRECISION,
    xsect_max_m2            DOUBLE PRECISION,
    xsect_min_m2            DOUBLE PRECISION,
    active                  BOOLEAN,
    pred_decay_date         TEXT,
    orbit_epoch             TEXT,
    sma_m                   DOUBLE PRECISION,
    ecc                     DOUBLE PRECISION,
    inc_deg                 DOUBLE PRECISION,
    raan_deg                DOUBLE PRECISION,
    source                  TEXT DEFAULT 'DISCOSweb:/api/initial-orbits (GEO)',
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_esalog_satno ON external_discos_esalog (satno);
CREATE INDEX IF NOT EXISTS idx_esalog_class ON external_discos_esalog (object_class);
CREATE INDEX IF NOT EXISTS idx_esalog_sma ON external_discos_esalog (sma_m);
