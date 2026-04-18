"""Create materialized view v_unified_objects merging ST + UCS + ESA."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from database.db import get_engine

engine = get_engine()

STMTS = [
    "DROP MATERIALIZED VIEW IF EXISTS v_unified_objects CASCADE",
    "DROP VIEW IF EXISTS v_unified_objects CASCADE",
    r"""
CREATE MATERIALIZED VIEW v_unified_objects AS

WITH _cospar_country AS (
    -- Derive launch country for COSPAR prefixes by majority vote from Space-Track
    SELECT DISTINCT ON (lp) lp, country_code
    FROM (
        SELECT SUBSTRING(esa."cosparId" FROM 1 FOR 8) AS lp,
               co.country_code,
               COUNT(*) AS cnt
        FROM external_esa_discos esa
        JOIN catalog_objects co ON co.norad_cat_id = esa.satno::int
        WHERE esa."cosparId" IS NOT NULL AND co.country_code IS NOT NULL
        GROUP BY 1, 2
    ) sub
    ORDER BY lp, cnt DESC
),

_name_purpose AS (
    -- Infer purpose from well-known satellite name patterns
    SELECT v.*,
        CASE
            WHEN v._raw_name ~* 'starlink'          THEN 'Communications'
            WHEN v._raw_name ~* 'oneweb'             THEN 'Communications'
            WHEN v._raw_name ~* 'kuiper'             THEN 'Communications'
            WHEN v._raw_name ~* 'iridium'            THEN 'Communications'
            WHEN v._raw_name ~* 'globalstar'         THEN 'Communications'
            WHEN v._raw_name ~* 'orbcomm'            THEN 'Communications'
            WHEN v._raw_name ~* 'ses[- ]'            THEN 'Communications'
            WHEN v._raw_name ~* 'intelsat'           THEN 'Communications'
            WHEN v._raw_name ~* 'viasat'             THEN 'Communications'
            WHEN v._raw_name ~* 'telesat'            THEN 'Communications'
            WHEN v._raw_name ~* 'o3b'                THEN 'Communications'
            WHEN v._raw_name ~* 'hulianwang'         THEN 'Communications'
            WHEN v._raw_name ~* 'qianfan'            THEN 'Communications'
            WHEN v._raw_name ~* 'gps|beidou|galileo|glonass|navstar' THEN 'Navigation'
            WHEN v._raw_name ~* 'landsat|sentinel|worldview|planet|flock|dove|superview|jilin|gaofen' THEN 'Earth Observation'
            WHEN v._raw_name ~* 'goes|metop|noaa|fengyun|himawari|meteosat' THEN 'Earth Science'
            WHEN v._raw_name ~* 'cygnus|dragon|progress|htv|tianzhou' THEN 'Space Station Supply'
            WHEN v._raw_name ~* 'cosmos|yaogan|nrol|usa[ -]\d' THEN 'Military'
            WHEN v._raw_name ~* 'tdrs'               THEN 'Data Relay'
            WHEN v._raw_name ~* 'hubble|jwst|chandra|xmm' THEN 'Astronomy'
            ELSE NULL
        END AS _name_inferred
    FROM (

    -- 1) Space-Track as base
    SELECT
        co.norad_cat_id,
        co.name           AS _raw_name,
        co.object_type,
        CASE co.country_code
            WHEN 'USA' THEN 'US'
            WHEN 'China' THEN 'PRC'
            WHEN 'CN' THEN 'PRC'
            ELSE co.country_code
        END AS country_code,
        co.launch_date,
        co.decay_date,
        co.inclination,
        co.perigee_km,
        co.apogee_km,
        co.period_min,
        co.rcs_size,
        ucs.purpose             AS ucs_purpose,
        ucs.users               AS ucs_users,
        ucs.operator            AS ucs_operator,
        UPPER(ucs.orbit_class)  AS ucs_orbit_class,
        ucs.launch_mass_kg,
        ucs.expected_lifetime_yr,
        esa.mass                AS esa_mass_kg,
        esa."xSectAvg"          AS esa_cross_section_m2,
        esa.active              AS esa_active,
        esa.mission             AS esa_mission,
        'Space-Track'::text     AS primary_source,
        CASE WHEN ucs.norad_cat_id IS NOT NULL THEN true ELSE false END AS has_ucs,
        CASE WHEN esa.satno       IS NOT NULL THEN true ELSE false END AS has_esa
    FROM catalog_objects co
    LEFT JOIN (
        SELECT DISTINCT ON (norad_cat_id) *
        FROM external_ucs_satellites
        WHERE norad_cat_id IS NOT NULL
        ORDER BY norad_cat_id
    ) ucs ON ucs.norad_cat_id = co.norad_cat_id
    LEFT JOIN (
        SELECT DISTINCT ON (satno) *
        FROM external_esa_discos
        WHERE satno IS NOT NULL
        ORDER BY satno
    ) esa ON esa.satno = co.norad_cat_id

    UNION ALL

    -- 2) UCS-only
    SELECT
        ucs2.norad_cat_id::int,
        ucs2.name,
        'PAYLOAD'::varchar,
        CASE ucs2.country
            WHEN 'USA' THEN 'US'
            WHEN 'China' THEN 'PRC'
            WHEN 'CN' THEN 'PRC'
            ELSE ucs2.country
        END,
        ucs2.launch_date::date,
        NULL::date,
        ucs2.inclination_deg,
        ucs2.perigee_km,
        ucs2.apogee_km,
        NULL::double precision,
        NULL::varchar,
        ucs2.purpose,
        ucs2.users,
        ucs2.operator,
        UPPER(ucs2.orbit_class),
        ucs2.launch_mass_kg,
        ucs2.expected_lifetime_yr,
        NULL::double precision,
        NULL::double precision,
        NULL::boolean,
        NULL::text,
        'UCS'::text,
        true,
        false
    FROM external_ucs_satellites ucs2
    WHERE ucs2.norad_cat_id IS NOT NULL
      AND ucs2.norad_cat_id NOT IN (
          SELECT norad_cat_id FROM catalog_objects WHERE norad_cat_id IS NOT NULL)

    UNION ALL

    -- 3) ESA-only
    SELECT
        esa2.satno::int,
        esa2.name,
        CASE
            WHEN esa2."objectClass" IN ('Payload','Payload Mission Related Object')
                 THEN 'PAYLOAD'
            WHEN esa2."objectClass" = 'Rocket Body'
                 THEN 'ROCKET BODY'
            WHEN esa2."objectClass" IN (
                 'Payload Fragmentation Debris','Rocket Fragmentation Debris',
                 'Payload Debris','Rocket Debris','Rocket Mission Related Object',
                 'Other Debris')
                 THEN 'DEBRIS'
            WHEN esa2."objectClass" = 'Unknown' THEN 'UNKNOWN'
            ELSE 'UNKNOWN'
        END::varchar,
        CASE cc.country_code
            WHEN 'USA' THEN 'US'
            WHEN 'China' THEN 'PRC'
            WHEN 'CN' THEN 'PRC'
            ELSE cc.country_code
        END,
        CASE WHEN esa2."firstEpoch" ~ '^\d{4}' THEN esa2."firstEpoch"::date ELSE NULL END,
        CASE WHEN esa2."predDecayDate" ~ '^\d{4}' THEN esa2."predDecayDate"::date ELSE NULL END,
        esa2.inclination::double precision,
        esa2.perigee_km::double precision,
        esa2.apogee_km::double precision,
        NULL::double precision,
        NULL::varchar,
        NULL::text, NULL::text, NULL::text, NULL::text,
        NULL::double precision, NULL::double precision,
        esa2.mass,
        esa2."xSectAvg",
        esa2.active,
        esa2.mission,
        'ESA-DISCOS'::text,
        false,
        true
    FROM external_esa_discos esa2
    LEFT JOIN _cospar_country cc ON cc.lp = SUBSTRING(esa2."cosparId" FROM 1 FOR 8)
    WHERE esa2.satno IS NOT NULL
      AND esa2.satno NOT IN (
          SELECT norad_cat_id FROM catalog_objects WHERE norad_cat_id IS NOT NULL)
      AND esa2.satno NOT IN (
          SELECT norad_cat_id FROM external_ucs_satellites WHERE norad_cat_id IS NOT NULL)

    ) v
)
SELECT
    norad_cat_id,
    _raw_name                                       AS name,
    object_type,
    country_code,
    launch_date,
    decay_date,
    inclination,
    perigee_km,
    apogee_km,
    period_min,
    rcs_size,
    ucs_purpose,
    ucs_users,
    ucs_operator,
    ucs_orbit_class,
    launch_mass_kg,
    expected_lifetime_yr,
    esa_mass_kg,
    esa_cross_section_m2,
    esa_active,
    esa_mission,
    primary_source,
    has_ucs,
    has_esa,
    COALESCE(ucs_purpose, esa_mission, _name_inferred) AS inferred_purpose,
    COALESCE(ucs_users,
        CASE
            WHEN _raw_name ~* 'cosmos|yaogan|nrol|usa[ -]\d' THEN 'Military'
            WHEN _raw_name ~* 'starlink|kuiper|oneweb|ses[- ]|intelsat|viasat|telesat|o3b|hulianwang|qianfan' THEN 'Commercial'
            WHEN _raw_name ~* 'landsat|goes|noaa|metop|fengyun|gaofen|sentinel|meteosat' THEN 'Government'
            WHEN _raw_name ~* 'gps|beidou|galileo|glonass|navstar|tdrs' THEN 'Government'
            ELSE NULL
        END
    ) AS inferred_users
FROM _name_purpose
""",
    "CREATE INDEX idx_unified_norad ON v_unified_objects (norad_cat_id)",
    "CREATE INDEX idx_unified_type  ON v_unified_objects (object_type)",
    "CREATE INDEX idx_unified_src   ON v_unified_objects (primary_source)",
]


def create():
    with engine.begin() as conn:
        for s in STMTS:
            s = s.strip()
            if s:
                print(f"  exec: {s[:60]}...")
                conn.execute(text(s))

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM v_unified_objects")).scalar()
        print(f"\nv_unified_objects total: {total:,}")

        rows = conn.execute(text("""
            SELECT primary_source, object_type, COUNT(*)
            FROM v_unified_objects GROUP BY 1,2 ORDER BY 1, 3 DESC
        """)).fetchall()
        print("\nSource × Type:")
        for r in rows:
            print(f"  {r[0]:15s} {r[1]:15s} {r[2]:>8,}")

        rows2 = conn.execute(text("""
            SELECT object_type, COUNT(*) FROM v_unified_objects GROUP BY 1 ORDER BY 2 DESC
        """)).fetchall()
        print("\nBy type (total):")
        for r in rows2:
            print(f"  {r[0]:15s} {r[1]:>8,}")


if __name__ == "__main__":
    create()
