"""Create / refresh ``v_unified_objects`` materialized view.

Behavioural change (May 2026)
-----------------------------
Previously the view hard-coded a fixed Space-Track > UCS > ESA precedence and
always preferred a Space-Track row whenever one existed.  We now dedupe by
the per-source priority stored in ``datasource_priority`` so:

* operators can re-rank sources at runtime (via the API) and the next
  materialized-view refresh will reflect the change;
* user-defined sources (``user_data_objects``) participate as first-class
  citizens — when a user inserts a row with a higher priority than
  Space-Track, that row wins after the next view refresh.

To keep the implementation tractable the view still computes per-source
"candidate" rows in independent CTEs and finally picks the highest-priority
row per ``norad_cat_id``.  All side-table joins (UCS/ESA enrichment columns)
remain on the Space-Track row so existing dashboards keep working unchanged.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from database.db import get_engine, init_db

engine = get_engine()


SQL = r"""
DROP MATERIALIZED VIEW IF EXISTS v_unified_objects CASCADE;

CREATE MATERIALIZED VIEW v_unified_objects AS

WITH _prio AS (
    SELECT source, priority FROM datasource_priority
),

_cospar_country AS (
    SELECT DISTINCT ON (lp) lp, country_code FROM (
        SELECT SUBSTRING(esa."cosparId" FROM 1 FOR 8) AS lp,
               co.country_code,
               COUNT(*) AS cnt
        FROM external_esa_discos esa
        JOIN catalog_objects co ON co.norad_cat_id = esa.satno::int
        WHERE esa."cosparId" IS NOT NULL AND co.country_code IS NOT NULL
        GROUP BY 1, 2
    ) sub ORDER BY lp, cnt DESC
),

-- Candidate rows from every source (UNION ALL) — each carries its source
-- name + priority.  Final pick = lowest priority per norad_cat_id.

_st AS (
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
        co.launch_date, co.decay_date,
        co.inclination, co.perigee_km, co.apogee_km, co.period_min,
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
),

_ucs_only AS (
    SELECT
        ucs2.norad_cat_id::int                      AS norad_cat_id,
        ucs2.name                                    AS _raw_name,
        'PAYLOAD'::varchar                           AS object_type,
        CASE ucs2.country
            WHEN 'USA' THEN 'US'
            WHEN 'China' THEN 'PRC'
            WHEN 'CN' THEN 'PRC'
            ELSE ucs2.country
        END                                          AS country_code,
        ucs2.launch_date::date                       AS launch_date,
        NULL::date                                   AS decay_date,
        ucs2.inclination_deg                         AS inclination,
        ucs2.perigee_km, ucs2.apogee_km,
        NULL::double precision                       AS period_min,
        NULL::varchar                                AS rcs_size,
        ucs2.purpose, ucs2.users, ucs2.operator,
        UPPER(ucs2.orbit_class)                      AS ucs_orbit_class,
        ucs2.launch_mass_kg, ucs2.expected_lifetime_yr,
        NULL::double precision                       AS esa_mass_kg,
        NULL::double precision                       AS esa_cross_section_m2,
        NULL::boolean                                AS esa_active,
        NULL::text                                   AS esa_mission,
        'UCS'::text                                  AS primary_source,
        true                                         AS has_ucs,
        false                                        AS has_esa
    FROM external_ucs_satellites ucs2
    WHERE ucs2.norad_cat_id IS NOT NULL
),

_esa_only AS (
    SELECT
        esa2.satno::int                              AS norad_cat_id,
        esa2.name                                    AS _raw_name,
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
        END::varchar                                 AS object_type,
        CASE cc.country_code
            WHEN 'USA' THEN 'US'
            WHEN 'China' THEN 'PRC'
            WHEN 'CN' THEN 'PRC'
            ELSE cc.country_code
        END                                          AS country_code,
        CASE WHEN esa2."firstEpoch" ~ '^\d{4}'
             THEN esa2."firstEpoch"::date ELSE NULL END   AS launch_date,
        CASE WHEN esa2."predDecayDate" ~ '^\d{4}'
             THEN esa2."predDecayDate"::date ELSE NULL END AS decay_date,
        esa2.inclination::double precision           AS inclination,
        esa2.perigee_km::double precision            AS perigee_km,
        esa2.apogee_km::double precision             AS apogee_km,
        NULL::double precision                       AS period_min,
        NULL::varchar                                AS rcs_size,
        NULL::text AS ucs_purpose, NULL::text AS ucs_users,
        NULL::text AS ucs_operator, NULL::text AS ucs_orbit_class,
        NULL::double precision AS launch_mass_kg,
        NULL::double precision AS expected_lifetime_yr,
        esa2.mass                                    AS esa_mass_kg,
        esa2."xSectAvg"                              AS esa_cross_section_m2,
        esa2.active                                  AS esa_active,
        esa2.mission                                 AS esa_mission,
        'ESA-DISCOS'::text                           AS primary_source,
        false                                        AS has_ucs,
        true                                         AS has_esa
    FROM external_esa_discos esa2
    LEFT JOIN _cospar_country cc ON cc.lp = SUBSTRING(esa2."cosparId" FROM 1 FOR 8)
    WHERE esa2.satno IS NOT NULL
),

_user AS (
    -- All rows from user-managed sources, regardless of source name
    SELECT
        u.norad_cat_id,
        u.name                                       AS _raw_name,
        u.object_type,
        u.country_code,
        u.launch_date, u.decay_date,
        u.inclination, u.perigee_km, u.apogee_km, u.period_min,
        u.rcs_size,
        NULL::text AS ucs_purpose, NULL::text AS ucs_users,
        NULL::text AS ucs_operator, NULL::text AS ucs_orbit_class,
        NULL::double precision AS launch_mass_kg,
        NULL::double precision AS expected_lifetime_yr,
        NULL::double precision AS esa_mass_kg,
        NULL::double precision AS esa_cross_section_m2,
        NULL::boolean AS esa_active,
        NULL::text AS esa_mission,
        u.source                                     AS primary_source,
        false AS has_ucs,
        false AS has_esa
    FROM user_data_objects u
    WHERE u.norad_cat_id IS NOT NULL
),

_all AS (
    SELECT * FROM _st
    UNION ALL SELECT * FROM _ucs_only
    UNION ALL SELECT * FROM _esa_only
    UNION ALL SELECT * FROM _user
),

_ranked AS (
    SELECT a.*,
           COALESCE(p.priority, 9999) AS _prio
    FROM _all a
    LEFT JOIN _prio p ON p.source = a.primary_source
),

_winner AS (
    SELECT DISTINCT ON (norad_cat_id) *
    FROM _ranked
    ORDER BY norad_cat_id, _prio ASC, primary_source ASC
)

SELECT
    norad_cat_id,
    _raw_name                                   AS name,
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
    COALESCE(ucs_purpose, esa_mission,
        CASE
            WHEN _raw_name ~* 'starlink|kuiper|oneweb|iridium|globalstar|orbcomm|ses[- ]|intelsat|viasat|telesat|o3b|hulianwang|qianfan' THEN 'Communications'
            WHEN _raw_name ~* 'gps|beidou|galileo|glonass|navstar' THEN 'Navigation'
            WHEN _raw_name ~* 'landsat|sentinel|worldview|planet|flock|dove|superview|jilin|gaofen' THEN 'Earth Observation'
            WHEN _raw_name ~* 'goes|metop|noaa|fengyun|himawari|meteosat' THEN 'Earth Science'
            WHEN _raw_name ~* 'cygnus|dragon|progress|htv|tianzhou' THEN 'Space Station Supply'
            WHEN _raw_name ~* 'cosmos|yaogan|nrol|usa[ -]\d' THEN 'Military'
            WHEN _raw_name ~* 'tdrs' THEN 'Data Relay'
            WHEN _raw_name ~* 'hubble|jwst|chandra|xmm' THEN 'Astronomy'
            ELSE NULL
        END) AS inferred_purpose,
    COALESCE(ucs_users,
        CASE
            WHEN _raw_name ~* 'cosmos|yaogan|nrol|usa[ -]\d' THEN 'Military'
            WHEN _raw_name ~* 'starlink|kuiper|oneweb|ses[- ]|intelsat|viasat|telesat|o3b|hulianwang|qianfan' THEN 'Commercial'
            WHEN _raw_name ~* 'landsat|goes|noaa|metop|fengyun|gaofen|sentinel|meteosat' THEN 'Government'
            WHEN _raw_name ~* 'gps|beidou|galileo|glonass|navstar|tdrs' THEN 'Government'
            ELSE NULL
        END) AS inferred_users
FROM _winner;
"""

INDEXES = [
    "CREATE INDEX idx_unified_norad ON v_unified_objects (norad_cat_id)",
    "CREATE INDEX idx_unified_type  ON v_unified_objects (object_type)",
    "CREATE INDEX idx_unified_src   ON v_unified_objects (primary_source)",
]


def create():
    init_db()                # ensures datasource_priority + user_data_objects exist
    with engine.begin() as conn:
        # The whole DDL must run in a single transaction so the materialized
        # view + indexes appear atomically.
        for stmt in [SQL] + INDEXES:
            conn.execute(text(stmt))

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM v_unified_objects")).scalar()
        print(f"\nv_unified_objects total: {total:,}")

        rows = conn.execute(text("""
            SELECT primary_source, COUNT(*)
            FROM v_unified_objects GROUP BY 1 ORDER BY 2 DESC
        """)).fetchall()
        print("\nBy primary_source:")
        for r in rows:
            print(f"  {r[0]:20s} {r[1]:>8,}")


if __name__ == "__main__":
    create()
