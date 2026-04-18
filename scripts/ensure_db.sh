#!/bin/bash
# Ensure the space_debris database and PostGIS extension exist.
# Placed in /docker-entrypoint-initdb.d/ so it runs on every container start.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
    SELECT 'CREATE DATABASE space_debris'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'space_debris')\gexec
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname space_debris <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
EOSQL
