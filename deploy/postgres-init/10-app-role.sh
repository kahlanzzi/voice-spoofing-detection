#!/usr/bin/env bash
# First start of the db container only (the image runs /docker-entrypoint-initdb.d/* when the data folder is
# empty; docs/setup.md "Database"). Creates the API's role, vsd_app: it can log in and owns the vsd database
# (so the API's migration runner can create its tables in public), but it is not a superuser and cannot create
# databases or roles. The password comes from VSD_DB_APP_PASSWORD and goes to psql as a variable, never into
# the SQL text or a log.
set -euo pipefail
: "${VSD_DB_APP_PASSWORD:?VSD_DB_APP_PASSWORD is not set (see .env.example)}"

psql -v ON_ERROR_STOP=1 -v app_pw="$VSD_DB_APP_PASSWORD" \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE vsd_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'app_pw';
ALTER DATABASE vsd OWNER TO vsd_app;
SQL
