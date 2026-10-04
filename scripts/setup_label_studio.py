"""Initialize the separate Label Studio database and deployment secrets once."""
import os
import secrets
from pathlib import Path

import psycopg
from psycopg import sql


if __name__ == '__main__':
    env_path = Path(os.environ.get('DEPLOY_ENV_FILE', '/deploy/.env.production'))
    original = env_path.read_text()
    values = {}
    for line in original.splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    defaults = {
        'LABEL_STUDIO_DB': 'ailab_label_studio',
        'LABEL_STUDIO_DB_USER': 'ailab_label_studio',
        'LABEL_STUDIO_DB_PASSWORD': secrets.token_urlsafe(48),
        'LABEL_STUDIO_BRIDGE_SECRET': secrets.token_urlsafe(48),
        'LABEL_STUDIO_PUBLIC_URL': 'http://172.25.2.135:8080',
        'AILAB_PUBLIC_URL': values.get('NEXT_PUBLIC_APP_URL', 'http://172.25.2.135'),
    }
    missing = {key: value for key, value in defaults.items() if key not in values}
    values.update(missing)
    if any(not values[key] for key in defaults):
        raise ValueError('Label Studio deployment values must not be empty.')
    # Persist secrets before provisioning so a retry uses the same credentials.
    if missing:
        with env_path.open('a') as output:
            output.write('\n# Private Label Studio integration\n')
            output.writelines(f'{key}={value}\n' for key, value in missing.items())
    env_path.chmod(0o600)
    role, database = values['LABEL_STUDIO_DB_USER'], values['LABEL_STUDIO_DB']
    with psycopg.connect(os.environ['DATABASE_URL'], autocommit=True) as connection:
        exists = connection.execute('select 1 from pg_roles where rolname=%s', (role,)).fetchone()
        if not exists:
            connection.execute(sql.SQL('create role {} login password {} nosuperuser nocreatedb nocreaterole').format(
                sql.Identifier(role), sql.Literal(values['LABEL_STUDIO_DB_PASSWORD'])))
        owner = connection.execute('select pg_get_userbyid(datdba) from pg_database where datname=%s', (database,)).fetchone()
        if owner and owner[0] != role:
            raise ValueError('Existing database has a different owner; refusing to change it.')
        if not owner:
            connection.execute(sql.SQL('create database {} owner {}').format(sql.Identifier(database), sql.Identifier(role)))
    print('Separate Label Studio database and deployment values are ready. No existing tables were changed.')
