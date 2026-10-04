"""Idempotent, additive application migrations. No user media is changed."""
import os
from pathlib import Path

import psycopg


def apply(connection):
    root = Path(__file__).resolve().parents[1] / 'database'
    for name in ('app_schema.sql', 'label_studio_schema.sql', 'compute_schema.sql'):
        connection.execute((root / name).read_text())


if __name__ == '__main__':
    with psycopg.connect(os.environ['DATABASE_URL']) as connection:
        apply(connection)
    print('Application, Label Studio and compute schemas are ready.')
