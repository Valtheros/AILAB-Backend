"""Add integration tables only; never change or remove native annotation data."""
import os
from pathlib import Path

import psycopg


if __name__ == "__main__":
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(Path(__file__).resolve().parents[1].joinpath("database/label_studio_schema.sql").read_text())
    print("Label Studio integration schema is ready.")
