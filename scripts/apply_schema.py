import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resource_repository import resource_repository

schema = Path(__file__).resolve().parents[1] / "database" / "app_schema.sql"
with resource_repository._connect() as connection:
    connection.execute(schema.read_text(encoding="utf-8"))
print("Application schema is current")
