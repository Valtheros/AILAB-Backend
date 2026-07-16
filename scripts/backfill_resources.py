from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_utils import inspect_dataset  # noqa: E402
from dataset_storage import owner_dataset_path, registered_storage_path  # noqa: E402
from resource_repository import resource_repository  # noqa: E402
from settings import DATASET_DIR, REDIS_URL, RUNS_DIR  # noqa: E402


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def infer_run_status(path: Path, config: dict) -> str:
    job_id = str(config.get("job_id") or config.get("rq_job_id") or "")
    if job_id:
        try:
            from redis import Redis
            from rq.job import Job, JobStatus

            job = Job.fetch(job_id, connection=Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2))
            raw_status = job.get_status()
            status = raw_status.value if isinstance(raw_status, JobStatus) else str(raw_status)
            return {
                "queued": "queued",
                "deferred": "queued",
                "scheduled": "queued",
                "started": "running",
                "finished": "completed",
                "failed": "failed",
                "stopped": "stopped",
                "canceled": "cancelled",
            }.get(status, "recovery_pending")
        except Exception:
            pass
    log_path = path / "train.log"
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as file:
            file.seek(max(0, size - 1024 * 1024))
            tail = file.read().decode("utf-8", errors="replace")
        if "[Worker] Training completed:" in tail:
            return "completed"
        if "[Worker] FAILED:" in tail:
            return "failed"
    except OSError:
        pass
    if (path / "results.csv").is_file() and (path / "results.csv").stat().st_size > 0:
        return "completed"
    return "recovery_pending"


def iter_dataset_directories() -> list[Path]:
    paths = [
        path for path in DATASET_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ] if DATASET_DIR.exists() else []
    owners_root = DATASET_DIR / ".owners"
    if owners_root.is_dir():
        paths.extend(
            dataset
            for owner_root in owners_root.iterdir() if owner_root.is_dir()
            for dataset in owner_root.iterdir() if dataset.is_dir() and not dataset.name.startswith(".")
        )
    return sorted(paths)


def migrate_owner_scoped_storage(connection, valid_users: set[str], apply: bool, report: dict) -> None:
    rows = connection.execute(
        "select id, owner_user_id, slug, storage_path from datasets where status = 'active' and owner_user_id is not null"
    ).fetchall()
    for row in rows:
        owner = str(row["owner_user_id"])
        if owner not in valid_users:
            report["orphans"].append({"type": "dataset_registry", "path": row["storage_path"], "owner": owner})
            continue
        source = registered_storage_path(DATASET_DIR, row["storage_path"])
        destination = owner_dataset_path(DATASET_DIR, owner, str(row["slug"]))
        if source == destination:
            continue
        active = connection.execute(
            "select run_slug from training_tasks where dataset_id = %s and status = any(%s)",
            (row["id"], ["queued", "running", "started", "stopping", "recovery_pending"]),
        ).fetchall()
        if active:
            report.setdefault("deferred", []).append({"slug": row["slug"], "active_runs": [item["run_slug"] for item in active]})
            continue
        report.setdefault("moves", []).append({"slug": row["slug"], "from": str(source), "to": str(destination)})
        if not apply:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() and destination.exists():
            raise RuntimeError(f"Both legacy and owner-scoped storage exist for dataset '{row['slug']}'")
        moved = False
        if source.exists():
            source.replace(destination)
            moved = True
        elif not destination.exists():
            report["orphans"].append({"type": "dataset_missing", "path": str(source), "owner": owner})
            continue
        try:
            connection.execute("update datasets set storage_path = %s, updated_at = now() where id = %s", (str(destination), row["id"]))
            connection.commit()
        except Exception:
            connection.rollback()
            if moved and destination.exists() and not source.exists():
                destination.replace(source)
            raise
        for run in connection.execute("select storage_path from training_tasks where dataset_id = %s", (row["id"],)).fetchall():
            config_path = Path(run["storage_path"]) / "job_config.json"
            config = read_json(config_path)
            if not config:
                continue
            if config.get("source_dataset_path") == str(source):
                config["source_dataset_path"] = str(destination)
            if config.get("dataset_path") == str(source):
                config["dataset_path"] = str(destination)
            try:
                config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            except OSError:
                report.setdefault("warnings", []).append(f"Could not refresh historical run config at {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate and backfill AILAB filesystem resources")
    parser.add_argument("--apply", action="store_true", help="Write discovered resources to PostgreSQL")
    parser.add_argument("--migrate", action="store_true", help="Apply app_schema.sql first")
    args = parser.parse_args()
    if not resource_repository.enabled:
        raise SystemExit("DATABASE_URL and psycopg are required")

    if args.migrate:
        sql = (ROOT / "database" / "app_schema.sql").read_text(encoding="utf-8")
        with resource_repository._connect() as connection:
            connection.execute(sql)

    report = {"datasets": [], "runs": [], "orphans": []}
    with resource_repository._connect() as connection:
        user_table = connection.execute("select to_regclass('\"user\"') as name").fetchone()
        if not user_table or not user_table["name"]:
            print(json.dumps({**report, "warning": "Better Auth user table does not exist; run auth migration and rerun backfill."}, indent=2))
            return 0
        valid_users = {row["id"] for row in connection.execute('select id from "user"').fetchall()}
        migrate_owner_scoped_storage(connection, valid_users, args.apply, report)
        for path in iter_dataset_directories():
            meta = read_json(path / ".ailab_dataset.json")
            owner = str(meta.get("created_by") or "")
            if owner not in valid_users:
                report["orphans"].append({"type": "dataset", "path": str(path), "owner": owner or None})
                continue
            inspected = inspect_dataset(path)
            report["datasets"].append({"slug": path.name, "owner": owner, "path": str(path)})
            if args.apply:
                resource_repository.upsert_dataset(owner, meta.get("created_by_email"), path.name, path, inspected, connection)

        for path in sorted(RUNS_DIR.iterdir() if RUNS_DIR.exists() else []):
            if not path.is_dir() or path.name.startswith("."):
                continue
            config = read_json(path / "job_config.json")
            owner = str(config.get("created_by") or "")
            if owner not in valid_users:
                report["orphans"].append({"type": "run", "path": str(path), "owner": owner or None})
                continue
            report["runs"].append({"slug": path.name, "owner": owner, "dataset": config.get("dataset_name")})
            if args.apply:
                dataset = resource_repository.get_dataset(owner, str(config.get("dataset_name") or ""), connection)
                inferred_status = infer_run_status(path, config)
                connection.execute(
                    """
                    insert into training_tasks
                      (dataset_id, dataset_slug, rq_job_id, run_slug, display_name, task_type, model_type, model_name,
                       params, status, storage_path, created_by, owner_user_id)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    on conflict (owner_user_id, run_slug) where owner_user_id is not null do nothing
                    """,
                    (dataset.get("id") if dataset else None, config.get("dataset_name"), config.get("job_id"), path.name,
                     path.name.rsplit("_", 1)[0] if path.name.rsplit("_", 1)[-1].isdigit() else path.name,
                     config.get("task_type") or "unknown", config.get("model_type") or "unknown", config.get("model_name"),
                     json.dumps(config.get("extra_args") or {}), inferred_status, str(path),
                     config.get("created_by_email") or owner, owner),
                )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
