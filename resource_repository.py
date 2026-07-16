from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from settings import DATABASE_URL, DATASET_DIR
from dataset_storage import owner_dataset_path, registered_storage_path

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # Local unit tests may run without production dependencies.
    psycopg = None
    dict_row = None


ACTIVE_RUN_STATUSES = ("queued", "running", "started", "stopping", "recovery_pending")


class ResourceRepository:
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url

    @property
    def enabled(self) -> bool:
        return bool(self.database_url and psycopg is not None)

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("DATABASE_URL and psycopg are required for the resource registry")
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _lock_key(value: str) -> int:
        return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big", signed=True)

    @contextmanager
    def dataset_guard(self, owner_id: str, slug: str) -> Iterator[Any | None]:
        if not self.enabled:
            yield None
            return
        with self._connect() as connection:
            with connection.transaction():
                connection.execute("select pg_advisory_xact_lock(%s)", (self._lock_key(f"user-resources:{owner_id}"),))
                connection.execute("select pg_advisory_xact_lock(%s)", (self._lock_key(f"dataset:{owner_id}:{slug}"),))
                yield connection

    def get_dataset(self, owner_id: str, slug: str, connection=None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        owned = connection is None
        connection = connection or self._connect()
        try:
            row = connection.execute(
                "select * from datasets where owner_user_id = %s and slug = %s and status = 'active' limit 1",
                (owner_id, slug),
            ).fetchone()
            return dict(row) if row else None
        finally:
            if owned:
                connection.close()

    def list_datasets(self, owner_id: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "select * from datasets where owner_user_id = %s and status = 'active' order by created_at desc",
                (owner_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def active_runs_for_dataset(self, dataset_id: Any, connection=None) -> list[dict[str, Any]]:
        if not self.enabled or dataset_id is None:
            return []
        owned = connection is None
        connection = connection or self._connect()
        try:
            rows = connection.execute(
                "select id, run_slug, rq_job_id, status from training_runs where dataset_id = %s and status = any(%s)",
                (dataset_id, list(ACTIVE_RUN_STATUSES)),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            if owned:
                connection.close()

    def upsert_dataset(self, owner_id: str, owner_email: str | None, slug: str, path: Path, metadata: dict[str, Any], connection=None) -> dict[str, Any]:
        if not self.enabled:
            return {"id": slug, "slug": slug, "owner_user_id": owner_id, "storage_path": str(path)}
        owned = connection is None
        connection = connection or self._connect()
        try:
            row = connection.execute(
                """
                insert into datasets (name, slug, storage_path, formats, tasks, classes, created_by, owner_user_id, status, metadata)
                values (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, 'active', %s::jsonb)
                on conflict (owner_user_id, slug) where owner_user_id is not null do update set
                  name = excluded.name, storage_path = excluded.storage_path, formats = excluded.formats,
                  tasks = excluded.tasks, classes = excluded.classes, status = 'active', metadata = excluded.metadata,
                  updated_at = now()
                returning *
                """,
                (slug, slug, str(path), json.dumps(metadata.get("formats", [])), json.dumps(metadata.get("tasks", [])),
                 json.dumps(metadata.get("classes", [])), owner_email or owner_id, owner_id, json.dumps(metadata)),
            ).fetchone()
            if owned:
                connection.commit()
            return dict(row)
        finally:
            if owned:
                connection.close()

    def update_dataset_metadata(self, dataset_id: Any, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute(
                """
                update datasets set formats = %s::jsonb, tasks = %s::jsonb, classes = %s::jsonb,
                  metadata = %s::jsonb, updated_at = now() where id = %s
                """,
                (
                    json.dumps(metadata.get("formats", [])),
                    json.dumps(metadata.get("tasks", [])),
                    json.dumps(metadata.get("classes", [])),
                    json.dumps(metadata),
                    dataset_id,
                ),
            )

    def mark_dataset_deleted(self, dataset_id: Any, connection=None) -> None:
        if not self.enabled:
            return
        owned = connection is None
        connection = connection or self._connect()
        try:
            connection.execute("update datasets set status = 'deleted', updated_at = now() where id = %s", (dataset_id,))
            if owned:
                connection.commit()
        finally:
            if owned:
                connection.close()

    def create_run(self, *, owner_id: str, owner_email: str | None, dataset: dict[str, Any], project_name: str,
                   task_type: str, model_type: str, model_name: str, params: dict[str, Any], storage_path: Path,
                   job_id: str | None = None, connection=None) -> dict[str, Any]:
        if not self.enabled:
            return {"id": project_name, "run_slug": project_name}
        owned = connection is None
        connection = connection or self._connect()
        try:
            row = connection.execute(
                """
                insert into training_runs
                  (dataset_id, dataset_slug, rq_job_id, run_slug, task_type, model_type, model_name, params, status,
                   storage_path, created_by, owner_user_id)
                values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'queued', %s, %s, %s)
                returning *
                """,
                (dataset.get("id"), dataset.get("slug"), job_id, project_name, task_type, model_type, model_name,
                 json.dumps(params), str(storage_path), owner_email or owner_id, owner_id),
            ).fetchone()
            if owned:
                connection.commit()
            return dict(row)
        finally:
            if owned:
                connection.close()

    def set_run_job(self, run_id: Any, job_id: str, connection=None) -> None:
        if not self.enabled:
            return
        if connection is not None:
            connection.execute("update training_runs set rq_job_id = %s, updated_at = now() where id = %s", (job_id, run_id))
            return
        with self._connect() as owned_connection:
            owned_connection.execute("update training_runs set rq_job_id = %s, updated_at = now() where id = %s", (job_id, run_id))

    def get_run_owner_by_slug(self, run_slug: str) -> str | None:
        if not self.enabled:
            return None
        with self._connect() as connection:
            row = connection.execute("select owner_user_id from training_runs where run_slug = %s limit 1", (run_slug,)).fetchone()
            return str(row["owner_user_id"]) if row and row["owner_user_id"] else None

    def get_run_owner_by_job(self, job_id: str) -> str | None:
        if not self.enabled:
            return None
        with self._connect() as connection:
            row = connection.execute("select owner_user_id from training_runs where rq_job_id = %s limit 1", (job_id,)).fetchone()
            return str(row["owner_user_id"]) if row and row["owner_user_id"] else None

    def list_runs(self, owner_id: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "select * from training_runs where owner_user_id = %s order by created_at desc", (owner_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def list_queued_runs(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "select id, rq_job_id, run_slug from training_runs where status = 'queued'"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_run(self, owner_id: str, run_slug: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "select * from training_runs where owner_user_id = %s and run_slug = %s limit 1", (owner_id, run_slug)
            ).fetchone()
            return dict(row) if row else None

    def delete_run_by_slug(self, owner_id: str, run_slug: str) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute("delete from training_runs where owner_user_id = %s and run_slug = %s", (owner_id, run_slug))

    def delete_run_record(self, run_id: Any) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute("delete from training_runs where id = %s", (run_id,))

    def update_run_status(self, job_id: str, status: str, error_detail: str | None = None) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute(
                "update training_runs set status = %s, error_detail = %s, updated_at = now(), finished_at = case when %s = any(%s) then now() else finished_at end where rq_job_id = %s",
                (status, error_detail, status, ["completed", "failed", "stopped", "cancelled"], job_id),
            )

    def update_run_status_by_id(self, run_id: Any, status: str, error_detail: str | None = None) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute(
                "update training_runs set status = %s, error_detail = %s, updated_at = now(), "
                "finished_at = case when %s = any(%s) then now() else finished_at end where id = %s",
                (status, error_detail, status, ["completed", "failed", "stopped", "cancelled"], run_id),
            )

    def assert_admin(self, user_id: str) -> None:
        if not self.enabled:
            raise PermissionError("Resource database is unavailable")
        with self._connect() as connection:
            row = connection.execute('select role from "user" where id = %s', (user_id,)).fetchone()
            if not row or row["role"] != "admin":
                raise PermissionError("System admin role is required")

    def user_impact(self, user_id: str) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Resource database is unavailable")
        with self._connect() as connection:
            workspace_memberships = connection.execute(
                "select count(*)::int as count from workspace_members where user_id = %s", (user_id,)
            ).fetchone()["count"]
            projects = connection.execute(
                "select count(*)::int as count from projects where created_by in (%s, (select email from \"user\" where id = %s))",
                (user_id, user_id),
            ).fetchone()["count"]
            datasets = connection.execute(
                "select id, slug, storage_path, status from datasets where owner_user_id = %s", (user_id,)
            ).fetchall()
            runs = connection.execute(
                "select id, run_slug, rq_job_id, status, storage_path from training_runs where owner_user_id = %s", (user_id,)
            ).fetchall()
        paths = {str(row["storage_path"]) for row in [*datasets, *runs]}
        total_bytes = 0
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                total_bytes += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        active = [dict(row) for row in runs if row["status"] in ACTIVE_RUN_STATUSES]
        deleted_datasets = sum(1 for row in datasets if row["status"] == "deleted")
        return {
            "workspaceMemberships": workspace_memberships,
            "projects": projects,
            "datasets": len(datasets),
            "deletedDatasets": deleted_datasets,
            "runs": len(runs),
            "activeJobs": active,
            "totalBytes": total_bytes,
        }

    def transfer_user_resources(self, source_user_id: str, target_user_id: str) -> dict[str, int]:
        if source_user_id == target_user_id:
            raise ValueError("Source and target users must be different")
        moves: list[tuple[Path, Path]] = []
        try:
            with self._connect() as connection:
                for user_id in sorted((source_user_id, target_user_id)):
                    connection.execute("select pg_advisory_xact_lock(%s)", (self._lock_key(f"user-resources:{user_id}"),))
                target = connection.execute('select id, email from "user" where id = %s', (target_user_id,)).fetchone()
                if not target:
                    raise FileNotFoundError("Transfer target user was not found")
                conflict = connection.execute(
                    "select d.slug from datasets d join datasets target on target.owner_user_id = %s and target.slug = d.slug where d.owner_user_id = %s limit 1",
                    (target_user_id, source_user_id),
                ).fetchone()
                if conflict:
                    raise ValueError(f"Target user already has dataset '{conflict['slug']}'")
                datasets = connection.execute(
                    "select id, slug, storage_path, status from datasets where owner_user_id = %s for update",
                    (source_user_id,),
                ).fetchall()
                active = connection.execute(
                    "select run_slug from training_runs where owner_user_id = %s and status = any(%s) limit 1",
                    (source_user_id, list(ACTIVE_RUN_STATUSES)),
                ).fetchone()
                if active:
                    raise ValueError(f"Stop active training run '{active['run_slug']}' before transferring ownership")
                for row in datasets:
                    source = registered_storage_path(DATASET_DIR, row["storage_path"])
                    destination = owner_dataset_path(DATASET_DIR, target_user_id, row["slug"])
                    if row["status"] != "deleted" and source != destination:
                        if not source.is_dir():
                            raise FileNotFoundError(f"Dataset storage for '{row['slug']}' was not found")
                        if destination.exists():
                            raise ValueError(f"Target storage already contains dataset '{row['slug']}'")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source.replace(destination)
                        moves.append((source, destination))
                    connection.execute(
                        "update datasets set owner_user_id = %s, created_by = %s, storage_path = %s, updated_at = now() where id = %s",
                        (target_user_id, target["email"], str(destination), row["id"]),
                    )
                runs = connection.execute(
                    "update training_runs set owner_user_id = %s, created_by = %s, updated_at = now() where owner_user_id = %s returning storage_path",
                    (target_user_id, target["email"], source_user_id),
                ).fetchall()
                projects = connection.execute(
                    "update projects set created_by = %s, updated_at = now() where created_by in (%s, (select email from \"user\" where id = %s)) returning id",
                    (target_user_id, source_user_id, source_user_id),
                ).fetchall()
                memberships = connection.execute("select workspace_id, role from workspace_members where user_id = %s", (source_user_id,)).fetchall()
                for membership in memberships:
                    connection.execute(
                        "insert into workspace_members (workspace_id, user_id, role) values (%s, %s, %s) on conflict (workspace_id, user_id) do nothing",
                        (membership["workspace_id"], target_user_id, membership["role"]),
                    )
                connection.execute("delete from workspace_members where user_id = %s", (source_user_id,))
        except Exception:
            for source, destination in reversed(moves):
                if destination.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            raise
        destination_by_slug = {row["slug"]: owner_dataset_path(DATASET_DIR, target_user_id, row["slug"]) for row in datasets}
        for row in datasets:
            metadata_path = destination_by_slug[row["slug"]] / ".ailab_dataset.json"
            try:
                data = {}
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                data.update({"created_by": target_user_id, "created_by_email": target["email"]})
                metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                continue
        for row in runs:
            config_path = Path(row["storage_path"]) / "job_config.json"
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                data.update({"created_by": target_user_id, "created_by_email": target["email"]})
                dataset_slug = str(data.get("dataset_name") or "")
                if dataset_slug in destination_by_slug:
                    old_source = str(data.get("source_dataset_path") or "")
                    new_source = str(destination_by_slug[dataset_slug])
                    data["source_dataset_path"] = new_source
                    if str(data.get("dataset_path") or "") == old_source:
                        data["dataset_path"] = new_source
                config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                continue
        return {"datasets": len(datasets), "runs": len(runs), "projects": len(projects), "workspaceMemberships": len(memberships)}

    def delete_user_resource_records(self, user_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._connect() as connection:
            connection.execute("select pg_advisory_xact_lock(%s)", (self._lock_key(f"user-resources:{user_id}"),))
            datasets = [dict(row) for row in connection.execute(
                "select * from datasets where owner_user_id = %s for update", (user_id,)
            ).fetchall()]
            connection.execute(
                "update datasets set status = 'deleting', updated_at = now() where owner_user_id = %s and status <> 'deleted' returning *", (user_id,)
            ).fetchall()
            runs = [dict(row) for row in connection.execute(
                "update training_runs set status = case when status = any(%s) then 'stopping' else status end, updated_at = now() where owner_user_id = %s returning *",
                (list(ACTIVE_RUN_STATUSES), user_id),
            ).fetchall()]
        return datasets, runs

    def finalize_user_resource_delete(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute("delete from training_runs where owner_user_id = %s", (user_id,))
            connection.execute("delete from datasets where owner_user_id = %s", (user_id,))
            connection.execute("delete from projects where created_by in (%s, (select email from \"user\" where id = %s))", (user_id, user_id))
            connection.execute("delete from workspace_members where user_id = %s", (user_id,))


resource_repository = ResourceRepository()
