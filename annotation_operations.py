from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job

from annotation_service import annotation_service
from resource_repository import resource_repository
from security_utils import contained_path, safe_extract_zip, staging_directory
from settings import DATASET_DIR, REDIS_URL


QUEUE_NAME = "annotation"
def _dict(row: Any) -> dict[str, Any]:
    result = dict(row)
    result["id"] = str(result["id"])
    if result.get("project_id"):
        result["project_id"] = str(result["project_id"])
    result["result"] = result.get("result") if isinstance(result.get("result"), dict) else json.loads(result.get("result") or "{}")
    for key in ("created_at", "updated_at", "started_at", "finished_at"):
        if result.get(key):
            result[key] = result[key].isoformat()
    result.pop("upload_path", None)
    return result


class AnnotationOperations:
    def _connect(self):
        return resource_repository._connect()

    def create(self, owner_id: str, project_id: str, kind: str, upload_path: Path | None = None) -> dict[str, Any]:
        if kind not in {"import", "publish"}:
            raise ValueError("Unsupported annotation operation")
        with self._connect() as connection:
            with connection.transaction():
                project = connection.execute(
                    "select id from annotation_projects where owner_user_id = %s and id = %s for update",
                    (owner_id, project_id),
                ).fetchone()
                if not project:
                    raise FileNotFoundError("Annotation project was not found")
                active = connection.execute(
                    """select id from annotation_operations
                       where project_id = %s and status in ('queued', 'running') for update""",
                    (project_id,),
                ).fetchone()
                if active:
                    raise FileExistsError("This annotation project already has an active operation")
                row = connection.execute(
                    """insert into annotation_operations
                       (owner_user_id, project_id, kind, upload_path)
                       values (%s, %s, %s, %s) returning *""",
                    (owner_id, project_id, kind, str(upload_path) if upload_path else None),
                ).fetchone()
        operation_id = str(row["id"])
        self.enqueue(operation_id)
        return self.get(owner_id, operation_id) or _dict(row)

    def enqueue(self, operation_id: str) -> bool:
        try:
            queue = Queue(QUEUE_NAME, connection=Redis.from_url(REDIS_URL))
            job = queue.enqueue(
                "annotation_operations.run_operation",
                operation_id,
                job_timeout=24 * 60 * 60,
                result_ttl=24 * 60 * 60,
                failure_ttl=7 * 24 * 60 * 60,
            )
            with self._connect() as connection:
                connection.execute(
                    "update annotation_operations set rq_job_id = %s, error_detail = null, updated_at = now() where id = %s and status = 'queued'",
                    (job.id, operation_id),
                )
            return True
        except Exception as exc:
            with self._connect() as connection:
                connection.execute(
                    """update annotation_operations
                       set error_detail = %s, updated_at = now()
                       where id = %s and status = 'queued'""",
                    (f"Waiting for annotation queue: {str(exc)[:300]}", operation_id),
                )
            return False

    def get(self, owner_id: str, operation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "select * from annotation_operations where owner_user_id = %s and id = %s",
                (owner_id, operation_id),
            ).fetchone()
        return _dict(row) if row else None

    def list_active(self, owner_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """select * from annotation_operations
                   where owner_user_id = %s
                   order by created_at desc limit 50""",
                (owner_id,),
            ).fetchall()
        return [_dict(row) for row in rows]

    def retry(self, owner_id: str, operation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            current = connection.execute(
                "select * from annotation_operations where owner_user_id = %s and id = %s",
                (owner_id, operation_id),
            ).fetchone()
            if not current or current["status"] not in {"failed", "cancelled", "queued"}:
                raise FileExistsError("Only failed, cancelled, or waiting operations can be retried")
            if current["kind"] == "import" and (
                not current["upload_path"] or not Path(current["upload_path"]).is_file()
            ):
                raise FileNotFoundError("The uploaded ZIP is no longer available; upload it again")
            row = connection.execute(
                """update annotation_operations set status = 'queued', progress = 0, processed_count = 0,
                          skipped_count = 0, cancel_requested = false, error_detail = null,
                          started_at = null, finished_at = null, updated_at = now()
                   where owner_user_id = %s and id = %s and status in ('failed', 'cancelled', 'queued')
                   returning *""",
                (owner_id, operation_id),
            ).fetchone()
        self.enqueue(operation_id)
        return self.get(owner_id, operation_id) or _dict(row)

    def cancel(self, owner_id: str, operation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            with connection.transaction():
                row = connection.execute(
                    """select * from annotation_operations
                       where owner_user_id = %s and id = %s for update""",
                    (owner_id, operation_id),
                ).fetchone()
                if not row:
                    raise FileNotFoundError("Annotation operation was not found")
                if row["status"] not in {"queued", "running"}:
                    return _dict(row)
                status = "cancelled" if row["status"] == "queued" else "running"
                row = connection.execute(
                    """update annotation_operations set status = %s, cancel_requested = true,
                              finished_at = case when %s = 'cancelled' then now() else finished_at end,
                              updated_at = now() where id = %s returning *""",
                    (status, status, operation_id),
                ).fetchone()
        if row.get("rq_job_id"):
            try:
                Job.fetch(row["rq_job_id"], connection=Redis.from_url(REDIS_URL)).cancel()
            except Exception:
                pass
        if row["status"] == "cancelled" and row.get("upload_path"):
            Path(row["upload_path"]).unlink(missing_ok=True)
        return _dict(row)

    def is_cancelled(self, operation_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "select cancel_requested from annotation_operations where id = %s",
                (operation_id,),
            ).fetchone()
        return not row or bool(row["cancel_requested"])

    def progress(self, operation_id: str, processed: int, total: int, skipped: int) -> None:
        percent = min(99, round(processed * 100 / total)) if total else 0
        with self._connect() as connection:
            connection.execute(
                """update annotation_operations set progress = %s, processed_count = %s,
                          total_count = %s, skipped_count = %s, updated_at = now()
                   where id = %s and status = 'running'""",
                (percent, processed, total, skipped, operation_id),
            )

    def recover(self) -> int:
        with self._connect() as connection:
            with connection.transaction():
                connection.execute(
                    """update annotation_operations set status = 'queued', started_at = null,
                              error_detail = 'Recovered after annotation worker restart', updated_at = now()
                       where status = 'running'""",
                )
                rows = connection.execute(
                    "select id from annotation_operations where status = 'queued' order by created_at"
                ).fetchall()
        for row in rows:
            self.enqueue(str(row["id"]))
        return len(rows)


annotation_operations = AnnotationOperations()


def run_operation(operation_id: str) -> dict[str, Any]:
    with annotation_operations._connect() as connection:
        row = connection.execute(
            """update annotation_operations set status = 'running', started_at = now(),
                      attempt = attempt + 1, error_detail = null, updated_at = now()
               where id = %s and status = 'queued' and not cancel_requested returning *""",
            (operation_id,),
        ).fetchone()
    if not row:
        return {"status": "ignored"}

    operation = dict(row)
    upload_path = Path(operation["upload_path"]) if operation.get("upload_path") else None
    extract_stage: Path | None = None
    try:
        callback = lambda processed, total, skipped: annotation_operations.progress(
            operation_id, processed, total, skipped
        )
        cancelled = lambda: annotation_operations.is_cancelled(operation_id)
        if operation["kind"] == "import":
            if not upload_path or not upload_path.is_file():
                raise FileNotFoundError("Uploaded ZIP is no longer available")
            extract_stage = staging_directory(contained_path(DATASET_DIR, ".annotation-work"))
            extract_dir = contained_path(extract_stage, "extracted")
            with zipfile.ZipFile(upload_path, "r") as archive:
                safe_extract_zip(archive, extract_dir)
            result = annotation_service.import_directory(
                operation["owner_user_id"], str(operation["project_id"]), operation_id,
                extract_dir, callback, cancelled,
            )
        else:
            with annotation_operations._connect() as connection:
                owner = connection.execute(
                    'select email from "user" where id = %s',
                    (operation["owner_user_id"],),
                ).fetchone()
            result = annotation_service.publish(
                operation["owner_user_id"], owner["email"] if owner else None,
                str(operation["project_id"]), callback, cancelled
            )
        # Service return means files and DB rows have committed. Late cancellation
        # must not report that committed work was discarded.
        with annotation_operations._connect() as connection:
            connection.execute(
                """update annotation_operations set status = 'completed', progress = 100,
                          result = %s::jsonb, error_detail = null, finished_at = now(), updated_at = now()
                   where id = %s""",
                (json.dumps(result), operation_id),
            )
        if upload_path:
            upload_path.unlink(missing_ok=True)
        return result
    except InterruptedError as exc:
        with annotation_operations._connect() as connection:
            connection.execute(
                """update annotation_operations set status = 'cancelled', error_detail = %s,
                          finished_at = now(), updated_at = now() where id = %s""",
                (str(exc), operation_id),
            )
        if upload_path:
            upload_path.unlink(missing_ok=True)
        return {"status": "cancelled"}
    except Exception as exc:
        with annotation_operations._connect() as connection:
            connection.execute(
                """update annotation_operations set status = 'failed', error_detail = %s,
                          finished_at = now(), updated_at = now() where id = %s""",
                (str(exc)[:2000], operation_id),
            )
        raise
    finally:
        if extract_stage:
            shutil.rmtree(extract_stage, ignore_errors=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "recover":
    count = annotation_operations.recover()
    print(f"Recovered or queued {count} annotation operations")
