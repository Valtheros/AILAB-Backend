# AILAB Backend

FastAPI backend for AILAB, a no-code Computer Vision training platform. It
accepts uploaded datasets, detects supported formats, queues training jobs with
Redis/RQ, streams progress, and serves trained artifacts back to the frontend.

## What It Does

- Upload and validate dataset ZIP files.
- Expose the model catalog used by the UI.
- Queue training jobs for the Computer Vision worker.
- Stream live logs, metrics, status, and heartbeats through SSE.
- Store run outputs such as logs, metrics, weights, and reports.
- Protect user-owned datasets and runs when requests include authenticated user
  headers from the frontend proxy.

## Supported Tasks

| Task | Models | Dataset formats |
| --- | --- | --- |
| Image Classification | ResNet, EfficientNet | ImageFolder |
| Segmentation | DeepLabV3+, Mask R-CNN | semantic masks, COCO instances |
| Object Detection | YOLOv11, Faster R-CNN | YOLO detection |

## Run Locally

Docker Compose is the recommended local runtime because the API depends on
Redis, PostgreSQL, and worker services.

```powershell
docker compose up --build
```

Useful URLs:

- API root: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Model catalog: `http://localhost:8000/api/model-catalog`
- RQ dashboard: `http://localhost:9181`

## Environment

Copy `.env.example` and adjust values when needed.

```text
APP_BASE_DIR=/app
STORAGE_DIR=/app/storage
DATASET_DIR=/app/dataset
RUNS_DIR=/app/runs
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://ailab:ailab_dev_password@postgres:5432/ailab
BACKEND_INTERNAL_TOKEN=
CORS_ORIGINS=http://localhost:3000
```

Set the same `BACKEND_INTERNAL_TOKEN` in the frontend and backend for
production-like deployments so browser traffic goes through the authenticated
Next.js proxy instead of calling FastAPI directly.

## Important Routes

| Route | Purpose |
| --- | --- |
| `GET /api/model-catalog` | Tasks, models, and parameter specs |
| `POST /api/upload-dataset` | Upload a dataset ZIP |
| `GET /api/datasets` | List visible datasets from the PostgreSQL resource registry |
| `POST /api/datasets/inspect-upload` | Upload, validate, and stage a ZIP once |
| `POST /api/datasets/import` | Commit a staged upload token |
| `POST /api/train` | Queue a training job |
| `GET /api/compute/devices` | GPU and CPU worker availability and coordinator status |
| `GET /api/jobs/{job_id}/events` | Live SSE updates |
| `GET /api/status/{job_id}` | Job snapshot |
| `GET /api/logs/{job_id}` | Training logs |
| `GET /api/metrics/{project_name}` | Metrics rows |
| `GET /api/runs` | Run artifacts |

Compute readiness is derived from worker and coordinator heartbeats; a busy GPU
may still accept queued jobs. The `insights` fields in metrics and comparison
responses remain empty for compatibility: the UI must not infer model quality
from hardcoded heuristics. Label Studio class names are displayed from dataset
metadata while the existing training folder layout stays unchanged.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m py_compile main.py security_utils.py sse_utils.py model_catalog.py dataset_utils.py services\training_service.py worker\worker_app.py worker\security_utils.py worker\trainers\*.py
```

Keep Redis internal to Docker Compose and bind local-only services to
`127.0.0.1` unless the stack is behind proper authentication and network
controls.

## Dataset Versions And Evaluation

The backend owns dataset collections, immutable version records, manifests,
image indexes, evaluations, compute jobs and notifications. The frontend owns
the views; Label Studio owns its own projects and source uploads. The root
repository owns deployment configuration only.

`database/dataset_versions_schema.sql`, `dataset_index_schema.sql`,
`evaluations_schema.sql`, `notifications_schema.sql` and `model_api_schema.sql`
are additive and are applied by `python -m scripts.apply_schema`. Back up the
database and media before deployment. To inspect the legacy version backfill,
run `python -m scripts.backfill_dataset_versions` first; add `--apply` only
after reviewing the output. It keeps existing dataset IDs and paths and does
not claim the contents of historical runs were verified.

`database/storage_schema.sql` and `storage_reservations.py` provide an
inactive foundation for per-owner storage accounting. Project IDs are unique
and never reused by the database sequence. Reservations are serialized across
users, retain the full estimated amount until all writing stops, and protect
against reserving the final 20 GiB of filesystem free space. This reserve is
not guaranteed until every writer is bounded by its reservation. A reservation
is not a filesystem quota:
upload handlers, Label Studio, worker outputs, account transfers and cleanup
are not wired to this ledger yet. Accounts with a storage record cannot be
transferred or deleted until project-quota cleanup is implemented; this avoids
discarding a quota identity or deleting external files prematurely.
The ownership inventory and production rollout gates are documented in
[`docs/storage-quota.md`](docs/storage-quota.md).

Run `python -B -m scripts.storage_quota_preflight --root <dataset-dir> --root
<runs-dir> --root <label-studio-data-dir>` to inspect existing storage roots.
It is read-only, requires permission to inspect the backing device and project
quota, and distinguishes filesystem readiness from application enforcement.
All user-owned roots must be covered by the same quota accounting, including
temporary uploads, Label Studio media, exports and worker-generated files.
Native Label Studio uploads and training outputs cannot be assumed to pass
through an AILAB upload endpoint. Legacy Annotation projects and their source
files may still exist even when Label Studio is active; inventory their owners
and retain their data during backfill. An ext4 project-inheriting directory also
rejects renames across different project IDs, so imports and transfers need
owner-scoped staging and recoverable cross-owner copies.

`tests/test_filesystem_project_quota.py` is an opt-in kernel integration test:
it formats and mounts only a temporary image under `/home/trainai`, then checks
quota enforcement, cross-owner rename and preserving pre-existing data above a
new limit. It requires a disposable loop device and mount permissions; it must
not be pointed at a production block device. The current root filesystem has
not been migrated. Do not enable project quota on it until a tested backup,
an offline maintenance plan, complete writer integration and recovery tests are
in place.

Integration tests with `COMPUTE_TESTS=1` truncate the dedicated test database.
They require the loopback `ailab_test` user and `ailab_compute_test_pg`
database, and the Model API tests require loopback Redis database 15. Set
`AILAB_TEST_ROOT` to a disposable directory containing a
`.ailab-disposable-test` file with the text `ailab-disposable-test-v1`; set
`RUNS_DIR`, `DATASET_DIR`, `STORAGE_DIR`, `TMPDIR`, and, for API tests,
`APP_BASE_DIR` to subdirectories under that root. Never point this test stack
or its volumes at production services or media.

Imports reserve a collection version under the owner lock. Source manifests
hash source files, classes and annotation schema, excluding only generated
AILAB cache directories. Dataset indexing uses the CPU data queue and stores
generated thumbnails under `.ailab_index/<generation>`; tracked unfinished
generations are reclaimed after a successful retry. Source images and labels
are never edited by indexing. Dataset deletion and ownership transfer must
wait for active indexes, training and evaluations. Account removal and transfer
also stop when a dataset import is unfinished; restore the account and retry
that import before proceeding rather than moving its partially committed files.

Completed compute-based training creates one automatic test evaluation;
manual evaluations may target another compatible dataset version. Evaluation
checks the manifest and checkpoint before and after inference, maps class names explicitly, and never
substitutes validation for a missing test split. Results and per-image records
are retained under the run until the owner deletes the evaluation or run.
New runs remain `unverified` until their worker has checked the source manifest
both before and after successful training; historical runs are not retroactively
verified. YOLO training rebuilds its label index in memory and neither reads nor
writes Ultralytics pickle caches from uploaded datasets.
Temporary compute-job files, including batch inputs and outputs, expire 24
hours after a job finishes. Notifications are owner-scoped and expire after
90 days; status changes and events are committed together.

`tests/test_real_classification_smoke.py` and `tests/test_real_vision_smoke.py`
run real CPU training, evaluation and batch prediction when
`AILAB_REAL_MODEL_TEST=1` is set in the compute image. They use generated data;
they do not replace a CUDA or production workflow test. The read-only
`tests/load/read-only.js` k6 script requires an isolated loopback API, a test
user and internal token. Its small default load is a regression check, not a
capacity or SLA measurement.

The public model API implementation is **not enabled for deployment**:
`STORAGE_ENFORCEMENT_IMPLEMENTED` remains false until the storage quota is
enforced for every upload and worker path, including Label Studio. Setting
`MODEL_API_ENABLED` alone cannot bypass that guard. Once storage enforcement,
security tests and rollout checks are complete, model API keys are scoped to
one endpoint, stored only as hashes, and issued with a 90-day expiry. Each job
is scoped to the key that created it; checkpoint hashes are checked before
and after execution. Disable an endpoint before deleting its run; ownership
transfer disables endpoints and revokes their existing keys.

Rollback requires stopping new work, draining affected workers, restoring the
previous application images and retaining the additive tables and existing
media. Do not roll back by dropping tables or deleting source datasets. New
features must remain disabled until their migration, quotas, ownership tests,
and end-to-end GPU verification have passed.
