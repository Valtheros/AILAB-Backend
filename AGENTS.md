# AILAB Backend Agent Guide

## Scope

This repository contains the AILAB FastAPI API, PostgreSQL resource registry,
dataset inspection/import logic, Redis/RQ job orchestration, CV/OCR workers, and
trainer implementations.

The deployment repository is `/home/trainai/AILAB`. The frontend is a separate
Git repository at `/home/trainai/AILAB/frontend`. Run Git commands from this
directory; do not assume the parent repository tracks backend changes.

## Runtime Architecture

- `backend`: FastAPI API; validates trusted identity, owns metadata and enqueue
- `postgres`: source of truth for dataset/run ownership, paths, and status
- `redis`: RQ queues and job state
- `worker`: `cv_training` queue for classification, detection, segmentation
- `ocr-worker`: `ocr_training` queue for PaddleOCR/Tesseract
- host filesystem: dataset/run bytes mounted into API and workers

Production mounts are configured in `/home/trainai/AILAB/.env.production` and
`docker-compose.prod.yml`. Defaults are:

- `/data/ailab/dataset` -> `/app/dataset`
- `/data/ailab/runs` -> `/app/runs`
- `/data/ailab/storage` -> `/app/storage`

Files may appear hidden under `.owners`; use `ls -la`. Never assume runtime
data lives inside the Git checkout or Docker writable layer.

## Sources Of Truth

- PostgreSQL `datasets` and `training_runs` rows decide ownership, visibility,
  stable IDs, storage paths, and lifecycle status.
- Filesystem metadata such as `.ailab_dataset.json` and `job_config.json` is for
  workers/recovery, not authorization.
- Redis is queue transport, not durable ownership storage.
- `model_catalog.py` owns supported model/task/parameter contracts.
- `settings.py` owns resolved runtime roots and connection settings.

Do not list resources by scanning directories and do not reconstruct an
existing path from a user-provided name. Resolve the DB row, verify owner, then
validate its registered path with containment helpers.

## Request Trust Boundary

Browser traffic should pass through the authenticated Next.js proxy. FastAPI
accepts identity headers only when the request also carries the matching
`BACKEND_INTERNAL_TOKEN`.

- Normal users may access only their own dataset/run rows.
- Admin endpoints require the internal token and a fresh database role check.
- Missing/invalid auth must fail closed in production.
- Never weaken containment, archive, image, label, or resource-budget checks to
  make malformed input pass.
- Keep Redis/PostgreSQL/backend ports private or loopback-bound behind the web
  proxy.

## Main Files

- `main.py`: routes, auth dependency, staged upload/import, deletion, SSE/admin
- `settings.py`: directories, Redis/PostgreSQL URLs, internal token, CORS
- `resource_repository.py`: PostgreSQL dataset/run repository and locks
- `dataset_storage.py`: owner-scoped paths and registered-path validation
- `staged_uploads.py`: one-time user-bound upload tokens, expiry, cleanup
- `dataset_utils.py`: lightweight upload inspection, metadata, train-time export
- `security_utils.py`: ZIP/path/file/image safety primitives
- `resource_guard.py`: hardware profile and OOM-safe training limits
- `sse_utils.py`: bounded log tail/chunk handling
- `model_catalog.py`: model catalog and parameter schemas
- `services/training_service.py`: run reservation, RQ enqueue/stop/status/results
- `database/app_schema.sql`: idempotent application schema/migrations
- `scripts/backfill_resources.py`: dry-run/apply migration of legacy resources
- `worker/worker_app.py`: RQ training entry point and lifecycle updates
- `worker/trainers/`: trainer implementations and shared datasets/limits
- `tests/`: stdlib `unittest` regression suite

## Dataset Upload And Storage

Current staged flow:

1. `POST /api/datasets/inspect-upload` streams one ZIP to owner-bound staging,
   performs bounded structural sampling, and returns a random token plus expiry.
2. `POST /api/datasets/import` consumes the token once, re-checks owner/expiry/
   fingerprint, acquires the owner+dataset lock, atomically commits storage, and
   writes the PostgreSQL row.
3. Model-specific exports are prepared lazily when training starts.

Rules:

- A user may have at most three staged uploads; tokens expire after 30 minutes.
- Different users may use the same dataset name. Physical paths are scoped as
  `.owners/<owner-hash>/<dataset-slug>`.
- Upload inspection samples enough files to identify a supported structure; it
  must not fully decode every large dataset. Full validation happens at train
  time, where bad samples are skipped or reported according to trainer rules.
- Unsupported structure, unsafe ZIP/path/symlink/compression behavior, or hard
  resource-budget violations still fail upload.
- Keep compatibility route `/api/upload-dataset` unless its callers are removed.

## Concurrency And Lifecycle Invariants

- Dataset enqueue, overwrite, and delete share the PostgreSQL advisory lock for
  that dataset.
- Reject overwrite/delete with `409` while a run is `queued`, `started`, or
  `stopping` and references the dataset.
- Create the `queued` training row inside the lock transaction before enqueue;
  this closes the delete/enqueue race.
- Reserve `project_name` globally before enqueue. Never let two users share a
  run directory.
- Use OS `flock` for process locks; process death must release the lock.
- Export cache writes to a temporary directory and becomes visible only through
  atomic rename after a complete manifest.
- Stop updates durable status and requests RQ cancellation. UI navigation must
  not be treated as cancellation.
- Account deletion order is disable/revoke, stop active jobs, remove files and
  resource rows, then delete the auth user. Partial cleanup stays retryable.
- Foreign keys use restrictive ownership semantics; do not cascade auth-user
  deletion over retained training data.

## Validation And Worker Rules

- Shared limits live in `worker/trainers/input_limits.py`; keep upload/worker
  behavior aligned where they validate the same format.
- Bound source pixels before image/mask decode, text/label bytes and line count,
  COCO JSON size, images, annotations, polygons, cumulative points, RLE counts,
  boxes, and masks per image.
- Reject NaN/Infinity and malformed coordinates before tensor conversion.
- Classification validation must reuse `train_dataset.class_to_idx`; validation
  may omit train classes but must not introduce unknown classes.
- Missing validation split may disable validation. A present but malformed split
  must fail the job with the real error; never swallow it as "no validation".
- SSE sends a bounded initial log tail, then bounded chunks from an offset. Do
  not read the whole log every polling interval.
- Redis connection is lazy/retryable with bounded backoff so recovery does not
  require an API restart.

## Supported Workloads

- Classification: ResNet, EfficientNet with ImageFolder
- Object detection: YOLOv11, Faster R-CNN with YOLO/COCO boxes
- Segmentation: DeepLabV3+ semantic masks, Mask R-CNN COCO instances
- OCR: PaddleOCR labels and Tesseract `.gt.txt`

CV and OCR use separate queues. A single GPU should normally run one worker job
at a time per queue configuration; do not add application-level parallelism
without checking GPU memory and RQ worker count.

## API Surface

- Catalog/profile: `GET /api/model-catalog`, `GET /api/resource-profile`
- Dataset: list, staged inspect/import, delete, metadata, compatibility
- Training: start, stop, status, logs, metrics, SSE events
- Results: list runs, delete run, download contained artifact path
- Admin: resource impact, transfer ownership, resource cleanup

Read route definitions in `main.py` before changing a contract. Keep frontend
compatibility when renaming fields or routes.

## Environment

Required production values:

- `DATABASE_URL`
- `REDIS_URL`
- `BACKEND_INTERNAL_TOKEN` (same value as frontend proxy)
- `DATASET_DIR`, `RUNS_DIR`, `STORAGE_DIR`
- `CORS_ORIGINS`

Do not commit or print real env files. Production startup requires PostgreSQL
migration to finish before API/worker traffic.

## Commands

```bash
# Fast regression suite; tests use unittest, not pytest
python3 -m unittest discover -s tests -v

# Syntax check
python3 -m compileall -q .

# Backfill preview; omit --apply to keep it read-only
python3 scripts/backfill_resources.py --migrate

# Production images
cd /home/trainai/AILAB
docker compose --env-file .env.production -f docker-compose.prod.yml build backend worker ocr-worker
```

Before restarting workers, inspect active queues/jobs. Do not interrupt a real
training run just to deploy unrelated code. For API-only changes, rebuild and
restart only `backend`; trainer/dependency changes require the relevant worker.

## Change Checklist

- Trace every caller before changing shared storage, auth, parser, or trainer code.
- Preserve DB ownership checks and path containment at every file boundary.
- Hold the correct DB lock across check plus state transition.
- Keep upload work bounded and defer complete dataset traversal to train time.
- Add/update the smallest regression test for non-trivial logic.
- Run the relevant tests, then the full `unittest` suite for shared code.
- Check `git status --short`; do not stage runtime data, env files, caches, or models.
