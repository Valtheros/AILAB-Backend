# Model Apps

Model Apps are session-authenticated, queued single-image inference, not a
dedicated deployment process or a latency-guaranteed serving endpoint. They
reuse `predict` jobs and the same CUDA executor as Test Model and Model API.

## Ownership And Access

- `/api/model-apps` manages the signed-in owner's apps and immutable releases.
- `/api/apps/{owner}/{slug}` shows a published app to any verified, active AILAB
  account with its link. There is no anonymous access or public directory.
- An owner can use `?version=N` to preview an unpublished release. Publication
  requires a completed owner preview and a checkpoint hash check. Failed or
  cancelled previews do not qualify. All versions of an app use the same task.
- The checkpoint includes the class mapping and inference configuration. A
  release stores its SHA-256 and path plus a copy of memory-estimator settings.
  The executor verifies the checkpoint before and after prediction.
- Switching releases changes only new admissions. Admission rechecks the app,
  owner, requester and active release inside the compute transaction.
- The model owner pays storage/GPU quotas. `app_requester_user_id`, not the
  resource owner, authorizes inputs, results, files, cancellation and notices.
  Owners see aggregate usage only. General compute routes exclude all app jobs.
  `is_model_app` survives deletion of app/release foreign keys so jobs cannot
  silently become ordinary owner-readable jobs.
- Response allowlists exclude checkpoint paths, run names and worker diagnostics.
  No owner API key is sent to visitors. All app responses are private/no-store.

## Admission And Retention

- Authentication and app resolution precede multipart parsing. Upload leases
  and spool files belong to the verified model owner, never a frontend user ID.
- Existing owner limits apply: three pending predictions/batches, one GPU job
  concurrently by default, kernel project quota, reservations and host limits.
- Each visitor also has at most three pending app jobs across owners, serialized
  by a visitor advisory lock. Submission shares the Model API owner budget of
  20/minute; visitors have a separate 20/minute budget. Reads/cancels are limited
  to 120/minute per app visitor session identity. Redis failure closes admission.
- Requests accept one supported image, up to 10 MiB and 25 million pixels.
  Auto GPU is used; there is no silent CPU fallback or extra serving reservation.
- Idempotency is namespaced by requester and app; changing bytes, threshold or
  release with the same key conflicts. Duplicate requests do not create jobs.
- Existing prediction cleanup removes files 24 hours after terminal status.
  APIs return 410 at expiry even before cleanup. Browser links retain job IDs.
  Unpublishing stops new visitors, not accepted work. A deleted app's requester
  can still read their existing job until expiry, using retained job metadata.
- Counts/queue times are actual compute records, not a response-time guarantee.

## Lifecycle

- Runs referenced by any app release cannot be deleted, including unpublished
  releases. Both application checks and a separate database trigger enforce it.
- Delete an app only after unpublishing and finishing/cancelling its jobs.
  Deleting the app removes release references, not trained weights.
- Account cleanup blocks active requester jobs as well as resource-owner jobs.
  It removes apps before protected runs. The admin impact includes app counts
  and active app jobs. Requester deletion never exposes their retained files to
  the model owner; the requester FK becomes null and access fails closed.
- Transfer checks address collisions before copying, transfers app ownership,
  disables publication and clears preview proof. The new owner must preview
  before republishing. Existing API keys are still revoked by the old lifecycle.

## Verification

`tests/test_model_apps_integration.py` covers preview/publish/rollback,
cross-user access, inherited Model API regression cases, retained jobs,
idempotency, cancellation, checkpoint changes, deletion and transfer.
`tests/test_storage_uploads.py` verifies model-owner spool ownership.
Use the disposable PostgreSQL/Redis/storage gates in `tests/isolation.py`.

Migration is additive and idempotent in `database/model_apps_schema.sql`.
Update backend and compute together: old workers cannot record preview proof
or route app notifications to the requester. Deploy only after draining jobs;
leave the new schema in place during image rollback. Preserve app metadata and
accepted jobs; do not drop new tables as rollback cleanup.

Deferred: anonymous/invited sharing, examples, persistent serving, GPU serving
reservations, per-class mask editing and model-app feedback. Model API batch
prediction remains separate; Model Apps deliberately accept one image at a time.
