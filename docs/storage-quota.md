# Storage quota ownership and rollout

## Current state

Storage enforcement and the queued Model API are enabled on 2026-11-06.
`STORAGE_ENFORCEMENT=true` is required in the backend, data worker, compute
workers/coordinator and Label Studio. `MODEL_API_ENABLED=true` additionally
opens the public API; disabling storage enforcement closes that API too.
The filesystem prerequisite alone is never sufficient to enable these flags.

On 2026-11-06, deployment moved dataset, runs, storage and Label Studio media
onto a dedicated 300 GiB ext4 filesystem with `project`, `quota` and `prjquota`.
Docker mounts the partition by UUID with volume subpaths; the host mount is
`/home/trainai/ailab-userdata`. PostgreSQL, Redis and shared pretrained model
cache remain on the OS disk. Source data was retained and copies were verified
with checksums under a write freeze. See the deployment repository's
`deploy/STORAGE.md` for rollback and exact mount contracts.

The subsequent stopped-writer backfill verified 83,774 entries for five
existing accounts, including legacy annotation data. It retained paths,
inodes and content hashes, assigned owner project IDs and recorded the
initial 20 GiB per-user limits. Existing data was not deleted. The private
deployment backup contains the original inventory, tag journal and databases.

## Enforcement Implementation

New deployments default to enforcement off until their own inventory,
backfill, operator and writer verification have completed.

- `storage_operator.py` runs without a network, database credentials, or
  `CAP_SYS_RESOURCE`. Its authenticated Unix socket can provision an owner-bound
  project, change its limit, prepare an empty owned directory, measure usage,
  remove a validated request-temporary directory, or retire one disabled
  owner's files. Paths are opened
  relative to pinned directory descriptors without following symlinks.
- `storage_reservations.py` holds owner and global PostgreSQL session locks.
  The database records a ceiling before the operator increases a limit. The
  operator lowers a limit before the database makes it reusable. A failed
  acknowledgement leaves charged headroom, never an unaccounted grant.
  `enforced_limit_bytes` is an upper bound on the kernel grant; `limit_bytes`
  is the user's configured quota. The two values need not be equal.
- Reservations use conservative write estimates for admission. Active owners
  can receive more headroom, up to their configured quota and available global
  capacity; all that headroom is accounted for when preserving the 20 GiB disk
  reserve. Job output is not artificially capped at its estimate. Shared owner
  headroom stays allocated until its last writer releases its reservation.
- `storage_uploads.py` runs after authentication and before multipart parsing.
  Context-local spools go to an owned directory, never a shared global tempdir.
  Both Content-Length and actual received bytes are bounded. Spools close
  before allowances return, including cancellation and disconnected clients.
- `storage_service.py` request leases use an OS file lock, plus a short registry
  lock covering creation. Recovery may reclaim them only after acquiring the
  same exclusive lock. A heartbeat timeout alone is not evidence of stopped
  writes. These small lock files are control metadata on the OS disk, not user
  media. Do not use the request-lease protocol for a background process whose
  writes can outlive its request.
- Compute output reservations start after the worker claims a job and are
  released after process termination and the authoritative terminal update.
  A bounded 4096-byte pipe carries the final outcome even when quota exhaustion
  prevents creating an error log or result-status file. Worker temporary files
  use the owned compute directory.

- Label Studio native multipart uploads reserve through the private backend
  bridge before Django parses them. Parser spools, generated ZIPs, backend
  extraction, imported originals and publish staging all inherit owner quotas.
  CSRF and project ownership checks remain in place. Control locks are shared;
  the Label Studio application never receives quota-administration capability.
- Dataset inspection/import metadata, training preparation, CPU indexing,
  Label Studio import/publish and all compute kinds use the same ledger.
  Indexing fails on quota exhaustion rather than calling every image corrupt.
- `/api/storage` reports kernel usage; `/api/storage/admin` permits admin-only
  quota changes. The account page and admin storage page use those endpoints.
  Category totals are dated snapshots refreshed by the CPU data worker, not
  the authority for admission. They may lag actual usage by five minutes.

## Account lifecycle

Disable the source account and stop all writers before deletion or transfer.
No live reservation or pending transfer may be silently discarded. Deletion
uses existing Label Studio cleanup first, then removes registered AILAB data
and retires remaining files by their kernel project ID. A failure retains the
account record so cleanup can be retried. The small immutable operator anchor
is retained and its numeric project ID is never reused.

Transfer checks the recipient's quota before copying. Copies get new inodes
under the recipient's project ID; original hashes are verified. The durable
`storage_transfers` journal moves to `committed` in the ownership transaction.
Recovery restores originals for an uncommitted transfer or removes originals
after a committed one. It must run before request-lease cleanup, and only
after acquiring the writer's OS lock. New reservations for either account are
blocked while the transfer remains pending. API keys are revoked and endpoints
disabled during transfer. Label Studio projects are not transferred by this
action; the admin dialog says so. Accounts with the separate legacy Annotation
system remain protected from deletion/transfer pending explicit legacy cleanup.

## Verification

The rollout passed 68 storage/worker tests, a separate real ext4/PostgreSQL
backfill-and-transfer test, eight API/auth/rate-limit tests and three frontend
tests. All four application production images built successfully. The native
Label Studio storage and ownership test suite passed separately.

Authenticated HTTPS smoke verified signup/login, ZIP preview/import, CPU
thumbnail indexing, CUDA ResNet training, automatic evaluation, native Label
Studio upload/annotation/publish/accept, queued API single and partial batch
prediction, duplicate-request handling, result archives and key revocation.
Lowering only the temporary test account's quota caused a readable 409 while
its existing data stayed readable. This is not a capacity benchmark or a new
six-model accuracy/long-training certification.

## Owned data

An owner's usage must include all of these, including temporary or retained
files, before a 20 GiB hard limit can be enabled:

- Dataset originals, versions, manifests, indexes, thumbnails, staged uploads,
  and Label Studio import/publish work under `DATASET_DIR`.
- Training artifacts, evaluation results, prediction inputs and results under
  `RUNS_DIR` and `STORAGE_DIR`.
- Label Studio source media and exports in the Label Studio data volume. Native
  uploads do not necessarily go through the AILAB backend.
- Upload parser spool files, worker temporary outputs and any per-user cache.
- Files and database records from the older in-app Annotation system, even if
  the deployed UI now uses Label Studio.

Map existing paths to owners using authoritative database records. A deleted
Label Studio project can still have files on disk; retain and account for them
until their ownership and cleanup policy are verified. Preserve unmatched,
missing, shared, hard-linked or otherwise ambiguous paths for manual review.
Never infer that a file is safe to delete from an absent UI entry alone.

On the inspection date the database contained 27 datasets, 58 runs, seven
active Label Studio projects, and one legacy Annotation project with 201 image
records. These are a snapshot, not a backfill manifest. Re-scan under a write
freeze before migration. A read-only top-level audit also found three empty
dataset owner directories with no matching current user; they were left in
place and must be reviewed instead of assigned or removed automatically.

The optional `python -m scripts.storage_ownership_audit --dataset-root
<dataset-dir> --runs-root <runs-dir> --label-upload-root <media-upload-dir>`
reads owner records in a read-only database transaction and reports unmatched
top-level paths, conflicts and symlinks. It does not recurse through all files,
verify hard links or prevent concurrent writes, and deliberately never reports
ownership as fully verified. Run it with read-only volume mounts; do not treat
an empty issue list as permission to enable quotas.

## Before enabling enforcement

1. Verify a restorable backup on storage separate from the VM, and provide a
   maintenance window with an offline recovery console and tested rollback.
2. Inventory every owner and writer above, including legacy and deleted-project
   data. Test an idempotent, recoverable backfill against a separate copy.
   Reconcile filesystem project IDs with the database sequence after a restore;
   a database backup can be older than the filesystem it is paired with.
3. Implement a narrowly privileged quota operator; do not grant the API or
   compute workers unrestricted filesystem administration capabilities.
4. Bound each writer by its reservation, account for parser spooling and child
   processes, and make cancellation and crash recovery release reservations
   only after writing has stopped.
5. Implement quota-aware user deletion and transfer. Project-inheriting
   directories reject cross-project renames, so transfer needs a verified,
   recoverable copy and a destination quota check.
6. Test existing users already over quota: preserve their files and allow read,
   download and deletion without permitting new over-limit writes.
7. Test upload, Label Studio, Train, Evaluate, Prediction, user deletion and
   transfer end-to-end with concurrent users and interrupted workers. Confirm
   rollback on an isolated copy before scheduling the production cutover.

`python -B -m scripts.storage_quota_preflight --root <dataset-dir> --root
<runs-dir> --root <label-studio-data-dir>` only checks filesystem prerequisites;
`applicationEnforcementVerified` is deliberately always false. It must be run
with permission to inspect the device. The kernel integration test uses a
disposable image under `/home/trainai`, never the production device.

Kernel tests must drop `CAP_SYS_RESOURCE` from their writing process; otherwise
host root can bypass quota limits. Production application/worker containers
explicitly drop this capability. Retain administrative capabilities only in the
isolated test harness or the separately privileged quota operator.

The application reservation ledger checks for a 20 GiB filesystem reserve,
but cannot enforce it against unmanaged writes from the OS, Docker or other
services. Monitor filesystem free space separately and stop accepting new
work when system capacity is unsafe.

## Offline filesystem change

The following applies only to the old root-volume alternative. The deployed
second disk already has the required filesystem features; do not change the
root filesystem to finish application quota integration.

Enabling ext4 filesystem features on the root volume is a maintenance task,
not a live API operation. The test image was unmounted for the feature change
and checked before mounting with project quotas. Verify the actual device,
boot procedure and recovery environment independently before changing the
root volume or `/etc/fstab`; never run filesystem repair against a mounted root.
Do not shrink, reformat, or delete the existing volume to implement this plan.
