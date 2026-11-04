# Storage quota ownership and rollout

## Current state

The storage reservation tables and ext4 project-quota adapter are foundations,
not production enforcement. The API remains disabled until every writer and
the account lifecycle use the same accounting. Do not mark storage accounts
`ready`, enable Model API, or enable project quotas for AILAB owners merely
because the filesystem preflight passes.

The deployment inspected on 2026-11-04 uses one ext4 filesystem for AILAB
data and the Label Studio Docker volume. Its filesystem does not yet have the
`project` and `quota` features. The existing application data is not moved or
deleted by the preflight or the isolated quota tests.

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

The application reservation ledger checks for a 20 GiB filesystem reserve,
but cannot enforce it against unmanaged writes from the OS, Docker or other
services. Monitor filesystem free space separately and stop accepting new
work when system capacity is unsafe.

## Offline filesystem change

Enabling ext4 filesystem features on the root volume is a maintenance task,
not a live API operation. The test image was unmounted for the feature change
and checked before mounting with project quotas. Verify the actual device,
boot procedure and recovery environment independently before changing the
root volume or `/etc/fstab`; never run filesystem repair against a mounted root.
Do not shrink, reformat, or delete the existing volume to implement this plan.
