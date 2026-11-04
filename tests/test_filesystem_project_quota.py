"""Opt-in kernel test using a disposable ext4 image, never the host filesystem."""

import errno
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from storage_project_quota import (ProjectQuotaMeasurement, assign_empty_directory, directory_project,
                                   project_usage, set_project_limit)
from tests.isolation import require_test_database


_ENABLED = os.environ.get("AILAB_REAL_QUOTA_TEST") == "1"
_LOOP_DEVICE = "/dev/loop0"


def _write_file(path, size):
    with path.open("wb", buffering=0) as stream:
        chunk = b"x" * 65536
        while size:
            written = stream.write(chunk[: min(size, len(chunk))])
            if not written:
                raise OSError(errno.EIO, "Write made no progress")
            size -= written
        os.fsync(stream.fileno())


@unittest.skipUnless(_ENABLED, "requires an isolated, privileged ext4 quota lab")
class FilesystemProjectQuotaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ["AILAB_QUOTA_TEST_ROOT"]).resolve()
        if not base.is_relative_to(Path("/home/trainai")) or not base.is_dir():
            raise RuntimeError("Quota test directory must already exist under /home/trainai")
        if subprocess.run(["losetup", _LOOP_DEVICE], capture_output=True).returncode == 0:
            raise RuntimeError("Quota test loop device is already in use")
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="ext4-quota-", dir=base))
        cls.image = cls.temp_dir / "test.img"
        cls.mount = cls.temp_dir / "mounted"
        cls.attached = False
        cls.mounted = False
        cls.addClassCleanup(cls._cleanup)
        cls.mount.mkdir()
        with cls.image.open("xb") as image:
            image.truncate(128 * 1024 * 1024)
        subprocess.run(["mkfs.ext4", "-q", "-F", "-I", "256", "-O", "^project,^quota", cls.image], check=True)
        subprocess.run(["losetup", _LOOP_DEVICE, cls.image], check=True)
        cls.attached = True
        subprocess.run(["mount", "-t", "ext4", _LOOP_DEVICE, cls.mount], check=True)
        cls.mounted = True
        (cls.mount / "existing-data").write_bytes(b"existing dataset")
        subprocess.run(["umount", cls.mount], check=True)
        cls.mounted = False
        subprocess.run(["tune2fs", "-O", "project,quota", _LOOP_DEVICE], check=True)
        result = subprocess.run(["e2fsck", "-f", "-y", _LOOP_DEVICE], capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"Quota test filesystem check failed: {result.stderr}")
        subprocess.run(["mount", "-t", "ext4", "-o", "prjquota", _LOOP_DEVICE, cls.mount], check=True)
        cls.mounted = True

    @classmethod
    def _cleanup(cls):
        if cls.mounted:
            subprocess.run(["umount", cls.mount], check=True)
            cls.mounted = False
        if cls.attached:
            result = subprocess.run(["losetup", "-n", "-O", "BACK-FILE", _LOOP_DEVICE],
                                    check=True, capture_output=True, text=True)
            if Path(result.stdout.strip()).resolve() != cls.image.resolve():
                raise RuntimeError("Loop device changed; refusing to detach it")
            subprocess.run(["losetup", "-d", _LOOP_DEVICE], check=True)
            cls.attached = False
        shutil.rmtree(cls.temp_dir)

    def test_owner_limit_covers_multiple_directories_and_blocks_cross_owner_rename(self):
        from scripts.storage_quota_preflight import inspect

        report = inspect([str(self.mount)])
        self.assertTrue(report['filesystemsReady'], report['errors'])
        self.assertFalse(report['applicationEnforcementVerified'])
        self.assertEqual((self.mount / "existing-data").read_bytes(), b"existing dataset")
        owner_data = self.mount / "owner-data"
        owner_runs = self.mount / "owner-runs"
        other_data = self.mount / "other-data"
        for path, project_id in ((owner_data, 11001), (owner_runs, 11001), (other_data, 11002)):
            path.mkdir()
            assign_empty_directory(path, project_id)
            self.assertEqual(directory_project(path), (project_id, True))
            set_project_limit(_LOOP_DEVICE, path, project_id, 3 * 1024 * 1024)

        _write_file(owner_data / "input", 1024 * 1024)
        _write_file(owner_runs / "output", 1024 * 1024)
        with self.assertRaises(OSError) as caught:
            _write_file(owner_runs / "overflow", 2 * 1024 * 1024)
        self.assertIn(caught.exception.errno, {errno.EDQUOT, errno.ENOSPC})

        used, hard = project_usage(_LOOP_DEVICE, owner_data, 11001)
        self.assertEqual(hard, 3 * 1024 * 1024)
        self.assertGreaterEqual(used, 2 * 1024 * 1024)
        self.assertLessEqual(used, 3 * 1024 * 1024)
        with self.assertRaises(ValueError):
            project_usage(_LOOP_DEVICE, owner_data, 11002)
        with self.assertRaises(ValueError):
            set_project_limit(_LOOP_DEVICE, self.temp_dir, 11001, 1024 * 1024)
        _write_file(other_data / "input", 1024 * 1024)

        with self.assertRaises(OSError) as cross_owner:
            (owner_data / "input").rename(other_data / "moved")
        self.assertEqual(cross_owner.exception.errno, errno.EXDEV)
        (owner_data / "input").rename(owner_runs / "moved")
        assign_empty_directory(owner_runs, 11001)
        with self.assertRaises(ValueError):
            assign_empty_directory(owner_runs, 11002)

    def test_enabling_limit_for_existing_over_quota_data_never_deletes_it(self):
        owner = self.mount / "existing-owner"
        owner.mkdir()
        assign_empty_directory(owner, 11003)
        original = owner / "existing-file"
        _write_file(original, 2 * 1024 * 1024)
        set_project_limit(_LOOP_DEVICE, owner, 11003, 1024 * 1024)

        with self.assertRaises(OSError) as caught:
            _write_file(owner / "new-file", 65536)
        self.assertIn(caught.exception.errno, {errno.EDQUOT, errno.ENOSPC})
        self.assertEqual(original.stat().st_size, 2 * 1024 * 1024)
        self.assertEqual(original.read_bytes(), b"x" * (2 * 1024 * 1024))
        original.unlink()
        _write_file(owner / "after-delete", 65536)

    def test_refuses_to_reassign_existing_data_or_follow_final_symlinks(self):
        legacy = self.mount / "untagged-legacy"
        legacy.mkdir()
        (legacy / "existing-file").write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "offline, verified backfill"):
            assign_empty_directory(legacy, 11004)
        self.assertEqual(directory_project(legacy), (0, False))
        self.assertEqual((legacy / "existing-file").read_bytes(), b"preserve")

        link = self.mount / "legacy-link"
        link.symlink_to(legacy, target_is_directory=True)
        with self.assertRaises(OSError):
            assign_empty_directory(link, 11004)
        self.assertEqual(directory_project(legacy), (0, False))

    def test_dataset_staging_and_import_rename_stay_within_owner_quota(self):
        from dataset_storage import owner_dataset_root
        from staged_uploads import StagedUploadStore

        datasets = self.mount / "datasets"
        datasets.mkdir()
        store = StagedUploadStore(datasets)
        owner = owner_dataset_root(datasets, "owner-a")
        owner.mkdir(parents=True)
        assign_empty_directory(owner, 11005)
        set_project_limit(_LOOP_DEVICE, owner, 11005, 10 * 1024 * 1024)
        source = owner / "incoming"
        source.mkdir()
        _write_file(source / "image", 65536)
        manifest = store.create("owner-a", "dataset", source, {})
        stage = store.peek(manifest["token"], "owner-a")[2]
        self.assertTrue(stage.is_relative_to(owner))
        _, dataset, _ = store.consume(manifest["token"], "owner-a")
        dataset.rename(owner / "imported")
        self.assertEqual((owner / "imported" / "image").stat().st_size, 65536)
        self.assertGreater(project_usage(_LOOP_DEVICE, owner, 11005)[0], 0)

    @unittest.skipUnless(os.environ.get('COMPUTE_TESTS') == '1', 'requires isolated PostgreSQL test stack')
    def test_kernel_usage_and_database_reservations_are_checked_together(self):
        from resource_repository import resource_repository as registry
        from storage_reservations import StorageReservationError, StorageReservations

        require_test_database(registry.database_url)
        owner = f'quota-lab-{uuid.uuid4().hex}'
        with registry._connect() as connection:
            connection.execute('insert into "user"(id,email) values(%s,%s)', (owner, f'{owner}@example.invalid'))

        def cleanup_database():
            with registry._connect() as connection:
                connection.execute('delete from storage_reservations where owner_user_id=%s', (owner,))
                connection.execute('delete from storage_accounts where owner_user_id=%s', (owner,))
                connection.execute('delete from "user" where id=%s', (owner,))

        self.addCleanup(cleanup_database)
        anchors = self.mount / 'anchors'
        anchors.mkdir()
        measure = ProjectQuotaMeasurement(_LOOP_DEVICE, anchors, self.mount)
        ledger = StorageReservations(registry._connect, measure, free_reserve_bytes=8 * 1024 * 1024)
        account = ledger.prepare_account(owner)
        project_id = account['project_id']
        anchor = anchors / str(project_id)
        anchor.mkdir()
        data = self.mount / 'account-data'
        data.mkdir()
        assign_empty_directory(anchor, project_id)
        assign_empty_directory(data, project_id)
        set_project_limit(_LOOP_DEVICE, anchor, project_id, 8 * 1024 * 1024)
        with registry._connect() as connection:
            connection.execute("update storage_accounts set state='ready',limit_bytes=%s where owner_user_id=%s",
                               (8 * 1024 * 1024, owner))

        used, hard_limit, free = measure(project_id)
        filesystem = os.statvfs(self.mount)
        self.assertEqual(hard_limit, 8 * 1024 * 1024)
        self.assertGreaterEqual(used, 0)
        self.assertEqual(free, filesystem.f_bavail * filesystem.f_frsize)
        with self.assertRaisesRegex(ValueError, 'unrestricted root'):
            ProjectQuotaMeasurement(_LOOP_DEVICE, anchors, data)(project_id)
        (anchors / str(project_id + 1)).symlink_to(anchor, target_is_directory=True)
        with self.assertRaises(OSError):
            measure(project_id + 1)

        ledger.reserve(owner, 'train:1', 'train', 3 * 1024 * 1024)
        _write_file(data / 'artifact', 2 * 1024 * 1024)
        with self.assertRaises(StorageReservationError) as exceeded:
            ledger.reserve(owner, 'upload:1', 'upload', 4 * 1024 * 1024)
        self.assertEqual(exceeded.exception.code, 'STORAGE_QUOTA_EXCEEDED')
        ledger.release(owner, 'train:1')
        ledger.reserve(owner, 'upload:2', 'upload', 5 * 1024 * 1024)
        with registry._connect() as connection:
            connection.execute('update storage_accounts set limit_bytes=%s where owner_user_id=%s',
                               (9 * 1024 * 1024, owner))
        with self.assertRaises(StorageReservationError) as mismatched:
            ledger.reserve(owner, 'upload:3', 'upload', 1024)
        self.assertEqual(mismatched.exception.code, 'STORAGE_UNAVAILABLE')


if __name__ == "__main__":
    unittest.main()
