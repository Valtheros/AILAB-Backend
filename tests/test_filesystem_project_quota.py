"""Opt-in kernel test using a disposable ext4 image, never the host filesystem."""

import errno
import os
import shutil
import subprocess
import tempfile
import time
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

    def test_operator_binds_owner_and_tags_only_empty_directories(self):
        from storage_operator import QuotaOperator

        operator = QuotaOperator(self.mount, Path(_LOOP_DEVICE))
        request = {'projectId': 12001, 'owner': 'alice'}
        initial = operator.dispatch({'operation': 'provision', **request})
        self.assertGreater(initial['hardLimitBytes'], 0)
        self.assertEqual(initial, operator.dispatch({'operation': 'provision', **request}))
        with self.assertRaisesRegex(ValueError, 'different owner'):
            operator.dispatch({'operation': 'provision', **request, 'owner': 'bob'})
        operator.dispatch({'operation': 'limit', **request, 'limitBytes': 2 * 1024 * 1024})
        (self.mount / 'storage').mkdir(exist_ok=True)
        prepared = {'operation': 'prepare', **request, 'namespace': 'storage', 'path': 'owners/alice'}
        operator.dispatch(prepared)
        owner = self.mount / 'storage/owners/alice'
        self.assertEqual(directory_project(owner), (12001, True))
        _write_file(owner / 'saved', 1024 * 1024)
        operator.dispatch(prepared)
        with self.assertRaises(OSError) as error:
            _write_file(owner / 'too-large', 2 * 1024 * 1024)
        self.assertIn(error.exception.errno, {errno.EDQUOT, errno.ENOSPC})
        self.assertEqual((owner / 'saved').stat().st_size, 1024 * 1024)

        other = {'projectId': 12002, 'owner': 'bob'}
        operator.dispatch({'operation': 'provision', **other})
        operator.dispatch({'operation': 'limit', **other, 'limitBytes': 2 * 1024 * 1024})
        with self.assertRaisesRegex(ValueError, 'different owner'):
            operator.dispatch({**prepared, **other})
        untagged = self.mount / 'storage/legacy'
        untagged.mkdir()
        (untagged / 'keep').write_bytes(b'unchanged')
        with self.assertRaisesRegex(ValueError, 'backfill'):
            operator.dispatch({**prepared, 'path': 'legacy'})
        self.assertEqual((untagged / 'keep').read_bytes(), b'unchanged')

    def test_operator_cleanup_only_removes_owned_request_temp_files(self):
        from storage_operator import QuotaOperator, owner_digest

        operator = QuotaOperator(self.mount, Path(_LOOP_DEVICE))
        identity = {'projectId': 12003, 'owner': 'cleanup-owner'}
        operator.dispatch({'operation': 'provision', **identity})
        operator.dispatch({'operation': 'limit', **identity, 'limitBytes': 2 * 1024 * 1024})
        (self.mount / 'storage').mkdir(exist_ok=True)
        parent = f'.uploads/{owner_digest(identity["owner"])[:32]}'
        request_id = 'a' * 32
        for path in (parent, parent + '/' + request_id):
            operator.dispatch({'operation': 'prepare', **identity, 'namespace': 'storage', 'path': path})
        temporary = self.mount / 'storage' / parent / request_id
        (temporary / 'file').write_bytes(b'discard')
        preserved = self.mount / 'original-image'
        preserved.write_bytes(b'preserve')
        (temporary / 'link').symlink_to(preserved)
        request = {'operation': 'cleanup-upload', **identity, 'namespace': 'storage', 'requestId': request_id}
        operator.dispatch(request)
        self.assertFalse(temporary.exists())
        self.assertEqual(preserved.read_bytes(), b'preserve')
        operator.dispatch(request)
        for invalid in ({'namespace': 'runs'}, {'requestId': '../original-image'}, {'owner': 'different-owner'}):
            with self.assertRaises(ValueError):
                operator.dispatch({**request, **invalid})

    def test_offline_backfill_preserves_content_and_is_repeatable(self):
        from storage_inventory import scan
        from scripts.storage_backfill import tag_records, verify_inventory
        from storage_operator import QuotaOperator
        from storage_project_quota import _attributes

        root = self.mount / 'backfill'
        root.mkdir()
        for name in ('dataset', 'runs', 'storage', 'label-studio'):
            (root / name).mkdir()
        data = root / 'dataset/legacy'
        data.mkdir()
        _write_file(data / 'image', 65536)
        runs = root / 'runs/legacy'
        runs.mkdir()
        os.link(data / 'image', runs / 'image-copy')
        roots = {'dataset/legacy': 'legacy-owner', 'runs/legacy': 'legacy-owner'}
        plan = scan(root, roots, attributes=_attributes)
        self.assertEqual(plan['issues'], [])
        operator = QuotaOperator(root, Path(_LOOP_DEVICE))
        identity = {'projectId': 14001, 'owner': 'legacy-owner'}
        operator.dispatch({'operation': 'provision', **identity})
        operator.dispatch({'operation': 'limit', **identity, 'limitBytes': 4 * 1024**2})
        for _ in range(2):
            tag_records(root, plan['files'], {'legacy-owner': 14001})
            verified = verify_inventory(root, plan, roots)
            self.assertTrue(all(row['project'] == 14001 for row in verified['files'] if row['owner']))
        self.assertEqual((data / 'image').read_bytes(), b'x' * 65536)
        tag_records(root, plan['files'], {'legacy-owner': 14001}, restore=True)
        verified = verify_inventory(root, plan, roots)
        self.assertTrue(all(row['project'] == 0 for row in verified['files']))
        (data / 'image').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Files changed'):
            verify_inventory(root, plan, roots)

    def test_retire_removes_only_its_owner_and_can_be_retried(self):
        from storage_operator import QuotaOperator

        root = self.mount / 'retire'
        root.mkdir()
        for namespace in ('dataset', 'runs', 'storage', 'label-studio'):
            (root / namespace).mkdir()
        operator = QuotaOperator(root, Path(_LOOP_DEVICE))
        alice, bob = {'projectId': 15001, 'owner': 'alice'}, {'projectId': 15002, 'owner': 'bob'}
        for identity in (alice, bob):
            operator.dispatch({'operation': 'provision', **identity})
            operator.dispatch({'operation': 'limit', **identity, 'limitBytes': 4 * 1024**2})
            for namespace in ('dataset', 'runs', 'storage', 'label-studio'):
                operator.dispatch({'operation': 'prepare', **identity, 'namespace': namespace, 'path': identity['owner']})
                (root / namespace / identity['owner'] / 'saved').write_bytes(identity['owner'].encode())
        for _ in range(2):
            result = operator.dispatch({'operation': 'retire', **alice})
            self.assertEqual(result['usedBytes'], result['hardLimitBytes'])
            for namespace in ('dataset', 'runs', 'storage', 'label-studio'):
                self.assertFalse((root / namespace / 'alice').exists())
                self.assertEqual((root / namespace / 'bob/saved').read_bytes(), b'bob')
        with self.assertRaisesRegex(ValueError, 'different owner'):
            operator.dispatch({'operation': 'retire', **alice, 'owner': 'bob'})

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

    @unittest.skipUnless(os.environ.get('AILAB_QUOTA_OPERATOR_IMAGE'), 'requires the quota operator image')
    def test_operator_container_uses_private_socket_without_network_or_quota_bypass(self):
        from storage_operator_client import StorageOperatorClient, StorageOperatorError

        image = os.environ['AILAB_QUOTA_OPERATOR_IMAGE']
        name = 'ailab-quota-operator-lab-' + uuid.uuid4().hex[:12]
        control = self.temp_dir / 'control'
        control.mkdir()
        command = ['docker', 'run', '-d', '--name', name, '--network', 'none', '--read-only',
                   '--cap-drop', 'ALL', '--cap-add', 'SYS_ADMIN', '--cap-add', 'CHOWN',
                   '--cap-add', 'FOWNER', '--cap-add', 'DAC_OVERRIDE', '--security-opt', 'no-new-privileges',
                   '--memory', '64m', '--pids-limit', '16', '--cpus', '0.25',
                   '--device', _LOOP_DEVICE + ':/dev/ailab-data:r',
                   '--mount', f'type=bind,source={self.mount},target=/userdata',
                   '--mount', f'type=bind,source={control},target=/storage-control',
                   image, '--root', '/userdata', '--device', '/dev/ailab-data', '--control', '/storage-control']
        subprocess.run(command, check=True, capture_output=True)
        self.addCleanup(lambda: subprocess.run(['docker', 'rm', '-f', name], check=True, capture_output=True))
        client = StorageOperatorClient(control)
        for attempt in range(50):
            try:
                client.call('capacity', 0, '')
                break
            except StorageOperatorError:
                time.sleep(0.1)
        else:
            logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True)
            self.fail(logs.stderr)
        client.call('provision', 13001, 'container-owner')
        client.call('limit', 13001, 'container-owner', limitBytes=2 * 1024 * 1024)
        (self.mount / 'runs').mkdir(exist_ok=True)
        client.call('prepare', 13001, 'container-owner', namespace='runs', path='container-test')
        self.assertEqual(directory_project(self.mount / 'runs/container-test'), (13001, True))
        result = subprocess.run(['docker', 'exec', '--user', '1001:0', name, 'python', '-B', '-c',
            "from storage_operator_client import StorageOperatorClient; assert StorageOperatorClient().call('capacity',0,'')['freeBytes'] > 0"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = subprocess.run(['docker', 'exec', name, 'python', '-B', '-c',
            "from pathlib import Path; c=int(next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('CapEff:')).split()[1],16); assert not c & (1 << 24)"],
            capture_output=True, text=True)
        self.assertEqual(status.returncode, 0, status.stderr)

    @unittest.skipUnless(os.environ.get('AILAB_QUOTA_BACKEND_IMAGE'), 'requires disposable database and backend image')
    def test_transfer_with_real_database_and_kernel_quota(self):
        root = self.mount / 'application-test'
        root.mkdir()
        (root / '.ailab-disposable-test').write_text('ailab-disposable-test-v1')
        for name in ('runs', 'dataset', 'storage', 'tmp', 'app'):
            (root / name).mkdir()
        source = Path(__file__).resolve().parents[1]
        command = ['docker', 'run', '--rm', '--network', 'container:ailab-storage-lab-postgres',
                   '--cap-add', 'SYS_ADMIN', '--cap-drop', 'SYS_RESOURCE', '--device', '/dev/loop0:/dev/loop0:r',
                   '--memory', '1g', '--mount', f'type=bind,source={root},target={root}',
                   '--mount', f'type=bind,source={source},target=/app,readonly']
        for key, value in {'COMPUTE_TESTS': '1', 'AILAB_REAL_QUOTA_RUNTIME': '1',
                'DATABASE_URL': 'postgresql://ailab_test:disposable-storage-test@localhost/ailab_compute_test_pg',
                'PYTHONDONTWRITEBYTECODE': '1', 'AILAB_TEST_ROOT': str(root), 'TMPDIR': str(root / 'tmp'),
                'RUNS_DIR': str(root / 'runs'), 'DATASET_DIR': str(root / 'dataset'),
                'STORAGE_DIR': str(root / 'storage'), 'APP_BASE_DIR': str(root / 'app')}.items():
            command.extend(['-e', key + '=' + value])
        command.extend([os.environ['AILAB_QUOTA_BACKEND_IMAGE'], 'python', '-B', '-m', 'unittest',
            'tests.test_storage_reservations_integration.StorageReservationsIntegrationTests.test_transfer_on_real_ext4_assigns_every_copy_and_reclaims_source_usage', '-v'])
        subprocess.run(command, check=True)

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
        from storage_operator import QuotaOperator
        operator = QuotaOperator(self.mount, Path(_LOOP_DEVICE))

        class DirectClient:
            def call(self, operation, project_id, subject, **values):
                return operator.dispatch({'operation': operation, 'projectId': project_id, 'owner': subject, **values})

        ledger = StorageReservations(registry._connect, DirectClient(), free_reserve_bytes=8 * 1024 * 1024)
        account = ledger.prepare_account(owner)
        project_id = account['project_id']
        anchor = self.mount / '.quota' / str(project_id)
        data = self.mount / 'account-data'
        data.mkdir()
        ledger.set_limit(owner, 8 * 1024 * 1024)
        ledger.reserve(owner, 'setup:1', 'upload', 65536)
        assign_empty_directory(anchor, project_id)
        assign_empty_directory(data, project_id)
        ledger.release(owner, 'setup:1')

        measure = ProjectQuotaMeasurement(_LOOP_DEVICE, self.mount / '.quota', self.mount)
        used, hard_limit, free = measure(project_id)
        filesystem = os.statvfs(self.mount)
        self.assertEqual(hard_limit, used)
        self.assertGreaterEqual(used, 0)
        self.assertEqual(free, filesystem.f_bavail * filesystem.f_frsize)
        with self.assertRaisesRegex(ValueError, 'unrestricted root'):
            ProjectQuotaMeasurement(_LOOP_DEVICE, anchors, data)(project_id)
        (self.mount / '.quota' / str(project_id + 1)).symlink_to(anchor, target_is_directory=True)
        with self.assertRaises(OSError):
            measure(project_id + 1)

        ledger.reserve(owner, 'train:1', 'train', 3 * 1024 * 1024)
        _write_file(data / 'artifact', 2 * 1024 * 1024)
        with self.assertRaises(StorageReservationError) as exceeded:
            ledger.reserve(owner, 'upload:1', 'upload', 4 * 1024 * 1024)
        self.assertEqual(exceeded.exception.code, 'STORAGE_QUOTA_EXCEEDED')
        ledger.release(owner, 'train:1')
        ledger.reserve(owner, 'upload:2', 'upload', 5 * 1024 * 1024)
        set_project_limit(_LOOP_DEVICE, anchor, project_id, 9 * 1024 * 1024)
        with self.assertRaises(StorageReservationError) as mismatched:
            ledger.reserve(owner, 'upload:3', 'upload', 1024)
        self.assertEqual(mismatched.exception.code, 'STORAGE_UNAVAILABLE')


if __name__ == "__main__":
    unittest.main()
