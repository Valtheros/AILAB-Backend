"""Inspect project-quota prerequisites without changing files or mount settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from storage_project_quota import project_quota_active



def _mount_info(path: Path) -> dict:
    result = subprocess.run(['findmnt', '--json', '--target', str(path), '--output',
                             'TARGET,SOURCE,FSTYPE,OPTIONS'], capture_output=True, text=True, check=True)
    mounts = json.loads(result.stdout).get('filesystems') or []
    if len(mounts) != 1:
        raise ValueError('Expected one filesystem for the path')
    return mounts[0]


def _filesystem_features(source: str) -> set[str]:
    result = subprocess.run(['tune2fs', '-l', source], capture_output=True, text=True, check=True,
                            env={**os.environ, 'LC_ALL': 'C'})
    for line in result.stdout.splitlines():
        if line.startswith('Filesystem features:'):
            return set(line.split(':', 1)[1].split())
    raise ValueError('Filesystem features are unavailable')


def inspect(paths: list[str]) -> dict:
    report = {'filesystemsReady': False, 'applicationEnforcementVerified': False, 'paths': [], 'errors': []}
    if not paths:
        report['errors'].append('At least one storage path is required.')
        return report
    devices = set()
    for raw in paths:
        try:
            path = Path(raw).resolve(strict=True)
            if not path.is_dir():
                raise ValueError('Storage path is not a directory')
            device = path.stat().st_dev
            mount = _mount_info(path)
            source = mount['source']
            entry = {'path': str(path), 'mount': mount['target'], 'source': source,
                     'filesystem': mount['fstype'], 'features': [], 'projectQuotaActive': False}
            devices.add(device)
            report['paths'].append(entry)
            if mount['fstype'] != 'ext4':
                raise ValueError('The planned quota configuration requires ext4')
            features = _filesystem_features(source)
            entry['features'] = sorted(features.intersection({'project', 'quota'}))
            if not {'project', 'quota'}.issubset(features):
                raise ValueError('Filesystem project and quota features are not both enabled')
            if not {'prjquota', 'pquota'}.intersection(mount.get('options', '').split(',')):
                raise ValueError('Project quota is not enabled in mount options')
            entry['projectQuotaActive'] = project_quota_active(source)
            if not entry['projectQuotaActive']:
                raise ValueError('Project quota is not active in the kernel')
        except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report['errors'].append(f'{raw}: {exc}')
    if len(devices) > 1:
        report['errors'].append('Storage paths are on different filesystems; one project quota cannot enforce their combined usage.')
    report['filesystemsReady'] = len(report['paths']) == len(paths) and not report['errors']
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True, help='Existing user-data directory; repeat for each storage root')
    args = parser.parse_args()
    report = inspect(args.root)
    print(json.dumps(report, indent=2))
    return 0 if report['filesystemsReady'] else 1


if __name__ == '__main__':
    sys.exit(main())
