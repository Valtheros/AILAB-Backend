"""Integration tests require a disposable PostgreSQL database with 'test' in its name."""
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

import label_studio_jobs as jobs


DSN = os.getenv('AILAB_LABEL_ACCOUNT_TEST_DSN', '')


@unittest.skipUnless(DSN, 'Requires a disposable PostgreSQL test database')
class LabelAccountCleanupIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'test' not in urlsplit(DSN).path.lower():
            raise RuntimeError('Refusing to use a database without test in its name')
        cls.schema = 'label_cleanup_' + uuid.uuid4().hex
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute(sql.SQL('create schema {}').format(sql.Identifier(cls.schema)))
            c.execute(sql.SQL('set search_path to {}, public').format(sql.Identifier(cls.schema)))
            c.execute('create table "user" (id text primary key, banned boolean not null default false)')
            c.execute((Path(__file__).resolve().parents[1] / 'database/label_studio_schema.sql').read_text())

    @classmethod
    def tearDownClass(cls):
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute(sql.SQL('drop schema {} cascade').format(sql.Identifier(cls.schema)))

    def connect(self):
        return psycopg.connect(DSN, row_factory=dict_row, options=f'-c search_path={self.schema},public')

    def setUp(self):
        with self.connect() as c:
            c.execute('truncate label_studio_operations, label_studio_projects, label_studio_accounts, "user" cascade')
            c.execute('insert into "user"(id,banned) values (%s,true),(%s,true)', ('alice', 'bob'))
            c.execute('insert into label_studio_accounts(owner_user_id,ls_user_id) values (%s,1),(%s,2)', ('alice', 'bob'))
            self.alice_project = c.execute(
                "insert into label_studio_projects(owner_user_id,ls_project_id,name,kind) values('alice',1,'Alice','object_detection') returning id"
            ).fetchone()['id']
            self.bob_project = c.execute(
                "insert into label_studio_projects(owner_user_id,ls_project_id,name,kind) values('bob',2,'Bob','object_detection') returning id"
            ).fetchone()['id']

    def test_removes_foreign_keys_only_for_owner_and_is_retryable(self):
        with self.connect() as c:
            for owner, project in (('alice', self.alice_project), ('bob', self.bob_project)):
                c.execute("insert into label_studio_operations(owner_user_id,project_id,kind,status) values(%s,%s,'publish','completed')", (owner, project))
        with patch.object(jobs.registry, '_connect', side_effect=self.connect), patch.object(jobs, 'bridge') as bridge:
            jobs.delete_account_data('alice')
            jobs.delete_account_data('alice')
            bridge.assert_called_once_with('alice', 'delete-account')
        with self.connect() as c:
            self.assertIsNone(c.execute("select 1 from label_studio_accounts where owner_user_id='alice'").fetchone())
            self.assertIsNotNone(c.execute("select 1 from label_studio_operations where owner_user_id='bob'").fetchone())
            c.execute("delete from \"user\" where id='alice'")
            self.assertIsNotNone(c.execute("select 1 from \"user\" where id='bob'").fetchone())

    def test_active_operation_and_bridge_failure_preserve_records(self):
        with self.connect() as c:
            c.execute("insert into label_studio_operations(owner_user_id,project_id,kind,status) values('alice',%s,'publish','running')", (self.alice_project,))
        with patch.object(jobs.registry, '_connect', side_effect=self.connect), patch.object(jobs, 'bridge') as bridge:
            with self.assertRaisesRegex(ValueError, 'active Label Studio'):
                jobs.delete_account_data('alice')
            bridge.assert_not_called()
        with self.connect() as c:
            c.execute("update label_studio_operations set status='completed' where owner_user_id='alice'")
        with patch.object(jobs.registry, '_connect', side_effect=self.connect), patch.object(jobs, 'bridge', side_effect=RuntimeError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                jobs.delete_account_data('alice')
        with self.connect() as c:
            self.assertIsNotNone(c.execute("select 1 from label_studio_accounts where owner_user_id='alice'").fetchone())
            self.assertIsNotNone(c.execute("select 1 from label_studio_operations where owner_user_id='alice'").fetchone())


if __name__ == '__main__':
    unittest.main()
