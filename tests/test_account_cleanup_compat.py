import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from main import _assert_legacy_user_cleanup_supported, resource_repository


class AccountCleanupCompatibilityTests(unittest.TestCase):
    def test_absent_or_other_users_new_tables_do_not_block_cleanup(self):
        context = MagicMock()
        connection = context.__enter__.return_value

        def execute(query, params):
            result = MagicMock()
            result.fetchone.return_value = {'name': 'evaluations'} if 'to_regclass' in query and params[0] == 'evaluations' else ({'name': None} if 'to_regclass' in query else None)
            return result

        connection.execute.side_effect = execute
        with patch.object(resource_repository, '_connect', return_value=context):
            _assert_legacy_user_cleanup_supported('owner')

    def test_owned_new_feature_data_blocks_cleanup(self):
        context = MagicMock()
        connection = context.__enter__.return_value

        def execute(query, params):
            result = MagicMock()
            result.fetchone.return_value = {'name': 'evaluations'} if 'to_regclass' in query and params[0] == 'evaluations' else ({'name': None} if 'to_regclass' in query else {'exists': 1})
            return result

        connection.execute.side_effect = execute
        with patch.object(resource_repository, '_connect', return_value=context), self.assertRaises(HTTPException) as error:
            _assert_legacy_user_cleanup_supported('owner')
        self.assertEqual(error.exception.status_code, 409)
        self.assertIn('newer AILAB feature', error.exception.detail)


if __name__ == '__main__':
    unittest.main()
