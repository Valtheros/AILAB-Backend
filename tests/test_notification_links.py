import unittest
from notification_api import target_path


class NotificationLinkTests(unittest.TestCase):
    def test_old_compute_events_link_to_the_correct_job(self):
        job = '4e3d82ca-1aa6-4db1-9581-0087e3b931a4'
        row = {'target_path': '/tasks/run', 'event_key': f'compute:{job}:completed', 'kind': 'evaluate'}
        self.assertEqual(target_path(row), f'/tasks/run?evaluationJob={job}')
        row['kind'] = 'batch_predict'
        self.assertEqual(target_path(row), f'/tasks/run?predictionJob={job}&predictionKind=batch_predict')
        row['kind'] = 'predict'
        self.assertEqual(target_path(row), f'/tasks/run?predictionJob={job}&predictionKind=predict')
        row['event_key'] = 'unrecognized'
        self.assertEqual(target_path(row), '/tasks/run')


if __name__ == '__main__':
    unittest.main()
