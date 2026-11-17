"""Model Apps reuse the API fixtures and must preserve their isolation guarantees."""
import io
import json
import uuid
from unittest.mock import patch

from PIL import Image
import tests.test_model_api_integration as fixtures


class ModelAppIntegrationTests(fixtures.ModelAPIIntegrationTests):
    def setUp(self):
        super().setUp()
        # No GPU is needed to test queue admission and ownership.
        self.target = patch('compute_repository.validate_target')
        self.target.start()
        self.addCleanup(self.target.stop)

    def app(self, subject='alice', slug='test-model'):
        response = self.client.post('/api/model-apps', headers=self.session(subject), json={
            'name': 'Test model', 'slug': slug, 'runSlug': self.runs[subject]['slug']})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def predict_app(self, subject='alice', slug='test-model', version=1, key='once'):
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(image, format='PNG')
        return self.client.post(f'/api/apps/alice/{slug}/predict' + (f'?version={version}' if version else ''),
            headers={**self.session(subject), 'idempotency-key': key}, files={'file': ('input.png', image.getvalue(), 'image/png')})

    def complete(self, job_id):
        import compute_repository as compute
        token = uuid.uuid4()
        with self.registry._connect() as c:
            c.execute("update compute_jobs set status='running',dispatch_token=%s,started_at=now() where id=%s", (token, job_id))
        (compute.job_directory(job_id) / 'result.json').write_text(json.dumps({
            'taskType': 'image_classification', 'predictions': [{'className': 'red', 'confidence': .9}],
            'model': {'checkpoint': '/private/best.pt'}, 'runSlug': 'private-run'}))
        compute.finish_job(job_id, token, 'completed')

    def publish(self, app):
        job = self.predict_app().json()
        self.complete(job['id'])
        response = self.client.patch(f"/api/model-apps/{app['id']}", headers=self.session('alice'),
                                    json={'version': 1, 'published': True})
        self.assertEqual(response.status_code, 200, response.text)

    def test_preview_publish_and_results_are_requester_private(self):
        app = self.app()
        self.assertEqual(self.client.patch(f"/api/model-apps/{app['id']}", headers=self.session('alice'),
                         json={'version': 1, 'published': True}).json()['code'], 'APP_TEST_REQUIRED')
        self.assertEqual(self.client.get('/api/apps/alice/test-model?version=1', headers=self.session('bob')).status_code, 404)
        self.publish(app)
        response = self.client.get('/api/apps/alice/test-model', headers=self.session('bob'))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn('releases', response.json())
        self.assertNotIn('usage', response.json())
        submitted = self.predict_app('bob', version=None)
        self.assertEqual(submitted.status_code, 202, submitted.text)
        job_id = submitted.json()['id']
        self.assertEqual(self.predict_app('bob', version=None).json()['id'], job_id)
        self.assertEqual(self.client.get(f'/api/compute/jobs/{job_id}', headers=self.session('alice')).status_code, 404)
        for suffix in ('', '/result', '/files/input'):
            self.assertEqual(self.client.get(f'/api/apps/jobs/{job_id}{suffix}', headers=self.session('alice')).status_code, 404)
        self.complete(job_id)
        result = self.client.get(f'/api/apps/jobs/{job_id}/result', headers=self.session('bob')).json()
        self.assertNotIn('model', result)
        self.assertNotIn('runSlug', result)
        self.assertEqual(result['version'], 1)
        with self.registry._connect() as c:
            row = c.execute('select * from compute_jobs where id=%s', (job_id,)).fetchone()
            self.assertEqual((row['owner_user_id'], row['app_requester_user_id']), ('alice', 'bob'))
            notices = c.execute("select owner_user_id,target_path from notifications where event_key=%s", (f'compute:{job_id}:completed',)).fetchall()
            self.assertEqual(notices[0]['owner_user_id'], 'bob')
            self.assertIn('/apps/alice/test-model?job=', notices[0]['target_path'])

    def test_unpublish_does_not_cancel_or_expose_existing_requests(self):
        app = self.app()
        self.publish(app)
        job = self.predict_app('bob', version=None).json()
        self.assertEqual(self.client.delete(f"/api/model-apps/{app['id']}", headers=self.session('alice')).status_code, 409)
        self.client.patch(f"/api/model-apps/{app['id']}", headers=self.session('alice'), json={'published': False})
        self.assertEqual(self.predict_app('bob', version=None, key='new').status_code, 404)
        self.assertEqual(self.client.post(f"/api/apps/jobs/{job['id']}/cancel", headers=self.session('alice')).status_code, 404)
        self.assertEqual(self.client.post(f"/api/apps/jobs/{job['id']}/cancel", headers=self.session('bob')).json()['status'], 'cancelled')
        self.assertEqual(self.client.delete(f"/api/model-apps/{app['id']}", headers=self.session('alice')).status_code, 200)
        self.assertEqual(self.client.get(f"/api/compute/jobs/{job['id']}", headers=self.session('alice')).status_code, 404)
        retained = self.client.get(f"/api/apps/alice/test-model?job={job['id']}", headers=self.session('bob'))
        self.assertEqual(retained.status_code, 200, retained.text)
        self.assertFalse(retained.json()['published'])
        with self.registry._connect() as c:
            c.execute("update compute_jobs set finished_at=now()-interval '25 hours' where id=%s", (job['id'],))
        self.assertEqual(self.client.get(f"/api/apps/jobs/{job['id']}", headers=self.session('bob')).status_code, 410)

    def test_account_cleanup_checks_visitor_jobs_before_deleting_label_studio_data(self):
        import main
        from api_errors import PublicError
        self.publish(self.app())
        self.assertEqual(self.predict_app('bob', version=None).status_code, 202)
        with patch('main._require_admin'), patch('label_studio_jobs.delete_account_data') as delete_labels:
            with self.assertRaises(PublicError):
                main.admin_delete_user_resources('bob', None)
            delete_labels.assert_not_called()

    def test_session_auth_errors_in_middleware_are_not_server_errors(self):
        self.app()
        anonymous = dict(self.session('bob'))
        anonymous.pop('x-user-id')
        self.assertEqual(self.client.get('/api/apps/alice/test-model', headers=anonymous).status_code, 401)
        with self.registry._connect() as c:
            c.execute('update "user" set banned=true where id=%s', ('bob',))
        self.assertEqual(self.client.get('/api/apps/alice/test-model', headers=self.session('bob')).status_code, 403)
        self.assertEqual(self.predict_app('bob', version=None).status_code, 403)

    def test_run_deletion_ownership_and_input_validation(self):
        app = self.app()
        for method, path, data in [('PATCH', f"/api/model-apps/{app['id']}", {'published': True}),
                                  ('DELETE', f"/api/model-apps/{app['id']}", None),
                                  ('POST', f"/api/model-apps/{app['id']}/releases", {'runSlug': self.runs['bob']['slug']})]:
            self.assertEqual(self.client.request(method, path, headers=self.session('bob'), json=data).status_code, 404)
        with self.registry._connect() as c:
            with self.assertRaises(Exception):
                c.execute('delete from training_tasks where id=%s', (self.runs['alice']['id'],))
        bad = self.client.post('/api/apps/alice/test-model/predict?version=1', headers={**self.session('alice'), 'idempotency-key': 'bad'},
                               files={'file': ('bad.png', b'invalid', 'image/png')})
        self.assertEqual(bad.status_code, 400, bad.text)
        self.assertEqual(bad.json()['code'], 'INVALID_IMAGE')
        self.assertEqual(self.client.get('/api/model-apps', headers={'x-user-id': 'alice'}).status_code, 401)
        with self.registry._connect() as c:
            c.execute('update "user" set banned=true where id=%s', ('alice',))
        self.assertEqual(self.client.get('/api/apps/alice/test-model', headers=self.session('bob')).status_code, 404)

    def test_release_switch_keeps_queued_checkpoint_and_rollback(self):
        app = self.app()
        self.publish(app)
        waiting = self.predict_app('bob', version=None).json()
        old_hash = self.client.get('/api/model-apps', headers=self.session('alice')).json()['apps'][0]['releases'][0]['checkpointSha256']
        # Use a different completed run, not a modified checkpoint of the old release.
        with self.registry._connect() as c:
            c.execute("update training_tasks set owner_user_id='alice' where id=%s", (self.runs['bob']['id'],))
        response = self.client.post(f"/api/model-apps/{app['id']}/releases", headers=self.session('alice'), json={'runSlug': self.runs['bob']['slug']})
        self.assertEqual(response.status_code, 201, response.text)
        preview = self.predict_app(version=2, key='second').json()
        self.complete(preview['id'])
        promoted = self.client.patch(f"/api/model-apps/{app['id']}", headers=self.session('alice'), json={'version': 2})
        self.assertEqual(promoted.json()['activeVersion'], 2)
        with self.registry._connect() as c:
            row = c.execute('select payload from compute_jobs where id=%s', (waiting['id'],)).fetchone()
            self.assertEqual((row['payload']['app_version'], row['payload']['checkpoint_sha256']), (1, old_hash))
        rolled_back = self.client.patch(f"/api/model-apps/{app['id']}", headers=self.session('alice'), json={'version': 1})
        self.assertEqual(rolled_back.json()['activeVersion'], 1)

    def test_changed_checkpoint_and_pending_visitor_deletion_are_blocked(self):
        self.app()
        job = self.predict_app().json()
        with self.assertRaisesRegex(Exception, 'active Model App'):
            self.registry.delete_user_resource_records('alice')
        self.client.post(f"/api/apps/jobs/{job['id']}/cancel", headers=self.session('alice'))
        (self.runs['alice']['path'] / 'best.pt').write_bytes(b'changed')
        response = self.predict_app(key='changed')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['code'], 'CHECKPOINT_CHANGED')

    def test_transfer_disables_app_and_requires_new_owner_preview(self):
        app = self.app()
        self.publish(app)
        self.registry.transfer_user_resources('alice', 'bob')
        response = self.client.get('/api/model-apps', headers=self.session('bob'))
        moved = response.json()['apps'][0]
        self.assertFalse(moved['published'])
        self.assertIsNone(moved['releases'][0]['testedAt'])
        self.assertEqual(moved['path'], '/apps/bob/test-model')
        self.assertEqual(self.client.get('/api/apps/alice/test-model', headers=self.session('bob')).status_code, 404)

    def test_account_cleanup_removes_app_before_protected_run(self):
        self.app()
        self.registry.delete_user_resource_records('alice')
        self.registry.finalize_user_resource_delete('alice')
        with self.registry._connect() as c:
            self.assertFalse(c.execute("select 1 from model_apps where owner_user_id='alice'").fetchone())
            self.assertTrue(c.execute("select 1 from training_tasks where owner_user_id='bob'").fetchone())

    def test_unpublished_app_retains_requester_result_page(self):
        app = self.app()
        self.publish(app)
        job = self.predict_app('bob', version=None).json()
        self.client.patch(f"/api/model-apps/{app['id']}", headers=self.session('alice'), json={'published': False})
        page = self.client.get(f"/api/apps/alice/test-model?job={job['id']}", headers=self.session('bob'))
        self.assertEqual(page.status_code, 200, page.text)
        self.assertFalse(page.json()['published'])
        self.assertEqual(self.client.get(f"/api/apps/bob/test-model?job={job['id']}", headers=self.session('bob')).status_code, 404)
