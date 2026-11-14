import http from 'k6/http';
import { check, sleep } from 'k6';

const base = __ENV.AILAB_LOAD_BASE_URL || 'http://127.0.0.1:18080';
const user = __ENV.AILAB_LOAD_USER;
const token = __ENV.AILAB_LOAD_INTERNAL_TOKEN;
if (!/^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(base) || !user || !token) {
  throw new Error('Use an isolated loopback server and supply AILAB_LOAD_USER and AILAB_LOAD_INTERNAL_TOKEN.');
}

export const options = {
  vus: 5,
  duration: __ENV.AILAB_LOAD_DURATION || '20s',
  summaryTrendStats: ['avg', 'med', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    'http_req_duration{name:catalog}': ['p(99)<5000'],
    'http_req_duration{name:datasets}': ['p(99)<5000'],
    'http_req_duration{name:devices}': ['p(99)<5000'],
    'http_req_duration{name:notifications}': ['p(99)<5000'],
  },
};

export default function () {
  const headers = { 'x-internal-token': token, 'x-user-id': user };
  const routes = [
    ['catalog', '/api/model-catalog'], ['notifications', '/api/notifications'],
    ['devices', '/api/compute/devices'], ['datasets', '/api/datasets'],
  ];
  const responses = http.batch(routes.map(([name, path]) => ({
    method: 'GET', url: base + path, params: { headers, tags: { name }, timeout: '10s' },
  })));
  for (const response of responses) check(response, { 'HTTP 200': (item) => item.status === 200 });
  sleep(0.3);
}
