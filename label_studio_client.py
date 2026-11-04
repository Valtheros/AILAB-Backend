"""Server-only bridge to our pinned Label Studio fork, never to user supplied URLs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler


class BridgeError(Exception):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def public_url():
    value = os.environ.get('LABEL_STUDIO_PUBLIC_URL', '').rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BridgeError('Label Studio public URL is not configured correctly.')
    return value


def bridge(owner_id: str, action: str, payload: dict | None = None, *, content: Path | None = None):
    base = os.environ.get("LABEL_STUDIO_INTERNAL_URL", "http://label-studio:8080").rstrip("/")
    secret = os.environ.get("LABEL_STUDIO_BRIDGE_SECRET", "")
    if not secret:
        raise BridgeError("Label Studio is not configured.")
    headers = {"Authorization": f"Bearer {secret}", "X-AILAB-User": owner_id}
    if content is not None:
        data = content.open("rb")
        headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(content.stat().st_size)})
        url = f"{base}/ailab/internal/{action}/?{urlencode(payload or {})}"
    else:
        data = json.dumps(payload or {}, allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
        url = f"{base}/ailab/internal/{action}/"
    try:
        with build_opener(NoRedirect).open(Request(url, data=data, headers=headers, method="POST"), timeout=90) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            message = json.loads(exc.read(8192)).get("detail", "Label Studio request failed.")
        except (ValueError, AttributeError):
            message = "Label Studio request failed."
        raise BridgeError(str(message), exc.code if 400 <= exc.code < 500 else 503) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise BridgeError("Label Studio is unavailable. Your saved work has not been removed.") from exc
    finally:
        if content is not None:
            data.close()


def download(owner_id: str, action: str, payload: dict, target: Path, max_bytes: int = 150_000_000):
    base = os.environ.get("LABEL_STUDIO_INTERNAL_URL", "http://label-studio:8080").rstrip("/")
    secret = os.environ.get("LABEL_STUDIO_BRIDGE_SECRET", "")
    if not secret:
        raise BridgeError("Label Studio is not configured.")
    request = Request(f"{base}/ailab/internal/{action}/", data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {secret}", "X-AILAB-User": owner_id, "Content-Type": "application/json",
    })
    try:
        with build_opener(NoRedirect).open(request, timeout=90) as response, target.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Downloaded image exceeds the allowed size.")
                output.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        target.unlink(missing_ok=True)
        raise BridgeError("Could not download the project image. Retry the operation.") from exc
