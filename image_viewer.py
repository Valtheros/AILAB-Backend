"""Display source pixels in the same orientation used by trainers."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.responses import FileResponse, Response
from PIL import Image

from api_errors import PublicError
from worker.trainers.input_limits import open_image_checked


def viewer_image(path: Path):
    try:
        with open_image_checked(path) as image:
            orientation = image.getexif().get(274, 1)
            if orientation not in (None, 1):
                buffer = BytesIO()
                image.convert('RGB').save(buffer, format='JPEG', quality=95)
                return Response(buffer.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'private, no-store'})
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise PublicError('RESULT_UNAVAILABLE', 'This image is unavailable.', status=410) from exc
    return FileResponse(path, headers={'Cache-Control': 'private, no-store'})
