from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from fastapi.responses import FileResponse, Response

from image_viewer import viewer_image


class ImageViewerTests(unittest.TestCase):
    def test_exif_orientation_is_removed_without_rotating_trainer_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'image.jpg'
            image = Image.new('RGB', (20, 10), 'red')
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            response = viewer_image(path)
            self.assertIsInstance(response, Response)
            with Image.open(io.BytesIO(response.body)) as visible:
                self.assertEqual(visible.size, (20, 10))
                self.assertNotIn(274, visible.getexif())
            with Image.open(path) as original:
                self.assertEqual(original.getexif()[274], 6)

    def test_unrotated_source_is_served_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'image.png'
            Image.new('RGB', (4, 4)).save(path)
            self.assertIsInstance(viewer_image(path), FileResponse)
