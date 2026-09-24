from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_service import THUMBNAIL_SIZE, _project_root, _sha256
from resource_repository import resource_repository
from security_utils import contained_path


updated = 0
skipped = 0
with resource_repository._connect() as connection:
    rows = connection.execute(
        """select i.id, i.storage_path, i.thumbnail_path, i.content_sha256,
                  p.id project_id, p.owner_user_id
           from annotation_images i join annotation_projects p on p.id = i.project_id
           where i.content_sha256 is null or i.thumbnail_path is null"""
    ).fetchall()
    for row in rows:
        root = _project_root(row["owner_user_id"], str(row["project_id"]))
        try:
            source = contained_path(root, Path(row["storage_path"]))
            if not source.is_file():
                raise FileNotFoundError(source)
            content_hash = row["content_sha256"] or _sha256(source)
            thumbnail = (
                contained_path(root, Path(row["thumbnail_path"]))
                if row["thumbnail_path"]
                else contained_path(root, "thumbnails", f"{row['id']}.webp")
            )
            with Image.open(source) as opened:
                preview = ImageOps.exif_transpose(opened)
                preview.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                if preview.mode != "RGB":
                    preview = preview.convert("RGB")
                thumbnail.parent.mkdir(parents=True, exist_ok=True)
                if not thumbnail.exists():
                    preview.save(thumbnail, "WEBP", quality=82)
                width, height = preview.size
            connection.execute(
                """update annotation_images set content_sha256 = %s, thumbnail_path = %s,
                          thumbnail_width = %s, thumbnail_height = %s
                   where id = %s""",
                (content_hash, str(thumbnail), width, height, row["id"]),
            )
            connection.commit()
            updated += 1
        except Exception as exc:
            connection.rollback()
            skipped += 1
            print(f"Skipped {row['id']}: {exc}")

print(f"Annotation media backfill updated={updated} skipped={skipped}")
