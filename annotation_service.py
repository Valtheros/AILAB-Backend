from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from dataset_storage import owner_dataset_path, owner_storage_key
from dataset_utils import (
    DATASET_METADATA_VERSION,
    MAX_COCO_ANNOTATIONS_PER_IMAGE,
    MAX_COCO_IMAGES,
    MAX_COCO_MASKS_PER_IMAGE,
    MAX_COCO_POLYGON_POINTS,
    MAX_SOURCE_IMAGE_PIXELS,
    dataset_workflow_metadata,
    inspect_dataset,
    safe_dataset_name,
    validate_dataset_for_upload,
)
from model_catalog import get_catalog
from resource_repository import resource_repository
from security_utils import (
    ReversibleDirectoryRemoval,
    contained_path,
    named_file_lock,
    staging_directory,
    validate_slug,
)
from settings import DATASET_DIR


TASK_TYPES = {
    "image_classification",
    "object_detection",
    "semantic_segmentation",
    "instance_segmentation",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")
MAX_CLASSES = 1_000
MAX_REVISIONS = 20
REVISION_INTERVAL_SECONDS = 30
THUMBNAIL_SIZE = 512
OperationCallback = Callable[[int, int, int], None]
CancelCallback = Callable[[], bool]


def _json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def _class_list(value: Any) -> list[dict[str, str]]:
    raw = _json_value(value, [])
    if not isinstance(raw, list) or not raw or len(raw) > MAX_CLASSES:
        raise ValueError(f"Classes must contain between 1 and {MAX_CLASSES} entries")
    result: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            item = {"id": f"class-{index + 1}", "name": item}
        if not isinstance(item, dict):
            raise ValueError("Each class must be an object")
        class_id = str(item.get("id") or f"class-{index + 1}").strip()
        name = str(item.get("name") or "").strip()
        color = str(item.get("color") or "#22c55e").strip()
        if not class_id or len(class_id) > 80 or not name or len(name) > 100:
            raise ValueError("Every class requires a short id and name")
        if class_id in seen_ids or name.casefold() in seen_names:
            raise ValueError("Class ids and names must be unique")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError(f"Invalid class color for {name}")
        seen_ids.add(class_id)
        seen_names.add(name.casefold())
        result.append({"id": class_id, "name": name, "color": color.lower()})
    return result


def _validate_class_count(task_type: str, classes: list[dict[str, str]]) -> None:
    if task_type == "image_classification" and len(classes) < 2:
        raise ValueError("Image classification requires at least two classes")


def _ratios(train: int, val: int, test: int) -> tuple[int, int, int]:
    if train < 1 or val < 0 or test < 0 or train + val + test != 100:
        raise ValueError("Train, validation, and test ratios must total 100; train must be positive")
    return train, val, test


def _project_root(owner_id: str, project_id: str) -> Path:
    return contained_path(DATASET_DIR, ".annotations", owner_storage_key(owner_id), project_id)


def _safe_class_folder(name: str, index: int) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return f"{index:04d}_{normalized[:90] or 'class'}"


def _allocation(total: int, ratios: tuple[int, int, int]) -> tuple[int, int, int]:
    exact = [total * ratio / 100 for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    for index in sorted(range(3), key=lambda item: exact[item] - counts[item], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    if total and counts[0] == 0:
        donor = max(range(1, 3), key=lambda item: counts[item])
        if counts[donor]:
            counts[donor] -= 1
            counts[0] = 1
    return counts[0], counts[1], counts[2]


def _stable_rows(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}\0{row['file_name']}".encode()).digest())


def _assign_initial_splits(rows: list[dict[str, Any]], project: dict[str, Any]) -> dict[str, str]:
    if project.get("task_type") == "image_classification":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            annotations = _json_value(row.get("annotations"), [])
            class_id = str(annotations[0].get("classId")) if annotations else ""
            if not class_id:
                break
            grouped[class_id].append(row)
        else:
            assigned: dict[str, str] = {}
            for class_id, class_rows in grouped.items():
                ordered = _stable_rows(class_rows, f"{project['split_seed']}\0{class_id}")
                train_count, val_count, _ = _allocation(
                    len(ordered), (project["train_ratio"], project["val_ratio"], project["test_ratio"])
                )
                assigned.update({
                    str(row["id"]): "train" if index < train_count else "val" if index < train_count + val_count else "test"
                    for index, row in enumerate(ordered)
                })
            return assigned

    ordered = _stable_rows(rows, project["split_seed"])
    train_count, val_count, _ = _allocation(
        len(ordered), (project["train_ratio"], project["val_ratio"], project["test_ratio"])
    )
    return {
        str(row["id"]): "train" if index < train_count else "val" if index < train_count + val_count else "test"
        for index, row in enumerate(ordered)
    }


def _assign_splits(rows: list[dict[str, Any]], project: dict[str, Any], rebalance_all: bool = False) -> dict[str, str]:
    automatic = _assign_initial_splits(rows, project)
    if rebalance_all:
        return automatic
    return {
        str(row["id"]): (
            str(row["split"])
            if row.get("split_source") == "manual" and row.get("split") in SPLITS
            else automatic[str(row["id"])]
        )
        for row in rows
    }


def _clean_points(points: Any, minimum: int, remaining: int, label: str) -> list[list[float]]:
    if not isinstance(points, list) or len(points) < minimum or len(points) > remaining:
        raise ValueError(f"{label} exceeds the cumulative {MAX_COCO_POLYGON_POINTS} point limit")
    cleaned: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{label} points must be [x, y] pairs")
        x, y = point
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in (x, y)):
            raise ValueError(f"{label} points must be finite and inside the image")
        cleaned.append([float(x), float(y)])
    return cleaned


def _validate_annotations(task_type: str, annotations: Any, marked_empty: bool, class_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(annotations, list):
        raise ValueError("Annotations must be a list")
    if task_type == "image_classification" and marked_empty:
        raise ValueError("Classification images cannot be marked empty")
    if marked_empty and annotations:
        raise ValueError("An empty image cannot also contain annotations")
    limit = MAX_COCO_MASKS_PER_IMAGE if "segmentation" in task_type else MAX_COCO_ANNOTATIONS_PER_IMAGE
    if len(annotations) > limit:
        raise ValueError("Image contains too many annotations")

    cleaned: list[dict[str, Any]] = []
    point_count = 0
    for item in annotations:
        if not isinstance(item, dict):
            raise ValueError("Each annotation must be an object")
        annotation_type = str(item.get("type") or "")
        allowed = (
            {"classification"} if task_type == "image_classification"
            else {"rectangle"} if task_type == "object_detection"
            else {"polygon", "brush"}
        )
        if annotation_type not in allowed:
            raise ValueError(f"{task_type} does not support {annotation_type or 'unknown'} annotations")
        class_id = str(item.get("classId") or "")
        if class_id not in class_ids:
            raise ValueError("Annotation references an unknown class")
        annotation_id = str(item.get("id") or uuid.uuid4())
        if len(annotation_id) > 128:
            raise ValueError("Annotation id is too long")

        if annotation_type == "classification":
            cleaned.append({"id": annotation_id, "type": annotation_type, "classId": class_id})
            continue
        if annotation_type == "rectangle":
            values = [item.get(key) for key in ("x", "y", "width", "height")]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                raise ValueError("Rectangle coordinates must be finite numbers")
            x, y, width, height = (float(value) for value in values)
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
                raise ValueError("Rectangle must stay inside the image")
            cleaned.append({"id": annotation_id, "type": annotation_type, "classId": class_id, "x": x, "y": y, "width": width, "height": height})
            continue

        minimum = 3 if annotation_type == "polygon" else 2
        points = _clean_points(item.get("points"), minimum, MAX_COCO_POLYGON_POINTS - point_count, annotation_type.title())
        point_count += len(points)
        record: dict[str, Any] = {"id": annotation_id, "type": annotation_type, "classId": class_id, "points": points}
        if task_type == "instance_segmentation":
            instance_id = str(item.get("instanceId") or annotation_id)
            if not instance_id or len(instance_id) > 128:
                raise ValueError("Instance id is required and must be short")
            record["instanceId"] = instance_id
        if annotation_type == "brush":
            radius = item.get("radius")
            mode = str(item.get("mode") or "paint")
            if not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 < radius <= 0.25:
                raise ValueError("Brush radius must be between 0 and 0.25")
            if mode not in {"paint", "erase"}:
                raise ValueError("Brush mode must be paint or erase")
            record.update({"radius": float(radius), "mode": mode})
        cleaned.append(record)

    if task_type == "image_classification" and len(cleaned) > 1:
        raise ValueError("Classification accepts one class per image")
    return cleaned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_normalized_image(source: Path, destination: Path, thumbnail: Path) -> tuple[int, int, int, int]:
    with Image.open(source) as opened:
        width, height = opened.size
        if width <= 0 or height <= 0 or width * height > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError(f"Image exceeds the {MAX_SOURCE_IMAGE_PIXELS} pixel limit")
        image = ImageOps.exif_transpose(opened)
        image.load()
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError(f"Image exceeds the {MAX_SOURCE_IMAGE_PIXELS} pixel limit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        elif suffix == ".webp":
            image.convert("RGB").save(destination, format="WEBP", quality=95)
        else:
            if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
                image = image.convert("RGBA")
            image.save(destination, format="PNG")
        preview = image.copy()
        preview.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
        if preview.mode != "RGB":
            background = Image.new("RGB", preview.size, "black")
            if "A" in preview.getbands():
                background.paste(preview, mask=preview.getchannel("A"))
            else:
                background.paste(preview.convert("RGB"))
            preview = background
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        preview.save(thumbnail, format="WEBP", quality=82)
        return width, height, preview.width, preview.height


class AnnotationService:
    def _connect(self):
        return resource_repository._connect()

    @staticmethod
    def _editable_project(connection, owner_id: str, project_id: str):
        project = connection.execute(
            "select * from annotation_projects where owner_user_id = %s and id = %s for update",
            (owner_id, project_id),
        ).fetchone()
        if not project:
            raise FileNotFoundError("Annotation project was not found")
        if project["status"] == "publishing":
            raise RuntimeError("Project is being published; wait until publishing finishes")
        return project

    @staticmethod
    def validate_project_input(name: str, task_type: str, classes: Any, ratios: tuple[int, int, int]) -> tuple[str, list[dict[str, str]], tuple[int, int, int]]:
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("Project name must contain 1-100 characters")
        if task_type not in TASK_TYPES:
            raise ValueError("Unsupported annotation task type")
        class_list = _class_list(classes)
        _validate_class_count(task_type, class_list)
        return name, class_list, _ratios(*ratios)

    def create_project_record(self, owner_id: str, name: str, task_type: str, classes: Any, ratios: tuple[int, int, int]) -> dict[str, Any]:
        name, class_list, (train, val, test) = self.validate_project_input(name, task_type, classes, ratios)
        with self._connect() as connection:
            row = connection.execute(
                """insert into annotation_projects
                   (owner_user_id, name, task_type, classes, train_ratio, val_ratio, test_ratio)
                   values (%s, %s, %s, %s::jsonb, %s, %s, %s) returning *""",
                (owner_id, name, task_type, json.dumps(class_list), train, val, test),
            ).fetchone()
        project_id = str(row["id"])
        _project_root(owner_id, project_id).mkdir(parents=True, exist_ok=True)
        return self.get_project(owner_id, project_id) or self._project_dict(row)

    def import_directory(
        self,
        owner_id: str,
        project_id: str,
        operation_id: str,
        extracted_dir: Path,
        progress: OperationCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> dict[str, Any]:
        project = self.get_project(owner_id, project_id)
        if not project:
            raise FileNotFoundError("Annotation project was not found")
        source_files = sorted(
            path for path in extracted_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not source_files:
            raise ValueError("ZIP does not contain supported JPEG, PNG, or WebP images")
        if len(source_files) > MAX_COCO_IMAGES:
            raise ValueError(f"ZIP contains more than {MAX_COCO_IMAGES} images")

        root = _project_root(owner_id, project_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        with named_file_lock(root.parent, project_id, "annotation project"):
            stage = staging_directory(root.parent)
            image_stage = contained_path(stage, "images")
            thumb_stage = contained_path(stage, "thumbnails")
            rows: list[dict[str, Any]] = []
            skipped: list[dict[str, str]] = []
            seen_hashes: set[str] = set()
            try:
                with self._connect() as connection:
                    existing = connection.execute(
                        "select content_sha256 from annotation_images where project_id = %s and content_sha256 is not null",
                        (project_id,),
                    ).fetchall()
                    seen_hashes.update(row["content_sha256"] for row in existing)
                    start_order = connection.execute(
                        "select coalesce(max(sort_order), -1)::int value from annotation_images where project_id = %s",
                        (project_id,),
                    ).fetchone()["value"] + 1

                for index, source in enumerate(source_files):
                    if cancelled and cancelled():
                        raise InterruptedError("Import cancelled")
                    relative = source.relative_to(extracted_dir).as_posix()
                    image_id = str(uuid.uuid4())
                    suffix = source.suffix.lower()
                    if suffix == ".jpeg":
                        suffix = ".jpg"
                    destination = contained_path(image_stage, f"{image_id}{suffix}")
                    thumbnail = contained_path(thumb_stage, f"{image_id}.webp")
                    try:
                        width, height, thumbnail_width, thumbnail_height = _save_normalized_image(
                            source, destination, thumbnail
                        )
                        content_hash = _sha256(destination)
                        if content_hash in seen_hashes:
                            destination.unlink(missing_ok=True)
                            thumbnail.unlink(missing_ok=True)
                            skipped.append({"file": relative, "reason": "duplicate content"})
                        else:
                            seen_hashes.add(content_hash)
                            rows.append({
                                "id": image_id,
                                "file_name": relative,
                                "source_path": destination,
                                "thumbnail_source": thumbnail,
                                "width": width,
                                "height": height,
                                "thumbnail_width": thumbnail_width,
                                "thumbnail_height": thumbnail_height,
                                "content_sha256": content_hash,
                                "sort_order": start_order + len(rows),
                            })
                    except (UnidentifiedImageError, OSError, ValueError) as exc:
                        destination.unlink(missing_ok=True)
                        thumbnail.unlink(missing_ok=True)
                        skipped.append({"file": relative, "reason": str(exc)[:240]})
                    if progress:
                        progress(index + 1, len(source_files), len(skipped))

                if not rows:
                    raise ValueError("No usable new images were found in the ZIP")
                if cancelled and cancelled():
                    raise InterruptedError("Import cancelled")

                batch = contained_path(root, "batches", operation_id)
                batch.parent.mkdir(parents=True, exist_ok=True)
                if batch.exists():
                    raise FileExistsError("This import operation has already been committed")
                stage.replace(batch)
                try:
                    with self._connect() as connection:
                        with connection.transaction():
                            locked = connection.execute(
                                "select * from annotation_projects where owner_user_id = %s and id = %s for update",
                                (owner_id, project_id),
                            ).fetchone()
                            if not locked:
                                raise FileNotFoundError("Annotation project was not found")
                            for row in rows:
                                image_path = contained_path(batch, "images", row["source_path"].name)
                                thumb_path = contained_path(batch, "thumbnails", row["thumbnail_source"].name)
                                connection.execute(
                                    """insert into annotation_images
                                       (id, project_id, file_name, storage_path, width, height, sort_order,
                                        content_sha256, thumbnail_path, thumbnail_width, thumbnail_height, split_source)
                                       values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'auto')""",
                                    (
                                        row["id"], project_id, row["file_name"], str(image_path), row["width"], row["height"],
                                        row["sort_order"], row["content_sha256"], str(thumb_path),
                                        row["thumbnail_width"], row["thumbnail_height"],
                                    ),
                                )
                            all_rows = [dict(item) for item in connection.execute(
                                "select id, file_name, split, split_source, annotations from annotation_images where project_id = %s",
                                (project_id,),
                            ).fetchall()]
                            assignments = _assign_splits(all_rows, dict(locked))
                            for image_id, split in assignments.items():
                                connection.execute(
                                    """update annotation_images set split = %s
                                       where id = %s and split_source = 'auto'""",
                                    (split, image_id),
                                )
                            connection.execute(
                                "update annotation_projects set status = 'draft', updated_at = now() where id = %s",
                                (project_id,),
                            )
                except Exception:
                    shutil.rmtree(batch, ignore_errors=True)
                    raise
                return {
                    "imported": len(rows),
                    "skipped": len(skipped),
                    "skippedFiles": skipped[:50],
                    "total": len(source_files),
                }
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise

    def list_projects(self, owner_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """select p.*, count(i.id)::int image_count,
                          count(i.id) filter (where i.is_labeled)::int labeled_count,
                          count(i.id) filter (where i.is_excluded)::int excluded_count,
                          count(i.id) filter (where i.is_labeled or i.is_excluded)::int completed_count
                   from annotation_projects p left join annotation_images i on i.project_id = p.id
                   where p.owner_user_id = %s group by p.id order by p.updated_at desc""",
                (owner_id,),
            ).fetchall()
            return [self._project_dict(row) for row in rows]

    def get_project(self, owner_id: str, project_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """select p.*, count(i.id)::int image_count,
                          count(i.id) filter (where i.is_labeled)::int labeled_count,
                          count(i.id) filter (where i.is_excluded)::int excluded_count,
                          count(i.id) filter (where i.is_labeled or i.is_excluded)::int completed_count
                   from annotation_projects p left join annotation_images i on i.project_id = p.id
                   where p.owner_user_id = %s and p.id = %s group by p.id""",
                (owner_id, project_id),
            ).fetchone()
            return self._project_dict(row) if row else None

    @staticmethod
    def _project_dict(row: Any) -> dict[str, Any]:
        result = dict(row)
        result["id"] = str(result["id"])
        result["classes"] = _json_value(result.get("classes"), [])
        for key in ("created_at", "updated_at"):
            if result.get(key):
                result[key] = result[key].isoformat()
        return result

    @staticmethod
    def _image_dict(row: Any, summary: bool = False) -> dict[str, Any]:
        result = dict(row)
        result["id"] = str(result["id"])
        result["project_id"] = str(result["project_id"])
        if not summary:
            result["annotations"] = _json_value(result.get("annotations"), [])
        else:
            result.pop("annotations", None)
        result.pop("storage_path", None)
        result.pop("thumbnail_path", None)
        if result.get("updated_at"):
            result["updated_at"] = result["updated_at"].isoformat()
        result["status"] = (
            "excluded" if result.get("is_excluded")
            else "empty" if result.get("marked_empty")
            else "labeled" if result.get("is_labeled")
            else "unlabeled"
        )
        return result

    def list_images(
        self,
        owner_id: str,
        project_id: str,
        status_filter: str = "all",
        split_filter: str = "all",
        search: str = "",
        sort: str = "original",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        project = self.get_project(owner_id, project_id)
        if not project:
            raise FileNotFoundError("Annotation project was not found")
        status_clauses = {
            "labeled": "is_labeled and not is_excluded",
            "unlabeled": "not is_labeled and not is_excluded",
            "empty": "marked_empty and not is_excluded",
            "excluded": "is_excluded",
        }
        if status_filter not in {"all", *status_clauses}:
            raise ValueError("Invalid image status filter")
        if split_filter not in {"all", *SPLITS}:
            raise ValueError("Invalid split filter")
        order = {
            "original": "sort_order asc",
            "name_asc": "lower(file_name) asc, sort_order asc",
            "name_desc": "lower(file_name) desc, sort_order asc",
            "updated_desc": "updated_at desc, sort_order asc",
        }.get(sort)
        if not order:
            raise ValueError("Invalid image sort")

        clauses = ["project_id = %s"]
        params: list[Any] = [project_id]
        if status_filter != "all":
            clauses.append(status_clauses[status_filter])
        if split_filter != "all":
            clauses.append("split = %s")
            params.append(split_filter)
        if search.strip():
            clauses.append("file_name ilike %s")
            params.append(f"%{search.strip()[:200]}%")
        where = " and ".join(clauses)
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        with self._connect() as connection:
            total = connection.execute(
                f"select count(*)::int count from annotation_images where {where}", params
            ).fetchone()["count"]
            split_counts = {"all": 0, "train": 0, "val": 0, "test": 0}
            for row in connection.execute(
                """select coalesce(split, 'train') split, count(*)::int count
                   from annotation_images where project_id = %s group by split""",
                (project_id,),
            ).fetchall():
                if row["split"] in SPLITS:
                    split_counts[row["split"]] = row["count"]
                    split_counts["all"] += row["count"]
            rows = connection.execute(
                f"""select id, project_id, file_name, width, height, sort_order, split, split_source,
                           marked_empty, is_labeled, is_excluded, revision, thumbnail_width,
                           thumbnail_height, updated_at
                    from annotation_images where {where} order by {order} limit %s offset %s""",
                [*params, limit, offset],
            ).fetchall()
        return {
            "project": project,
            "images": [self._image_dict(row, summary=True) for row in rows],
            "total": total,
            "splitCounts": split_counts,
        }

    def get_image(self, owner_id: str, project_id: str, image_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """select i.* from annotation_images i join annotation_projects p on p.id = i.project_id
                   where p.owner_user_id = %s and p.id = %s and i.id = %s""",
                (owner_id, project_id, image_id),
            ).fetchone()
            return self._image_dict(row) if row else None

    def image_path(self, owner_id: str, project_id: str, image_id: str) -> Path:
        row = self._owned_media_row(owner_id, project_id, image_id)
        path = contained_path(_project_root(owner_id, project_id), Path(row["storage_path"]))
        if not path.is_file():
            raise FileNotFoundError("Annotation image file was not found")
        return path

    def thumbnail_path(self, owner_id: str, project_id: str, image_id: str) -> Path:
        row = self._owned_media_row(owner_id, project_id, image_id)
        root = _project_root(owner_id, project_id)
        thumbnail = contained_path(root, Path(row["thumbnail_path"])) if row.get("thumbnail_path") else contained_path(root, "thumbnails", f"{image_id}.webp")
        if not thumbnail.is_file():
            source = contained_path(root, Path(row["storage_path"]))
            if not source.is_file():
                raise FileNotFoundError("Annotation image file was not found")
            thumbnail.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as opened:
                preview = ImageOps.exif_transpose(opened)
                preview.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                if preview.mode != "RGB":
                    preview = preview.convert("RGB")
                preview.save(thumbnail, "WEBP", quality=82)
                thumb_width, thumb_height = preview.size
            with self._connect() as connection:
                connection.execute(
                    """update annotation_images set thumbnail_path = %s, thumbnail_width = %s, thumbnail_height = %s
                       where id = %s""",
                    (str(thumbnail), thumb_width, thumb_height, image_id),
                )
        return thumbnail

    def _owned_media_row(self, owner_id: str, project_id: str, image_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """select i.storage_path, i.thumbnail_path from annotation_images i
                   join annotation_projects p on p.id = i.project_id
                   where p.owner_user_id = %s and p.id = %s and i.id = %s""",
                (owner_id, project_id, image_id),
            ).fetchone()
        if not row:
            raise FileNotFoundError("Annotation image was not found")
        return dict(row)

    def save_image(
        self,
        owner_id: str,
        project_id: str,
        image_id: str,
        revision: int,
        annotations: Any,
        marked_empty: bool,
        is_excluded: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            with connection.transaction():
                project = self._editable_project(connection, owner_id, project_id)
                current = connection.execute(
                    "select * from annotation_images where project_id = %s and id = %s for update",
                    (project_id, image_id),
                ).fetchone()
                if not current:
                    raise FileNotFoundError("Annotation image was not found")
                if int(current["revision"]) != revision and not force:
                    raise FileExistsError("Annotation changed in another tab")
                classes = _json_value(project["classes"], [])
                cleaned = _validate_annotations(
                    project["task_type"], annotations, marked_empty, {item["id"] for item in classes}
                )
                is_labeled = bool(cleaned) or marked_empty
                last_checkpoint = connection.execute(
                    "select created_at from annotation_image_revisions where image_id = %s order by created_at desc limit 1",
                    (image_id,),
                ).fetchone()
                if not last_checkpoint or (time.time() - last_checkpoint["created_at"].timestamp()) >= REVISION_INTERVAL_SECONDS:
                    connection.execute(
                        """insert into annotation_image_revisions
                           (image_id, revision, annotations, marked_empty, is_excluded)
                           values (%s, %s, %s::jsonb, %s, %s)
                           on conflict (image_id, revision) do nothing""",
                        (
                            image_id, current["revision"], json.dumps(_json_value(current["annotations"], [])),
                            current["marked_empty"], current["is_excluded"],
                        ),
                    )
                    connection.execute(
                        """delete from annotation_image_revisions where image_id = %s and id not in (
                             select id from annotation_image_revisions where image_id = %s
                             order by created_at desc limit %s
                           )""",
                        (image_id, image_id, MAX_REVISIONS),
                    )
                row = connection.execute(
                    """update annotation_images set annotations = %s::jsonb, marked_empty = %s,
                              is_labeled = %s, is_excluded = %s, revision = revision + 1, updated_at = now()
                       where id = %s returning *""",
                    (json.dumps(cleaned), marked_empty, is_labeled, is_excluded, image_id),
                ).fetchone()
                connection.execute(
                    "update annotation_projects set status = 'draft', updated_at = now() where id = %s",
                    (project_id,),
                )
                return self._image_dict(row)

    def set_split(self, owner_id: str, project_id: str, image_id: str, split: str) -> dict[str, Any]:
        if split not in SPLITS:
            raise ValueError("Split must be train, val, or test")
        with self._connect() as connection:
            with connection.transaction():
                self._editable_project(connection, owner_id, project_id)
                row = connection.execute(
                    """update annotation_images set split = %s, split_source = 'manual', updated_at = now()
                       where project_id = %s and id = %s returning *""",
                    (split, project_id, image_id),
                ).fetchone()
                if not row:
                    raise FileNotFoundError("Annotation image was not found")
                connection.execute(
                    "update annotation_projects set status = 'draft', updated_at = now() where id = %s",
                    (project_id,),
                )
                return self._image_dict(row)

    def bulk_update(
        self,
        owner_id: str,
        project_id: str,
        image_ids: list[str],
        split: str | None = None,
        excluded: bool | None = None,
        automatic: bool = False,
    ) -> int:
        image_ids = list(dict.fromkeys(image_ids))
        if not image_ids or len(image_ids) > 500:
            raise ValueError("Choose between 1 and 500 images")
        if split is not None and split not in SPLITS:
            raise ValueError("Split must be train, val, or test")
        if (split is not None) + (excluded is not None) + automatic != 1:
            raise ValueError("Choose exactly one bulk action")
        with self._connect() as connection:
            with connection.transaction():
                project = self._editable_project(connection, owner_id, project_id)
                owned = connection.execute(
                    "select id from annotation_images where project_id = %s and id = any(%s)",
                    (project_id, image_ids),
                ).fetchall()
                if len(owned) != len(image_ids):
                    raise FileNotFoundError("One or more annotation images were not found")
                if split is not None:
                    connection.execute(
                        """update annotation_images set split = %s, split_source = 'manual', updated_at = now()
                           where project_id = %s and id = any(%s)""",
                        (split, project_id, image_ids),
                    )
                elif excluded is not None:
                    connection.execute(
                        "update annotation_images set is_excluded = %s, updated_at = now() where project_id = %s and id = any(%s)",
                        (excluded, project_id, image_ids),
                    )
                else:
                    connection.execute(
                        "update annotation_images set split_source = 'auto', updated_at = now() where project_id = %s and id = any(%s)",
                        (project_id, image_ids),
                    )
                    rows = [dict(row) for row in connection.execute(
                        "select id, file_name, split, split_source, annotations from annotation_images where project_id = %s",
                        (project_id,),
                    ).fetchall()]
                    assignments = _assign_splits(rows, dict(project))
                    for image_id in image_ids:
                        connection.execute(
                            "update annotation_images set split = %s where id = %s",
                            (assignments[image_id], image_id),
                        )
                connection.execute(
                    "update annotation_projects set status = 'draft', updated_at = now() where id = %s",
                    (project_id,),
                )
        return len(image_ids)

    def rebalance(self, owner_id: str, project_id: str) -> int:
        with self._connect() as connection:
            with connection.transaction():
                project = self._editable_project(connection, owner_id, project_id)
                rows = [dict(row) for row in connection.execute(
                    "select id, file_name, split, split_source, annotations from annotation_images where project_id = %s",
                    (project_id,),
                ).fetchall()]
                assignments = _assign_splits(rows, dict(project), rebalance_all=True)
                for image_id, split in assignments.items():
                    connection.execute(
                        "update annotation_images set split = %s, split_source = 'auto', updated_at = now() where id = %s",
                        (split, image_id),
                    )
                connection.execute(
                    "update annotation_projects set status = 'draft', updated_at = now() where id = %s",
                    (project_id,),
                )
                return len(rows)

    def update_project(self, owner_id: str, project_id: str, name: str, classes: Any, ratios: tuple[int, int, int]) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("Project name must contain 1-100 characters")
        class_list = _class_list(classes)
        train, val, test = _ratios(*ratios)
        with self._connect() as connection:
            with connection.transaction():
                project = self._editable_project(connection, owner_id, project_id)
                _validate_class_count(project["task_type"], class_list)
                used_rows = connection.execute(
                    """select distinct item->>'classId' class_id from annotation_images i
                       cross join lateral jsonb_array_elements(i.annotations) item where i.project_id = %s""",
                    (project_id,),
                ).fetchall()
                removed = {row["class_id"] for row in used_rows} - {item["id"] for item in class_list}
                if removed:
                    raise ValueError("Remove annotations that use a class before deleting that class")
                row = connection.execute(
                    """update annotation_projects set name = %s, classes = %s::jsonb, train_ratio = %s,
                              val_ratio = %s, test_ratio = %s, status = 'draft', updated_at = now()
                       where id = %s returning *""",
                    (name, json.dumps(class_list), train, val, test, project_id),
                ).fetchone()
                rows = [dict(item) for item in connection.execute(
                    "select id, file_name, split, split_source, annotations from annotation_images where project_id = %s",
                    (project_id,),
                ).fetchall()]
                assignments = _assign_splits(rows, dict(row))
                for image_id, split in assignments.items():
                    connection.execute(
                        "update annotation_images set split = %s where id = %s and split_source = 'auto'",
                        (split, image_id),
                    )
        return self.get_project(owner_id, project_id) or self._project_dict(row)

    def list_revisions(self, owner_id: str, project_id: str, image_id: str) -> list[dict[str, Any]]:
        if not self.get_image(owner_id, project_id, image_id):
            raise FileNotFoundError("Annotation image was not found")
        with self._connect() as connection:
            rows = connection.execute(
                """select id, revision, marked_empty, is_excluded, created_at
                   from annotation_image_revisions where image_id = %s order by created_at desc limit %s""",
                (image_id, MAX_REVISIONS),
            ).fetchall()
        return [
            {
                **dict(row),
                "id": str(row["id"]),
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def restore_revision(self, owner_id: str, project_id: str, image_id: str, revision_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            with connection.transaction():
                project = self._editable_project(connection, owner_id, project_id)
                snapshot = connection.execute(
                    "select * from annotation_image_revisions where image_id = %s and id = %s",
                    (image_id, revision_id),
                ).fetchone()
                current = connection.execute(
                    "select * from annotation_images where project_id = %s and id = %s for update",
                    (project_id, image_id),
                ).fetchone()
                if not snapshot or not current:
                    raise FileNotFoundError("Annotation revision was not found")
                connection.execute(
                    """insert into annotation_image_revisions
                       (image_id, revision, annotations, marked_empty, is_excluded)
                       values (%s, %s, %s::jsonb, %s, %s)
                       on conflict (image_id, revision) do nothing""",
                    (
                        image_id, current["revision"], json.dumps(_json_value(current["annotations"], [])),
                        current["marked_empty"], current["is_excluded"],
                    ),
                )
                annotations = _json_value(snapshot["annotations"], [])
                row = connection.execute(
                    """update annotation_images set annotations = %s::jsonb, marked_empty = %s,
                              is_excluded = %s, is_labeled = %s, revision = revision + 1, updated_at = now()
                       where id = %s returning *""",
                    (
                        json.dumps(annotations), snapshot["marked_empty"], snapshot["is_excluded"],
                        bool(annotations) or snapshot["marked_empty"], image_id,
                    ),
                ).fetchone()
                connection.execute(
                    """delete from annotation_image_revisions where image_id = %s and id not in (
                         select id from annotation_image_revisions where image_id = %s
                         order by created_at desc limit %s
                       )""",
                    (image_id, image_id, MAX_REVISIONS),
                )
                connection.execute(
                    "update annotation_projects set status = 'draft', updated_at = now() where id = %s",
                    (project_id,),
                )
                return self._image_dict(row)

    def neighbor_ids(
        self,
        owner_id: str,
        project_id: str,
        image_id: str,
        split: str = "all",
        status: str = "all",
        search: str = "",
        sort: str = "original",
    ) -> dict[str, str | None]:
        ids: list[str] = []
        offset = 0
        while True:
            listing = self.list_images(owner_id, project_id, status, split, search, sort, offset, 200)
            ids.extend(item["id"] for item in listing["images"])
            if len(ids) >= listing["total"]:
                break
            offset += 200
        if image_id not in ids:
            return {"previousId": None, "nextId": None}
        index = ids.index(image_id)
        return {
            "previousId": ids[index - 1] if index > 0 else None,
            "nextId": ids[index + 1] if index + 1 < len(ids) else None,
        }

    def delete_project(self, owner_id: str, project_id: str) -> None:
        root = _project_root(owner_id, project_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        with named_file_lock(root.parent, project_id, "annotation project"):
            removal = ReversibleDirectoryRemoval(root) if root.exists() else None
            with self._connect() as connection:
                try:
                    with connection.transaction():
                        project = connection.execute(
                            "select id from annotation_projects where owner_user_id = %s and id = %s for update",
                            (owner_id, project_id),
                        ).fetchone()
                        if not project:
                            raise FileNotFoundError("Annotation project was not found")
                        active = connection.execute(
                            """select count(*)::int count from annotation_operations
                               where project_id = %s and status in ('queued', 'running')""",
                            (project_id,),
                        ).fetchone()["count"]
                        if active:
                            raise FileExistsError("Cancel active annotation operations before deleting this project")
                        if removal:
                            removal.apply()
                        connection.execute(
                            "delete from annotation_projects where owner_user_id = %s and id = %s",
                            (owner_id, project_id),
                        )
                    if removal:
                        removal.commit()
                except Exception:
                    if removal:
                        removal.rollback()
                    raise

    def publish(
        self,
        owner_id: str,
        owner_email: str | None,
        project_id: str,
        progress: OperationCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> dict[str, Any]:
        root = _project_root(owner_id, project_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        with named_file_lock(root.parent, project_id, "annotation project"):
            with self._connect() as connection:
                with connection.transaction():
                    project_row = connection.execute(
                        "select * from annotation_projects where owner_user_id = %s and id = %s for update",
                        (owner_id, project_id),
                    ).fetchone()
                    if not project_row:
                        raise FileNotFoundError("Annotation project was not found")
                    incomplete = connection.execute(
                        """select count(*)::int count from annotation_images
                           where project_id = %s and not is_excluded and not is_labeled""",
                        (project_id,),
                    ).fetchone()["count"]
                    if incomplete:
                        raise ValueError(f"Label, mark empty, or exclude all images before publishing ({incomplete} remaining)")
                    usable = connection.execute(
                        "select count(*)::int count from annotation_images where project_id = %s and not is_excluded",
                        (project_id,),
                    ).fetchone()["count"]
                    if not usable:
                        raise ValueError("Project has no images to publish")
                    connection.execute(
                        "update annotation_projects set status = 'publishing', updated_at = now() where id = %s",
                        (project_id,),
                    )
            stage = None
            moved_target = False
            try:
                project = dict(project_row)
                project["classes"] = _json_value(project["classes"], [])
                with self._connect() as connection:
                    rows = [dict(row) for row in connection.execute(
                        """select * from annotation_images
                           where project_id = %s and not is_excluded order by sort_order""",
                        (project_id,),
                    ).fetchall()]
                split_by_id = _assign_splits(rows, project)
                if not any(split == "train" for split in split_by_id.values()):
                    raise ValueError("Move at least one usable image to Train before publishing")
                if project["task_type"] == "image_classification":
                    train_class_ids = {
                        _json_value(row["annotations"], [])[0]["classId"]
                        for row in rows
                        if split_by_id[str(row["id"])] == "train"
                    }
                    missing = [item["name"] for item in project["classes"] if item["id"] not in train_class_ids]
                    if missing:
                        raise ValueError(
                            "Every classification class needs at least one training image. Missing: " + ", ".join(missing)
                        )
                version = int(project["published_version"]) + 1
                base_slug = safe_dataset_name(project["name"])
                dataset_slug = f"{base_slug[:110]}-v{version}"
                validate_slug(dataset_slug, "dataset name")
                target = owner_dataset_path(DATASET_DIR, owner_id, dataset_slug)
                stage = staging_directory(target.parent)
                if cancelled and cancelled():
                    raise InterruptedError("Publish cancelled")
                self._write_dataset(stage, project, rows, split_by_id, progress, cancelled)
                metadata = inspect_dataset(stage)
                metadata["classes"] = [item["name"] for item in project["classes"]]
                metadata["label_schema"] = {
                    "origin": "ailab_label",
                    "task": project["task_type"],
                    "format": {
                        "image_classification": "imagefolder",
                        "object_detection": "coco_boxes",
                        "semantic_segmentation": "semantic_masks",
                        "instance_segmentation": "coco_instances",
                    }[project["task_type"]],
                    "background_id": 0 if project["task_type"] == "semantic_segmentation" else None,
                    "classes": [
                        {
                            "id": item["id"],
                            "name": item["name"],
                            "train_id": index + (1 if project["task_type"] != "image_classification" else 0),
                        }
                        for index, item in enumerate(project["classes"])
                    ],
                }
                metadata["metadata_version"] = DATASET_METADATA_VERSION
                validate_dataset_for_upload(stage, metadata)
                workflow = dataset_workflow_metadata(metadata, get_catalog())
                (stage / ".ailab_dataset.json").write_text(json.dumps({
                    "created_by": owner_id,
                    "created_by_email": owner_email,
                    "source_annotation_project_id": project_id,
                    "source_annotation_version": version,
                    "label_schema": metadata["label_schema"],
                    "workflow": workflow,
                }, indent=2), encoding="utf-8")
                with resource_repository.dataset_guard(owner_id, dataset_slug) as connection:
                    if cancelled and cancelled():
                        raise InterruptedError("Publish cancelled")
                    if resource_repository.get_dataset(owner_id, dataset_slug, connection) or target.exists():
                        raise FileExistsError(f"Dataset {dataset_slug} already exists")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stage.replace(target)
                    moved_target = True
                    record = resource_repository.upsert_dataset(
                        owner_id, owner_email, dataset_slug, target, metadata, connection
                    )
                    connection.execute(
                        """update annotation_projects set status = 'published', published_version = %s,
                                  updated_at = now() where id = %s""",
                        (version, project_id),
                    )
                    for image_id, split in split_by_id.items():
                        connection.execute(
                            """update annotation_images set split = %s
                               where id = %s and split_source = 'auto'""",
                            (split, image_id),
                        )
                return {"datasetId": str(record["id"]), "datasetName": dataset_slug, "version": version}
            except Exception:
                if stage is not None:
                    shutil.rmtree(stage, ignore_errors=True)
                if moved_target and target.exists() and not resource_repository.get_dataset(owner_id, dataset_slug):
                    shutil.rmtree(target, ignore_errors=True)
                with self._connect() as connection:
                    connection.execute(
                        """update annotation_projects
                           set status = 'draft',
                               updated_at = now()
                           where owner_user_id = %s and id = %s""",
                        (owner_id, project_id),
                    )
                raise

    @staticmethod
    def _draw_brush(draw: ImageDraw.ImageDraw, annotation: dict[str, Any], width: int, height: int, fill: int) -> None:
        points = [(point[0] * width, point[1] * height) for point in annotation["points"]]
        radius = max(1, round(annotation["radius"] * min(width, height)))
        draw.line(points, fill=fill, width=radius * 2, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    @staticmethod
    def _polygon_area(points: list[list[float]]) -> float:
        return abs(sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )) / 2

    def _write_instance_annotations(
        self,
        document: dict[str, Any],
        annotations: list[dict[str, Any]],
        row: dict[str, Any],
        class_index: dict[str, int],
        image_id: int,
        next_id: int,
    ) -> int:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            grouped[str(annotation.get("instanceId") or annotation["id"])].append(annotation)
        for instance in grouped.values():
            class_ids = {item["classId"] for item in instance}
            if len(class_ids) != 1:
                raise ValueError("One instance cannot use multiple classes")
            category_id = class_index[next(iter(class_ids))]
            has_brush = any(item["type"] == "brush" for item in instance)
            if has_brush:
                import numpy as np
                from pycocotools import mask as mask_utils

                mask = Image.new("1", (row["width"], row["height"]), 0)
                draw = ImageDraw.Draw(mask)
                for item in instance:
                    if item["type"] == "polygon":
                        points = [(point[0] * row["width"], point[1] * row["height"]) for point in item["points"]]
                        draw.polygon(points, fill=1)
                    else:
                        self._draw_brush(
                            draw, item, row["width"], row["height"],
                            0 if item.get("mode") == "erase" else 1,
                        )
                array = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
                encoded = mask_utils.encode(array)
                encoded["counts"] = encoded["counts"].decode("ascii")
                bbox = [float(value) for value in mask_utils.toBbox(encoded)]
                area = float(mask_utils.area(encoded))
                if area <= 0:
                    continue
                segmentation: Any = encoded
            else:
                segmentation = [
                    [coordinate for point in item["points"] for coordinate in (
                        point[0] * row["width"], point[1] * row["height"]
                    )]
                    for item in instance
                ]
                flat_points = [
                    [point[0] * row["width"], point[1] * row["height"]]
                    for item in instance for point in item["points"]
                ]
                xs = [point[0] for point in flat_points]
                ys = [point[1] for point in flat_points]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                area = sum(self._polygon_area([
                    [point[0] * row["width"], point[1] * row["height"]]
                    for point in item["points"]
                ]) for item in instance)
            document["annotations"].append({
                "id": next_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0,
            })
            next_id += 1
        return next_id

    def _write_dataset(
        self,
        root: Path,
        project: dict[str, Any],
        rows: list[dict[str, Any]],
        splits: dict[str, str],
        progress: OperationCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> None:
        classes = project["classes"]
        class_index = {item["id"]: index + 1 for index, item in enumerate(classes)}
        coco = {
            split: {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": index + 1, "name": item["name"], "supercategory": "none"}
                    for index, item in enumerate(classes)
                ],
            }
            for split in SPLITS
        }
        annotation_id = 1
        for index, row in enumerate(rows):
            if cancelled and cancelled():
                raise InterruptedError("Publish cancelled")
            split = splits[str(row["id"])]
            source = contained_path(
                _project_root(project["owner_user_id"], str(project["id"])),
                Path(row["storage_path"]),
            )
            target_name = f"{row['id']}{source.suffix.lower()}"
            annotations = _json_value(row["annotations"], [])
            if project["task_type"] == "image_classification":
                class_id = annotations[0]["classId"]
                class_position = next(position for position, item in enumerate(classes) if item["id"] == class_id)
                destination = contained_path(
                    root, split, _safe_class_folder(classes[class_position]["name"], class_position), target_name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            else:
                images_dir = contained_path(root, split, "images")
                images_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, contained_path(images_dir, target_name))
                if project["task_type"] == "semantic_segmentation":
                    masks_dir = contained_path(root, split, "masks")
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    mask = Image.new("I", (row["width"], row["height"]), 0)
                    draw = ImageDraw.Draw(mask)
                    for annotation in annotations:
                        fill = 0 if annotation.get("mode") == "erase" else class_index[annotation["classId"]]
                        if annotation["type"] == "polygon":
                            points = [
                                (point[0] * row["width"], point[1] * row["height"])
                                for point in annotation["points"]
                            ]
                            draw.polygon(points, fill=fill)
                        else:
                            self._draw_brush(draw, annotation, row["width"], row["height"], fill)
                    mask.convert("I;16").save(contained_path(masks_dir, f"{row['id']}.png"))
                else:
                    document = coco[split]
                    coco_image_id = row["sort_order"] + 1
                    document["images"].append({
                        "id": coco_image_id,
                        "file_name": f"images/{target_name}",
                        "width": row["width"],
                        "height": row["height"],
                    })
                    if project["task_type"] == "object_detection":
                        for annotation in annotations:
                            x = annotation["x"] * row["width"]
                            y = annotation["y"] * row["height"]
                            width = annotation["width"] * row["width"]
                            height = annotation["height"] * row["height"]
                            document["annotations"].append({
                                "id": annotation_id,
                                "image_id": coco_image_id,
                                "category_id": class_index[annotation["classId"]],
                                "bbox": [x, y, width, height],
                                "area": width * height,
                                "iscrowd": 0,
                            })
                            annotation_id += 1
                    else:
                        annotation_id = self._write_instance_annotations(
                            document, annotations, row, class_index, coco_image_id, annotation_id
                        )
            if progress:
                progress(index + 1, len(rows), 0)

        if project["task_type"] in {"object_detection", "instance_segmentation"}:
            for split, document in coco.items():
                if document["images"]:
                    contained_path(root, split, "_annotations.coco.json").write_text(
                        json.dumps(document), encoding="utf-8"
                    )


annotation_service = AnnotationService()
