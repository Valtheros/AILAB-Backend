"""Single-image inference for completed object-detection runs.

Two model families are produced by the training worker, saved in two different
formats:

* **YOLO (Ultralytics)** — ``weights/best.pt`` in the Ultralytics format,
  loaded with ``ultralytics.YOLO`` (see ``worker/trainers/yolo_trainer.py``).
* **Faster R-CNN (torchvision)** — ``best.pt`` holding
  ``{"model": state_dict, "classes": [...], "config": job_config}``, mirroring
  ``worker/trainers/detection_common.py``.

Boxes are returned in the ORIGINAL image's pixel coordinates so the frontend
can overlay them at any display size by scaling with the reported image size.

The Faster R-CNN builder below intentionally mirrors ``build_faster_rcnn`` in
``worker/trainers/detection_common.py``. It is duplicated rather than imported
because importing the worker trainer package pulls in the full training
dependency stack. If the head layout changes there, change it here too or the
saved ``state_dict`` will no longer load.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Reuse the classification module's hardened image reader, error types, torch
# guard, and cache size so both endpoints behave identically for uploads.
from inference_service import (
    InferenceError,
    InferenceUnavailable,
    MODEL_CACHE_SIZE,
    _open_verified_image,
    _require_torch,
)

DETECTION_MODEL_TYPES = {"yolo", "faster_rcnn"}
DETECTION_TASK_TYPE = "object_detection"

# Default confidence floor; matches Ultralytics' own predict default. The caller
# may override it per request. Boxes below this score are dropped before return.
DEFAULT_SCORE_THRESHOLD = 0.25
# Hard cap so a noisy image cannot return thousands of boxes to the browser.
MAX_DETECTIONS = 300


_cache_lock = threading.Lock()
_model_cache: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()


def _clamp_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SCORE_THRESHOLD
    if threshold != threshold:  # NaN
        return DEFAULT_SCORE_THRESHOLD
    return min(max(threshold, 0.0), 1.0)


def _yolo_checkpoint_path(run_dir: Path) -> Path:
    """YOLO writes into a weights/ subfolder; fall back to the run root."""
    for candidate in (
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
        run_dir / "best.pt",
        run_dir / "last.pt",
    ):
        if candidate.is_file():
            return candidate
    raise InferenceError(
        "This run has no saved YOLO model file (weights/best.pt). It may have "
        "failed before the first epoch finished."
    )


def _rcnn_checkpoint_path(run_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise InferenceError(
        "This run has no saved model file (best.pt). It may have failed before "
        "the first epoch finished."
    )


def _build_faster_rcnn(config: dict, num_classes: int):
    """Recreate the trained Faster R-CNN head. Mirrors detection_common."""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    args = (config or {}).get("extra_args") or {}
    # Weights are overwritten by the trained state_dict, so none are downloaded.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=int(args.get("image_size", 640)),
        max_size=int(args.get("max_size", 1333)),
        box_score_thresh=float(args.get("box_score_thresh", 0.05)),
        box_nms_thresh=float(args.get("box_nms_thresh", 0.5)),
        box_detections_per_img=int(args.get("detections_per_img", 100)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _cache_get(key: tuple) -> dict[str, Any] | None:
    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            _model_cache.move_to_end(key)
        return cached


def _cache_put(key: tuple, entry: dict[str, Any]) -> None:
    with _cache_lock:
        _model_cache[key] = entry
        _model_cache.move_to_end(key)
        while len(_model_cache) > MODEL_CACHE_SIZE:
            _model_cache.popitem(last=False)


def _load_yolo(run_dir: Path) -> dict[str, Any]:
    _require_torch()  # ultralytics needs torch; surface the same clear error.
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on the built image
        raise InferenceUnavailable(
            "Ultralytics is not installed in the API image. Rebuild the backend "
            "image so YOLO model testing can run."
        ) from exc

    path = _yolo_checkpoint_path(run_dir)
    stat = path.stat()
    key = ("yolo", str(path), stat.st_mtime_ns, stat.st_size)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        model = YOLO(str(path))
    except Exception as exc:
        raise InferenceError("The saved YOLO model file could not be loaded.") from exc
    names = getattr(model, "names", None) or {}
    classes = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    entry = {
        "kind": "yolo",
        "model": model,
        "classes": classes,
        "device": "cpu",
        "architecture": Path(path).parent.parent.name or "yolo",
        "checkpoint": path.name,
    }
    _cache_put(key, entry)
    return entry


def _load_faster_rcnn(run_dir: Path) -> dict[str, Any]:
    torch = _require_torch()
    path = _rcnn_checkpoint_path(run_dir)
    stat = path.stat()
    key = ("faster_rcnn", str(path), stat.st_mtime_ns, stat.st_size)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise InferenceError("The saved model file is not a supported AILAB detection checkpoint.")
    classes = list(checkpoint.get("classes") or [])
    if not classes:
        raise InferenceError("The saved model file does not record its class names.")
    config = checkpoint.get("config") or {}

    # Head size at training time was len(classes) + 1 (index 0 = background).
    num_classes = len(classes) + 1
    model = _build_faster_rcnn(config, num_classes)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise InferenceError(
            "The saved weights do not match the recorded architecture "
            f"({len(missing)} missing / {len(unexpected)} unexpected parameters)."
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    entry = {
        "kind": "faster_rcnn",
        "model": model,
        "classes": classes,
        "device": str(device),
        "architecture": "fasterrcnn_resnet50_fpn_v2",
        "checkpoint": path.name,
    }
    _cache_put(key, entry)
    return entry


def load_detector(run_dir: Path, model_type: str) -> dict[str, Any]:
    kind = (model_type or "").lower()
    if kind == "yolo":
        return _load_yolo(run_dir)
    if kind == "faster_rcnn":
        return _load_faster_rcnn(run_dir)
    raise InferenceError(f"Model testing does not support detection model '{model_type}'.")


def _predict_yolo(entry: dict[str, Any], image, threshold: float) -> list[dict[str, Any]]:
    import numpy as np

    # Ultralytics accepts an RGB ndarray and returns boxes in the input image's
    # own pixel space, which is exactly the original size we pass here.
    results = entry["model"].predict(source=np.asarray(image), conf=threshold, verbose=False)
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = results[0].names or {}
    detections = []
    for xyxy, conf, cls in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ):
        index = int(cls)
        detections.append({
            "className": names.get(index, str(index)) if isinstance(names, dict) else str(index),
            "confidence": round(float(conf), 6),
            "percent": round(float(conf) * 100, 2),
            "box": [round(float(v), 1) for v in xyxy],
        })
    return detections


def _predict_faster_rcnn(entry: dict[str, Any], image, threshold: float) -> list[dict[str, Any]]:
    torch = _require_torch()
    import torchvision.transforms.functional as F

    # torchvision detection models take a list of [0,1] tensors of any size and
    # map predicted boxes back to the original pixels internally.
    tensor = F.to_tensor(image).to(entry["device"])
    with torch.no_grad():
        output = entry["model"]([tensor])[0]

    classes = entry["classes"]
    detections = []
    for box, label, score in zip(
        output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()
    ):
        if float(score) < threshold:
            continue
        # Label 0 is background; class names are 1-indexed against `classes`.
        index = int(label) - 1
        name = classes[index] if 0 <= index < len(classes) else str(label)
        detections.append({
            "className": name,
            "confidence": round(float(score), 6),
            "percent": round(float(score) * 100, 2),
            "box": [round(float(v), 1) for v in box],
        })
    return detections


def predict_detection(
    run_dir: Path,
    payload: bytes,
    model_type: str,
    score_threshold: Any = DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Return every detected box (above threshold) for one uploaded image."""
    started = time.perf_counter()
    threshold = _clamp_threshold(score_threshold)
    image = _open_verified_image(payload)
    original_size = list(image.size)

    entry = load_detector(run_dir, model_type)
    model_ready = time.perf_counter()

    if entry["kind"] == "yolo":
        detections = _predict_yolo(entry, image, threshold)
    else:
        detections = _predict_faster_rcnn(entry, image, threshold)

    # Most-confident first, then cap so the payload stays small.
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    detections = detections[:MAX_DETECTIONS]

    counts: dict[str, int] = {}
    for item in detections:
        counts[item["className"]] = counts.get(item["className"], 0) + 1

    finished = time.perf_counter()
    return {
        "predictions": detections,
        "count": len(detections),
        "counts": counts,
        "threshold": threshold,
        "classes": entry["classes"],
        "model": {
            "architecture": entry["architecture"],
            "checkpoint": entry["checkpoint"],
            "device": entry["device"],
        },
        "image": {"width": original_size[0], "height": original_size[1]},
        "timingMs": {
            "modelLoad": round((model_ready - started) * 1000, 1),
            "inference": round((finished - model_ready) * 1000, 1),
            "total": round((finished - started) * 1000, 1),
        },
    }
