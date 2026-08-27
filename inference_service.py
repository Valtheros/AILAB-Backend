"""Single-image inference for completed image-classification runs.

The training worker saves each checkpoint as
``{"model": state_dict, "classes": [...], "config": job_config}`` (see
``worker/trainers/classification_common.py``). This module rebuilds the same
architecture from that metadata, loads the trained weights, and returns the
top-k softmax predictions for one uploaded image.

Scope: image classification only (ResNet / EfficientNet). Detection and
segmentation runs are rejected by the caller.

Note: the architecture builders below intentionally mirror ``_build_resnet``
and ``_build_efficientnet`` in ``worker/trainers/classification_common.py``.
They are duplicated rather than imported because the worker trainer package
pulls in the full training dependency stack on import. If the classifier head
layout changes there, it must be changed here as well or the saved
``state_dict`` will no longer load.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque
from io import BytesIO
from pathlib import Path
from typing import Any

# ImageNet normalisation constants, identical to the validation transform used
# during training. Changing these silently degrades accuracy.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_MODEL_TYPES = {"resnet", "efficientnet"}
SUPPORTED_TASK_TYPE = "image_classification"


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# An uploaded test image is a single photo, so it is capped far below the
# dataset archive limit in security_utils.py.
MAX_INFERENCE_IMAGE_BYTES = _env_int("AILAB_MAX_INFERENCE_IMAGE_BYTES", 10 * 1024 * 1024)
# Matches AILAB_MAX_SOURCE_IMAGE_PIXELS used by the dataset inspectors so an
# image accepted here could also have been accepted at training time.
MAX_INFERENCE_IMAGE_PIXELS = _env_int("AILAB_MAX_SOURCE_IMAGE_PIXELS", 25_000_000)
# Each cached ResNet-50 holds roughly 100 MB of weights, so the cache is kept
# deliberately small.
MODEL_CACHE_SIZE = _env_int("AILAB_INFERENCE_CACHE_SIZE", 2)
# Per-user sliding window. Loading a checkpoint is IO and CPU heavy, so this
# stops one account from monopolising the API worker pool.
RATE_LIMIT_REQUESTS = _env_int("AILAB_INFERENCE_RATE_LIMIT", 20)
RATE_LIMIT_WINDOW_SECONDS = _env_int("AILAB_INFERENCE_RATE_WINDOW", 60)


class InferenceError(ValueError):
    """Raised for problems the caller should surface as a 4xx response."""


class InferenceUnavailable(RuntimeError):
    """Raised when the runtime cannot serve predictions at all (missing torch)."""


class RateLimitExceeded(RuntimeError):
    """Raised when a single user calls the predict endpoint too often."""


_cache_lock = threading.Lock()
_model_cache: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()

_rate_lock = threading.Lock()
_rate_hits: dict[str, deque] = {}


def check_rate_limit(owner_id: str) -> None:
    """Allow at most RATE_LIMIT_REQUESTS predictions per user per window.

    In-process only. The deployment runs a single backend container, so this is
    sufficient today; a multi-replica deployment would need a shared counter in
    Redis or PostgreSQL (see lib/rate-limit.ts for the pattern used elsewhere).
    """
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        hits = _rate_hits.setdefault(owner_id, deque())
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(hits[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1)
            raise RateLimitExceeded(
                f"Too many prediction requests. Try again in {retry_after} seconds."
            )
        hits.append(now)
        # Drop idle users so the dict cannot grow without bound.
        if len(_rate_hits) > 1000:
            for key in [key for key, value in _rate_hits.items() if not value]:
                _rate_hits.pop(key, None)


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the built image
        raise InferenceUnavailable(
            "PyTorch is not installed in the API image. Rebuild the backend image "
            "so model testing can run."
        ) from exc
    import torch

    return torch


def _build_classifier(architecture: str, num_classes: int, dropout: float):
    """Recreate the trained architecture with randomly initialised weights.

    Pretrained weights are never downloaded here: the trained ``state_dict``
    overwrites every parameter anyway, and the API container must not depend on
    outbound network access at request time.
    """
    import torch.nn as nn
    import torchvision.models as models

    builder = getattr(models, architecture, None)
    if builder is None:
        raise InferenceError(f"Unsupported architecture '{architecture}'.")
    model = builder(weights=None)

    if hasattr(model, "fc"):  # ResNet family
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif hasattr(model, "classifier"):  # EfficientNet family
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise InferenceError(f"Architecture '{architecture}' has no supported classifier head.")
    return model


def _checkpoint_path(run_dir: Path) -> Path:
    """Prefer the best checkpoint, fall back to the final epoch."""
    for name in ("best.pt", "last.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise InferenceError(
        "This run has no saved model file (best.pt). It may have failed before the first epoch finished."
    )


def load_classifier(run_dir: Path) -> dict[str, Any]:
    """Load (or reuse) the trained classifier for one run directory.

    The cache key includes the checkpoint size and modification time so that
    retraining the same run slug invalidates the cached weights automatically.
    """
    torch = _require_torch()
    path = _checkpoint_path(run_dir)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)

    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            _model_cache.move_to_end(key)
            return cached

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise InferenceError("The saved model file is not a supported AILAB classification checkpoint.")

    config = checkpoint.get("config") or {}
    extra_args = config.get("extra_args") or {}
    classes = list(checkpoint.get("classes") or [])
    if not classes:
        raise InferenceError("The saved model file does not record its class names.")

    model_type = str(config.get("model_type") or "").lower()
    if model_type and model_type not in SUPPORTED_MODEL_TYPES:
        raise InferenceError(f"Model testing supports classification models only, not '{model_type}'.")

    architecture = str(extra_args.get("architecture") or config.get("model_name") or "")
    if not architecture:
        raise InferenceError("The saved model file does not record its architecture.")

    image_size = int(extra_args.get("image_size", 224) or 224)
    dropout = float(extra_args.get("dropout", 0.2) or 0.0)

    model = _build_classifier(architecture, len(classes), dropout)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise InferenceError(
            "The saved weights do not match the recorded architecture "
            f"({len(missing)} missing / {len(unexpected)} unexpected parameters)."
        )

    # Use the GPU when the image provides one; the CPU path is the norm because
    # the API image ships CPU-only wheels to avoid competing for VRAM with an
    # active training job.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    entry = {
        "model": model,
        "classes": classes,
        "image_size": image_size,
        "architecture": architecture,
        "device": str(device),
        "checkpoint": path.name,
    }
    with _cache_lock:
        _model_cache[key] = entry
        _model_cache.move_to_end(key)
        while len(_model_cache) > MODEL_CACHE_SIZE:
            _model_cache.popitem(last=False)
    return entry


def _open_verified_image(payload: bytes):
    """Reject anything that is not a decodable image within the pixel budget."""
    from PIL import Image

    if not payload:
        raise InferenceError("The uploaded file is empty.")
    if len(payload) > MAX_INFERENCE_IMAGE_BYTES:
        limit_mb = MAX_INFERENCE_IMAGE_BYTES / 1024 / 1024
        raise InferenceError(f"The image is larger than the {limit_mb:.0f} MB limit.")

    try:
        with Image.open(BytesIO(payload)) as probe:
            probe.verify()  # structural check; consumes the file object
    except Exception as exc:
        raise InferenceError("The uploaded file is not a readable image.") from exc

    # verify() leaves the image unusable, so decode again for the real pixels.
    try:
        image = Image.open(BytesIO(payload))
        width, height = image.size
        if width <= 0 or height <= 0:
            raise InferenceError("The uploaded image has no pixels.")
        if width * height > MAX_INFERENCE_IMAGE_PIXELS:
            raise InferenceError(
                f"The image has {width * height} pixels, above the "
                f"{MAX_INFERENCE_IMAGE_PIXELS} pixel limit."
            )
        return image.convert("RGB")
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError("The uploaded image could not be decoded.") from exc


def predict_image(run_dir: Path, payload: bytes, top_k: int = 3) -> dict[str, Any]:
    """Return the top-k softmax predictions for one image."""
    torch = _require_torch()
    import torchvision.transforms as transforms

    started = time.perf_counter()
    image = _open_verified_image(payload)
    original_size = list(image.size)

    entry = load_classifier(run_dir)
    model_ready = time.perf_counter()

    # Mirrors the validation transform in classification_common.py exactly.
    transform = transforms.Compose(
        [
            transforms.Resize((entry["image_size"], entry["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(entry["device"])

    with torch.no_grad():
        logits = model_forward(entry["model"], tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    classes = entry["classes"]
    count = max(1, min(int(top_k), len(classes)))
    scores, indices = torch.topk(probabilities, count)
    predictions = [
        {
            "className": classes[int(index)],
            "confidence": round(float(score), 6),
            "percent": round(float(score) * 100, 2),
        }
        for score, index in zip(scores.tolist(), indices.tolist())
    ]

    finished = time.perf_counter()
    return {
        "predictions": predictions,
        "top": predictions[0] if predictions else None,
        "classes": classes,
        "model": {
            "architecture": entry["architecture"],
            "checkpoint": entry["checkpoint"],
            "imageSize": entry["image_size"],
            "device": entry["device"],
        },
        "image": {"width": original_size[0], "height": original_size[1]},
        "timingMs": {
            "modelLoad": round((model_ready - started) * 1000, 1),
            "inference": round((finished - model_ready) * 1000, 1),
            "total": round((finished - started) * 1000, 1),
        },
    }


def model_forward(model, tensor):
    """Isolated so tests can stub the forward pass without a real model."""
    return model(tensor)
