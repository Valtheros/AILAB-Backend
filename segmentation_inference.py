"""Single-image inference for completed segmentation runs.

Two model families:

* **DeepLabV3+ (semantic)** — ``best.pt`` holds ``{"model": state_dict,
  "config": job_config}`` (no class names); the model is rebuilt with
  ``segmentation_models_pytorch`` exactly as in
  ``worker/trainers/deeplabv3plus_trainer.py``. Output is one class id per
  pixel.
* **Mask R-CNN (instance)** — ``best.pt`` holds ``{"model", "classes",
  "config"}`` and outputs per-instance boxes, labels, scores, and masks,
  mirroring ``worker/trainers/detection_common.build_mask_rcnn``.

Rather than shipping raw pixel data to the browser, this module composites a
translucent RGBA overlay (a PNG the size of the original image) and returns it
as a data URI. The frontend lays that PNG over the uploaded image, so a mask
lines up regardless of display scaling. A colour legend accompanies it.

The builders below mirror the trainers and are duplicated for the same reason
as detection_inference: importing the worker trainer package pulls in the full
training stack.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from inference_service import (
    InferenceError,
    InferenceUnavailable,
    MODEL_CACHE_SIZE,
    _open_verified_image,
    _require_torch,
)
from detection_inference import _clamp_threshold

SEGMENTATION_TASK_TYPE = "segmentation"
SEGMENTATION_MODEL_TYPES = {"deeplabv3plus", "mask_rcnn"}

# Alpha applied to overlay colours (0-255); low enough to read the photo beneath.
OVERLAY_ALPHA = 130
# Distinct hues cycled per class, matching the frontend's detection palette.
PALETTE = [
    (239, 68, 68), (59, 130, 246), (34, 197, 94), (245, 158, 11),
    (168, 85, 247), (236, 72, 153), (6, 182, 212), (132, 204, 22),
]


_cache_lock = threading.Lock()
_model_cache: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()


def _hex(colour: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % colour


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


def _checkpoint_path(run_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise InferenceError(
        "This run has no saved model file (best.pt). It may have failed before "
        "the first epoch finished."
    )


def _build_deeplabv3plus(config: dict, num_classes: int):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - depends on the built image
        raise InferenceUnavailable(
            "segmentation-models-pytorch is not installed in the API image. "
            "Rebuild the backend image so DeepLabV3+ testing can run."
        ) from exc

    args = (config or {}).get("extra_args") or {}
    atrous_rates = tuple(
        int(value.strip())
        for value in str(args.get("decoder_atrous_rates", "12,24,36")).split(",")
        if value.strip()
    ) or (12, 24, 36)
    return smp.DeepLabV3Plus(
        encoder_name=str(args.get("encoder_name", "resnet34")),
        encoder_depth=int(args.get("encoder_depth", 5)),
        encoder_weights=None,  # overwritten by the trained state_dict
        encoder_output_stride=int(args.get("encoder_output_stride", 16)),
        decoder_channels=int(args.get("decoder_channels", 256)),
        decoder_atrous_rates=atrous_rates,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def _build_mask_rcnn(config: dict, num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    args = (config or {}).get("extra_args") or {}
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
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
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def _semantic_class_names(config: dict, num_classes: int) -> list[str]:
    """DeepLabV3+ does not store class names; recover them from the job config.

    By convention class 0 is background and the dataset's declared classes fill
    indices 1..N. Missing entries fall back to a generic label.
    """
    declared = list((config or {}).get("dataset_metadata", {}).get("classes") or [])
    names = ["background"] + declared
    if len(names) < num_classes:
        names += [f"class {i}" for i in range(len(names), num_classes)]
    return names[:num_classes]


def _load_deeplabv3plus(run_dir: Path) -> dict[str, Any]:
    torch = _require_torch()
    path = _checkpoint_path(run_dir)
    stat = path.stat()
    key = ("deeplabv3plus", str(path), stat.st_mtime_ns, stat.st_size)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise InferenceError("The saved model file is not a supported AILAB segmentation checkpoint.")
    config = checkpoint.get("config") or {}
    args = config.get("extra_args") or {}
    num_classes = int(args.get("num_classes", 2))
    image_size = int(args.get("image_size", 512))

    model = _build_deeplabv3plus(config, num_classes)
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
        "kind": "semantic",
        "model": model,
        "classes": _semantic_class_names(config, num_classes),
        "num_classes": num_classes,
        "image_size": image_size,
        "device": str(device),
        "architecture": f"deeplabv3plus/{args.get('encoder_name', 'resnet34')}",
        "checkpoint": path.name,
    }
    _cache_put(key, entry)
    return entry


def _load_mask_rcnn(run_dir: Path) -> dict[str, Any]:
    torch = _require_torch()
    path = _checkpoint_path(run_dir)
    stat = path.stat()
    key = ("mask_rcnn", str(path), stat.st_mtime_ns, stat.st_size)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise InferenceError("The saved model file is not a supported AILAB segmentation checkpoint.")
    classes = list(checkpoint.get("classes") or [])
    if not classes:
        raise InferenceError("The saved model file does not record its class names.")
    config = checkpoint.get("config") or {}

    num_classes = len(classes) + 1  # index 0 = background
    model = _build_mask_rcnn(config, num_classes)
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
        "kind": "instance",
        "model": model,
        "classes": classes,
        "device": str(device),
        "architecture": "maskrcnn_resnet50_fpn_v2",
        "checkpoint": path.name,
    }
    _cache_put(key, entry)
    return entry


def load_segmenter(run_dir: Path, model_type: str) -> dict[str, Any]:
    kind = (model_type or "").lower()
    if kind == "deeplabv3plus":
        return _load_deeplabv3plus(run_dir)
    if kind == "mask_rcnn":
        return _load_mask_rcnn(run_dir)
    raise InferenceError(f"Model testing does not support segmentation model '{model_type}'.")


def _png_data_uri(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _predict_semantic(entry: dict[str, Any], image, _threshold: float) -> dict[str, Any]:
    torch = _require_torch()
    import numpy as np
    import torchvision.transforms.functional as F
    from torchvision.transforms import InterpolationMode
    from PIL import Image

    width, height = image.size
    size = entry["image_size"]
    # Mirrors the training transform: resize (bilinear) then to_tensor, NO
    # ImageNet normalisation (the trainer normalises nothing on the input).
    resized = F.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR)
    tensor = F.to_tensor(resized).unsqueeze(0).to(entry["device"])
    with torch.no_grad():
        logits = entry["model"](tensor)
    prediction = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)

    # Back to the original resolution with nearest-neighbour so class edges stay
    # crisp, then colour every non-background class into an RGBA overlay.
    class_map = np.asarray(
        Image.fromarray(prediction, mode="L").resize((width, height), Image.NEAREST)
    )
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    total = float(width * height) or 1.0
    legend = []
    for class_id in range(1, entry["num_classes"]):
        mask = class_map == class_id
        count = int(mask.sum())
        if count == 0:
            continue
        colour = PALETTE[(class_id - 1) % len(PALETTE)]
        overlay[mask] = (*colour, OVERLAY_ALPHA)
        name = entry["classes"][class_id] if class_id < len(entry["classes"]) else f"class {class_id}"
        legend.append({"name": name, "color": _hex(colour), "percent": round(count / total * 100, 2)})

    legend.sort(key=lambda item: item["percent"], reverse=True)
    return {
        "segmentation": {
            "kind": "semantic",
            "overlay": _png_data_uri(Image.fromarray(overlay, mode="RGBA")),
            "legend": legend,
        },
        "count": len(legend),
    }


def _predict_instance(entry: dict[str, Any], image, threshold: float) -> dict[str, Any]:
    torch = _require_torch()
    import numpy as np
    import torchvision.transforms.functional as F
    from PIL import Image, ImageDraw

    width, height = image.size
    tensor = F.to_tensor(image).to(entry["device"])
    with torch.no_grad():
        output = entry["model"]([tensor])[0]

    classes = entry["classes"]
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    instances = []
    counts: dict[str, int] = {}

    boxes = output["boxes"].tolist()
    labels = output["labels"].tolist()
    scores = output["scores"].tolist()
    masks = output["masks"].to("cpu").numpy()  # (N, 1, H, W) in [0, 1]
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if float(score) < threshold:
            continue
        class_index = int(label) - 1  # 0 = background
        name = classes[class_index] if 0 <= class_index < len(classes) else str(label)
        colour = PALETTE[class_index % len(PALETTE)] if class_index >= 0 else PALETTE[0]

        binary = masks[index, 0] > 0.5
        if binary.any():
            tint = np.zeros((height, width, 4), dtype=np.uint8)
            tint[binary] = (*colour, OVERLAY_ALPHA)
            overlay.alpha_composite(Image.fromarray(tint, mode="RGBA"))
        draw.rectangle(box, outline=(*colour, 255), width=2)
        draw.text((box[0] + 2, box[1] + 2), f"{name} {round(float(score) * 100)}%", fill=(255, 255, 255, 255))

        instances.append({
            "className": name,
            "confidence": round(float(score), 6),
            "percent": round(float(score) * 100, 2),
            "box": [round(float(v), 1) for v in box],
            "color": _hex(colour),
        })
        counts[name] = counts.get(name, 0) + 1

    return {
        "segmentation": {
            "kind": "instance",
            "overlay": _png_data_uri(overlay),
            "instances": instances,
            "counts": counts,
        },
        "count": len(instances),
    }


def predict_segmentation(
    run_dir: Path,
    payload: bytes,
    model_type: str,
    score_threshold: Any = None,
) -> dict[str, Any]:
    """Return a colour overlay (and legend) for one uploaded image."""
    started = time.perf_counter()
    threshold = _clamp_threshold(score_threshold if score_threshold is not None else 0.5)
    image = _open_verified_image(payload)
    original_size = list(image.size)

    entry = load_segmenter(run_dir, model_type)
    model_ready = time.perf_counter()

    if entry["kind"] == "semantic":
        payload_out = _predict_semantic(entry, image, threshold)
    else:
        payload_out = _predict_instance(entry, image, threshold)

    finished = time.perf_counter()
    return {
        **payload_out,
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
