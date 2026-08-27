from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from typing import Any


DEFAULT_GPU_VRAM_MB = 12282
DEFAULT_SYSTEM_RAM_GB = 16
SAFE_GPU_VRAM_MB = 10500
SAFE_SYSTEM_RAM_GB = 12
SAFE_WORKERS_MAX = 4


class ResourcePlanError(ValueError):
    def __init__(self, errors: list[str], suggestions: list[str] | None = None):
        self.errors = errors
        self.suggestions = suggestions or []
        message = " ".join(errors)
        if self.suggestions:
            message = f"{message} Suggestion: {' '.join(self.suggestions)}"
        super().__init__(message)


def _read_meminfo_gb(field: str) -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                key, _, value = line.partition(":")
                if key != field:
                    continue
                amount_kb = int(value.strip().split()[0])
                return round(amount_kb / 1024 / 1024, 1)
    except Exception:
        return None
    return None


def _detected_gpus() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 3)]
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        index, name, total_mb, free_mb = parts
        try:
            total = int(total_mb)
            free = int(free_mb)
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": name,
                "vram_total_mb": total,
                "vram_free_mb": free,
                "vram_total_gb": round(total / 1024, 1),
                "vram_free_gb": round(free / 1024, 1),
            }
        )
    return gpus


def get_resource_profile() -> dict[str, Any]:
    gpus = _detected_gpus()
    ram_total_gb = _read_meminfo_gb("MemTotal") or DEFAULT_SYSTEM_RAM_GB
    ram_available_gb = _read_meminfo_gb("MemAvailable") or max(ram_total_gb - 3, 1)
    primary_gpu = gpus[0] if gpus else None
    safe_vram_mb = min(
        SAFE_GPU_VRAM_MB,
        int((primary_gpu["vram_total_mb"] if primary_gpu else DEFAULT_GPU_VRAM_MB) * 0.86),
    )
    safe_ram_gb = min(SAFE_SYSTEM_RAM_GB, max(int(ram_total_gb - 4), 4))
    return {
        "hardware": {
            "gpus": gpus,
            "gpu_available": bool(gpus),
            "primary_gpu": primary_gpu,
            "default_device": primary_gpu["index"] if primary_gpu else "cpu",
            "system_ram_total_gb": ram_total_gb,
            "system_ram_available_gb": ram_available_gb,
        },
        "safe_limits": {
            "gpu_vram_mb": safe_vram_mb,
            "system_ram_gb": safe_ram_gb,
            "workers": SAFE_WORKERS_MAX,
        },
        "policy": "auto_safe",
        "notes": [
            "Limits are conservative heuristics to reduce training memory errors.",
            "Backend validation always runs before a training job is queued.",
        ],
    }


RESOURCE_METADATA: dict[str, dict[str, Any]] = {
    "resnet": {
        "safe_defaults": {"batch_size": 16, "image_size": 224, "workers": 4, "amp": True},
        "hard_limits": {"batch_size": 64, "image_size": 1024, "workers": 4},
        "memory_notes": ["224px works comfortably; reduce batch size for 512px or larger images."],
    },
    "efficientnet": {
        "safe_defaults": {"batch_size": 16, "image_size": 224, "workers": 4, "amp": True},
        "hard_limits": {"batch_size": 64, "image_size": 1024, "workers": 4},
        "memory_notes": ["B0-B3 are appropriate defaults; use smaller batches for larger images."],
    },
    "yolo": {
        "safe_defaults": {"batch_size": 16, "imgsz": 640, "model_size": "n", "workers": 4, "amp": True, "cache": False},
        "hard_limits": {"imgsz": 1024, "workers": 4, "cache": False},
        "memory_notes": ["YOLOv11 n/s are the safest defaults; m/l/x need smaller batches.", "Image cache is disabled by default to reduce memory errors."],
    },
    "faster_rcnn": {
        "safe_defaults": {"batch_size": 2, "image_size": 640, "max_size": 1333, "workers": 4, "amp": True},
        "hard_limits": {"batch_size": 4, "image_size": 1024, "max_size": 1600, "workers": 4},
        "memory_notes": ["Two-stage detectors are memory-heavy; batch 1-2 is the safest starting range."],
    },
    "mask_rcnn": {
        "safe_defaults": {"batch_size": 2, "image_size": 640, "max_size": 1333, "workers": 4, "amp": True},
        "hard_limits": {"batch_size": 4, "image_size": 1024, "max_size": 1600, "workers": 4},
        "memory_notes": ["Instance masks add memory pressure; use batch 1-2 for high-resolution datasets."],
    },
    "deeplabv3plus": {
        "safe_defaults": {"batch_size": 2, "image_size": 512, "workers": 4, "amp": True},
        "hard_limits": {"batch_size": 4, "image_size": 1024, "workers": 4},
        "memory_notes": ["Semantic segmentation scales quickly with image size; 1024px should use batch 1."],
    },
}


def catalog_resource_metadata(model_type: str) -> dict[str, Any]:
    metadata = deepcopy(RESOURCE_METADATA.get(model_type, {}))
    if metadata:
        metadata["resource_profile"] = {
            "target_gpu_vram_mb": DEFAULT_GPU_VRAM_MB,
            "safe_gpu_vram_mb": SAFE_GPU_VRAM_MB,
            "target_system_ram_gb": DEFAULT_SYSTEM_RAM_GB,
            "safe_system_ram_gb": SAFE_SYSTEM_RAM_GB,
            "policy": "auto_safe",
        }
    return metadata


def enrich_catalog_resources(catalog: dict[str, Any]) -> dict[str, Any]:
    for task in catalog.get("tasks", []):
        for model in task.get("models", []):
            model.update(catalog_resource_metadata(str(model.get("id", ""))))
    return catalog


def _num(params: dict[str, Any], key: str, default: int | float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(params: dict[str, Any], key: str, default: int) -> int:
    return int(_num(params, key, default))


def _device(params: dict[str, Any]) -> str:
    value = str(params.get("device", "0"))
    return "cpu" if value.lower() == "cpu" else value


def _batch_cap_for_classification(image_size: int, cpu: bool) -> int:
    if cpu:
        return 8 if image_size <= 224 else 2
    if image_size <= 224:
        return 64
    if image_size <= 512:
        return 16
    return 4


def _batch_cap_for_yolo(model_size: str, imgsz: int, cpu: bool) -> int:
    if cpu:
        return 4 if imgsz <= 640 and model_size in {"n", "s"} else 1
    base = {"n": 16, "s": 16, "m": 8, "l": 4, "x": 4}.get(model_size, 8)
    if imgsz <= 640:
        return base
    return max(1, min(base, {"n": 4, "s": 4, "m": 2, "l": 1, "x": 1}.get(model_size, 2)))


def _batch_cap_for_detection(image_size: int, cpu: bool) -> int:
    if cpu:
        return 1
    return 4 if image_size <= 640 else 2


def _batch_cap_for_deeplab(image_size: int, cpu: bool) -> int:
    if cpu:
        return 1
    if image_size <= 512:
        return 4
    if image_size <= 768:
        return 2
    return 1


def validate_resource_plan(
    model_type: str,
    params: dict[str, Any] | None = None,
    batch_size: int | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile or get_resource_profile()
    normalized_params = dict(params or {})
    normalized_batch_size = int(batch_size or _int(normalized_params, "batch_size", 1))
    warnings: list[str] = []
    errors: list[str] = []
    suggestions: list[str] = []
    model_type = str(model_type)
    is_cpu = _device(normalized_params) == "cpu"

    workers = _int(normalized_params, "workers", 4)
    worker_cap = 2 if is_cpu else SAFE_WORKERS_MAX
    if workers > worker_cap:
        normalized_params["workers"] = worker_cap
        warnings.append(f"Data workers reduced from {workers} to {worker_cap} for the available system RAM.")

    if model_type in {"resnet", "efficientnet"}:
        image_size = _int(normalized_params, "image_size", 224)
        if image_size > 1024:
            errors.append(f"{model_type} image_size {image_size} exceeds the safe 1024 cap to prevent memory errors.")
            suggestions.append("Use image_size 224-512, or batch 1-4 for larger images.")
        batch_cap = _batch_cap_for_classification(image_size, is_cpu)
        if normalized_batch_size > batch_cap:
            errors.append(f"{model_type} batch_size {normalized_batch_size} is too high for image_size {image_size}.")
            suggestions.append(f"Use batch_size <= {batch_cap}.")

    elif model_type == "yolo":
        imgsz = _int(normalized_params, "imgsz", 640)
        model_size = str(normalized_params.get("model_size", "n")).replace("yolo11", "").replace(".pt", "") or "n"
        if imgsz > 1024:
            errors.append(f"YOLOv11 imgsz {imgsz} exceeds the safe 1024 cap to prevent memory errors.")
            suggestions.append("Use imgsz 640 for normal training, or 1024 with a much smaller batch.")
        batch_cap = _batch_cap_for_yolo(model_size, imgsz, is_cpu)
        if normalized_batch_size > batch_cap:
            errors.append(f"YOLOv11{model_size} imgsz {imgsz} batch_size {normalized_batch_size} exceeds the safe limit.")
            suggestions.append(f"Use batch_size <= {batch_cap} for YOLOv11{model_size} at imgsz {min(imgsz, 1024)}.")
        if normalized_params.get("cache") is True and profile["hardware"].get("system_ram_total_gb", 16) <= 16:
            normalized_params["cache"] = False
            warnings.append("YOLO image cache was disabled because safe caching can cause memory errors on this machine.")

    elif model_type in {"faster_rcnn", "mask_rcnn"}:
        image_size = _int(normalized_params, "image_size", 640)
        max_size = _int(normalized_params, "max_size", 1333)
        if image_size > 1024:
            errors.append(f"{model_type} image_size {image_size} exceeds the safe 1024 cap to prevent memory errors.")
            suggestions.append("Use image_size 640-1024.")
        if max_size > 1600:
            errors.append(f"{model_type} max_size {max_size} exceeds the safe 1600 long-side cap.")
            suggestions.append("Use max_size <= 1600.")
        batch_cap = _batch_cap_for_detection(image_size, is_cpu)
        if normalized_batch_size > batch_cap:
            errors.append(f"{model_type} batch_size {normalized_batch_size} is too high for image_size {image_size}.")
            suggestions.append(f"Use batch_size <= {batch_cap}; batch 1-2 is recommended.")

    elif model_type == "deeplabv3plus":
        image_size = _int(normalized_params, "image_size", 512)
        if image_size > 1024:
            errors.append(f"DeepLabV3+ image_size {image_size} exceeds the safe 1024 cap to prevent memory errors.")
            suggestions.append("Use image_size 512 batch_size 2, or image_size 1024 batch_size 1.")
        batch_cap = _batch_cap_for_deeplab(image_size, is_cpu)
        if normalized_batch_size > batch_cap:
            errors.append(f"DeepLabV3+ image_size {image_size} with batch_size {normalized_batch_size} exceeds the safe limit.")
            suggestions.append(f"Use batch_size <= {batch_cap}.")

    estimated_vram_mb = estimate_training_memory(model_type, normalized_params, normalized_batch_size)
    safe_vram_mb = int(profile["safe_limits"]["gpu_vram_mb"])
    if not is_cpu and estimated_vram_mb > safe_vram_mb:
        errors.append(
            f"Estimated GPU memory {estimated_vram_mb} MB exceeds the safe {safe_vram_mb} MB limit."
        )
        suggestions.append("Reduce batch size or image size until the estimated GPU memory fits the safe limit.")
    plan = {
        "policy": "auto_safe",
        "ok": not errors,
        "model_type": model_type,
        "device": "cpu" if is_cpu else f"gpu:{_device(normalized_params)}",
        "batch_size": normalized_batch_size,
        "normalized_params": normalized_params,
        "warnings": warnings,
        "errors": errors,
        "suggestions": suggestions,
        "estimated_vram_mb": estimated_vram_mb,
        "safe_vram_mb": safe_vram_mb,
        "safe_system_ram_gb": profile["safe_limits"]["system_ram_gb"],
    }
    return plan


def enforce_resource_plan(
    model_type: str,
    params: dict[str, Any] | None = None,
    batch_size: int | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = validate_resource_plan(model_type, params=params, batch_size=batch_size, profile=profile)
    if not plan["ok"]:
        raise ResourcePlanError(plan["errors"], plan["suggestions"])
    return plan


def estimate_training_memory(model_type: str, params: dict[str, Any], batch_size: int) -> int:
    if model_type in {"resnet", "efficientnet"}:
        image_size = _int(params, "image_size", 224)
        return int(1200 + batch_size * (image_size / 224) ** 2 * 220)
    if model_type == "yolo":
        imgsz = _int(params, "imgsz", 640)
        model_size = str(params.get("model_size", "n"))
        factor = {"n": 360, "s": 440, "m": 650, "l": 900, "x": 1150}.get(model_size, 500)
        return int(1800 + batch_size * (imgsz / 640) ** 2 * factor)
    if model_type in {"faster_rcnn", "mask_rcnn"}:
        image_size = _int(params, "image_size", 640)
        max_size = _int(params, "max_size", 1333)
        factor = 1800 if model_type == "faster_rcnn" else 2200
        # Activation memory scales with the resized image area, which depends on
        # BOTH the short side (min_size) and the long-side cap (max_size), not
        # the short side alone. The factor is calibrated at the 640x1333 default,
        # so this equals the previous short-side-only estimate there while now
        # responding to a wider long-side cap.
        return int(2500 + batch_size * (image_size / 640) * (max_size / 1333) * factor)
    if model_type == "deeplabv3plus":
        image_size = _int(params, "image_size", 512)
        return int(2000 + batch_size * (image_size / 512) ** 2 * 2100)
    return 0
