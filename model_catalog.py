from __future__ import annotations

from copy import deepcopy
import subprocess
from typing import Any

from resource_guard import enrich_catalog_resources


def _number(
    key: str,
    label: str,
    default: int | float,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    description: str = "",
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "key": key,
        "label": label,
        "type": "number",
        "default": default,
        "description": description,
    }
    if minimum is not None:
        item["min"] = minimum
    if maximum is not None:
        item["max"] = maximum
    if step is not None:
        item["step"] = step
    return item


def _boolean(key: str, label: str, default: bool, description: str = "") -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "type": "boolean",
        "default": default,
        "description": description,
    }


def _select(
    key: str,
    label: str,
    default: str,
    options: list[dict[str, str]],
    description: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "type": "select",
        "default": default,
        "options": options,
        "description": description,
    }


def _text(key: str, label: str, default: str = "", description: str = "") -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "type": "text",
        "default": default,
        "description": description,
    }


def _detected_gpu_options() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return []

    options: list[dict[str, str]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        index, name, memory_mb = parts
        try:
            memory_gb = int(memory_mb) / 1024
            label = f"GPU {index} - {name} ({memory_gb:.1f} GB)"
        except ValueError:
            label = f"GPU {index} - {name}"
        options.append({"value": index, "label": label})
    return options


def _device_options() -> list[dict[str, str]]:
    return _detected_gpu_options() + [{"value": "cpu", "label": "CPU"}]


def _common_training_params() -> list[dict[str, Any]]:
    device_options = _device_options()
    default_device = device_options[0]["value"]
    return [
        _number("epochs", "Epochs", 50, 1, 2000, 1, "Maximum training epochs."),
        _number("batch_size", "Batch size", 16, 1, 256, 1, "Images processed in each training step."),
        _select(
            "device",
            "Device",
            default_device,
            device_options,
            "Available training devices detected from the deployment container.",
        ),
        _number("workers", "Data workers", 4, 0, 32, 1, "DataLoader worker processes."),
        _boolean("amp", "Mixed precision", True, "Use automatic mixed precision when CUDA is available."),
        _number("seed", "Random seed", 0, 0, 999999, 1, "Seed for reproducible runs."),
    ]


COMMON_TRAINING_PARAMS = _common_training_params()


OPTIMIZER_PARAMS = [
    _select(
        "optimizer",
        "Optimizer",
        "sgd",
        [
            {"value": "sgd", "label": "SGD"},
            {"value": "adam", "label": "Adam"},
            {"value": "adamw", "label": "AdamW"},
            {"value": "rmsprop", "label": "RMSprop"},
        ],
    ),
    _number("learning_rate", "Learning rate", 0.001, 0.000001, 1, 0.0001),
    _number("momentum", "Momentum", 0.9, 0, 0.999, 0.001, "Used by SGD/RMSprop."),
    _number("weight_decay", "Weight decay", 0.0005, 0, 0.1, 0.0001),
    _select(
        "scheduler",
        "LR scheduler",
        "cosine",
        [
            {"value": "none", "label": "None"},
            {"value": "step", "label": "StepLR"},
            {"value": "cosine", "label": "Cosine annealing"},
        ],
    ),
]


CLASSIFICATION_PARAMS = [
    _select(
        "architecture",
        "Architecture",
        "resnet50",
        [
            {"value": "resnet18", "label": "ResNet-18"},
            {"value": "resnet34", "label": "ResNet-34"},
            {"value": "resnet50", "label": "ResNet-50"},
            {"value": "resnet101", "label": "ResNet-101"},
        ],
    ),
    _number("image_size", "Image size", 224, 64, 1024, 1),
    _boolean("pretrained", "ImageNet weights", True),
    _boolean("freeze_backbone", "Freeze backbone", False),
    _number("dropout", "Classifier dropout", 0.2, 0, 0.8, 0.05),
    _number("label_smoothing", "Label smoothing", 0.0, 0, 0.5, 0.01),
    _number("random_rotation", "Random rotation", 0, 0, 180, 1),
    _number("horizontal_flip", "Horizontal flip probability", 0.5, 0, 1, 0.05),
    _number("color_jitter", "Color jitter strength", 0.1, 0, 1, 0.05),
] + OPTIMIZER_PARAMS


EFFICIENTNET_PARAMS = [
    _select(
        "architecture",
        "Architecture",
        "efficientnet_b0",
        [
            {"value": "efficientnet_b0", "label": "EfficientNet-B0"},
            {"value": "efficientnet_b1", "label": "EfficientNet-B1"},
            {"value": "efficientnet_b2", "label": "EfficientNet-B2"},
            {"value": "efficientnet_b3", "label": "EfficientNet-B3"},
        ],
    ),
] + [p for p in CLASSIFICATION_PARAMS if p["key"] != "architecture"]


DETECTION_PARAMS = [
    _boolean("pretrained", "COCO weights", True),
    _number("image_size", "Short side", 640, 256, 1600, 32),
    _number("max_size", "Long side cap", 1333, 512, 2400, 32),
    _number("trainable_backbone_layers", "Trainable backbone layers", 3, 0, 5, 1),
    _number("rpn_nms_thresh", "RPN NMS threshold", 0.7, 0.1, 1, 0.05),
    _number("box_score_thresh", "Box score threshold", 0.05, 0, 1, 0.01),
    _number("box_nms_thresh", "Box NMS threshold", 0.5, 0.1, 1, 0.05),
    _number("detections_per_img", "Detections per image", 100, 1, 1000, 1),
] + OPTIMIZER_PARAMS


DEEPLAB_PARAMS = [
    _select(
        "encoder_name",
        "Encoder",
        "resnet34",
        [
            {"value": "resnet18", "label": "ResNet-18"},
            {"value": "resnet34", "label": "ResNet-34"},
            {"value": "resnet50", "label": "ResNet-50"},
            {"value": "efficientnet-b0", "label": "EfficientNet-B0"},
            {"value": "mobilenet_v2", "label": "MobileNetV2"},
        ],
    ),
    _select(
        "encoder_weights",
        "Encoder weights",
        "imagenet",
        [
            {"value": "imagenet", "label": "ImageNet"},
            {"value": "none", "label": "Random init"},
        ],
    ),
    _number("image_size", "Image size", 512, 128, 2048, 32),
    _number("num_classes", "Mask classes", 2, 1, 1000, 1),
    _number("ignore_index", "Ignored mask value", 255, -1, 255, 1),
    _number("encoder_depth", "Encoder depth", 5, 3, 5, 1),
    _select(
        "encoder_output_stride",
        "Output stride",
        "16",
        [
            {"value": "8", "label": "8"},
            {"value": "16", "label": "16"},
        ],
    ),
    _number("decoder_channels", "Decoder channels", 256, 32, 1024, 32),
    _text("decoder_atrous_rates", "Atrous rates", "12,24,36"),
    _select(
        "loss",
        "Loss",
        "cross_entropy",
        [
            {"value": "cross_entropy", "label": "Cross entropy"},
            {"value": "dice", "label": "Dice"},
        ],
    ),
] + OPTIMIZER_PARAMS


CV_MODEL_CATALOG: dict[str, Any] = {
    "version": "2026-06-13",
    "common_params": COMMON_TRAINING_PARAMS,
    "tasks": [
        {
            "id": "image_classification",
            "label": "Image Classification",
            "description": "Fine-tune image classifiers from folder-per-class datasets.",
            "dataset_formats": ["imagefolder"],
            "models": [
                {
                    "id": "resnet",
                    "label": "ResNet",
                    "model_name": "resnet50",
                    "runtime": "pytorch/pytorch + torchvision",
                    "dataset_formats": ["imagefolder"],
                    "reason": "TorchVision provides pretrained weights and the PyTorch Docker images are already aligned with this stack.",
                    "params": CLASSIFICATION_PARAMS,
                },
                {
                    "id": "efficientnet",
                    "label": "EfficientNet",
                    "model_name": "efficientnet_b0",
                    "runtime": "pytorch/pytorch + torchvision",
                    "dataset_formats": ["imagefolder"],
                    "reason": "Popular compact classifiers with pretrained TorchVision weights.",
                    "params": EFFICIENTNET_PARAMS,
                },
            ],
        },
        {
            "id": "segmentation",
            "label": "Semantic / Instance Segmentation",
            "description": "Train semantic masks or instance masks depending on the chosen model.",
            "dataset_formats": ["semantic_masks", "coco_instances"],
            "models": [
                {
                    "id": "deeplabv3plus",
                    "label": "DeepLabV3+",
                    "model_name": "deeplabv3plus",
                    "runtime": "pytorch/pytorch + segmentation-models-pytorch",
                    "dataset_formats": ["semantic_masks"],
                    "reason": "A widely used semantic segmentation architecture with flexible encoders.",
                    "params": DEEPLAB_PARAMS,
                },
                {
                    "id": "mask_rcnn",
                    "label": "Mask R-CNN",
                    "model_name": "maskrcnn_resnet50_fpn_v2",
                    "runtime": "pytorch/pytorch + torchvision",
                    "dataset_formats": ["coco_instances"],
                    "reason": "TorchVision includes pretrained Mask R-CNN and a standard fine-tuning path for instance segmentation.",
                    "params": DETECTION_PARAMS,
                },
            ],
        },
        {
            "id": "object_detection",
            "label": "Object Detection",
            "description": "Train bounding-box detectors. YOLO supports YOLO datasets, while Faster R-CNN supports YOLO and COCO boxes.",
            "dataset_formats": ["yolo_detection", "coco_instances"],
            "models": [
                {
                    "id": "yolo",
                    "label": "YOLOv11",
                    "model_name": "yolo11n",
                    "runtime": "pytorch/pytorch + ultralytics",
                    "dataset_formats": ["yolo_detection"],
                    "reason": "Fast YOLOv11 detector for bounding-box object detection with a practical speed and accuracy baseline.",
                    "params": [
                        _select(
                            "model_size",
                            "Model size",
                            "n",
                            [
                                {"value": "n", "label": "Nano"},
                                {"value": "s", "label": "Small"},
                                {"value": "m", "label": "Medium"},
                                {"value": "l", "label": "Large"},
                                {"value": "x", "label": "XLarge"},
                            ],
                        ),
                        _number("imgsz", "Image size", 640, 64, 2048, 32),
                        _select(
                            "optimizer",
                            "Optimizer",
                            "auto",
                            [
                                {"value": "auto", "label": "Auto"},
                                {"value": "SGD", "label": "SGD"},
                                {"value": "Adam", "label": "Adam"},
                                {"value": "AdamW", "label": "AdamW"},
                            ],
                        ),
                        _number("lr0", "Initial LR", 0.01, 0.000001, 1, 0.0001),
                        _number("lrf", "Final LR factor", 0.01, 0, 1, 0.001),
                        _number("momentum", "Momentum", 0.937, 0, 0.999, 0.001),
                        _number("weight_decay", "Weight decay", 0.0005, 0, 0.1, 0.0001),
                        _number("patience", "Early stop patience", 100, 0, 1000, 1),
                        _boolean("pretrained", "Pretrained weights", True),
                        _boolean("cache", "Cache images", False),
                        _number("degrees", "Rotation", 0, 0, 180, 1),
                        _number("translate", "Translate", 0.1, 0, 1, 0.01),
                        _number("scale", "Scale", 0.5, 0, 1, 0.01),
                        _number("fliplr", "Horizontal flip", 0.5, 0, 1, 0.05),
                        _number("mosaic", "Mosaic", 1.0, 0, 1, 0.05),
                        _number("mixup", "MixUp", 0.0, 0, 1, 0.05),
                    ],
                },
                {
                    "id": "faster_rcnn",
                    "label": "Faster R-CNN",
                    "model_name": "fasterrcnn_resnet50_fpn_v2",
                    "runtime": "pytorch/pytorch + torchvision",
                    "dataset_formats": ["yolo_detection", "coco_instances"],
                    "reason": "The most established two-stage detector in the provided choices, available directly in TorchVision, and compatible with COCO boxes.",
                    "params": DETECTION_PARAMS,
                },
            ],
        },
    ],
}


DATASET_INTERFACES = {
    "resnet": {
        "dataset_task": "image_classification",
        "required_annotations": ["one class label per image"],
        "accepted_source_formats": ["imagefolder"],
        "accepted_canonical_formats": ["imagefolder"],
        "canonical_format": "imagefolder",
        "conversion_targets": ["imagefolder"],
        "train_export_format": "imagefolder",
    },
    "efficientnet": {
        "dataset_task": "image_classification",
        "required_annotations": ["one class label per image"],
        "accepted_source_formats": ["imagefolder"],
        "accepted_canonical_formats": ["imagefolder"],
        "canonical_format": "imagefolder",
        "conversion_targets": ["imagefolder"],
        "train_export_format": "imagefolder",
    },
    "yolo": {
        "dataset_task": "object_detection",
        "required_annotations": ["bounding boxes"],
        "accepted_source_formats": ["yolo", "coco", "cvat_coco", "label_studio_coco", "roboflow_coco"],
        "accepted_canonical_formats": ["object_detection_boxes"],
        "canonical_format": "object_detection_boxes",
        "conversion_targets": ["yolo_detection"],
        "train_export_format": "yolo_detection",
    },
    "faster_rcnn": {
        "dataset_task": "object_detection",
        "required_annotations": ["bounding boxes"],
        "accepted_source_formats": ["yolo", "coco", "cvat_coco", "label_studio_coco", "roboflow_coco"],
        "accepted_canonical_formats": ["object_detection_boxes", "coco_instance_masks"],
        "canonical_format": "object_detection_boxes",
        "conversion_targets": ["coco_instances", "yolo_detection"],
        "train_export_format": "coco_instances_or_yolo_detection",
    },
    "deeplabv3plus": {
        "dataset_task": "semantic_segmentation",
        "required_annotations": ["one class ID per pixel mask"],
        "accepted_source_formats": ["semantic_masks"],
        "accepted_canonical_formats": ["semantic_masks"],
        "canonical_format": "semantic_masks",
        "conversion_targets": ["semantic_masks"],
        "train_export_format": "semantic_masks",
    },
    "mask_rcnn": {
        "dataset_task": "instance_segmentation",
        "required_annotations": ["instance masks", "bounding boxes"],
        "accepted_source_formats": ["coco", "cvat_coco", "label_studio_coco", "roboflow_coco"],
        "accepted_canonical_formats": ["coco_instance_masks"],
        "canonical_format": "coco_instance_masks",
        "conversion_targets": ["coco_instances"],
        "train_export_format": "coco_instances",
    },
}


def _enrich_catalog_interfaces(catalog: dict[str, Any]) -> dict[str, Any]:
    for task in catalog.get("tasks", []):
        for model in task.get("models", []):
            model.update(DATASET_INTERFACES.get(model.get("id"), {}))
    return catalog


def get_catalog() -> dict[str, Any]:
    return enrich_catalog_resources(_enrich_catalog_interfaces(deepcopy(CV_MODEL_CATALOG)))


def flatten_models() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    catalog = get_catalog()
    for task in catalog["tasks"]:
        for model in task["models"]:
            entry = deepcopy(model)
            entry["task_type"] = task["id"]
            entry["task_label"] = task["label"]
            entry["dataset_formats"] = entry.get("dataset_formats", task["dataset_formats"])
            models[model["id"]] = entry
    return models


def get_model(model_type: str) -> dict[str, Any] | None:
    return flatten_models().get(model_type)


def validate_model_params(model_type: str, params: dict[str, Any]) -> dict[str, Any]:
    model = get_model(model_type)
    if model is None:
        raise ValueError(f"Unsupported model_type: {model_type}")

    specs = {item["key"]: item for item in COMMON_TRAINING_PARAMS + model.get("params", [])}
    unknown = sorted(set(params) - set(specs))
    if unknown:
        raise ValueError(f"Unsupported params for {model_type}: {', '.join(unknown)}")

    validated: dict[str, Any] = {}
    for key, value in params.items():
        spec = specs[key]
        if spec["type"] == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Param '{key}' must be a number")
            if "min" in spec and value < spec["min"]:
                raise ValueError(f"Param '{key}' must be at least {spec['min']}")
            if "max" in spec and value > spec["max"]:
                raise ValueError(f"Param '{key}' must be at most {spec['max']}")
        elif spec["type"] == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Param '{key}' must be true or false")
        elif spec["type"] == "select":
            choices = {option["value"] for option in spec.get("options", [])}
            if str(value) not in choices:
                raise ValueError(f"Param '{key}' must be one of: {', '.join(sorted(choices))}")
        elif spec["type"] == "text" and not isinstance(value, str):
            raise ValueError(f"Param '{key}' must be text")
        validated[key] = value
    return validated
