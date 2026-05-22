from __future__ import annotations

from copy import deepcopy
from typing import Any


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


COMMON_TRAINING_PARAMS = [
    _number("epochs", "Epochs", 50, 1, 2000, 1, "Maximum training epochs."),
    _number("batch_size", "Batch size", 16, -1, 256, 1, "-1 lets compatible trainers auto-size."),
    _select(
        "device",
        "Device",
        "0",
        [
            {"value": "0", "label": "GPU 0"},
            {"value": "0,1", "label": "GPU 0,1"},
            {"value": "cpu", "label": "CPU"},
        ],
        "Torch-style device selector.",
    ),
    _number("workers", "Data workers", 4, 0, 32, 1, "DataLoader worker processes."),
    _boolean("amp", "Mixed precision", True, "Use automatic mixed precision when CUDA is available."),
    _number("seed", "Random seed", 0, 0, 999999, 1, "Seed for reproducible runs."),
]


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


OCR_COMMON_PARAMS = [
    _select(
        "ocr_task",
        "OCR task",
        "rec",
        [
            {"value": "det", "label": "Text detection"},
            {"value": "rec", "label": "Text recognition"},
            {"value": "e2e", "label": "End-to-end OCR"},
        ],
    ),
    _number("learning_rate", "Learning rate", 0.001, 0.000001, 1, 0.0001),
    _text("character_dict_path", "Character dictionary path", ""),
    _boolean("use_space_char", "Use space character", True),
    _number("max_text_length", "Max text length", 25, 1, 512, 1),
]


CV_MODEL_CATALOG: dict[str, Any] = {
    "version": "2026-05-18",
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
            "dataset_formats": ["semantic_masks", "coco_instances", "yolo_segmentation"],
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
            "id": "ocr",
            "label": "OCR / Document Vision",
            "description": "Train or fine-tune OCR engines with OCR-specific annotation files.",
            "dataset_formats": ["paddleocr_labels", "tesseract_ground_truth"],
            "models": [
                {
                    "id": "paddleocr",
                    "label": "PaddleOCR",
                    "model_name": "paddleocr",
                    "runtime": "paddlepaddle/paddle + PaddleOCR",
                    "dataset_formats": ["paddleocr_labels"],
                    "reason": "PaddleOCR has an official Docker-based workflow and covers detection, recognition, and document OCR training.",
                    "params": [
                        _text("config_path", "PaddleOCR config path", "/opt/PaddleOCR/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml"),
                        _text("pretrained_model", "Pretrained model path", ""),
                        _number("batch_size_per_card", "Batch per card", 32, 1, 512, 1),
                    ]
                    + OCR_COMMON_PARAMS,
                },
                {
                    "id": "tesseract",
                    "label": "Tesseract",
                    "model_name": "tesseract_lstm",
                    "runtime": "tesseract-ocr/tesstrain",
                    "dataset_formats": ["tesseract_ground_truth"],
                    "reason": "Tesseract training is supported through the official tesstrain workflow and is practical for language/font adaptation.",
                    "params": [
                        _text("model_name", "Output language/model code", "custom"),
                        _text("start_model", "Start model", "eng"),
                        _number("max_iterations", "Max iterations", 10000, 100, 1000000, 100),
                        _number("target_error_rate", "Target CER", 0.01, 0.0001, 1, 0.001),
                        _number("ratio_train", "Train split ratio", 0.9, 0.1, 0.99, 0.01),
                    ],
                },
            ],
        },
        {
            "id": "object_detection",
            "label": "Object Detection",
            "description": "Train bounding-box detectors. YOLO remains available, with Faster R-CNN added as the second detector family.",
            "dataset_formats": ["yolo_detection", "coco_instances"],
            "models": [
                {
                    "id": "yolo",
                    "label": "YOLOv11",
                    "model_name": "yolo11n",
                    "runtime": "pytorch/pytorch + ultralytics",
                    "dataset_formats": ["yolo_detection", "yolo_segmentation"],
                    "reason": "Existing platform model.",
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
                    "dataset_formats": ["yolo_detection"],
                    "reason": "The most established two-stage detector in the provided choices and available directly in TorchVision.",
                    "params": DETECTION_PARAMS,
                },
            ],
        },
    ],
}


def get_catalog() -> dict[str, Any]:
    return deepcopy(CV_MODEL_CATALOG)


def flatten_models() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for task in CV_MODEL_CATALOG["tasks"]:
        for model in task["models"]:
            entry = deepcopy(model)
            entry["task_type"] = task["id"]
            entry["task_label"] = task["label"]
            entry["dataset_formats"] = entry.get("dataset_formats", task["dataset_formats"])
            models[model["id"]] = entry
    return models


def get_model(model_type: str) -> dict[str, Any] | None:
    return flatten_models().get(model_type)
