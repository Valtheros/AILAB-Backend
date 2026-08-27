from __future__ import annotations

import json
from pathlib import Path

from .detection_datasets import CocoInstanceDataset, YoloBoxDataset
from .trainer_utils import (
    append_csv_row,
    format_epoch_metrics,
    collate_detection,
    extra,
    get_device,
    optimizer_for,
    require_positive_batch_size,
    read_yaml,
    runs_root,
    scheduler_for,
    set_seed,
)


def _empty_detection_metrics(include_masks: bool, value="") -> dict:
    metrics = {
        "metrics/precision(B)": value,
        "metrics/recall(B)": value,
        "metrics/mAP50(B)": value,
        "metrics/mAP50-95(B)": value,
    }
    if include_masks:
        metrics.update({"metrics/mAP50(M)": value, "metrics/mAP50-95(M)": value})
    return metrics


def _encode_coco_mask(mask) -> dict:
    import numpy as np
    from pycocotools import mask as coco_mask

    encoded = coco_mask.encode(np.asfortranarray(mask.numpy().astype("uint8")))
    if isinstance(encoded["counts"], bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _coco_ap(images: list[dict], annotations: list[dict], categories: list[dict], detections: list[dict], iou_type: str) -> tuple[float, float]:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not annotations or not detections:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {"images": images, "annotations": annotations, "categories": categories, "info": {}}
        ground_truth.createIndex()
        predictions = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, predictions, iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return max(float(evaluator.stats[1]), 0.0), max(float(evaluator.stats[0]), 0.0)


def _box_overlaps(box, candidates):
    import torch

    if not len(candidates):
        return torch.empty(0)
    top_left = torch.maximum(box[:2], candidates[:, :2])
    bottom_right = torch.minimum(box[2:], candidates[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(1)
    box_area = (box[2:] - box[:2]).clamp(min=0).prod()
    candidate_area = (candidates[:, 2:] - candidates[:, :2]).clamp(min=0).prod(1)
    return intersection / (box_area + candidate_area - intersection).clamp(min=1e-9)


def _evaluate_detection_metrics(
    model,
    loader,
    device,
    num_classes: int,
    include_masks: bool,
) -> dict:
    import torch

    images_json: list[dict] = []
    annotations: list[dict] = []
    box_detections: list[dict] = []
    mask_detections: list[dict] = []
    true_positives = false_positives = ground_truth_count = 0
    annotation_id = 1

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            device_images = [image.to(device) for image in images]
            outputs = [{key: value.detach().cpu() for key, value in output.items()} for output in model(device_images)]
            for image, target, output in zip(images, targets, outputs):
                image_id = int(target["image_id"].reshape(-1)[0].item())
                images_json.append({"id": image_id, "width": int(image.shape[-1]), "height": int(image.shape[-2])})
                boxes = target["boxes"].cpu()
                labels = target["labels"].cpu()
                ground_truth_count += len(boxes)
                for index, (box, label) in enumerate(zip(boxes, labels)):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": int(target.get("iscrowd", torch.zeros(len(boxes), dtype=torch.int64))[index]),
                    }
                    if include_masks:
                        annotation["segmentation"] = _encode_coco_mask(target["masks"][index].cpu())
                    annotations.append(annotation)
                    annotation_id += 1

                prediction_boxes = output.get("boxes", torch.empty((0, 4)))
                prediction_labels = output.get("labels", torch.empty(0, dtype=torch.int64))
                prediction_scores = output.get("scores", torch.empty(0))
                order = prediction_scores.argsort(descending=True)
                matched = torch.zeros(len(boxes), dtype=torch.bool)
                for index in order:
                    box = prediction_boxes[index]
                    label = prediction_labels[index]
                    x1, y1, x2, y2 = [float(value) for value in box]
                    detection = {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(prediction_scores[index]),
                    }
                    box_detections.append(detection)
                    candidates = torch.where((labels == label) & ~matched)[0]
                    if len(candidates):
                        overlaps = _box_overlaps(box, boxes[candidates])
                        best = int(overlaps.argmax().item())
                        if float(overlaps[best]) >= 0.5:
                            matched[candidates[best]] = True
                            true_positives += 1
                        else:
                            false_positives += 1
                    else:
                        false_positives += 1
                    if include_masks and "masks" in output:
                        predicted_mask = output["masks"][index, 0] >= 0.5
                        mask_detection = dict(detection)
                        mask_detection.pop("bbox")
                        mask_detection["segmentation"] = _encode_coco_mask(predicted_mask.to(torch.uint8))
                        mask_detections.append(mask_detection)

    categories = [{"id": label, "name": str(label)} for label in range(1, num_classes)]
    box_map50, box_map = _coco_ap(images_json, annotations, categories, box_detections, "bbox")
    metrics = _empty_detection_metrics(include_masks, 0.0)
    metrics.update({
        "metrics/precision(B)": true_positives / max(true_positives + false_positives, 1),
        "metrics/recall(B)": true_positives / max(ground_truth_count, 1),
        "metrics/mAP50(B)": box_map50,
        "metrics/mAP50-95(B)": box_map,
    })
    if include_masks:
        mask_map50, mask_map = _coco_ap(images_json, annotations, categories, mask_detections, "segm")
        metrics.update({"metrics/mAP50(M)": mask_map50, "metrics/mAP50-95(M)": mask_map})
    return metrics


def _num_classes_from_dataset(dataset, fallback: int = 2) -> int:
    classes = getattr(dataset, "classes", None) or []
    return max(len(classes) + 1, fallback)


def build_faster_rcnn(config: dict, num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    args = extra(config)
    pretrained = bool(args.get("pretrained", True))
    weights = "DEFAULT" if pretrained else None
    weights_backbone = None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=weights_backbone,
        min_size=int(args.get("image_size", 640)),
        max_size=int(args.get("max_size", 1333)),
        trainable_backbone_layers=int(args.get("trainable_backbone_layers", 3)),
        rpn_nms_thresh=float(args.get("rpn_nms_thresh", 0.7)),
        box_score_thresh=float(args.get("box_score_thresh", 0.05)),
        box_nms_thresh=float(args.get("box_nms_thresh", 0.5)),
        box_detections_per_img=int(args.get("detections_per_img", 100)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_mask_rcnn(config: dict, num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    args = extra(config)
    pretrained = bool(args.get("pretrained", True))
    weights = "DEFAULT" if pretrained else None
    weights_backbone = None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=weights_backbone,
        min_size=int(args.get("image_size", 640)),
        max_size=int(args.get("max_size", 1333)),
        trainable_backbone_layers=int(args.get("trainable_backbone_layers", 3)),
        rpn_nms_thresh=float(args.get("rpn_nms_thresh", 0.7)),
        box_score_thresh=float(args.get("box_score_thresh", 0.05)),
        box_nms_thresh=float(args.get("box_nms_thresh", 0.5)),
        box_detections_per_img=int(args.get("detections_per_img", 100)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def train_detection_model(config: dict, model_kind: str, log_path: Path | None, logger) -> dict:
    import torch
    from torch.utils.data import DataLoader

    args = extra(config)
    set_seed(int(args.get("seed", 0)))

    dataset_path = config["dataset_path"]
    data_yaml_path = config.get("data_yaml_path")
    dataset_formats = set(config.get("dataset_metadata", {}).get("formats", []))
    dataset_root = Path(dataset_path)
    has_val_dir = any((dataset_root / name).is_dir() for name in ("valid", "val", "validation"))
    has_val_coco = any(
        path.is_file()
        for name in ("instances_val.json", "instances_valid.json", "instances_validation.json", "val.json", "valid.json", "validation.json")
        for path in dataset_root.rglob(name)
    )
    has_val_yaml = bool(data_yaml_path and read_yaml(data_yaml_path).get("val"))
    has_validation = has_val_dir or has_val_coco or has_val_yaml
    use_coco_boxes = model_kind == "faster_rcnn" and "coco_instances" in dataset_formats and not data_yaml_path

    if model_kind == "mask_rcnn":
        train_dataset = CocoInstanceDataset(dataset_path, "train", include_masks=True)
        if has_validation:
            val_dataset = CocoInstanceDataset(
                dataset_path,
                "val",
                include_masks=True,
                category_to_label=train_dataset.category_to_label,
                classes=train_dataset.classes,
            )
        else:
            val_dataset = None
        model_builder = build_mask_rcnn
    elif use_coco_boxes:
        train_dataset = CocoInstanceDataset(dataset_path, "train", include_masks=False)
        if has_validation:
            val_dataset = CocoInstanceDataset(
                dataset_path,
                "val",
                include_masks=False,
                category_to_label=train_dataset.category_to_label,
                classes=train_dataset.classes,
            )
        else:
            val_dataset = None
        model_builder = build_faster_rcnn
    else:
        train_dataset = YoloBoxDataset(dataset_path, data_yaml_path, "train")
        if has_validation:
            val_dataset = YoloBoxDataset(dataset_path, data_yaml_path, "val")
        else:
            val_dataset = None
        model_builder = build_faster_rcnn

    dataset_warnings = list(getattr(train_dataset, "warnings", []))
    if val_dataset is not None:
        dataset_warnings.extend(getattr(val_dataset, "warnings", []))
    for warning in dataset_warnings[:100]:
        logger(log_path, f"[Dataset warning] {warning}")
    if len(dataset_warnings) > 100:
        logger(log_path, f"[Dataset warning] {len(dataset_warnings) - 100} additional unusable files were skipped.")

    num_classes = _num_classes_from_dataset(train_dataset)
    model = model_builder(config, num_classes)
    device = get_device(str(args.get("device", "0")))
    model.to(device)

    batch_size = require_positive_batch_size(int(config.get("batch_size", args.get("batch_size", 4))), model_kind)
    workers = int(args.get("workers", 4))
    worker_options = {"prefetch_factor": 1} if workers > 0 else {}
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_detection,
        **worker_options,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_detection,
            **worker_options,
        )
        if val_dataset is not None
        else None
    )

    optimizer = optimizer_for(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        str(args.get("optimizer", "sgd")),
        float(args.get("learning_rate", 0.005)),
        float(args.get("momentum", 0.9)),
        float(args.get("weight_decay", 0.0005)),
    )
    epochs = int(config.get("epochs", args.get("epochs", 20)))
    scheduler = scheduler_for(optimizer, str(args.get("scheduler", "cosine")), epochs)
    amp = bool(args.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    results_dir = runs_root() / config.get("project_name", "train_run")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "results.csv"
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(losses.item())
            batches += 1

        avg_train_loss = train_loss / max(batches, 1)
        val_loss = ""
        validation_metrics = _empty_detection_metrics(model_kind == "mask_rcnn")
        if val_loader is not None:
            model.train()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [image.to(device) for image in images]
                    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                    loss_dict = model(images, targets)
                    total_val_loss += float(sum(loss for loss in loss_dict.values()).item())
                    val_batches += 1
            val_loss = total_val_loss / max(val_batches, 1)
            try:
                validation_metrics = _evaluate_detection_metrics(
                    model,
                    val_loader,
                    device,
                    num_classes,
                    model_kind == "mask_rcnn",
                )
            except Exception as exc:
                logger(log_path, f"[{model_kind}] Validation metrics unavailable: {exc}")

        score_loss = float(val_loss) if val_loss != "" else avg_train_loss
        row = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "val/loss": val_loss,
            **validation_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_csv_row(metrics_path, row)
        if scheduler is not None:
            scheduler.step()
        logger(log_path, format_epoch_metrics(model_kind, epoch, epochs, row))

        if score_loss <= best_loss:
            best_loss = score_loss
            torch.save({"model": model.state_dict(), "classes": getattr(train_dataset, "classes", []), "config": config}, results_dir / "best.pt")
        torch.save({"model": model.state_dict(), "classes": getattr(train_dataset, "classes", []), "config": config}, results_dir / "last.pt")

    # Optional one-shot held-out test evaluation on best.pt, run once after the
    # training loop. torchvision detectors only expose losses (not mAP) without a
    # heavy COCO-eval dependency, so test loss is the held-out metric here,
    # mirroring the val-loss the loop already tracks. Skips when no test split.
    try:
        if model_kind == "mask_rcnn":
            test_dataset = CocoInstanceDataset(dataset_path, "test", include_masks=True,
                category_to_label=train_dataset.category_to_label, classes=train_dataset.classes)
        elif use_coco_boxes:
            test_dataset = CocoInstanceDataset(dataset_path, "test", include_masks=False,
                category_to_label=train_dataset.category_to_label, classes=train_dataset.classes)
        else:
            test_dataset = YoloBoxDataset(dataset_path, data_yaml_path, "test")
    except Exception as exc:
        test_dataset = None
        logger(log_path, f"[{model_kind}] No test split found; skipping held-out test evaluation. ({exc})")

    if test_dataset is not None and len(test_dataset) > 0:
        try:
            best_file = results_dir / "best.pt"
            if best_file.is_file():
                model.load_state_dict(torch.load(best_file, map_location=device)["model"])
            test_loader = DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=workers,
                collate_fn=collate_detection, **worker_options,
            )
            model.train()  # torchvision detectors return losses only in train mode
            total_test_loss = 0.0
            test_batches = 0
            with torch.no_grad():
                for images, targets in test_loader:
                    images = [image.to(device) for image in images]
                    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                    loss_dict = model(images, targets)
                    total_test_loss += float(sum(loss for loss in loss_dict.values()).item())
                    test_batches += 1
            test_loss = total_test_loss / max(test_batches, 1)
            result = {
                "task_type": "segmentation" if model_kind == "mask_rcnn" else "object_detection",
                "model_type": model_kind,
                "checkpoint": "best.pt",
                "test_loss": round(test_loss, 6),
                "test_images": len(test_dataset),
                "num_classes": num_classes,
                "classes": list(getattr(train_dataset, "classes", [])),
            }
            (results_dir / "test_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger(log_path, f"[{model_kind}] Held-out test loss={test_loss:.4f} on {len(test_dataset)} images (from best.pt)")
        except Exception as exc:
            logger(log_path, f"[{model_kind}] Held-out test evaluation skipped: {exc}")

    return {
        "status": "completed",
        "task_type": "segmentation" if model_kind == "mask_rcnn" else "object_detection",
        "model_type": model_kind,
        "project_name": config.get("project_name", "train_run"),
        "results_dir": str(results_dir),
        "best_loss": best_loss,
    }
