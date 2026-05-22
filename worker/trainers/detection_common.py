from __future__ import annotations

from pathlib import Path

from .detection_datasets import CocoInstanceDataset, YoloBoxDataset
from .trainer_utils import append_csv_row, collate_detection, extra, get_device, optimizer_for, set_seed


def _num_classes_from_dataset(dataset, fallback: int = 2) -> int:
    classes = getattr(dataset, "classes", None) or []
    return max(len(classes) + 1, fallback)


def build_faster_rcnn(config: dict, num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    args = extra(config)
    pretrained = bool(args.get("pretrained", True))
    weights = "DEFAULT" if pretrained else None
    weights_backbone = None if pretrained else "DEFAULT"
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
    weights_backbone = None if pretrained else "DEFAULT"
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
    if model_kind == "mask_rcnn":
        train_dataset = CocoInstanceDataset(dataset_path, "train")
        try:
            val_dataset = CocoInstanceDataset(dataset_path, "val")
        except Exception:
            val_dataset = None
        model_builder = build_mask_rcnn
    else:
        train_dataset = YoloBoxDataset(dataset_path, data_yaml_path, "train")
        try:
            val_dataset = YoloBoxDataset(dataset_path, data_yaml_path, "val")
        except Exception:
            val_dataset = None
        model_builder = build_faster_rcnn

    num_classes = _num_classes_from_dataset(train_dataset)
    model = model_builder(config, num_classes)
    device = get_device(str(args.get("device", "0")))
    model.to(device)

    batch_size = int(config.get("batch_size", args.get("batch_size", 4)))
    workers = int(args.get("workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_detection,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate_detection)
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
    amp = bool(args.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    results_dir = Path("/app/runs") / config.get("project_name", "train_run")
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

        score_loss = float(val_loss) if val_loss != "" else avg_train_loss
        row = {"epoch": epoch, "train/loss": avg_train_loss, "val/loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        append_csv_row(metrics_path, row)
        logger(log_path, f"[{model_kind}] epoch={epoch}/{epochs} train_loss={avg_train_loss:.4f} val_loss={val_loss}")

        if score_loss <= best_loss:
            best_loss = score_loss
            torch.save({"model": model.state_dict(), "classes": getattr(train_dataset, "classes", []), "config": config}, results_dir / "best.pt")
        torch.save({"model": model.state_dict(), "classes": getattr(train_dataset, "classes", []), "config": config}, results_dir / "last.pt")

    return {
        "status": "completed",
        "task_type": "segmentation" if model_kind == "mask_rcnn" else "object_detection",
        "model_type": model_kind,
        "project_name": config.get("project_name", "train_run"),
        "results_dir": str(results_dir),
        "best_loss": best_loss,
    }
