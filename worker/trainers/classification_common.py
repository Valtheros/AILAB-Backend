from __future__ import annotations

from pathlib import Path

from .trainer_utils import (
    append_csv_row,
    extra,
    get_device,
    optimizer_for,
    require_positive_batch_size,
    runs_root,
    scheduler_for,
    set_seed,
)


def _weights_for(torchvision_models, architecture: str, pretrained: bool):
    if not pretrained:
        return None
    weight_name = f"{architecture}_Weights"
    enum_name = "".join(part.capitalize() for part in weight_name.split("_"))
    weights_enum = getattr(torchvision_models, enum_name, None)
    if weights_enum is None:
        return "DEFAULT"
    return weights_enum.DEFAULT


def _build_resnet(architecture: str, num_classes: int, pretrained: bool, dropout: float):
    import torch.nn as nn
    import torchvision.models as models

    builder = getattr(models, architecture)
    model = builder(weights=_weights_for(models, architecture, pretrained))
    if pretrained:
        in_features = model.fc.in_features
    else:
        in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def _build_efficientnet(architecture: str, num_classes: int, pretrained: bool, dropout: float):
    import torch.nn as nn
    import torchvision.models as models

    builder = getattr(models, architecture)
    model = builder(weights=_weights_for(models, architecture, pretrained))
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if len(model.classifier) > 1 and hasattr(model.classifier[0], "p"):
        model.classifier[0].p = dropout
    return model


def _batch_size_for(config: dict, args: dict, family: str) -> int:
    return require_positive_batch_size(int(config.get("batch_size", args.get("batch_size", 16))), family)


def train_classifier(config: dict, family: str, log_path: Path | None, logger) -> dict:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    args = extra(config)
    seed = int(args.get("seed", 0))
    set_seed(seed)

    dataset_path = Path(config["dataset_path"])
    train_dir = dataset_path / "train"
    if not train_dir.exists():
        train_dir = dataset_path / "training"
    val_dir = dataset_path / "valid"
    if not val_dir.exists():
        val_dir = dataset_path / "val"

    if not train_dir.exists():
        raise ValueError("Image classification requires train/<class_name> image folders.")

    image_size = int(args.get("image_size", 224))
    color_jitter = float(args.get("color_jitter", 0.1))
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(float(args.get("horizontal_flip", 0.5))),
            transforms.RandomRotation(float(args.get("random_rotation", 0))),
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(color_jitter / 2, 0.5),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform) if val_dir.exists() else None
    num_classes = len(train_dataset.classes)
    if num_classes < 2:
        raise ValueError("Image classification requires at least two class folders.")

    architecture = str(args.get("architecture") or config.get("model_name"))
    pretrained = bool(args.get("pretrained", True))
    dropout = float(args.get("dropout", 0.2))
    if family == "resnet":
        model = _build_resnet(architecture, num_classes, pretrained, dropout)
    else:
        model = _build_efficientnet(architecture, num_classes, pretrained, dropout)

    if bool(args.get("freeze_backbone", False)):
        for name, parameter in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                parameter.requires_grad = False

    device = get_device(str(args.get("device", "0")))
    model.to(device)

    batch_size = _batch_size_for(config, args, family)
    workers = int(args.get("workers", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        if val_dataset is not None
        else None
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.get("label_smoothing", 0.0)))
    optimizer = optimizer_for(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        str(args.get("optimizer", "sgd")),
        float(args.get("learning_rate", 0.001)),
        float(args.get("momentum", 0.9)),
        float(args.get("weight_decay", 0.0005)),
    )
    epochs = int(config.get("epochs", args.get("epochs", 50)))
    scheduler = scheduler_for(optimizer, str(args.get("scheduler", "cosine")), epochs)
    amp = bool(args.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    results_dir = runs_root() / config.get("project_name", "train_run")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "results.csv"
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * labels.size(0)
            train_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            train_total += labels.size(0)

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += float(loss.item()) * labels.size(0)
                    val_correct += int((outputs.argmax(dim=1) == labels).sum().item())
                    val_total += labels.size(0)
        if scheduler is not None:
            scheduler.step()

        train_accuracy = train_correct / max(train_total, 1)
        val_accuracy = val_correct / max(val_total, 1) if val_total else train_accuracy
        row = {
            "epoch": epoch,
            "train/loss": train_loss / max(train_total, 1),
            "train/accuracy": train_accuracy,
            "val/loss": val_loss / max(val_total, 1) if val_total else "",
            "val/accuracy": val_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_csv_row(metrics_path, row)
        logger(log_path, f"[{family}] epoch={epoch}/{epochs} train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}")

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model": model.state_dict(), "classes": train_dataset.classes, "config": config}, results_dir / "best.pt")
        torch.save({"model": model.state_dict(), "classes": train_dataset.classes, "config": config}, results_dir / "last.pt")

    return {
        "status": "completed",
        "task_type": "image_classification",
        "model_type": family,
        "project_name": config.get("project_name", "train_run"),
        "results_dir": str(results_dir),
        "best_accuracy": best_accuracy,
    }
