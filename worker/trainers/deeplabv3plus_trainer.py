from __future__ import annotations

import json
from pathlib import Path

from .base_trainer import BaseTrainer
from .semantic_dataset import SemanticMaskDataset
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


def _evaluate_semantic_test_split(
    model, dataset_path, image_size, num_classes, ignore_index, batch_size, results_dir, device, logger, log_path
):
    """One-shot held-out test evaluation, run once AFTER training (not in the
    loop). Reports pixel accuracy and mean IoU; skips cleanly when the dataset
    has no test split. Writes the same test_evaluation.json contract the UI reads."""
    import torch
    from torch.utils.data import DataLoader

    try:
        test_dataset = SemanticMaskDataset(dataset_path, "test", image_size, num_classes, ignore_index)
    except Exception as exc:
        logger(log_path, f"[deeplabv3plus] No test split found; skipping held-out test evaluation. ({exc})")
        return None

    loader = DataLoader(test_dataset, batch_size=max(1, min(batch_size, 8)), shuffle=False, num_workers=0)
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = model(images).argmax(dim=1)
            valid = masks != ignore_index
            indices = masks[valid] * num_classes + predictions[valid]
            confusion += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    confusion = confusion.cpu()
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    pixel_accuracy = correct / max(total, 1)
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    mean_iou = float((intersection[present].float() / union[present].float()).mean().item()) if bool(present.any()) else 0.0

    result = {
        "task_type": "segmentation",
        "model_type": "deeplabv3plus",
        "checkpoint": "best.pt",
        "test_pixel_accuracy": round(pixel_accuracy, 6),
        "test_mean_iou": round(mean_iou, 6),
        "test_images": len(test_dataset),
        "num_classes": num_classes,
    }
    (results_dir / "test_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger(
        log_path,
        f"[deeplabv3plus] Held-out test pixel_acc={pixel_accuracy:.4f} mIoU={mean_iou:.4f} "
        f"on {len(test_dataset)} images (from best.pt)",
    )
    return result


class DeepLabV3PlusTrainer(BaseTrainer):
    def validate_config(self, config: dict) -> None:
        if not config.get("dataset_path"):
            raise ValueError("DeepLabV3PlusTrainer requires dataset_path")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "segmentation-models-pytorch is required for DeepLabV3+. "
                "Rebuild the worker image after installing worker requirements."
            ) from exc

        self.validate_config(config)
        args = extra(config)
        set_seed(int(args.get("seed", 0)))

        image_size = int(args.get("image_size", 512))
        num_classes = int(args.get("num_classes", 2))
        ignore_index = int(args.get("ignore_index", 255))
        train_dataset = SemanticMaskDataset(config["dataset_path"], "train", image_size, num_classes, ignore_index)
        has_val_split = any((Path(config["dataset_path"]) / name).is_dir() for name in ("valid", "val", "validation"))
        if has_val_split:
            val_dataset = SemanticMaskDataset(config["dataset_path"], "val", image_size, num_classes, ignore_index)
        else:
            val_dataset = None
        for warning in train_dataset.warnings + (val_dataset.warnings if val_dataset is not None else []):
            self._write_log(log_path, f"[Dataset warning] {warning}")

        encoder_weights = args.get("encoder_weights", "imagenet")
        if encoder_weights == "none":
            encoder_weights = None

        atrous_rates = tuple(
            int(value.strip())
            for value in str(args.get("decoder_atrous_rates", "12,24,36")).split(",")
            if value.strip()
        )
        model = smp.DeepLabV3Plus(
            encoder_name=str(args.get("encoder_name", "resnet34")),
            encoder_depth=int(args.get("encoder_depth", 5)),
            encoder_weights=encoder_weights,
            encoder_output_stride=int(args.get("encoder_output_stride", 16)),
            decoder_channels=int(args.get("decoder_channels", 256)),
            decoder_atrous_rates=atrous_rates or (12, 24, 36),
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        device = get_device(str(args.get("device", "0")))
        model.to(device)
        batch_size = require_positive_batch_size(int(config.get("batch_size", args.get("batch_size", 8))), "deeplabv3plus")
        workers = int(args.get("workers", 4))
        worker_options = {"prefetch_factor": 1} if workers > 0 else {}
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, **worker_options
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, **worker_options)
            if val_dataset is not None
            else None
        )

        if str(args.get("loss", "cross_entropy")) == "dice":
            from segmentation_models_pytorch.losses import DiceLoss

            criterion = DiceLoss(mode="multiclass", ignore_index=ignore_index)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        optimizer = optimizer_for(
            model.parameters(),
            str(args.get("optimizer", "adamw")),
            float(args.get("learning_rate", 0.001)),
            float(args.get("momentum", 0.9)),
            float(args.get("weight_decay", 0.0001)),
        )
        epochs = int(config.get("epochs", args.get("epochs", 50)))
        scheduler = scheduler_for(optimizer, str(args.get("scheduler", "cosine")), epochs)
        amp = bool(args.get("amp", True)) and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        results_dir = runs_root() / config.get("project_name", "train_run")
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = results_dir / "results.csv"
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_pixels = 0
            correct_pixels = 0
            batches = 0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item())
                predictions = logits.argmax(dim=1)
                valid_pixels = masks != ignore_index
                correct_pixels += int(((predictions == masks) & valid_pixels).sum().item())
                total_pixels += int(valid_pixels.sum().item())
                batches += 1

            val_loss = ""
            val_pixel_accuracy = ""
            if val_loader is not None:
                model.eval()
                total_val_loss = 0.0
                val_batches = 0
                val_correct = 0
                val_pixels = 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(device)
                        masks = masks.to(device)
                        logits = model(images)
                        loss = criterion(logits, masks)
                        total_val_loss += float(loss.item())
                        valid_pixels = masks != ignore_index
                        val_correct += int(((logits.argmax(dim=1) == masks) & valid_pixels).sum().item())
                        val_pixels += int(valid_pixels.sum().item())
                        val_batches += 1
                val_loss = total_val_loss / max(val_batches, 1)
                val_pixel_accuracy = val_correct / max(val_pixels, 1)

            train_loss = total_loss / max(batches, 1)
            train_pixel_accuracy = correct_pixels / max(total_pixels, 1)
            score_loss = float(val_loss) if val_loss != "" else train_loss
            append_csv_row(
                metrics_path,
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/pixel_accuracy": train_pixel_accuracy,
                    "val/loss": val_loss,
                    "val/pixel_accuracy": val_pixel_accuracy,
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )
            if scheduler is not None:
                scheduler.step()
            self._write_log(
                log_path,
                f"[deeplabv3plus] epoch={epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss}",
            )

            if score_loss <= best_loss:
                best_loss = score_loss
                torch.save({"model": model.state_dict(), "config": config}, results_dir / "best.pt")
            torch.save({"model": model.state_dict(), "config": config}, results_dir / "last.pt")

        # Optional one-shot held-out test evaluation on best.pt, after the loop.
        best_path = results_dir / "best.pt"
        if best_path.is_file():
            try:
                model.load_state_dict(torch.load(best_path, map_location=device)["model"])
                _evaluate_semantic_test_split(
                    model, config["dataset_path"], image_size, num_classes, ignore_index,
                    batch_size, results_dir, device, self._write_log, log_path,
                )
            except Exception as exc:
                self._write_log(log_path, f"[deeplabv3plus] Held-out test evaluation skipped: {exc}")

        return {
            "status": "completed",
            "task_type": "segmentation",
            "model_type": "deeplabv3plus",
            "project_name": config.get("project_name", "train_run"),
            "results_dir": str(results_dir),
            "best_loss": best_loss,
        }
