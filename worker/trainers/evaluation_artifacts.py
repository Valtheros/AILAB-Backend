from __future__ import annotations

import json
from pathlib import Path


class CurveAccumulator:
    """Streaming PR/F1 counts at fixed confidence thresholds."""

    def __init__(self, steps: int = 101):
        import torch

        self.thresholds = torch.linspace(0, 1, steps)
        self.tp = torch.zeros(steps, dtype=torch.int64)
        self.fp = torch.zeros(steps, dtype=torch.int64)
        self.positives = 0

    def update(self, scores, matches, positive_count: int) -> None:
        import torch

        scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
        matches = torch.as_tensor(matches, dtype=torch.bool).reshape(-1).cpu()
        if scores.numel() != matches.numel():
            raise ValueError("scores and matches must have the same length")
        self.positives += int(positive_count)
        if not scores.numel():
            return
        for start in range(0, scores.numel(), 100_000):
            chunk_scores = scores[start:start + 100_000]
            chunk_matches = matches[start:start + 100_000]
            selected = chunk_scores[:, None] >= self.thresholds[None, :]
            self.tp += (selected & chunk_matches[:, None]).sum(0)
            self.fp += (selected & ~chunk_matches[:, None]).sum(0)

    def as_dict(self, curve_id: str, label: str) -> dict:
        tp = self.tp.float()
        fp = self.fp.float()
        fn = (self.positives - self.tp).clamp(min=0).float()
        precision = tp / (tp + fp).clamp(min=1)
        recall = tp / (tp + fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        values = lambda tensor: [round(float(value), 6) for value in tensor]
        return {
            "id": curve_id,
            "label": label,
            "thresholds": values(self.thresholds),
            "precision": values(precision),
            "recall": values(recall),
            "f1": values(f1),
        }


def confusion_dict(matrix, labels: list[str]) -> dict:
    import torch

    matrix = torch.as_tensor(matrix, dtype=torch.int64).cpu()
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("confusion matrix dimensions must match labels")
    totals = matrix.sum(1, keepdim=True).clamp(min=1)
    normalized = matrix.float() / totals
    return {
        "labels": labels,
        "values": matrix.tolist(),
        "normalized": [[round(float(value), 6) for value in row] for row in normalized],
    }


def segmentation_classes(matrix, labels: list[str]) -> list[dict]:
    import torch

    matrix = torch.as_tensor(matrix, dtype=torch.float32).cpu()
    intersection = matrix.diag()
    target = matrix.sum(1)
    predicted = matrix.sum(0)
    union = target + predicted - intersection
    denominator = target + predicted
    return [
        {
            "label": label,
            "iou": round(float(intersection[index] / union[index]), 6) if union[index] else 0.0,
            "dice": round(float(2 * intersection[index] / denominator[index]), 6) if denominator[index] else 0.0,
            "support": int(target[index]),
        }
        for index, label in enumerate(labels)
    ]


def write_evaluation_artifact(results_dir: Path, payload: dict) -> None:
    path = results_dir / "evaluation_curves.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"version": 1, **payload}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
