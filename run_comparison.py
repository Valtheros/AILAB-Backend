"""Build side-by-side metric series for two or more completed runs.

Everything here reads what the trainers already wrote: `results.csv` (parsed by
TrainingService.get_training_metrics) and `job_config.json`. No metric is
recomputed or invented — the primary metric for each run is resolved from the
columns that actually exist in that run's CSV, so a run that never recorded an
accuracy column falls back to the loss it did record.
"""

from __future__ import annotations

from typing import Any

# Ordered candidates per task type. The first entry whose value column is
# present in the run's results.csv wins, so a DeepLabV3+ run (which records
# pixel accuracy) and a Mask R-CNN run (loss only) both resolve correctly.
METRIC_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "image_classification": [
        {
            "key": "accuracy",
            "label": "Accuracy",
            "direction": "higher",
            "valColumn": "val/accuracy",
            "trainColumn": "train/accuracy",
        },
        {
            "key": "loss",
            "label": "Loss",
            "direction": "lower",
            "valColumn": "val/loss",
            "trainColumn": "train/loss",
        },
    ],
    "object_detection": [
        {
            # Ultralytics records mAP on the validation pass only, so there is
            # no training counterpart to plot for this metric.
            "key": "map50",
            "label": "mAP@50",
            "direction": "higher",
            "valColumn": "metrics/mAP50(B)",
            "trainColumn": None,
        },
        {
            "key": "loss",
            "label": "Box loss",
            "direction": "lower",
            "valColumn": "val/box_loss",
            "trainColumn": "train/box_loss",
        },
        {
            "key": "loss",
            "label": "Loss",
            "direction": "lower",
            "valColumn": "val/loss",
            "trainColumn": "train/loss",
        },
    ],
    "segmentation": [
        {
            "key": "pixel_accuracy",
            "label": "Pixel accuracy",
            "direction": "higher",
            "valColumn": "val/pixel_accuracy",
            "trainColumn": "train/pixel_accuracy",
        },
        {
            "key": "loss",
            "label": "Loss",
            "direction": "lower",
            "valColumn": "val/loss",
            "trainColumn": "train/loss",
        },
    ],
}

# Used when a run's task type is not in the table above.
FALLBACK_CANDIDATES = METRIC_CANDIDATES["image_classification"]

# Subset of extra_args worth showing next to the chart. Everything else stays
# in job_config.json.
HIGHLIGHT_PARAM_KEYS = (
    "learning_rate",
    "image_size",
    "imgsz",
    "optimizer",
    "scheduler",
    "architecture",
    "model_size",
    "amp",
    "device",
)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def resolve_metric(task_type: str, columns: set[str]) -> dict[str, Any] | None:
    """Pick the primary metric this run actually recorded."""
    for candidate in METRIC_CANDIDATES.get(task_type or "", FALLBACK_CANDIDATES):
        if candidate["valColumn"] in columns or (candidate["trainColumn"] or "") in columns:
            return candidate
    return None


def _series_for(rows: list[dict[str, str]], column: str | None) -> list[float | None] | None:
    if not column:
        return None
    if not any(column in row for row in rows):
        return None
    return [_to_float(row.get(column)) for row in rows]


def _best_point(
    epochs: list[float | None],
    values: list[float | None] | None,
    direction: str,
) -> dict[str, Any] | None:
    if not values:
        return None
    points = [(value, epochs[index]) for index, value in enumerate(values) if value is not None]
    if not points:
        return None
    chooser = max if direction == "higher" else min
    value, epoch = chooser(points, key=lambda item: item[0])
    return {"value": value, "epoch": epoch}


def _final_point(values: list[float | None] | None) -> dict[str, Any] | None:
    if not values:
        return None
    for value in reversed(values):
        if value is not None:
            return {"value": value}
    return None


def build_run_comparison(
    *,
    task_row: dict[str, Any],
    metric_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one run's entry for the comparison response.

    `task_row` is the training_tasks record, `metric_rows` is the parsed
    results.csv, and `config` is job_config.json.
    """
    extra_args = dict(config.get("extra_args") or {})
    params = dict(task_row.get("params") or {})
    task_type = str(task_row.get("task_type") or config.get("task_type") or "")

    columns: set[str] = set()
    for row in metric_rows:
        columns.update(row.keys())

    metric = resolve_metric(task_type, columns)
    epochs = [_to_float(row.get("epoch")) for row in metric_rows]

    train_series = _series_for(metric_rows, metric["trainColumn"]) if metric else None
    val_series = _series_for(metric_rows, metric["valColumn"]) if metric else None
    direction = metric["direction"] if metric else "higher"

    # Prefer the validation curve for best/final; fall back to training when a
    # run has no validation split.
    scored = val_series if val_series and any(v is not None for v in val_series) else train_series

    architecture = str(
        extra_args.get("architecture")
        or task_row.get("model_name")
        or config.get("model_name")
        or ""
    )

    highlights = {
        key: extra_args[key] for key in HIGHLIGHT_PARAM_KEYS if key in extra_args
    }

    return {
        "runId": str(task_row.get("id")),
        "runSlug": task_row.get("run_slug"),
        "displayName": task_row.get("display_name") or task_row.get("run_slug"),
        "taskType": task_type,
        "modelType": task_row.get("model_type") or config.get("model_type"),
        "architecture": architecture,
        "datasetSlug": task_row.get("dataset_slug") or config.get("dataset_name"),
        "metric": metric and {
            "key": metric["key"],
            "label": metric["label"],
            "direction": metric["direction"],
            "trainColumn": metric["trainColumn"],
            "valColumn": metric["valColumn"],
        },
        "series": {
            "epoch": epochs,
            "trainMetric": train_series,
            "valMetric": val_series,
        },
        "bestMetric": _best_point(epochs, scored, direction),
        "finalMetric": _final_point(scored),
        "epochsRecorded": len(metric_rows),
        "trainingParams": {
            "epochs": params.get("epochs", config.get("epochs")),
            "batchSize": params.get("batch_size", config.get("batch_size")),
            "learningRate": extra_args.get("learning_rate"),
            **highlights,
        },
    }


def summarise_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Top-level facts the UI needs to render an honest chart."""
    metric_keys = {run["metric"]["key"] for run in runs if run.get("metric")}
    labels = {run["metric"]["label"] for run in runs if run.get("metric")}
    task_types = {run["taskType"] for run in runs if run.get("taskType")}
    shared = len(metric_keys) == 1 and len(labels) == 1

    warnings: list[str] = []
    if not metric_keys:
        warnings.append("None of the selected runs recorded a comparable metric.")
    elif not shared:
        warnings.append(
            "The selected runs recorded different metrics "
            f"({', '.join(sorted(labels))}), so the curves are not directly comparable."
        )
    if len(task_types) > 1:
        warnings.append(
            f"The selected runs have different task types ({', '.join(sorted(task_types))})."
        )

    first = next((run["metric"] for run in runs if run.get("metric")), None)
    return {
        "sharedMetric": {
            "key": first["key"],
            "label": first["label"],
            "direction": first["direction"],
        }
        if shared and first
        else None,
        "warnings": warnings,
    }
