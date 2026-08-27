"""Rule-based, plain-language insights over metrics a run already recorded.

Pure if/else comparisons on numbers already in results.csv and
test_evaluation.json. No model, no external call, no new metric. The functions
return structured items ``{"level", "code", "params"}`` so the frontend owns
the bilingual wording via i18n; the backend never emits display strings.

Reuses ``resolve_metric`` from run_comparison so the accuracy-like column is
found the same way in both places.
"""

from __future__ import annotations

from typing import Any

from run_comparison import resolve_metric

# Thresholds (percentage points unless noted). Kept here so they are easy to tune.
OVERFIT_HIGH = 15.0
OVERFIT_MILD = 8.0
UNSTABLE_STD = 5.0          # std of last-5 validation accuracy, in percent
TEST_GAP_HIGH = 10.0


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _series(rows: list[dict[str, str]], column: str | None) -> list[float | None]:
    if not column:
        return []
    return [_f(row.get(column)) for row in rows]


def _last_valid(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def _std_percent(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    mean = sum(present) / len(present)
    variance = sum((v - mean) ** 2 for v in present) / len(present)
    return (variance ** 0.5) * 100.0


def analyse_run(rows: list[dict[str, str]], task_type: str, test_eval: dict | None = None) -> dict[str, Any]:
    """Insights for one run. Returns ``{"metric", "insights": [...]}``.

    ``insights`` items are ``{"level": danger|warning|success|info, "code", "params"}``.
    """
    insights: list[dict[str, Any]] = []
    if not rows:
        return {"metric": None, "insights": insights}

    columns: set[str] = set()
    for row in rows:
        columns.update(row.keys())
    metric = resolve_metric(task_type or "", columns)
    epochs = _series(rows, "epoch")

    val_series = _series(rows, metric["valColumn"]) if metric else []
    train_series = _series(rows, metric["trainColumn"]) if metric else []
    higher_is_better = bool(metric and metric["direction"] == "higher")

    # 1. Best epoch (by the primary metric). Reported as a percentage only for
    #    accuracy-like (higher-is-better) metrics; loss gets a value variant.
    scored = val_series if any(v is not None for v in val_series) else train_series
    if metric and any(v is not None for v in scored):
        points = [(v, epochs[i]) for i, v in enumerate(scored) if v is not None]
        chooser = max if higher_is_better else min
        best_value, best_epoch = chooser(points, key=lambda item: item[0])
        insights.append({
            "level": "info",
            "code": "best_epoch_acc" if higher_is_better else "best_epoch_loss",
            "params": {
                "epoch": int(best_epoch) if best_epoch is not None else None,
                "value": round(best_value * 100, 2) if higher_is_better else round(best_value, 4),
                "metric": metric["label"],
            },
        })

    # 2. Overfitting: final train vs final val, only for accuracy-like metrics
    #    that record BOTH a train and a val column (e.g. classification,
    #    DeepLabV3+). Detection mAP (val-only) is skipped honestly.
    final_train = _last_valid(train_series) if higher_is_better else None
    final_val = _last_valid(val_series) if higher_is_better else None
    if higher_is_better and final_train is not None and final_val is not None:
        gap = (final_train - final_val) * 100.0
        if gap > OVERFIT_HIGH:
            insights.append({"level": "danger", "code": "overfitting.high", "params": {"gap": round(gap, 1)}})
        elif gap >= OVERFIT_MILD:
            insights.append({"level": "warning", "code": "overfitting.mild", "params": {"gap": round(gap, 1)}})
        else:
            insights.append({"level": "success", "code": "overfitting.none", "params": {"gap": round(gap, 1)}})

    # 4. Validation instability over the last 5 epochs.
    std5 = _std_percent(val_series[-5:]) if higher_is_better else None
    if std5 is not None and std5 > UNSTABLE_STD:
        insights.append({"level": "warning", "code": "val_unstable", "params": {"std": round(std5, 1)}})

    # 5. Held-out test vs best validation, only when a test evaluation exists.
    #    Emit both real numbers (not just the gap) and name it "best validation"
    #    with its epoch: the Latest Metrics box shows the *final* epoch's val,
    #    while this gap is measured against the best epoch used to pick the
    #    checkpoint, so the two would otherwise look mismatched to the user.
    test_acc = _f((test_eval or {}).get("test_accuracy"))
    best_val = None
    best_val_epoch = None
    if higher_is_better:
        for i, v in enumerate(val_series):
            if v is not None and (best_val is None or v > best_val):
                best_val, best_val_epoch = v, (epochs[i] if i < len(epochs) else None)
    if test_acc is not None and best_val is not None:
        gap_test = (best_val - test_acc) * 100.0
        test_params = {
            "gap": round(abs(gap_test), 1),
            "test": round(test_acc * 100, 2),
            "best_val": round(best_val * 100, 2),
            "best_epoch": int(best_val_epoch) if best_val_epoch is not None else None,
        }
        if gap_test > TEST_GAP_HIGH:
            insights.append({"level": "warning", "code": "test_gap.high", "params": test_params})
        else:
            insights.append({"level": "success", "code": "test_gap.ok", "params": test_params})

    return {"metric": metric and {"key": metric["key"], "label": metric["label"], "direction": metric["direction"]}, "insights": insights}


def _run_stats(run: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the numbers needed for cross-run comparison from a compare-run entry."""
    metric = run.get("metric")
    if not metric:
        return None
    higher = metric["direction"] == "higher"
    epochs = run.get("series", {}).get("epoch") or []
    val = run.get("series", {}).get("valMetric")
    train = run.get("series", {}).get("trainMetric")
    val = val if val and any(v is not None for v in val) else train
    if not val:
        return None
    best = run.get("bestMetric") or {}
    best_value = best.get("value")
    if best_value is None:
        return None

    # Final overfit gap (accuracy-like, both curves present).
    gap = None
    if higher and train and any(t is not None for t in train):
        ft = _last_valid(train)
        fv = _last_valid(val)
        if ft is not None and fv is not None:
            gap = (ft - fv) * 100.0

    epochs_used = run.get("epochsRecorded") or (int(max((e for e in epochs if e is not None), default=0)) or None)
    return {
        "name": run.get("displayName") or run.get("runSlug"),
        "higher": higher,
        "best_value": best_value,
        "gap": gap,
        "epochs": epochs_used,
    }


def compare_insights(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-run insights for the compare view. Only fires when comparable."""
    stats = [s for s in (_run_stats(run) for run in runs) if s]
    insights: list[dict[str, Any]] = []
    if len(stats) < 2:
        return insights
    # Only compare runs that share metric direction (all accuracy or all loss).
    directions = {s["higher"] for s in stats}
    if len(directions) != 1:
        return insights
    higher = stats[0]["higher"]

    # Overfit most / least (needs the train-vs-val gap for at least two runs).
    with_gap = [s for s in stats if s["gap"] is not None]
    if len(with_gap) >= 2:
        most = max(with_gap, key=lambda s: s["gap"])
        least = min(with_gap, key=lambda s: s["gap"])
        if most["name"] != least["name"]:
            insights.append({"level": "warning", "code": "compare.overfit_most", "params": {"run": most["name"], "gap": round(most["gap"], 1)}})
            # "Least overfit in this group" can still exceed the single-run red
            # threshold; when it does, say so against the SAME threshold rather
            # than flashing a green all-clear that misreads as "this one is fine".
            if least["gap"] > OVERFIT_HIGH:
                insights.append({"level": "warning", "code": "compare.overfit_least_high",
                                 "params": {"run": least["name"], "gap": round(least["gap"], 1), "threshold": round(OVERFIT_HIGH)}})
            else:
                insights.append({"level": "success", "code": "compare.overfit_least", "params": {"run": least["name"], "gap": round(least["gap"], 1)}})

    # Best headline value, and the most efficient (best value per epoch).
    best = (max if higher else min)(stats, key=lambda s: s["best_value"])
    insights.append({
        "level": "success",
        "code": "compare.best_value",
        "params": {"run": best["name"], "value": round(best["best_value"] * 100, 2) if higher else round(best["best_value"], 4)},
    })
    if higher:
        efficient = max((s for s in stats if s["epochs"]), key=lambda s: s["best_value"] / s["epochs"], default=None)
        if efficient and efficient["name"] != best["name"]:
            insights.append({
                "level": "info",
                "code": "compare.efficient",
                "params": {"run": efficient["name"], "value": round(efficient["best_value"] * 100, 2), "epochs": efficient["epochs"]},
            })
    return insights
