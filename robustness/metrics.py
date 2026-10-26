"""Метрики бинарного классификатора с перцентильным бутстрапом."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def rates(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    """Точка на одной выборке пар (expected, pred)."""
    tp = sum(1 for expected, pred in pairs if expected and pred)
    fp = sum(1 for expected, pred in pairs if not expected and pred)
    tn = sum(1 for expected, pred in pairs if not expected and not pred)
    fn = sum(1 for expected, pred in pairs if expected and not pred)
    return {
        "accuracy": _divide(tp + tn, len(pairs)),
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
        "fpr": _divide(fp, fp + tn),
    }


def metrics(expected: list[bool], pred: list[bool | None], n_boot: int, seed: int) -> dict:
    """Точка + перцентильный 95% CI; сбои (pred=None) исключаются и считаются в failed."""
    pairs = [(e, p) for e, p in zip(expected, pred) if p is not None]
    if not pairs:
        return {"n": 0, "failed": len(expected), "value": {}, "ci95": None}

    rng = random.Random(seed)
    with warnings.catch_warnings():  # nanpercentile ругается на метрики с нулевым знаменателем
        warnings.simplefilter("ignore", RuntimeWarning)
        boot = [rates([rng.choice(pairs) for _ in pairs]) for _ in range(n_boot)]
        ci95 = {
            name: [
                float(np.nanpercentile([sample[name] for sample in boot], 2.5)),
                float(np.nanpercentile([sample[name] for sample in boot], 97.5)),
            ]
            for name in rates(pairs)
        }
    return {
        "n": len(pairs),
        "failed": len(expected) - len(pairs),
        "value": rates(pairs),
        "ci95": ci95,
    }


def write_json(path: Path, report: dict, indent: int = 2) -> None:
    """JSON-отчёт с nan -> null."""

    def _clean(value):
        if isinstance(value, dict):
            return {key: _clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_clean(item) for item in value]
        return None if isinstance(value, float) and np.isnan(value) else value

    path.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=indent))
