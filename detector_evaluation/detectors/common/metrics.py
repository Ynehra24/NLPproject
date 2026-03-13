"""Evaluation metrics for detector quality and attack robustness."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve

from .config import EPS, SOURCE_MAP


def encode_source_labels(series) -> np.ndarray:
    if series is None:
        raise ValueError("Cannot encode labels from None")
    values = series.astype(str).str.lower().map(SOURCE_MAP)
    if values.isna().any():
        bad = sorted(series[values.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unknown source labels found: {bad}")
    return values.to_numpy(dtype=int)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.linspace(0.0, 1.0, 401)
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) == 0:
        return 0.0
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = int(np.clip(idx, 0, len(tpr) - 1))
    return float(tpr[idx])


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    if threshold is None:
        threshold = find_best_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    out = {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
        "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_ai": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    out["tpr_at_fpr_1"] = tpr_at_fpr(y_true, y_score, 0.01)
    out["tpr_at_fpr_5"] = tpr_at_fpr(y_true, y_score, 0.05)
    return out


def compute_attack_success_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ai_mask = y_true == 1
    if ai_mask.sum() == 0:
        return 0.0
    flipped_to_human = (y_pred[ai_mask] == 0).sum()
    return float(flipped_to_human / (ai_mask.sum() + EPS))
