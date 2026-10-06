from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import gammaincc

from .dataset import EMGPNError, _real_array, _labels

__all__ = ["perform_measure", "midrank", "group_test"]


def midrank(value: np.ndarray) -> np.ndarray:

    value = np.asarray(value, dtype=float).ravel()
    if not np.all(np.isfinite(value)):
        raise EMGPNError("midrank requires finite values.")
    n = value.size
    if n == 0:
        return np.empty(0)
    order = np.argsort(value, kind="stable")
    sorted_v = value[order]
    is_start = np.concatenate([[True], sorted_v[1:] != sorted_v[:-1]])
    start = np.flatnonzero(is_start)
    stop = np.concatenate([start[1:] - 1, [n - 1]])
    group = np.cumsum(is_start) - 1
    rank = np.empty(n)
    rank[order] = (start[group] + stop[group]) / 2.0 + 1.0
    return rank


def _binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = label > 0
    num_pos = int(label.sum())
    num_neg = label.size - num_pos
    if num_pos == 0 or num_neg == 0:
        return float("nan")
    rank = midrank(score)
    return float((rank[label].sum() - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def _average_precision(score: np.ndarray, label: np.ndarray) -> float:
    label = label.ravel() > 0
    num_pos = int(label.sum())
    if num_pos == 0:
        return float("nan")
    order = np.argsort(-score.ravel(), kind="stable")
    sorted_s = score.ravel()[order]
    hit = label[order]
    tp = np.cumsum(hit)
    fp = np.cumsum(~hit)
    last = np.concatenate([sorted_s[:-1] != sorted_s[1:], [True]])
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / num_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape)
    ok = b > 0
    out[ok] = a[ok] / b[ok]
    return out


def perform_measure(P: np.ndarray, y: np.ndarray, decision: str = "argmax",
                    threshold: Optional[float] = None) -> Dict[str, object]:


    P = _real_array(P, "P")
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] == 0:
        raise EMGPNError("P must be a nonempty class-by-participant matrix with at least two classes.")
    c, n = P.shape
    y = _labels(np.asarray(y).ravel(), n, c)
    if not np.all(np.isfinite(P)):
        raise EMGPNError("P must be finite.")
    Y = np.zeros((c, n))
    Y[y, np.arange(n)] = 1.0
    present = Y.sum(axis=1) > 0
    if threshold is None:
        threshold = 1.0 / c
    if np.any(P < 0) or np.any(P > 1):
        raise EMGPNError("P must contain probabilities in [0, 1].")
    if not np.isscalar(threshold) or not np.isreal(threshold) or not np.isfinite(threshold) \
            or not 0 <= threshold <= 1:
        raise EMGPNError("threshold must lie in [0, 1].")
    perf: Dict[str, object] = {}
    class_auroc = np.array([_binary_auroc(P[k], Y[k]) for k in range(c)])
    class_auprc = np.array([_average_precision(P[k], Y[k]) for k in range(c)])
    valid = ~np.isnan(class_auroc)
    perf["ClassAUROC"], perf["ClassAUPRC"] = class_auroc, class_auprc
    perf["AUROC"] = float(class_auroc[valid].mean()) if valid.any() else float("nan")
    perf["AUPRC"] = float(class_auprc[valid].mean()) if valid.any() else float("nan")
    perf["MicroAUROC"] = _binary_auroc(P.ravel(order="F"), Y.ravel(order="F"))
    perf["MicroAUPRC"] = _average_precision(P.ravel(order="F"), Y.ravel(order="F"))
    yhat = np.argmax(P, axis=0)
    perf["Accuracy"] = float(np.mean(yhat == y))
    confusion = np.zeros((c, c))
    np.add.at(confusion, (y, yhat), 1.0)
    perf["Confusion"] = confusion
    if decision == "argmax":
        tp = np.diag(confusion).copy()
        fp = confusion.sum(axis=0) - tp
        fn = confusion.sum(axis=1) - tp
    elif decision == "ovr":
        d = P >= threshold
        tp = np.sum(d & (Y == 1), axis=1).astype(float)
        fp = np.sum(d & (Y == 0), axis=1).astype(float)
        fn = np.sum(~d & (Y == 1), axis=1).astype(float)
    else:
        raise EMGPNError("decision must be 'argmax' or 'ovr'.")
    perf["ClassPrecision"] = _safe_divide(tp, tp + fp)
    perf["ClassRecall"] = _safe_divide(tp, tp + fn)
    perf["ClassF1"] = _safe_divide(2 * tp, 2 * tp + fp + fn)
    label = present | ((tp + fp) > 0)
    perf["Precision"] = float(perf["ClassPrecision"][label].mean())
    perf["Recall"] = float(perf["ClassRecall"][label].mean())
    perf["F1"] = float(perf["ClassF1"][label].mean())
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    perf["BalancedAccuracy"] = float(recall[present].mean())
    perf["LogLoss"] = float(-np.mean(np.log(np.maximum(P[y, np.arange(n)], np.finfo(float).tiny))))
    return perf


def group_test(z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    z = np.atleast_2d(np.asarray(z, dtype=float))
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise EMGPNError("z must be a finite row-by-participant matrix.")
    num_row, n = z.shape
    y = np.asarray(y).ravel()
    if y.size != n:
        raise EMGPNError(f"y must have {n} elements.")
    _, group = np.unique(y, return_inverse=True)
    num_group = int(group.max()) + 1 if n else 0
    count = np.bincount(group, minlength=num_group).astype(float)
    pvalue = np.full(num_row, np.nan)
    stat = np.full(num_row, np.nan)
    if num_group < 2 or n < 3:
        return pvalue, stat
    for q in range(num_row):
        rank = midrank(z[q])
        sum_rank = np.bincount(group, weights=rank, minlength=num_group)
        h = 12.0 / (n * (n + 1)) * np.sum(sum_rank ** 2 / count) - 3.0 * (n + 1)
        srt = np.sort(z[q])
        start = np.flatnonzero(np.concatenate([[True], srt[1:] != srt[:-1]]))
        tie = np.diff(np.concatenate([start, [n]])).astype(float)
        correction = 1.0 - np.sum(tie ** 3 - tie) / (n ** 3 - n)
        if correction > 0:
            h = max(h / correction, 0.0)
            pvalue[q] = gammaincc((num_group - 1) / 2.0, h / 2.0)
        else:
            h, pvalue[q] = 0.0, 1.0
        stat[q] = h
    return pvalue, stat
