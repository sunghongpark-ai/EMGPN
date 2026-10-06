from __future__ import annotations

from typing import Dict, List, Optional
from numbers import Real

import numpy as np

from .dataset import EMGPNError, _labels
from .metrics import group_test, perform_measure
from .model import Model, classifier_weight, logit_softmax, model_forward, param_reshape
from .rng import random_order, random_stream, uniform

__all__ = ["model_explain", "explain_summary", "key_region", "subject_explain"]

PERMUTATION_SEED_OFFSET = 7919


def _perm_score(s: np.ndarray, y: np.ndarray, metric: str) -> float:
    if metric == "loss":
        shift = s - s.max(axis=0, keepdims=True)
        logp = shift - np.log(np.exp(shift).sum(axis=0, keepdims=True))
        return float(-np.mean(logp[y, np.arange(y.size)]))
    return float(perform_measure(logit_softmax(s).p, y)["AUROC"])


def model_explain(model: Model, x: List[np.ndarray], y: np.ndarray, group: bool = True) -> Dict[str, object]:


    p = model.params
    if p.integration == "concat":
        raise EMGPNError("model_explain requires ROI-level integrated features (integration 'softmax' or 'mean').")
    model = param_reshape(model)
    out = model_forward(model, x)
    z, s, prob = out.z, out.s, out.p
    r, n = z.shape
    c = model.num_class
    y = _labels(np.asarray(y).ravel(), n, c)
    theta, zbar = model.theta, model.z_mean
    part: Dict[str, object] = {
        "ModalityName": list(model.design.modality_name), "ClassName": list(model.design.class_name),
        "RoiName": list(model.design.roi_name), "ModalityImportance": model.lambda_prob.copy(),
        "Steadiness": model.phi.copy() if p.propagation == "gpn" else None, "RiskEffect": theta.copy(),
    }
    part["FeatureImportance"] = np.mean(np.abs(theta), axis=1) * np.mean(np.abs(z - zbar[:, None]), axis=1)

    stream = random_stream((model.rand_seed + PERMUTATION_SEED_OFFSET) % 2 ** 32)
    base = _perm_score(s, y, p.perm_metric)
    pi = np.zeros(r)
    for q in range(r):
        perm = random_order(uniform(stream, n, p.num_permute), axis=0)
        theta_q = theta[q][:, None]
        for b in range(p.num_permute):
            pi[q] += _perm_score(s + theta_q * (z[q, perm[:, b]] - z[q])[None, :], y, p.perm_metric) - base
    if p.perm_metric == "auroc":
        pi = -pi
    part["PermutationImportance"] = pi / p.num_permute

    baseline = zbar if p.prob_baseline == "mean" else np.zeros(r)
    change = np.full((r, c), np.nan)
    change_abs = np.full((r, c), np.nan)
    for q in range(r):
        ablated = logit_softmax(s - theta[q][:, None] * (z[q] - baseline[q])[None, :])
        pq = ablated.p
        for k in range(c):
            sub = (y == k) if p.prob_subset == "class" else np.ones(n, dtype=bool)
            if sub.any():


                with np.errstate(over="ignore"):
                    ratio_change = np.expm1(out.logp[k, sub] - ablated.logp[k, sub])
                change[q, k] = 100.0 * np.mean(ratio_change)
                change_abs[q, k] = 100.0 * np.mean(prob[k, sub] - pq[k, sub])
    part["ProbChange"], part["ProbChangeAbs"] = change, change_abs
    part["GroupPValue"] = group_test(z, y)[0] if group else np.full(r, np.nan)
    return explain_summary(part)


def _min_max(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def _mean_rows(a: np.ndarray, rows: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a.T).T if a.ndim == 1 else a
    return a[rows].mean(axis=0) if rows.any() else np.full(a.shape[1], np.nan)


def key_region(fi: np.ndarray, pi: np.ndarray, prob_change: np.ndarray,
               group_logp: Optional[np.ndarray] = None) -> Dict[str, object]:

    fi, pi = np.asarray(fi, float).ravel(), np.asarray(pi, float).ravel()
    if group_logp is None:
        group_logp = np.full(fi.shape, np.nan)
    key: Dict[str, object] = {"FeatureImportanceNorm": _min_max(fi), "PermutationImportanceNorm": _min_max(pi)}
    key["CombinedImportance"] = (key["FeatureImportanceNorm"] + key["PermutationImportanceNorm"]) / 2.0
    is_key = (fi > fi.mean()) & (pi > pi.mean())
    key["IsKey"] = is_key
    idx = np.flatnonzero(is_key)
    key["KeyRoi"] = idx[np.argsort(-key["CombinedImportance"][idx], kind="stable")]
    key["SubtypeKeyRoi"] = [key["KeyRoi"][prob_change[key["KeyRoi"], k] > 0] for k in range(prob_change.shape[1])]
    key["ProbChangeKey"] = _mean_rows(prob_change, is_key)
    key["ProbChangeRest"] = _mean_rows(prob_change, ~is_key)
    key["GroupLogPKey"] = float(_mean_rows(group_logp[:, None], is_key)[0])
    key["GroupLogPRest"] = float(_mean_rows(group_logp[:, None], ~is_key)[0])
    return key


def explain_summary(part: Dict[str, object]) -> Dict[str, object]:

    ex = dict(part)
    lp = np.asarray(part["ModalityImportance"], float)
    m_ = lp.shape[1]
    ex["ModalityImportanceMean"] = lp.mean(axis=0)
    top = lp.max(axis=1, keepdims=True)
    dominant = np.argmax(lp, axis=1).astype(float)
    tie = np.sum(lp >= top - 1e-12, axis=1) > 1
    dominant[tie] = np.nan
    ex["DominantModality"] = dominant
    ex["DominantCount"] = np.bincount(dominant[~tie].astype(int), minlength=m_)
    theta = np.asarray(part["RiskEffect"], float)
    ex["RiskEffectMean"] = theta.mean(axis=0)
    ex["RiskPositiveCount"] = np.sum(theta > 0, axis=0)
    ex["RiskNegativeCount"] = np.sum(theta < 0, axis=0)
    ex["RiskPositiveMean"] = np.sum(theta * (theta > 0), axis=0) / np.maximum(ex["RiskPositiveCount"], 1)
    ex["RiskNegativeMean"] = np.sum(theta * (theta < 0), axis=0) / np.maximum(ex["RiskNegativeCount"], 1)
    pv = np.asarray(part["GroupPValue"], float)
    logp = -np.log10(np.maximum(pv, np.finfo(float).tiny))
    logp[np.isnan(pv)] = np.nan
    ex["GroupLogP"] = logp
    ex.update(key_region(part["FeatureImportance"], part["PermutationImportance"],
                         np.asarray(part["ProbChange"], float), logp))
    return ex


def subject_explain(model: Model, x: List[np.ndarray], j: int) -> Dict[str, object]:


    model = param_reshape(model)
    n = x[0].shape[1]
    if isinstance(j, bool) or not isinstance(j, Real) or not np.isfinite(j) or j != int(j) \
            or not 0 <= int(j) < n:
        raise EMGPNError(f"j must lie in 0..{n - 1}.")
    out = model_forward(model, [xm[:, [int(j)]] for xm in x])
    r, m_, c = model.num_roi, model.num_modal, model.num_class
    res: Dict[str, object] = {"Risk": out.p[:, 0], "Logit": out.s[:, 0]}
    res["PredictedClass"] = int(np.argmax(out.p[:, 0]))
    res["PredictedName"] = model.design.class_name[res["PredictedClass"]]
    base = model.theta.T @ model.z_mean
    if model.params.use_bias:
        base = base + model.bias[:, 0]
    res["BaseLogit"] = base
    res["Propagated"] = np.column_stack([h[:, 0] for h in out.h])
    res["ModalityImportance"] = model.lambda_prob.copy()
    mc = np.zeros((r, m_, c))
    for m in range(m_):
        mc[:, m, :] = classifier_weight(model, m) * (out.h[m][:, 0] - model.h_mean[m])[:, None]
    res["ModalityContribution"] = mc
    res["Contribution"] = mc.sum(axis=1)
    res["Integrated"] = None if model.params.integration == "concat" else out.z[:, 0]
    res["DecompositionError"] = float(np.max(np.abs(base + res["Contribution"].sum(axis=0) - res["Logit"])))
    return res
