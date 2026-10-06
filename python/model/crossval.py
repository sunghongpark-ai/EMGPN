from __future__ import annotations

import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .config import Parameters, _integer
from .dataset import Dataset, EMGPNError
from .explain import explain_summary, model_explain
from .metrics import group_test, perform_measure
from .model import (Design, Model, data_indexing, data_normalize, graph_construct, model_initialize,
                    param_initialize, param_training)

__all__ = ["CVResult", "cross_validation", "result_summary", "METRIC_NAME"]

METRIC_NAME = ("AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1", "MicroAUROC", "MicroAUPRC",
               "BalancedAccuracy", "LogLoss", "PrecisionOvR", "RecallOvR", "F1OvR")


@dataclass
class CVResult:


    risk_test: List[np.ndarray]
    risk_mean: np.ndarray
    zoof: np.ndarray
    ydata: np.ndarray
    metric_name: List[str]
    metric: np.ndarray
    metric_mean: np.ndarray
    metric_std: np.ndarray
    explain: Optional[Dict[str, object]]
    fold: List[Dict[str, object]]
    parameters: Parameters
    modality_name: List[str]
    class_name: List[str]
    roi_name: List[str]
    subject_id: List[str]
    cv_index: np.ndarray
    info: Dict[str, object] = field(default_factory=dict)

    def metric_value(self, name: str) -> float:
        return float(self.metric_mean[self.metric_name.index(name)])


def _fold_compact(model: Model) -> Dict[str, object]:
    p = model.params
    f: Dict[str, object] = {
        "IdxModel": model.idx_model, "IdxIter": model.idx_iter, "IdxFold": model.idx_fold,
        "IdxTest": model.idx_test, "Ptest": model.test.p, "Ztest": model.test.z.astype(np.float32),
        "LambdaProb": model.lambda_prob, "Theta": model.theta,
        "Phi": model.phi if p.propagation == "gpn" else None,
        "Bias": model.bias if p.use_bias else None, "Sigma": model.sigma,
        "BestEpoch": model.best_epoch, "NumEpoch": model.num_epoch, "NumProject": model.num_project,
        "LossTrain": model.loss_train, "LossValid": model.loss_valid,
    }
    if p.integration == "concat":
        for name in ("FeatureImportance", "PermutationImportance", "ProbChange", "ProbChangeAbs"):
            f[name] = None
    else:
        ex = model_explain(model, model.xtest, model.ytest, group=False)
        for name in ("FeatureImportance", "PermutationImportance", "ProbChange", "ProbChangeAbs"):
            f[name] = ex[name]
    return f


_WORKER_DESIGN: Optional[Design] = None


def _init_worker(design: Design) -> None:
    global _WORKER_DESIGN
    _WORKER_DESIGN = design


def _train_one(idx_model: int, design: Optional[Design] = None) -> Dict[str, object]:
    design = design if design is not None else _WORKER_DESIGN
    model = data_indexing(design, idx_model)
    model = data_normalize(model)
    model = graph_construct(model)
    model = param_initialize(model)
    model = param_training(model)
    return _fold_compact(model)


def cross_validation(dataset: Dataset, params=None, n_jobs: int = 1):


    n_jobs = _integer(n_jobs, "n_jobs", 1)
    design = model_initialize(dataset, params)
    clock = time.perf_counter()
    if n_jobs == 1:
        fold = [_train_one(i, design) for i in range(design.num_model)]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(design,)) as pool:
            fold = list(pool.map(_train_one, range(design.num_model)))
    result = result_summary(design, fold)
    result.info["ElapsedSecond"] = time.perf_counter() - clock
    return result, design


def _fold_mean(fold: List[Dict[str, object]], name: str) -> Optional[np.ndarray]:
    if fold[0][name] is None:
        return None
    total = np.zeros(np.shape(fold[0][name]))
    count = np.zeros(total.shape)
    for f in fold:
        v = np.asarray(f[name], dtype=float)
        ok = ~np.isnan(v)
        total[ok] += v[ok]
        count += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out


def result_summary(design: Design, fold: List[Dict[str, object]]) -> CVResult:

    if len(fold) != design.num_model:
        raise EMGPNError("fold results must contain all CV models exactly once.")
    if sorted(f["IdxModel"] for f in fold) != list(range(design.num_model)):
        raise EMGPNError("CV model indices must be unique and cover the design.")
    p = design.params
    y = design.ydata
    n, c = design.num_subj, design.num_class
    risk_test = [np.full((c, n), np.nan) for _ in range(p.num_iter)]
    num_feat = fold[0]["Ztest"].shape[0]
    zsum = np.zeros((num_feat, n))
    zcount = np.zeros(n)
    for f in fold:
        risk_test[f["IdxIter"]][:, f["IdxTest"]] = f["Ptest"]
        zsum[:, f["IdxTest"]] += f["Ztest"].astype(float)
        zcount[f["IdxTest"]] += 1
        
    risk_mean = sum(risk_test) / p.num_iter
    zoof = zsum / np.maximum(zcount, 1)
    metric = np.full((p.num_iter, len(METRIC_NAME)), np.nan)
    for i in range(p.num_iter):
        if np.isnan(risk_test[i]).any():
            warnings.warn(f"Repetition {i + 1} lacks out-of-fold predictions.", stacklevel=2)
            continue
        perf = perform_measure(risk_test[i], y)
        ovr = perform_measure(risk_test[i], y, decision="ovr")
        perf["PrecisionOvR"], perf["RecallOvR"], perf["F1OvR"] = ovr["Precision"], ovr["Recall"], ovr["F1"]
        metric[i] = [perf[name] for name in METRIC_NAME]
    done = ~np.isnan(metric).any(axis=1)
    metric_mean = metric[done].mean(axis=0)
    metric_std = metric[done].std(axis=0, ddof=1) if done.sum() > 1 else np.zeros(len(METRIC_NAME))
    explain = None
    if p.integration != "concat":
        part = {"ModalityName": list(design.modality_name), "ClassName": list(design.class_name),
                "RoiName": list(design.roi_name), "ModalityImportance": _fold_mean(fold, "LambdaProb"),
                "Steadiness": _fold_mean(fold, "Phi"), "RiskEffect": _fold_mean(fold, "Theta"),
                "FeatureImportance": _fold_mean(fold, "FeatureImportance"),
                "PermutationImportance": _fold_mean(fold, "PermutationImportance"),
                "ProbChange": _fold_mean(fold, "ProbChange"), "ProbChangeAbs": _fold_mean(fold, "ProbChangeAbs"),
                "GroupPValue": group_test(zoof, y)[0]}
        explain = explain_summary(part)
    info = {"Method": "EMGPN (Park et al., Neural Networks 192:107971, 2025)",
            "Version": f"EMGPN-Python {__version__}",
            "Software": f"Python {platform.python_version()}, NumPy {np.__version__}",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "NumModel": len(fold),
            "MeanBestEpoch": float(np.mean([f["BestEpoch"] for f in fold])),
            "NumPhiProjection": int(sum(f["NumProject"] for f in fold))}
    return CVResult(risk_test=risk_test, risk_mean=risk_mean, zoof=zoof, ydata=y, metric_name=list(METRIC_NAME),
                    metric=metric, metric_mean=metric_mean, metric_std=metric_std, explain=explain,
                    fold=[{k: v for k, v in f.items() if k != "Ztest"} for f in fold],
                    parameters=p, modality_name=list(design.modality_name), class_name=list(design.class_name),
                    roi_name=list(design.roi_name), subject_id=list(design.subject_id), cv_index=design.cv_index,
                    info=info)
