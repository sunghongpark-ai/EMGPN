from __future__ import annotations

import math
from numbers import Real
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import lapack
from scipy import sparse
from scipy.sparse.linalg import splu

from .config import Parameters, resolve_parameters
from .dataset import Dataset, EMGPNError, _names, _real_array, _labels
from .graph import graph_laplacian, knn_graph
from .rng import random_order, random_stream, uniform

__all__ = [
    "Design", "Model", "Forward", "model_initialize", "data_indexing", "data_normalize",
    "data_transform", "graph_construct", "param_initialize", "param_reshape", "prop_solve",
    "model_forward", "forward_propagate", "loss_calculation", "backward_propagate",
    "parameter_update", "param_training", "model_fit", "risk_predict", "classifier_weight",
    "logit_softmax",
]


@dataclass
class Design:


    params: Parameters
    xraw: List[np.ndarray]
    ydata: np.ndarray
    wdata: Optional[List[np.ndarray]]
    roi_name: List[str]
    class_name: List[str]
    class_count: np.ndarray
    modality_name: List[str]
    modality_index: Tuple[int, ...]
    modality_name_total: List[str]
    subject_id: List[str]
    cv_index: np.ndarray
    cv_list: np.ndarray

    @property
    def num_roi(self) -> int:
        return self.xraw[0].shape[0]

    @property
    def num_subj(self) -> int:
        return self.xraw[0].shape[1]

    @property
    def num_class(self) -> int:
        return len(self.class_name)

    @property
    def num_modal(self) -> int:
        return len(self.xraw)

    @property
    def num_modal_total(self) -> int:
        return len(self.modality_name_total)

    @property
    def num_model(self) -> int:
        return self.cv_list.shape[0]


@dataclass
class Forward:


    s: np.ndarray
    p: np.ndarray
    logp: np.ndarray
    h: Optional[List[np.ndarray]] = None
    z: Optional[np.ndarray] = None


@dataclass
class Model:


    design: Design
    params: Parameters
    idx_model: int = -1
    idx_iter: int = -1
    idx_fold: int = -1
    rand_seed: int = 0
    idx_train: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    idx_valid: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    idx_test: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ytrain: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    yvalid: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ytest: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    norm_mean: Optional[List[np.ndarray]] = None
    norm_std: Optional[List[np.ndarray]] = None
    xtrain: Optional[List[np.ndarray]] = None
    xvalid: Optional[List[np.ndarray]] = None
    xtest: Optional[List[np.ndarray]] = None
    wgraph: Optional[List[np.ndarray]] = None
    lgraph: Optional[List[np.ndarray]] = None
    agcn: Optional[List[Optional[np.ndarray]]] = None
    sigma: Optional[np.ndarray] = None
    param_name: List[str] = field(default_factory=list)
    param_shape: List[Tuple[int, int]] = field(default_factory=list)
    param_slice: List[slice] = field(default_factory=list)
    num_feat: int = 0
    weight: Optional[np.ndarray] = None
    adam: Optional[dict] = None
    phi: Optional[np.ndarray] = None
    lam: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    lambda_prob: Optional[np.ndarray] = None
    prop_chol: Optional[list] = None
    loss_train: Optional[np.ndarray] = None
    loss_valid: Optional[np.ndarray] = None
    obj_train: Optional[np.ndarray] = None
    num_project: int = 0
    best_epoch: int = 0
    num_epoch: int = 0
    train: Optional[Forward] = None
    valid: Optional[Forward] = None
    test: Optional[Forward] = None
    z_mean: Optional[np.ndarray] = None
    h_mean: Optional[List[np.ndarray]] = None
    gradient: Optional[np.ndarray] = None
    cache_y: Optional[list] = None
    hfix_train: Optional[list] = None
    hfix_valid: Optional[list] = None

    def __getstate__(self):
        
        
        state = self.__dict__.copy()
        state["prop_chol"] = None
        return state

    
    @property
    def num_roi(self) -> int:
        return self.design.num_roi

    @property
    def num_class(self) -> int:
        return self.design.num_class

    @property
    def num_modal(self) -> int:
        return self.design.num_modal

    @property
    def num_train(self) -> int:
        return self.idx_train.size

    @property
    def num_valid(self) -> int:
        return self.idx_valid.size

    @property
    def num_test(self) -> int:
        return self.idx_test.size

    @property
    def phi_slice(self) -> Optional[slice]:
        return self.param_slice[self.param_name.index("Phi")] if "Phi" in self.param_name else None

    @property
    def num_param(self) -> int:
        return 0 if self.weight is None else self.weight.size


def _modality_index(selection, names: List[str]) -> Tuple[int, ...]:
    total = len(names)
    if selection is None:
        return tuple(range(total))
    index = []
    for s in selection:
        if isinstance(s, str):
            if s not in names:
                raise EMGPNError(f"Unknown modality '{s}'; available: {', '.join(names)}.")
            index.append(names.index(s))
        else:
            if isinstance(s, bool) or not isinstance(s, Real) or not math.isfinite(s) \
                    or int(s) != s or not 0 <= int(s) < total:
                raise EMGPNError(f"Modality indices must lie in 0..{total - 1}.")
            index.append(int(s))
    if len(set(index)) != len(index) or not index:
        raise EMGPNError("modality must list distinct modalities.")
    return tuple(sorted(index))


def model_initialize(dataset: Dataset, params=None) -> Design:

    p = resolve_parameters(params)
    if not isinstance(dataset, Dataset):
        raise EMGPNError("dataset must be an emgpn.Dataset.")
    if not isinstance(dataset.xdata, (list, tuple)) or not dataset.xdata:
        raise EMGPNError("dataset.xdata must hold a nonempty list of modality matrices.")
    xdata = [_real_array(x, "xdata") for x in dataset.xdata]
    if not xdata:
        raise EMGPNError("dataset.xdata must hold at least one modality.")
    total = len(xdata)
    names_total = list(dataset.modality_name) if dataset.modality_name else (
        ["sMRI", "fMRI", "PET"] if total == 3 else [f"Modality{m + 1}" for m in range(total)])
    names_total = _names(names_total, "modality_name", total)
    index = _modality_index(p.modality, names_total)
    if p.kernel_width is not None and len(p.kernel_width) not in (1, len(index), total):
        raise EMGPNError(f"kernel_width must have 1, {len(index)} or {total} values.")
    if xdata[index[0]].ndim != 2:
        raise EMGPNError("xdata must be ROI-by-participant matrices.")
    r, n = xdata[index[0]].shape
    for m in index:
        if xdata[m].ndim != 2 or xdata[m].shape != (r, n):
            raise EMGPNError(f"All modalities must be {r}-by-{n} (ROI x participant).")
        if np.isinf(xdata[m]).any():
            raise EMGPNError("xdata contains Inf; use NaN for missing values.")
    if r < 2:
        raise EMGPNError("At least two ROIs are required.")
    if dataset.ydata is None:
        raise EMGPNError("Training requires labels (dataset.ydata).")
    y = _labels(dataset.ydata, n)
    class_name = list(dataset.class_name) if dataset.class_name else [f"Class{k + 1}" for k in range(y.max() + 1)]
    class_name = _names(class_name, "class_name")
    if y.max() >= len(class_name):
        raise EMGPNError(f"Labels exceed the number of class names ({len(class_name)}).")
    if len(class_name) < 2:
        raise EMGPNError("At least two classes are required.")
    class_count = np.bincount(y, minlength=len(class_name))
    roi_name = list(dataset.roi_name) if dataset.roi_name else [f"ROI{q + 1:03d}" for q in range(r)]
    roi_name = _names(roi_name, "roi_name", r)
    subject_id = list(dataset.subject_id) if dataset.subject_id else [f"P{j + 1:04d}" for j in range(n)]
    subject_id = _names(subject_id, "subject_id", n)
    wdata = None
    if dataset.wdata is not None and not isinstance(dataset.wdata, (list, tuple)):
        raise EMGPNError("wdata must be a list of graphs or None.")
    if dataset.wdata:
        if len(dataset.wdata) != total:
            raise EMGPNError(f"wdata must hold {total} graphs.")
        wdata = [None if dataset.wdata[m] is None else
                 (dataset.wdata[m].copy() if sparse.issparse(dataset.wdata[m]) else
                  _real_array(dataset.wdata[m], "wdata").copy()) for m in index]

    if p.cv_index is not None:
        cv = np.atleast_2d(_real_array(p.cv_index, "cv_index"))
        if cv.ndim != 2 or not cv.size or cv.shape[1] != n or not np.all(np.isfinite(cv)) \
                or np.any(cv != np.round(cv)) or cv.min() < 0 or cv.max() >= n:
            raise EMGPNError(f"cv_index must be a num_iter-by-{n} matrix of folds 0..K-1.")
        cv = cv.astype(int)
        k = int(cv.max()) + 1
        if any(np.unique(row).size != k for row in cv):
            raise EMGPNError("Every row of cv_index must contain all folds 0..K-1.")
        p = p.replace(num_iter=cv.shape[0], num_fold=k)
    if p.use_valid and p.num_fold < 3:
        raise EMGPNError("num_fold must be >= 3 when use_valid is true (train/valid/test folds).")
    if np.any((class_count > 0) & (class_count < p.num_fold)):
        warnings.warn("Some classes have fewer members than folds; they are absent from some folds.",
                      stacklevel=2)
    if p.cv_index is None:
        cv = np.zeros((p.num_iter, n), dtype=int)
        for i in range(p.num_iter):
            stream = random_stream((p.seed + i + 1) % 2 ** 32)
            order = []
            for k in range(len(class_name)):
                member = np.flatnonzero(y == k)
                order.append(member[random_order(uniform(stream, 1, member.size)[0])])
            cv[i, np.concatenate(order)] = np.arange(n) % p.num_fold
    k = p.num_fold
    cv_list = np.array([(i, f, (f + 1) % k) for i in range(p.num_iter) for f in range(k)], dtype=int)
    return Design(params=p, xraw=[xdata[m].copy() for m in index], ydata=y, wdata=wdata,
                  roi_name=roi_name, class_name=class_name, class_count=class_count,
                  modality_name=[names_total[m] for m in index], modality_index=index,
                  modality_name_total=names_total, subject_id=subject_id, cv_index=cv, cv_list=cv_list)


def _index(idx, n: int, name: str) -> np.ndarray:
    if idx is None:
        return np.zeros(0, dtype=int)
    idx = np.asarray(idx)
    if idx.dtype == bool:
        if idx.size != n:
            raise EMGPNError(f"Boolean {name} must have {n} elements.")
        idx = np.flatnonzero(idx)
    idx = _real_array(idx, name).ravel()
    if not np.all(np.isfinite(idx)) or np.any(idx != np.round(idx)) or np.any(idx < 0) \
            or np.any(idx >= n) or np.unique(idx).size != idx.size:
        raise EMGPNError(f"{name} must contain distinct participant indices in 0..{n - 1}.")
    return idx.astype(int)


def data_indexing(design: Design, idx_model: Optional[int] = None, train=None, valid=None, test=None) -> Model:

    model = Model(design=design, params=design.params)
    n = design.num_subj
    if idx_model is not None:
        if isinstance(idx_model, bool) or not isinstance(idx_model, Real) or not math.isfinite(idx_model) \
                or idx_model != int(idx_model) or not 0 <= int(idx_model) < design.num_model:
            raise EMGPNError(f"idx_model must lie in 0..{design.num_model - 1}.")
        it, f_test, f_valid = design.cv_list[int(idx_model)]
        fold = design.cv_index[it]
        test = np.flatnonzero(fold == f_test)
        if design.params.use_valid:
            valid = np.flatnonzero(fold == f_valid)
            train = np.flatnonzero((fold != f_test) & (fold != f_valid))
        else:
            valid = None
            train = np.flatnonzero(fold != f_test)
        model.idx_model, model.idx_iter, model.idx_fold = int(idx_model), int(it), int(f_test)
        model.rand_seed = (design.params.seed + int(it) + 1) % 2 ** 32
    else:
        model.rand_seed = design.params.seed
    model.idx_train = _index(train, n, "train")
    model.idx_valid = _index(valid, n, "valid")
    model.idx_test = _index(test, n, "test")
    if model.idx_train.size == 0:
        raise EMGPNError("The training set is empty.")
    parts = (model.idx_train, model.idx_valid, model.idx_test)
    if np.unique(np.concatenate(parts)).size != sum(a.size for a in parts):
        raise EMGPNError("Training, validation and test sets must be disjoint.")
    y = design.ydata
    model.ytrain, model.yvalid, model.ytest = y[model.idx_train], y[model.idx_valid], y[model.idx_test]
    return model


def _one_hot(y: np.ndarray, c: int) -> np.ndarray:
    out = np.zeros((c, y.size))
    out[y, np.arange(y.size)] = 1.0
    return out


def data_normalize(model: Model) -> Model:

    model.norm_mean, model.norm_std = [], []
    for x in model.design.xraw:
        xtr = x[:, model.idx_train]
        valid = ~np.isnan(xtr)
        count = valid.sum(axis=1)
        x0 = np.where(valid, xtr, 0.0)
        mu = x0.sum(axis=1) / np.maximum(count, 1)
        dev = (x0 - mu[:, None]) * valid
        sd = np.sqrt((dev ** 2).sum(axis=1) / np.maximum(count - 1, 1))
        if not np.all(np.isfinite(mu[count > 0])) or not np.all(np.isfinite(sd)):
            raise EMGPNError("Training normalization statistics overflowed; rescale the data.")
        mu[count == 0] = np.nan
        sd[(count <= 1) | ~(sd > 0)] = 1.0
        model.norm_mean.append(mu)
        model.norm_std.append(sd)
    xnorm = data_transform(model, model.design.xraw)
    model.xtrain = [x[:, model.idx_train] for x in xnorm]
    model.xvalid = [x[:, model.idx_valid] for x in xnorm]
    model.xtest = [x[:, model.idx_test] for x in xnorm]
    return model


def _align(model: Model, data) -> List[np.ndarray]:
    design = model.design
    if isinstance(data, Dataset):
        xs = [_real_array(x, "xdata") for x in data.xdata]
        names = list(data.modality_name)
        if names:
            _names(names, "modality_name", len(xs))
            missing = [m for m in design.modality_name if m not in names]
            if missing:
                raise EMGPNError("Missing modalities: " + ", ".join(missing))
            xs = [xs[names.index(m)] for m in design.modality_name]
        elif len(xs) == design.num_modal_total:
            xs = [xs[m] for m in design.modality_index]
        if data.roi_name:
            _names(data.roi_name, "roi_name")
            if any(x.ndim not in (1, 2) or x.shape[0] != len(data.roi_name) for x in xs):
                raise EMGPNError("Each imaging matrix must have one row per roi_name.")
            pos = {name: q for q, name in enumerate(data.roi_name)}
            missing = [q for q in design.roi_name if q not in pos]
            if missing:
                raise EMGPNError(f"{len(missing)} ROIs are missing, e.g. {missing[0]}.")
            order = [pos[q] for q in design.roi_name]
            xs = [x[order] for x in xs]
        if len(xs) != design.num_modal:
            raise EMGPNError(f"Expected {design.num_modal} modalities.")
        return xs
    xs = [_real_array(x, "data") for x in (data if isinstance(data, (list, tuple)) else [data])]
    if len(xs) == design.num_modal_total and design.num_modal_total != design.num_modal:
        xs = [xs[m] for m in design.modality_index]
    elif len(xs) != design.num_modal:
        raise EMGPNError(f"Expected {design.num_modal} (or {design.num_modal_total}) modalities.")
    return xs


def data_transform(model: Model, data) -> List[np.ndarray]:


    out = []
    n = None
    for m, x in enumerate(_align(model, data)):
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != model.num_roi:
            raise EMGPNError(f"Modality {m} must have {model.num_roi} rows (ROIs).")
        if np.isinf(x).any():
            raise EMGPNError("Prediction data contains Inf; use NaN for missing values.")
        if n is None:
            n = x.shape[1]
        elif x.shape[1] != n:
            raise EMGPNError("Prediction modalities must have the same participant count.")
        if model.params.normalize in ("zscore-logistic", "zscore"):
            z = (x - model.norm_mean[m][:, None]) / model.norm_std[m][:, None]
        else:
            z = x.copy()
        z[np.isnan(z)] = 0.0
        if not np.all(np.isfinite(z)):
            raise EMGPNError("Normalization overflowed; rescale the data.")
        if model.params.normalize == "zscore-logistic":
            positive = z >= 0
            result = np.empty_like(z)
            result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
            ez = np.exp(z[~positive])
            result[~positive] = ez / (1.0 + ez)
            z = result
        out.append(z)
    return out


def graph_construct(model: Model) -> Model:

    p, design = model.params, model.design
    r = model.num_roi
    model.wgraph, model.lgraph, model.agcn = [], [], []
    model.sigma = np.full(model.num_modal, np.nan)
    for m in range(model.num_modal):
        if design.wdata is not None and design.wdata[m] is not None:
            raw = design.wdata[m]
            if sparse.issparse(raw):
                if raw.dtype.kind not in "biuf":
                    raise EMGPNError("wdata must be real numeric.")
                w = sparse.csc_matrix(raw, dtype=float)
                values = w.data
            else:
                w = _real_array(raw, "wdata")
                values = w
            if w.shape != (r, r) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise EMGPNError(f"wdata[{m}] must be a finite nonnegative {r}-by-{r} matrix.")
            w = (w + w.T) / 2.0
            if sparse.issparse(w):
                w.setdiag(0.0)
                w.eliminate_zeros()
            else:
                np.fill_diagonal(w, 0.0)
        else:
            kw = p.kernel_width
            if kw is None:
                sigma = None
            elif len(kw) == 1:
                sigma = kw[0]
            elif len(kw) == design.num_modal_total:
                sigma = kw[design.modality_index[m]]
            else:
                sigma = kw[m]
            w, sigma, _ = knn_graph(model.xtrain[m], p.num_neighbor, sigma)
            model.sigma[m] = sigma
        lap, ahat = graph_laplacian(w)
        model.wgraph.append(w)
        model.lgraph.append(lap)
        model.agcn.append(ahat if p.propagation == "gcn" else None)
    return model


def param_initialize(model: Model) -> Model:

    p = model.params
    r, m_, c = model.num_roi, model.num_modal, model.num_class
    model.num_feat = m_ * r if p.integration == "concat" else r
    init = []
    model.param_name, model.param_shape = [], []
    if p.propagation == "gpn":
        model.param_name.append("Phi")
        model.param_shape.append((r, m_))
        init.append(np.full((r, m_), p.init_steady))
    if p.integration == "softmax":
        model.param_name.append("Lambda")
        model.param_shape.append((r, m_))
        init.append(np.full((r, m_), p.init_integrate))
    if p.init_classifier == "glorot":
        stream = random_stream(model.rand_seed)
        theta = (2.0 * uniform(stream, model.num_feat, c) - 1.0) * math.sqrt(6.0 / (model.num_feat + c))
    else:
        theta = np.zeros((model.num_feat, c))
    model.param_name.append("Theta")
    model.param_shape.append((model.num_feat, c))
    init.append(theta)
    if p.use_bias:
        model.param_name.append("Bias")
        model.param_shape.append((c, 1))
        init.append(np.zeros((c, 1)))
    stop = np.cumsum([a * b for a, b in model.param_shape])
    start = np.concatenate([[0], stop[:-1]])
    model.param_slice = [slice(int(a), int(b)) for a, b in zip(start, stop)]
    model.weight = np.concatenate([v.ravel(order="F") for v in init])
    model.adam = {"t": 0, "m": np.zeros_like(model.weight), "v": np.zeros_like(model.weight)}
    model.loss_train = np.full(p.max_epoch, np.nan)
    model.loss_valid = np.full(p.max_epoch, np.nan)
    model.obj_train = np.full(p.max_epoch, np.nan)
    model.num_project = 0
    return model


def param_reshape(model: Model) -> Model:

    p = model.params
    if model.weight is None or model.weight.shape != (sum(a * b for a, b in model.param_shape),) \
            or not np.all(np.isfinite(model.weight)):
        raise EMGPNError("Model parameters must be a finite vector of the declared size.")
    value = {}
    for name, shape, sl in zip(model.param_name, model.param_shape, model.param_slice):
        value[name] = model.weight[sl].reshape(shape, order="F")
    model.phi, model.lam = value.get("Phi"), value.get("Lambda")
    model.theta, model.bias = value["Theta"], value.get("Bias")
    r, m_ = model.num_roi, model.num_modal
    if p.integration == "softmax":
        e = np.exp(model.lam - model.lam.max(axis=1, keepdims=True))
        model.lambda_prob = e / e.sum(axis=1, keepdims=True)
    elif p.integration == "mean":
        model.lambda_prob = np.full((r, m_), 1.0 / m_)
    else:
        model.lambda_prob = np.full((r, m_), np.nan)
    if p.propagation == "gpn":
        model.prop_chol = []
        for m in range(m_):
            if sparse.issparse(model.lgraph[m]):
                if np.any(model.phi[:, m] <= 0):
                    raise EMGPNError("Sparse Phi + L is not positive definite: phi must be positive.")
                a = model.lgraph[m] + sparse.diags(model.phi[:, m], format="csc")
                try:
                    model.prop_chol.append(splu(a.tocsc()))
                except RuntimeError as err:
                    raise EMGPNError(f"Sparse propagation factorization failed for modality {m}.") from err
                continue
            factor, info = lapack.dpotrf(model.lgraph[m] + np.diag(model.phi[:, m]), lower=0, clean=1)
            if info != 0:
                raise EMGPNError(f"Phi + L of modality {m} is not positive definite (min phi = "
                                 f"{model.phi[:, m].min():g}); keep phi_min > 0.")
            model.prop_chol.append(factor)
    return model


def prop_solve(model: Model, m: int, b: np.ndarray) -> np.ndarray:

    b = _real_array(b, "Propagation right-hand side")
    if b.ndim not in (1, 2) or b.shape[0] != model.num_roi or not np.all(np.isfinite(b)):
        raise EMGPNError("Propagation right-hand side must be finite with one row per ROI.")
    if sparse.issparse(model.lgraph[m]):
        y = model.prop_chol[m].solve(np.asarray(b, dtype=float))
        if not np.all(np.isfinite(y)):
            raise EMGPNError(f"Sparse propagation solve failed for modality {m}.")
        return y
    y, info = lapack.dpotrs(model.prop_chol[m], b, lower=0)
    if info != 0 or not np.all(np.isfinite(y)):
        raise EMGPNError(f"Propagation solve failed for modality {m} (LAPACK info {info}).")
    return y


def classifier_weight(model: Model, m: int) -> np.ndarray:

    if model.params.integration == "concat":
        r = model.num_roi
        return model.theta[m * r:(m + 1) * r, :]
    return model.lambda_prob[:, m:m + 1] * model.theta


def logit_softmax(s: np.ndarray) -> Forward:

    if s.ndim != 2 or not np.all(np.isfinite(s)):
        raise EMGPNError("Logits must be a finite class-by-participant matrix.")
    shift = s - s.max(axis=0, keepdims=True)
    expo = np.exp(shift)
    denom = expo.sum(axis=0, keepdims=True)
    return Forward(s=s, p=expo / denom, logp=shift - np.log(denom))


def model_forward(model: Model, x: List[np.ndarray]) -> Forward:

    if not isinstance(x, (list, tuple)) or len(x) != model.num_modal:
        raise EMGPNError(f"Normalized features must contain {model.num_modal} modalities.")
    x = [_real_array(value, "Normalized features") for value in x]
    if any(value.ndim != 2 or value.shape[0] != model.num_roi or not np.all(np.isfinite(value)) for value in x) \
            or any(value.shape[1] != x[0].shape[1] for value in x):
        raise EMGPNError("Normalized features must be finite ROI-by-participant matrices with matching shapes.")
    p = model.params
    h = []
    for m in range(model.num_modal):
        if p.propagation == "gpn":
            h.append(prop_solve(model, m, model.phi[:, m:m + 1] * x[m]))
        elif p.propagation == "gcn":
            h.append(model.agcn[m] @ x[m])
        else:
            h.append(np.array(x[m], dtype=float))
    if p.integration == "concat":
        z = np.vstack(h)
    else:
        z = model.lambda_prob[:, 0:1] * h[0]
        for m in range(1, model.num_modal):
            z = z + model.lambda_prob[:, m:m + 1] * h[m]
    s = model.theta.T @ z
    if p.use_bias:
        s = s + model.bias
    out = logit_softmax(s)
    out.h, out.z = h, z
    return out


def forward_propagate(model: Model) -> Model:


    p = model.params
    c = model.num_class
    if p.propagation != "gpn" and model.hfix_train is None:
        if p.propagation == "gcn":
            model.hfix_train = [a @ x for a, x in zip(model.agcn, model.xtrain)]
            model.hfix_valid = [a @ x for a, x in zip(model.agcn, model.xvalid)]
        else:
            model.hfix_train, model.hfix_valid = list(model.xtrain), list(model.xvalid)
    s_train = np.zeros((c, model.num_train))
    s_valid = np.zeros((c, model.num_valid))
    model.cache_y = [None] * model.num_modal
    for m in range(model.num_modal):
        w = classifier_weight(model, m)
        if p.propagation == "gpn":
            y = prop_solve(model, m, w)
            v = model.phi[:, m:m + 1] * y
            model.cache_y[m] = y
            s_train = s_train + v.T @ model.xtrain[m]
            if model.num_valid:
                s_valid = s_valid + v.T @ model.xvalid[m]
        else:
            s_train = s_train + w.T @ model.hfix_train[m]
            if model.num_valid:
                s_valid = s_valid + w.T @ model.hfix_valid[m]
    if p.use_bias:
        s_train = s_train + model.bias
        s_valid = s_valid + model.bias
    model.train = logit_softmax(s_train)
    model.valid = logit_softmax(s_valid) if model.num_valid else None
    return model


def _cross_entropy(logp: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(logp[y, np.arange(y.size)]))


def loss_calculation(model: Model, epoch: int) -> Model:

    model.loss_train[epoch] = _cross_entropy(model.train.logp, model.ytrain)
    reg = 0.0
    for name, val in (("Phi", model.phi), ("Lambda", model.lam), ("Theta", model.theta)):
        if name in model.param_name:
            reg += float(np.sum(val ** 2))
    model.obj_train[epoch] = model.loss_train[epoch] + model.params.reg_coeff * reg
    if model.num_valid:
        model.loss_valid[epoch] = _cross_entropy(model.valid.logp, model.yvalid)
    return model


def backward_propagate(model: Model) -> Model:


    p = model.params
    r, c, m_ = model.num_roi, model.num_class, model.num_modal
    delta = p.reg_coeff
    e = (model.train.p - _one_hot(model.ytrain, c)) / model.num_train
    grad_theta = np.zeros_like(model.theta)
    omega = np.zeros((r, m_))
    grad_phi = np.zeros((r, m_)) if p.propagation == "gpn" else None
    for m in range(m_):
        w = classifier_weight(model, m)
        if p.propagation == "gpn":
            xe = model.xtrain[m] @ e.T
            phi = model.phi[:, m:m + 1]
            he = prop_solve(model, m, phi * xe)
            if p.grad_mode == "exact":
                g = np.sum(model.cache_y[m] * (xe - he), axis=1)
            elif p.grad_mode == "published":
                g = np.sum(w * prop_solve(model, m, xe - he), axis=1)
            else:
                v = prop_solve(model, m, xe)
                g = np.sum(w * v, axis=1) - phi[:, 0] * np.sum(w * prop_solve(model, m, v), axis=1)
            grad_phi[:, m] = g + 2.0 * delta * model.phi[:, m]
        else:
            he = model.hfix_train[m] @ e.T
        if p.integration == "concat":
            grad_theta[m * r:(m + 1) * r, :] = he
        else:
            grad_theta = grad_theta + model.lambda_prob[:, m:m + 1] * he
            if p.integration == "softmax":
                omega[:, m] = np.sum(model.theta * he, axis=1)
    grad = {"Theta": grad_theta + 2.0 * delta * model.theta}
    if p.integration == "softmax":
        lp = model.lambda_prob
        grad["Lambda"] = lp * (omega - np.sum(lp * omega, axis=1, keepdims=True)) + 2.0 * delta * model.lam
    if grad_phi is not None:
        grad["Phi"] = grad_phi
    if p.use_bias:
        grad["Bias"] = e.sum(axis=1, keepdims=True)
    model.gradient = np.concatenate([grad[name].ravel(order="F") for name in model.param_name])
    return model


def parameter_update(model: Model) -> Model:

    p = model.params
    g = model.gradient
    if g is None or g.shape != model.weight.shape or not np.all(np.isfinite(g)):
        raise EMGPNError("Gradient must be finite and have the parameter-vector shape.")
    if p.optimizer == "adam":
        a = model.adam
        a["t"] += 1
        a["m"] = p.adam_beta1 * a["m"] + (1.0 - p.adam_beta1) * g
        a["v"] = p.adam_beta2 * a["v"] + (1.0 - p.adam_beta2) * (g ** 2)
        if not np.all(np.isfinite(a["m"])) or not np.all(np.isfinite(a["v"])):
            raise EMGPNError("Adam moments overflowed; check gradient and data scale.")
        mhat = a["m"] / (1.0 - p.adam_beta1 ** a["t"])
        vhat = a["v"] / (1.0 - p.adam_beta2 ** a["t"])
        model.weight = model.weight - p.learn_rate * mhat / (np.sqrt(vhat) + p.adam_epsilon)
    else:
        model.weight = model.weight - p.learn_rate * g
    if not np.all(np.isfinite(model.weight)):
        raise EMGPNError("Parameter update produced non-finite weights; check learning rate and data scale.")
    sl = model.phi_slice
    if sl is not None:
        phi = model.weight[sl]
        low = phi < p.phi_min
        if low.any():
            phi[low] = p.phi_min
            model.num_project += int(low.sum())
    return model


def param_training(model: Model) -> Model:

    p = model.params
    has_valid = model.num_valid > 0
    best_crit, best_epoch, best_weight = math.inf, 0, model.weight.copy()
    best_adam = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in model.adam.items()}
    wait = 0
    num_epoch = 0
    for epoch in range(p.max_epoch):
        model = param_reshape(model)
        model = forward_propagate(model)
        model = loss_calculation(model, epoch)
        if not np.isfinite(model.obj_train[epoch]) or (has_valid and not np.isfinite(model.loss_valid[epoch])):
            warnings.warn(f"Non-finite loss at epoch {epoch + 1}; training stopped.", stacklevel=2)
            break
        num_epoch = epoch + 1
        crit = model.loss_valid[epoch] if has_valid else -float(epoch + 1)
        if crit < best_crit:
            best_crit, best_epoch, best_weight, wait = crit, epoch + 1, model.weight.copy(), 0
            best_adam = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in model.adam.items()}
        else:
            wait += 1
            if wait >= p.patience:
                break
        if epoch + 1 == p.max_epoch:
            break
        model = backward_propagate(model)
        model = parameter_update(model)
        if p.verbose >= 2 and (epoch + 1) % 50 == 0:
            print(f"[EMGPN] epoch {epoch + 1:4d} | train loss {model.loss_train[epoch]:.4f} | "
                  f"objective {model.obj_train[epoch]:.4f} | valid loss {model.loss_valid[epoch]:.4f}")
    if best_epoch == 0:
        raise EMGPNError("No finite loss was obtained; check the data and hyperparameters.")
    model.num_epoch, model.best_epoch = num_epoch, best_epoch
    model.loss_train = model.loss_train[:num_epoch]
    model.loss_valid = model.loss_valid[:num_epoch]
    model.obj_train = model.obj_train[:num_epoch]
    model.weight = best_weight
    model.adam = best_adam
    model = param_reshape(model)
    model.train = model_forward(model, model.xtrain)
    model.valid = model_forward(model, model.xvalid) if has_valid else None
    model.z_mean = model.train.z.mean(axis=1)
    model.h_mean = [h.mean(axis=1) for h in model.train.h]
    model.test = model_forward(model, model.xtest) if model.num_test else None
    model.gradient = model.cache_y = model.hfix_train = model.hfix_valid = None
    if p.verbose >= 1:
        print(f"[EMGPN] iter {model.idx_iter + 1:3d} fold {model.idx_fold + 1:d} | best epoch "
              f"{best_epoch:4d} / {num_epoch:4d} | train loss {model.loss_train[best_epoch - 1]:.4f} | "
              f"valid loss {model.loss_valid[best_epoch - 1]:.4f}")
    return model


def model_fit(dataset: Dataset, params=None, train=None, valid=None, test=None) -> Model:

    p = resolve_parameters(params).replace(num_iter=1, cv_index=None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some classes have fewer members than folds")
        design = model_initialize(dataset, p)
    model = data_indexing(design, None, train, valid, test)
    model = data_normalize(model)
    model = graph_construct(model)
    model = param_initialize(model)
    return param_training(model)


def risk_predict(model: Model, data) -> Tuple[np.ndarray, Forward]:

    if model.weight is None or model.norm_mean is None or model.best_epoch == 0:
        raise EMGPNError("risk_predict requires a trained model (see model_fit).")
    model = param_reshape(model)
    out = model_forward(model, data_transform(model, data))
    return out.p, out
