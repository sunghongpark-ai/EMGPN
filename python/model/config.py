from __future__ import annotations

import math
from numbers import Real
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional, Sequence, Union

from .dataset import EMGPNError

__all__ = ["Parameters", "resolve_parameters"]

_CHOICE = {
    "optimizer": ("adam", "gd"),
    "init_classifier": ("glorot", "zeros"),
    "grad_mode": ("exact", "published", "legacy"),
    "normalize": ("zscore-logistic", "zscore", "none"),
    "propagation": ("gpn", "gcn", "none"),
    "integration": ("softmax", "mean", "concat"),
    "perm_metric": ("loss", "auroc"),
    "prob_baseline": ("zero", "mean"),
    "prob_subset": ("class", "all"),
}


@dataclass(frozen=True)
class Parameters:


    num_iter: int = 100
    num_fold: int = 5
    use_valid: bool = True
    seed: int = 0
    cv_index: Optional[Any] = None
    max_epoch: int = 500
    patience: float = math.inf
    optimizer: str = "adam"
    learn_rate: float = 0.005
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    reg_coeff: float = 1e-4
    init_steady: float = 1.0
    init_integrate: float = 0.0
    init_classifier: str = "glorot"
    grad_mode: str = "exact"
    phi_min: float = 1e-6
    num_neighbor: int = 10
    kernel_width: Optional[Union[float, Sequence[float]]] = None
    normalize: str = "zscore-logistic"
    modality: Optional[Sequence[Union[int, str]]] = None
    propagation: str = "gpn"
    integration: str = "softmax"
    use_bias: bool = False
    num_permute: int = 10
    perm_metric: str = "loss"
    prob_baseline: str = "zero"
    prob_subset: str = "class"
    verbose: int = 1

    def replace(self, **changes) -> "Parameters":
        return resolve_parameters({**asdict(self), **changes})

    def as_dict(self) -> dict:
        return asdict(self)


_FIELD = {f.name.replace("_", ""): f.name for f in fields(Parameters)}


def _integer(value, name, lower):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) \
            or value != int(value) or value < lower:
        raise EMGPNError(f"{name} must be an integer >= {lower}.")
    return int(value)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise EMGPNError(f"{name} must be a positive finite number.")
    return float(value)


def _number(value, name, lower=None, upper=None, allow_inf=False):
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value) \
            or (not allow_inf and not math.isfinite(value)) \
            or (lower is not None and value < lower) or (upper is not None and value >= upper):
        raise EMGPNError(f"{name} must be a real number in the documented range.")
    return float(value)


def _boolean(value, name):
    
    if isinstance(value, bool) or (type(value).__name__ == "bool_" and getattr(value, "ndim", 0) == 0):
        return bool(value)
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    raise EMGPNError(f"{name} must be a scalar boolean or 0/1.")


def resolve_parameters(params: Optional[Union[Parameters, Mapping[str, Any]]] = None, **kwargs) -> Parameters:


    if isinstance(params, Parameters):
        params = asdict(params)
    if params is not None and not isinstance(params, Mapping):
        raise EMGPNError("params must be a Parameters instance or a mapping.")
    merged = dict(params or {})
    merged.update(kwargs)
    value = {}
    for key, val in merged.items():
        name = _FIELD.get(str(key).replace("_", "").lower())
        if name is None:
            raise EMGPNError(f"Unknown parameter '{key}'. Valid names: {', '.join(sorted(_FIELD.values()))}.")
        value[name] = val
    p = {f.name: f.default for f in fields(Parameters)}
    p.update(value)
    p["num_iter"] = _integer(p["num_iter"], "num_iter", 1)
    p["num_fold"] = _integer(p["num_fold"], "num_fold", 2)
    p["max_epoch"] = _integer(p["max_epoch"], "max_epoch", 1)
    p["num_neighbor"] = _integer(p["num_neighbor"], "num_neighbor", 1)
    p["num_permute"] = _integer(p["num_permute"], "num_permute", 1)
    p["seed"] = _integer(p["seed"], "seed", 0)
    if p["seed"] >= 2 ** 32:
        raise EMGPNError("seed must be smaller than 2**32.")
    p["verbose"] = _integer(p["verbose"], "verbose", 0)
    for name in ("learn_rate", "init_steady", "adam_epsilon", "phi_min"):
        p[name] = _positive(p[name], name)
    if p["init_steady"] < p["phi_min"]:
        raise EMGPNError("init_steady must be >= phi_min.")
    p["patience"] = _number(p["patience"], "patience", lower=1, allow_inf=True)
    if math.isfinite(p["patience"]) and p["patience"] != int(p["patience"]):
        raise EMGPNError("patience must be an integer >= 1 or inf.")
    p["reg_coeff"] = _number(p["reg_coeff"], "reg_coeff", lower=0)
    for name in ("adam_beta1", "adam_beta2"):
        p[name] = _number(p[name], name, lower=0, upper=1)
    p["init_integrate"] = _number(p["init_integrate"], "init_integrate")
    if p["kernel_width"] is not None:
        raw = p["kernel_width"]
        try:
            kw = [_positive(v, "kernel_width") for v in (raw if hasattr(raw, "__len__") else [raw])]
        except (TypeError, ValueError) as err:
            raise EMGPNError("kernel_width must be None or positive real value(s).") from err
        if not kw:
            raise EMGPNError("kernel_width must be None or positive value(s).")
        p["kernel_width"] = tuple(kw)
    if p["modality"] is not None:
        mod = p["modality"]
        try:
            p["modality"] = tuple([mod] if isinstance(mod, (str, Real)) else mod)
        except TypeError as err:
            raise EMGPNError("modality must be names or integer indices.") from err
    p["use_valid"] = _boolean(p["use_valid"], "use_valid")
    p["use_bias"] = _boolean(p["use_bias"], "use_bias")
    p["learn_rate"] = float(p["learn_rate"])
    p["reg_coeff"] = float(p["reg_coeff"])
    p["init_steady"] = float(p["init_steady"])
    p["init_integrate"] = float(p["init_integrate"])
    for name, allowed in _CHOICE.items():
        val = str(p[name]).lower()
        if val not in allowed:
            raise EMGPNError(f"{name} must be one of: {', '.join(allowed)}.")
        p[name] = val
    return Parameters(**p)
