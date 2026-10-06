from __future__ import annotations

import numpy as np
from numbers import Real

__all__ = ["random_stream", "uniform", "random_order", "normal_quantile"]

_DEFAULT_SEED = 5489
_TINY = 2.0 ** -53


def random_stream(seed: int) -> np.random.RandomState:

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Real) or not np.isfinite(seed) \
            or seed != int(seed) or seed < 0 or seed >= 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    seed = int(seed)
    return np.random.RandomState(_DEFAULT_SEED if seed == 0 else seed)


def uniform(stream: np.random.RandomState, rows: int, cols: int) -> np.ndarray:

    for dimension in (rows, cols):
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Real) \
                or not np.isfinite(dimension) or dimension != int(dimension) or dimension < 0:
            raise ValueError("rows and cols must be nonnegative integers")
    values = stream.random_sample(int(rows) * int(cols))
    return values.reshape((int(rows), int(cols)), order="F")


def random_order(u: np.ndarray, axis: int = -1) -> np.ndarray:

    return np.argsort(u, axis=axis, kind="stable")


_A = (3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
      1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
      3.3430575583588128105e+4, 2.5090809287301226727e+3)
_B = (1.0, 4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
      2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4,
      5.2264952788528545610e+3)
_C = (1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
      3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
      2.27238449892691845833e-2, 7.74545014278341407640e-4)
_D = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
      1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
      1.05075007164441684324e-9)
_E = (6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
      2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
      2.71155556874348757815e-5, 2.01033439929228813265e-7)
_F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
      7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
      2.04426310338993978564e-15)


def _horner(coef: tuple, x: np.ndarray) -> np.ndarray:
    y = np.full_like(x, coef[7])
    for k in range(6, -1, -1):
        y = y * x + coef[k]
    return y


def normal_quantile(u: np.ndarray) -> np.ndarray:

    raw = np.asarray(u)
    if raw.dtype.kind not in "biuf" or not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("u must contain finite real probabilities in [0, 1]")
    p = np.clip(raw.astype(float), _TINY, 1.0 - _TINY)
    q = p - 0.5
    z = np.empty_like(p)
    central = np.abs(q) <= 0.425
    r = 0.180625 - q[central] * q[central]
    z[central] = q[central] * _horner(_A, r) / _horner(_B, r)
    tail = ~central
    qt = q[tail]
    r = np.sqrt(-np.log(np.where(qt < 0, p[tail], 1.0 - p[tail])))
    zt = np.empty_like(r)
    near = r <= 5.0
    rn = r[near] - 1.6
    zt[near] = _horner(_C, rn) / _horner(_D, rn)
    rf = r[~near] - 5.0
    zt[~near] = _horner(_E, rf) / _horner(_F, rf)
    zt[qt < 0] = -zt[qt < 0]
    z[tail] = zt
    return z
