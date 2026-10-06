from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse

from .dataset import EMGPNError, _real_array
from .config import _integer, _positive

__all__ = ["knn_graph", "graph_laplacian"]


def knn_graph(x: np.ndarray, num_neighbor: int, sigma: Optional[float] = None
              ) -> Tuple[np.ndarray, float, np.ndarray]:


    x = _real_array(x, "x")
    if x.ndim != 2 or x.shape[1] == 0:
        raise EMGPNError("Graph construction requires a nonempty ROI-by-participant matrix.")
    r = x.shape[0]
    if r < 2:
        raise EMGPNError("A graph needs at least two nodes.")
    if not np.all(np.isfinite(x)):
        raise EMGPNError("Graph construction requires finite (imputed) data.")
    k = min(_integer(num_neighbor, "num_neighbor", 1), r - 1)
    if sigma is not None:
        sigma = _positive(sigma, "sigma")
    xc = x - x.mean(axis=0, keepdims=True)
    sq = np.sum(xc * xc, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (xc @ xc.T)
    d2 = np.maximum((d2 + d2.T) / 2.0, 0.0)
    if not np.all(np.isfinite(d2)):
        raise EMGPNError("Pairwise graph distances overflowed; rescale the data.")
    np.fill_diagonal(d2, np.inf)
    neighbor = np.argsort(d2, axis=1, kind="stable")[:, :k]
    directed = np.zeros((r, r), dtype=bool)
    directed[np.repeat(np.arange(r), k), neighbor.ravel()] = True
    if sigma is None:
        sigma = float(np.mean(np.sqrt(d2.T[directed.T])))
        if not sigma > 0:
            sigma = 1.0
    edge = directed | directed.T
    w = np.zeros((r, r))
    w[edge] = np.exp(-d2[edge] / sigma ** 2)
    return w, float(sigma), neighbor


def graph_laplacian(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:


    is_sparse = sparse.issparse(w)
    if is_sparse:
        if w.dtype.kind not in "biuf":
            raise EMGPNError("w must be real numeric.")
        w = sparse.csc_matrix(w, dtype=float)
        values = w.data
    else:
        w = _real_array(w, "w")
        values = w
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] == 0 \
            or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise EMGPNError("w must be a finite nonnegative square matrix.")
    difference = w - w.T
    if (is_sparse and difference.nnz and np.max(np.abs(difference.data)) > 1e-12) \
            or (not is_sparse and not np.allclose(w, w.T, rtol=0, atol=1e-12)):
        raise EMGPNError("w must be symmetric; graph_construct symmetrizes user graphs.")
    r = w.shape[0]
    if is_sparse:
        degree = np.asarray(w.sum(axis=1)).ravel()
        if not np.all(np.isfinite(degree)):
            raise EMGPNError("Graph degrees overflowed; rescale weights.")
        s = np.zeros(r)
        s[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
        identity = sparse.eye(r, format="csc")
        edges = w.tocoo()
        normalized = sparse.csc_matrix((edges.data * (s[edges.row] * s[edges.col]),
                                        (edges.row, edges.col)), shape=w.shape)
        lap = identity - normalized
        wt = w + identity
        st = 1.0 / np.sqrt(degree + 1.0)
        edges = wt.tocoo()
        ahat = sparse.csc_matrix((edges.data * (st[edges.row] * st[edges.col]),
                                 (edges.row, edges.col)), shape=w.shape)
        return lap.tocsc(), ahat
    degree = w.sum(axis=1)
    if not np.all(np.isfinite(degree)):
        raise EMGPNError("Graph degrees overflowed; rescale weights.")
    s = np.zeros(r)
    s[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    lap = np.eye(r) - w * np.outer(s, s)
    wt = w + np.eye(r)
    st = 1.0 / np.sqrt(wt.sum(axis=1))
    return lap, wt * np.outer(st, st)
