from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._io import atomic_open

__all__ = ["Dataset", "read_csv", "write_csv", "EMGPNError"]

MISSING_TOKEN = frozenset({"", "NA", "NAN", "N/A", "NULL"})


class EMGPNError(ValueError):
    pass


def _names(values, name, count=None):

    result = list(values)
    if (count is not None and len(result) != count) or any(not isinstance(v, str) or not v.strip() for v in result) \
            or len(set(result)) != len(result):
        raise EMGPNError(f"{name} must contain distinct nonempty names" +
                         (f" ({count} entries)." if count is not None else "."))
    return result


def _real_array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "biuf":
            raise EMGPNError(f"{name} must be a real numeric array.")
        return raw.astype(float, copy=False)
    except (TypeError, ValueError) as err:
        raise EMGPNError(f"{name} must be a real numeric array.") from err


def _labels(value, n, c=None):
    y = _real_array(value, "ydata")
    if y.shape != (n,) or n == 0 or not np.all(np.isfinite(y)) or np.any(y != np.round(y)) \
            or np.any(y < 0) or np.any(y >= np.iinfo(np.intp).max) or (c is not None and np.any(y >= c)):
        raise EMGPNError(f"ydata must hold {n} finite class indices 0..c-1.")
    return y.astype(int)


@dataclass
class Dataset:


    xdata: List[np.ndarray]
    ydata: Optional[np.ndarray]
    modality_name: List[str]
    class_name: List[str]
    roi_name: List[str]
    subject_id: List[str]
    covariate: Dict[str, np.ndarray] = field(default_factory=dict)
    wdata: Optional[List[np.ndarray]] = None
    truth: Optional[dict] = None
    description: str = ""
    source: str = ""

    @property
    def num_subj(self) -> int:
        return int(self.xdata[0].shape[1])

    @property
    def num_roi(self) -> int:
        return int(self.xdata[0].shape[0])

    def subset(self, index: Sequence[int]) -> "Dataset":

        idx = np.asarray(index, dtype=int)
        return Dataset(
            xdata=[x[:, idx] for x in self.xdata],
            ydata=None if self.ydata is None else self.ydata[idx],
            modality_name=list(self.modality_name), class_name=list(self.class_name),
            roi_name=list(self.roi_name), subject_id=[self.subject_id[j] for j in idx],
            covariate={k: v[idx] for k, v in self.covariate.items()}, wdata=self.wdata,
            truth=None, description=self.description, source=self.source)


def _split_name(name: str):
    pos = name.find("_")
    if pos <= 0 or pos == len(name) - 1:
        return None, None
    return name[:pos], name[pos + 1:]


def _infer_modality(names: Sequence[str]) -> List[str]:
    prefix_order: List[str] = []
    suffix: Dict[str, List[str]] = {}
    for name in names:
        pre, suf = _split_name(name)
        if pre is None:
            continue
        if pre not in suffix:
            prefix_order.append(pre)
            suffix[pre] = []
        suffix[pre].append(suf)
    candidate = [p for p in prefix_order if len(suffix[p]) >= 2]
    if not candidate:
        raise EMGPNError("No imaging columns found; name them <Modality>_<ROI>, e.g. sMRI_ROI001.")
    keys = [tuple(sorted(suffix[p])) for p in candidate]
    best = None
    for key in keys:
        score = (keys.count(key), len(key))
        if best is None or score > best[0]:
            best = (score, key)
    modality = [p for p, key in zip(candidate, keys) if key == best[1]]
    ignored = [p for p, key in zip(candidate, keys) if key != best[1]]
    if ignored:
        warnings.warn("Column groups " + ", ".join(ignored) + " do not share the ROI set of "
                      + ", ".join(modality) + " and are kept as covariates.", stacklevel=3)
    return modality


def _to_float(text: np.ndarray, column: Sequence[str], row_offset: int = 2) -> np.ndarray:
    token = np.char.upper(np.char.strip(text.astype(str)))
    missing = np.isin(token, list(MISSING_TOKEN))
    value = np.full(text.shape, np.nan)
    try:
        value[~missing] = token[~missing].astype(float)
    except ValueError:
        for i, j in zip(*np.nonzero(~missing)):
            try:
                float(token[i, j])
            except ValueError:
                raise EMGPNError(f"Non-numeric value '{text[i, j]}' in row {i + row_offset}, "
                                 f"column {column[j]}.") from None
    if np.isinf(value).any():
        raise EMGPNError("Imaging values must be finite; leave missing values empty.")
    return value


def _parse_label(raw: Sequence[str], class_name: Optional[Sequence[str]]):
    raw = [s.strip() for s in raw]
    if any(s.upper() in MISSING_TOKEN for s in raw):
        raise EMGPNError("The label column has missing values.")
    if class_name is not None:
        class_name = _names(class_name, "class_name")
        if all(s in class_name for s in raw):
            return np.array([class_name.index(s) for s in raw]), class_name
        try:
            code = np.array([float(s) for s in raw])
        except ValueError:
            code = None
        if code is not None and np.all(code == np.round(code)) and code.min() >= 1 \
                and code.max() <= len(class_name):
            return code.astype(int) - 1, class_name
        unknown = sorted(set(raw) - set(class_name))
        raise EMGPNError("Labels not found in ClassName: " + ", ".join(unknown[:10]))
    try:
        numeric = np.array([float(s) for s in raw])
    except ValueError:
        numeric = None
    if numeric is not None:
        if not np.all(np.isfinite(numeric)):
            raise EMGPNError("Numeric diagnosis labels must be finite.")
        level = np.unique(numeric)
        names = [f"{v:g}" for v in level]
        return np.searchsorted(level, numeric), names
    level = sorted(set(raw))
    return np.array([level.index(s) for s in raw]), level


def read_csv(path: str, id_column: Optional[str] = "SubjectID", label_column: Optional[str] = "Diagnosis",
             class_name: Optional[Sequence[str]] = None, modality: Optional[Sequence[str]] = None,
             delimiter: str = ",") -> Dataset:


    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.reader(handle, delimiter=delimiter, strict=True) if any(c.strip() for c in row)]
    except csv.Error as err:
        raise EMGPNError(f"Invalid CSV in {path}: {err}") from err
    if len(rows) < 2:
        raise EMGPNError(f"{path} needs a header and at least one participant row.")
    header = [h.strip() for h in rows[0]]
    if len(set(header)) != len(header):
        raise EMGPNError("Column names must be unique.")
    if any(not h for h in header):
        raise EMGPNError("Column names must be nonempty.")
    body = rows[1:]
    for i, row in enumerate(body):
        if len(row) != len(header):
            raise EMGPNError(f"Row {i + 2} has {len(row)} fields; the header has {len(header)}.")
    table = np.array(body, dtype=object)
    n = table.shape[0]

    def column_index(name, role, required):
        if name is None or name == "":
            return None
        if name in header:
            return header.index(name)
        if required:
            raise EMGPNError(f"{role} column '{name}' not found.")
        return None

    id_col = column_index(id_column, "ID", False)
    label_col = column_index(label_column, "Label", True)
    if id_col is not None and id_col == label_col:
        raise EMGPNError("ID and label columns must be different.")
    reserved = {c for c in (id_col, label_col) if c is not None}
    free = [j for j in range(len(header)) if j not in reserved]
    if modality is None:
        modality = _infer_modality([header[j] for j in free])
    modality = _names(modality, "modality")
    if not modality:
        raise EMGPNError("At least one modality is required.")
    column_of: Dict[str, List[int]] = {}
    for m in modality:
        cols = [j for j in free if _split_name(header[j])[0] == m]
        if not cols:
            raise EMGPNError(f"No columns for modality {m}.")
        column_of[m] = cols
    roi_name = [_split_name(header[j])[1] for j in column_of[modality[0]]]
    xdata = []
    for m in modality:
        suffix = [_split_name(header[j])[1] for j in column_of[m]]
        if sorted(suffix) != sorted(roi_name) or len(set(suffix)) != len(suffix):
            raise EMGPNError(f"Modality {m} does not have the ROIs of {modality[0]}.")
        pos = {s: j for s, j in zip(suffix, column_of[m])}
        cols = [pos[s] for s in roi_name]
        xdata.append(_to_float(table[:, cols], [header[j] for j in cols]).T.copy())
    used = reserved | {j for m in modality for j in column_of[m]}
    covariate: Dict[str, np.ndarray] = {}
    for j in range(len(header)):
        if j in used:
            continue
        raw = np.array([str(v).strip() for v in table[:, j]], dtype=object)
        try:
            covariate[header[j]] = _to_float(table[:, [j]], [header[j]])[:, 0]
        except EMGPNError:
            covariate[header[j]] = raw
    subject_id = [str(v).strip() for v in table[:, id_col]] if id_col is not None \
        else [f"P{j + 1:04d}" for j in range(n)]
    subject_id = _names(subject_id, "subject_id", n)
    if label_col is not None:
        ydata, names = _parse_label([str(v) for v in table[:, label_col]], class_name)
    else:
        ydata, names = None, _names(class_name, "class_name") if class_name is not None else []
    return Dataset(xdata=xdata, ydata=ydata, modality_name=modality, class_name=list(names),
                   roi_name=roi_name, subject_id=subject_id, covariate=covariate,
                   description=f"Read from {os.path.basename(path)}", source=os.path.abspath(path))


def _format_value(v: float, precision: int) -> str:
    return "" if np.isnan(v) else f"{v:.{precision}f}"


def _format_column(values: np.ndarray, precision: int) -> List[str]:
    if values.dtype.kind in "fiu":
        v = values.astype(float)
        finite = v[np.isfinite(v)]
        if finite.size and np.all(finite == np.round(finite)):
            return ["" if np.isnan(x) else f"{int(x)}" for x in v]
        return [_format_value(x, precision) for x in v]
    return [str(x) for x in values]


def write_csv(dataset: Dataset, path: str, precision: int = 4, id_column: str = "SubjectID",
              label_column: str = "Diagnosis") -> None:

    if isinstance(precision, bool) or not isinstance(precision, (int, np.integer)) or not 0 <= precision <= 17:
        raise EMGPNError("precision must be an integer in 0..17.")
    if not isinstance(dataset, Dataset) or not dataset.xdata:
        raise EMGPNError("dataset must contain at least one imaging modality.")
    xdata = [_real_array(x, "xdata") for x in dataset.xdata]
    if xdata[0].ndim != 2:
        raise EMGPNError("xdata must be ROI-by-participant matrices.")
    r, n = xdata[0].shape
    if n == 0 or any(x.shape != (r, n) or np.isinf(x).any() for x in xdata):
        raise EMGPNError("All imaging matrices must share a nonempty shape and contain no Inf.")
    _names(dataset.subject_id, "subject_id", n)
    _names(dataset.modality_name, "modality_name", len(xdata))
    _names(dataset.roi_name, "roi_name", r)
    if dataset.ydata is not None:
        _names(dataset.class_name, "class_name")
        labels = _labels(dataset.ydata, n, len(dataset.class_name))
    header = [id_column]
    columns = [list(dataset.subject_id)]
    if dataset.ydata is not None:
        header.append(label_column)
        columns.append([dataset.class_name[k] for k in labels])
    for name, values in dataset.covariate.items():
        if np.asarray(values).shape != (n,):
            raise EMGPNError(f"Covariate {name} must hold {n} values.")
        header.append(name)
        columns.append(_format_column(np.asarray(values), precision))
    fmt = f"%.{precision}f"
    for m, modality in enumerate(dataset.modality_name):
        x = xdata[m]
        text = np.char.mod(fmt, x)
        text[np.isnan(x)] = ""
        for q, roi in enumerate(dataset.roi_name):
            header.append(f"{modality}_{roi}")
            columns.append(text[q].tolist())
    _names(header, "CSV header")
    with atomic_open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=",", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(zip(*columns))
