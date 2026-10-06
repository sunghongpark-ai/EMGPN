from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .dataset import Dataset, EMGPNError
from .rng import normal_quantile, random_order, random_stream, uniform
from .config import _integer, _number

__all__ = ["data_simulate", "GROUP_NAME", "CLASS_NAME", "MODALITY_NAME"]


CLASS_NAME = ("SCD", "MCI", "AD", "VCI", "VD")
CLASS_SIZE = (80, 237, 139, 124, 64)
FEMALE_COUNT = (64, 163, 95, 83, 40)
AMYLOID_COUNT = (0, 44, 129, 8, 16)
WMH_COUNT = ((59, 21, 0), (234, 2, 1), (93, 42, 4), (14, 101, 9), (2, 39, 23))
AGE_MEDIAN = (72.0, 71.0, 75.0, 76.0, 77.0)
AGE_IQR = (9.0, 10.0, 10.0, 9.0, 10.0)
CDRSB_MEDIAN = (1.0, 1.5, 4.0, 1.5, 4.5)
CDRSB_IQR = (0.5, 1.5, 3.0, 2.0, 3.5)
IQR_TO_SD = 1.349


MODALITY_NAME = ("sMRI", "fMRI", "PET")
GROUP_NAME = ("VIS", "SMN", "DAN", "VAN", "LIM", "FPN", "DMN", "HIP", "AMY", "BG", "THA", "CBL")
GROUP_SIZE = (30, 36, 26, 24, 12, 26, 46, 4, 4, 12, 16, 27)
NUM_CORTICAL = 7


PATTERN_AMYLOID = (0.00, 0.00, 0.90, 0.85, 0.80, 0.10, 1.20, 0.10, 0.60, 0.85, 0.00, 0.00)
PATTERN_ATROPHY = (0.55, 0.10, 0.05, 0.05, 0.15, 0.05, 0.10, 1.10, 0.20, 0.10, 0.40, 0.05)
PATTERN_VASCULAR = (0.05, 0.80, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.20, 0.90, 0.05)
PATTERN_FUNCTION = (0.05, 0.05, 0.05, 0.20, 0.05, 1.10, 0.15, 0.05, 0.05, 0.05, 0.05, 0.40)
PATTERN_PERFUSION = (0.05, 0.05, 0.05, 0.05, 0.05, 0.50, 0.05, 0.05, 0.05, 0.05, 0.05, 0.30)
SUBTYPE_EFFECT: Dict[Tuple[str, str, str], float] = {
    ("sMRI", "VIS", "SCD"): 0.30, ("sMRI", "SMN", "SCD"): 0.30,
    ("sMRI", "THA", "MCI"): -0.35, ("fMRI", "FPN", "MCI"): -0.35,
    ("sMRI", "HIP", "AD"): -0.25, ("PET", "DMN", "AD"): 0.25,
    ("sMRI", "SMN", "VCI"): -0.35, ("fMRI", "CBL", "VCI"): -0.30,
    ("sMRI", "HIP", "VD"): -0.35, ("fMRI", "FPN", "VD"): -0.30,
}
KAPPA_AMYLOID = 3.0
KAPPA_ATROPHY = 0.9
KAPPA_VASCULAR = 0.6
KAPPA_FUNCTION = 1.0
KAPPA_PERFUSION = 0.6
KAPPA_SUBTYPE = 1.0
KAPPA_AGE = 0.35
KAPPA_SEX = 0.3
NOISE_ROI = (1.0, 1.1, 0.9)
NOISE_GLOBAL = 0.3


def _round(x):

    return np.floor(np.asarray(x, dtype=float) + 0.5)


def _group_count(num_roi: int) -> np.ndarray:

    size = np.asarray(GROUP_SIZE, dtype=float)
    if num_roi == int(size.sum()):
        return size.astype(int)
    if num_roi < len(GROUP_SIZE):
        raise EMGPNError(f"num_roi must be at least {len(GROUP_SIZE)}.")
    quota = (num_roi - len(GROUP_SIZE)) * size / size.sum()
    count = 1.0 + np.floor(quota)
    remainder = quota - np.floor(quota)
    extra = int(num_roi - count.sum())
    count[random_order(-remainder)[:extra]] += 1.0
    return count.astype(int)


def _roi_layout(num_roi: int):

    name, group = [], []
    for g, total in enumerate(_group_count(num_roi)):
        if GROUP_NAME[g] == "CBL" and total % 2 == 1:
            part = (("L", (total - 1) // 2), ("R", (total - 1) // 2), ("V", 1))
        else:
            part = (("L", (total + 1) // 2), ("R", total // 2))
        for hemi, num in part:
            for i in range(num):
                name.append(f"{GROUP_NAME[g]}_{hemi}{i + 1:02d}")
                group.append(g)
    return name, np.asarray(group, dtype=int)


def _class_count(count: float, size: float, n_k: int) -> int:
    return int(_round(count / size * n_k))


def _subtype_table(modality: str, c: int) -> np.ndarray:
    table = np.zeros((len(GROUP_NAME), c))
    for (mod, grp, cls), value in SUBTYPE_EFFECT.items():
        k = CLASS_NAME.index(cls)
        if mod == modality and k < c:
            table[GROUP_NAME.index(grp), k] = value
    return table


def data_simulate(num_roi: int = 263, class_size: Sequence[int] = CLASS_SIZE, seed: int = 2025,
                  missing_rate: Sequence[float] = (0.0, 0.03, 0.02), dropout_rate: float = 0.02,
                  signal_scale: float = 1.0, noise_scale: float = 1.0) -> Dataset:


    try:
        class_size = tuple(_integer(v, "class_size", 1) for v in class_size)
        missing_rate = tuple(_number(v, "missing_rate", lower=0) for v in missing_rate)
    except TypeError as err:
        raise EMGPNError("class_size and missing_rate must be numeric sequences.") from err
    c = len(class_size)
    if not 2 <= c <= len(CLASS_NAME) or min(class_size) < 1:
        raise EMGPNError("class_size must hold 2 to 5 positive class sizes.")
    if len(missing_rate) != len(MODALITY_NAME):
        raise EMGPNError("missing_rate must have one value per modality.")
    dropout_rate = _number(dropout_rate, "dropout_rate", lower=0)
    if any(v > 1 for v in missing_rate) or dropout_rate > 1:
        raise EMGPNError("missing_rate and dropout_rate must lie in [0, 1].")
    if missing_rate[1] + dropout_rate > 1:
        raise EMGPNError("fMRI missing_rate + dropout_rate must be <= 1.")
    signal_scale = _number(signal_scale, "signal_scale", lower=0)
    noise_scale = _number(noise_scale, "noise_scale", lower=0)
    num_roi = _integer(num_roi, "num_roi", len(GROUP_SIZE))
    n = sum(class_size)
    if int(_round(missing_rate[1] * n)) + int(_round(dropout_rate * n)) > n:
        raise EMGPNError("Rounded fMRI missing and dropout counts exceed the cohort size.")
    roi_name, group = _roi_layout(num_roi)
    r, num_group = len(roi_name), len(GROUP_NAME)
    stream = random_stream(seed)

    y = np.repeat(np.arange(c), class_size)[random_order(uniform(stream, 1, n)[0])]
    female, amyloid, wmh = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(c):
        member = np.flatnonzero(y == k)
        n_k, size_k = member.size, CLASS_SIZE[k]
        perm = member[random_order(uniform(stream, 1, n_k)[0])]
        female[perm[:_class_count(FEMALE_COUNT[k], size_k, n_k)]] = 1.0
        perm = member[random_order(uniform(stream, 1, n_k)[0])]
        amyloid[perm[:_class_count(AMYLOID_COUNT[k], size_k, n_k)]] = 1.0
        perm = member[random_order(uniform(stream, 1, n_k)[0])]
        mild = _class_count(WMH_COUNT[k][0], size_k, n_k)
        moderate = min(_class_count(WMH_COUNT[k][1], size_k, n_k), n_k - mild)
        wmh[perm[mild:mild + moderate]] = 1.0
        wmh[perm[mild + moderate:]] = 2.0
    z = normal_quantile(uniform(stream, 1, n)[0])
    age = _round(np.clip(np.asarray(AGE_MEDIAN)[y] + np.asarray(AGE_IQR)[y] / IQR_TO_SD * z, 50.0, 95.0))
    z = normal_quantile(uniform(stream, 1, n)[0])
    severity = np.clip(np.asarray(CDRSB_MEDIAN)[y] + np.asarray(CDRSB_IQR)[y] / IQR_TO_SD * z, 0.0, 18.0)
    z = normal_quantile(uniform(stream, 1, n)[0])
    burden = amyloid * np.maximum(1.0 + 0.25 * z, 0.3)
    sev_c = (severity - 1.5) / 2.5
    drive = sev_c * (0.5 + 0.5 * amyloid)
    age_c = (age - 74.0) / 7.0

    gi = group[:, None]
    cortical = (group < NUM_CORTICAL)[:, None]
    cerebellum = (group == GROUP_NAME.index("CBL"))[:, None]
    kappa = float(signal_scale)
    xdata = []
    information = np.zeros((r, len(MODALITY_NAME)))
    for m, modality in enumerate(MODALITY_NAME):
        z_base = normal_quantile(uniform(stream, r, 1))
        z_jit = normal_quantile(uniform(stream, r, 1))
        u_load = uniform(stream, r, 1)
        z_net = normal_quantile(uniform(stream, num_group, n))
        z_glob = normal_quantile(uniform(stream, 1, n))
        z_roi = normal_quantile(uniform(stream, r, n))
        jitter = np.maximum(1.0 + 0.2 * z_jit, 0.0)
        load = 0.5 + 0.3 * u_load
        subtype = ((kappa * KAPPA_SUBTYPE) * _subtype_table(modality, c)[gi[:, 0]][:, y]) * jitter
        if modality == "sMRI":
            base = np.minimum(np.maximum(0.55 + 0.07 * z_base, 0.30), 0.80)
            scale = 0.09 * base
            w_a = (kappa * KAPPA_ATROPHY) * np.asarray(PATTERN_ATROPHY)[gi] * jitter
            w_v = (kappa * KAPPA_VASCULAR) * np.asarray(PATTERN_VASCULAR)[gi] * jitter
            effect = (subtype - w_a * drive - w_v * wmh - (kappa * KAPPA_AGE) * age_c
                      - (kappa * KAPPA_SEX) * female)
        elif modality == "fMRI":
            base = np.minimum(np.maximum(0.30 + 0.025 * z_base, 0.18), 0.42)
            scale = np.full((r, 1), 0.03)
            w_f = (kappa * KAPPA_FUNCTION) * np.asarray(PATTERN_FUNCTION)[gi] * jitter
            w_p = (kappa * KAPPA_PERFUSION) * np.asarray(PATTERN_PERFUSION)[gi] * jitter
            effect = subtype - w_f * sev_c - w_p * wmh
        else:
            mean = np.where(cortical, 1.15, np.where(cerebellum, 1.00, 1.35))
            spread = np.where(cortical, 0.05, np.where(cerebellum, 0.01, 0.08))
            base = mean + spread * z_base
            scale = np.where(cerebellum, 0.015, 0.06)
            w_b = (kappa * KAPPA_AMYLOID) * np.asarray(PATTERN_AMYLOID)[gi] * jitter
            effect = subtype + w_b * burden
        tau = noise_scale * NOISE_ROI[m]
        glob = noise_scale * NOISE_GLOBAL
        noise = load * z_net[group, :] + glob * z_glob + tau * z_roi
        xdata.append(base + scale * (effect + noise))
        information[:, m] = _information(effect, y, c, load[:, 0] ** 2 + glob ** 2 + tau ** 2)

    for m in range(len(MODALITY_NAME)):
        if missing_rate[m] > 0:
            miss = random_order(uniform(stream, 1, n)[0])[:int(_round(missing_rate[m] * n))]
            xdata[m][:, miss] = np.nan
    if dropout_rate > 0:
        f = MODALITY_NAME.index("fMRI")
        order = random_order(uniform(stream, 1, n)[0])
        order = order[~np.isnan(xdata[f][0, order])]
        drop = order[:int(_round(dropout_rate * n))]
        xdata[f][np.ix_(group == GROUP_NAME.index("LIM"), drop)] = np.nan

    truth = {"Group": group, "GroupName": list(GROUP_NAME), "Amyloid": amyloid, "Burden": burden,
             "WmhGrade": wmh, "Severity": severity, "Age": age, "Female": female,
             "Information": information, "TrueModality": np.argmax(information, axis=1)}
    return Dataset(xdata=xdata, ydata=y.astype(int), modality_name=list(MODALITY_NAME),
                   class_name=list(CLASS_NAME[:c]), roi_name=roi_name,
                   subject_id=[f"S{j + 1:04d}" for j in range(n)],
                   covariate={"Age": age.astype(float), "Sex": np.where(female > 0, "F", "M").astype(object)},
                   truth=truth,
                   description=(f"Synthetic EMGPN sample (DataSimulate, seed {seed}): {r} ROIs, {n} participants, "
                                f"{c} subtypes, {len(MODALITY_NAME)} modalities"))


def _information(effect: np.ndarray, y: np.ndarray, c: int, noise_var: np.ndarray) -> np.ndarray:

    n = effect.shape[1]
    grand = effect.mean(axis=1)
    between = np.zeros(effect.shape[0])
    within = np.zeros(effect.shape[0])
    for k in range(c):
        e_k = effect[:, y == k]
        if e_k.shape[1] == 0:
            continue
        mu_k = e_k.mean(axis=1)
        between += e_k.shape[1] * (mu_k - grand) ** 2
        within += ((e_k - mu_k[:, None]) ** 2).sum(axis=1)
    return (between / n) / (between / n + within / n + noise_var)
