from __future__ import annotations

import os
import csv
from typing import List, Optional

import numpy as np

from .crossval import CVResult
from ._io import atomic_open

__all__ = ["result_report", "result_export", "result_visualize"]

MODALITY_COLOR = ((0.91, 0.51, 0.37), (0.23, 0.61, 0.50), (0.36, 0.48, 0.56))


def result_report(result: CVResult, num_top: int = 10) -> str:

    out: List[str] = ["", "==================== EMGPN cross-validation ===================="]
    info = result.info
    out.append(f"{info['Version']} | {info['Software']}")
    out.append(f"Repetitions: {result.metric.shape[0]} | models: {info['NumModel']} | "
               f"mean best epoch: {info['MeanBestEpoch']:.1f}")
    out.append("")
    out.append("Performance (mean +/- SD over repetitions)")
    for name, mean, sd in zip(result.metric_name, result.metric_mean, result.metric_std):
        out.append(f"  {name:<17s} {mean:.4f} (+/- {sd:.4f})")
    ex = result.explain
    if ex is not None:
        count = ex["DominantCount"]
        out.append("")
        out.append("Modality importance (whole-brain mean; #ROIs dominated)")
        for m, name in enumerate(ex["ModalityName"]):
            out.append(f"  {name:<8s} {ex['ModalityImportanceMean'][m]:.4f}   {count[m]:4d} ROIs "
                       f"({100.0 * count[m] / max(count.sum(), 1):.1f}%)")
        out.append("")
        out.append("Risk effects (mean; #positive / #negative ROIs)")
        for k, name in enumerate(ex["ClassName"]):
            out.append(f"  {name:<8s} {ex['RiskEffectMean'][k]:+.4f}   {ex['RiskPositiveCount'][k]:4d} / "
                       f"{ex['RiskNegativeCount'][k]:4d}")
        out.append("")
        out.append(f"Key regions: {len(ex['KeyRoi'])} ROIs (mean -log10 P: key {ex['GroupLogPKey']:.2f} "
                   f"vs rest {ex['GroupLogPRest']:.2f})")
        for i, q in enumerate(ex["KeyRoi"][:num_top]):
            out.append(f"  {i + 1:2d}. {ex['RoiName'][q]:<24s} combined {ex['CombinedImportance'][q]:.3f}")
        out.append("Subtype-specific key regions (increase the subtype probability)")
        for k, name in enumerate(ex["ClassName"]):
            out.append(f"  {name:<8s} {len(ex['SubtypeKeyRoi'][k]):3d} ROIs")
    out.append("================================================================")
    text = "\n".join(out)
    print(text)
    return text


def _fmt(value, spec: str) -> str:

    value = float(value)
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return format(value, spec)


def _quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\r\n') else s


def result_export(result: CVResult, folder: str = "Result") -> List[str]:

    os.makedirs(folder, exist_ok=True)
    files = []
    name = os.path.join(folder, "Performance_Repetition.csv")
    with atomic_open(name, "w", encoding="utf-8", newline="") as f:
        f.write("Repetition," + ",".join(result.metric_name) + "\n")
        for i, row in enumerate(result.metric):
            f.write(f"{i + 1}" + "".join("," + _fmt(v, ".6f") for v in row) + "\n")
    files.append(name)
    name = os.path.join(folder, "Performance_Summary.csv")
    with atomic_open(name, "w", encoding="utf-8", newline="") as f:
        f.write("Metric,Mean,SD\n")
        for k, metric in enumerate(result.metric_name):
            f.write(f"{metric},{_fmt(result.metric_mean[k], '.6f')},{_fmt(result.metric_std[k], '.6f')}\n")
    files.append(name)
    name = os.path.join(folder, "Risk_OutOfFold.csv")
    pred = np.argmax(result.risk_mean, axis=0)
    with atomic_open(name, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["SubjectID", "Diagnosis"] + [f"Risk_{c}" for c in result.class_name] + ["Predicted"])
        for j, sid in enumerate(result.subject_id):
            writer.writerow([sid, result.class_name[result.ydata[j]]]
                            + [_fmt(v, ".6f") for v in result.risk_mean[:, j]] + [result.class_name[pred[j]]])
    files.append(name)
    ex = result.explain
    if ex is not None:
        name = os.path.join(folder, "Explain_ROI.csv")
        header = ["Index", "ROI"] + [f"Lambda_{m}" for m in ex["ModalityName"]] + ["DominantModality"]
        if ex["Steadiness"] is not None:
            header += [f"Phi_{m}" for m in ex["ModalityName"]]
        header += [f"Theta_{c}" for c in ex["ClassName"]]
        header += ["FeatureImportance", "PermutationImportance", "CombinedImportance", "IsKey"]
        header += [f"ProbChange_{c}" for c in ex["ClassName"]] + ["GroupPValue", "GroupLogP"]
        with atomic_open(name, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(header)
            for q, roi in enumerate(ex["RoiName"]):
                dom = ex["DominantModality"][q]
                line = [q + 1, roi] + [_fmt(v, ".6f") for v in ex["ModalityImportance"][q]]
                line += ["NA" if np.isnan(dom) else ex["ModalityName"][int(dom)]]
                if ex["Steadiness"] is not None:
                    line += [_fmt(v, ".6f") for v in ex["Steadiness"][q]]
                line += [_fmt(v, ".6f") for v in ex["RiskEffect"][q]]
                line += [_fmt(ex['FeatureImportance'][q], '.6g'), _fmt(ex['PermutationImportance'][q], '.6g'),
                         _fmt(ex['CombinedImportance'][q], '.6f'), int(ex['IsKey'][q])]
                line += [_fmt(v, ".4f") for v in ex["ProbChange"][q]]
                line += [_fmt(ex['GroupPValue'][q], '.6g'), _fmt(ex['GroupLogP'][q], '.4f')]
                writer.writerow(line)
        files.append(name)
    return files


def _curve(score: np.ndarray, label: np.ndarray):
    order = np.argsort(-score, kind="stable")
    srt, hit = score[order], label[order] > 0
    tp, fp = np.cumsum(hit), np.cumsum(~hit)
    last = np.concatenate([srt[:-1] != srt[1:], [True]])
    tp, fp = tp[last], fp[last]
    tpr = np.concatenate([[0.0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0.0], fp / max(fp[-1], 1)])
    prec = np.concatenate([[1.0], tp / (tp + fp)])
    return fpr, tpr, tpr.copy(), prec


def result_visualize(result: CVResult, folder: Optional[str] = None, show: bool = False):


    import matplotlib
    if not show:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figs = [_plot_performance(result, plt)]
    if result.explain is not None:
        figs += [_plot_modality(result.explain, plt), _plot_risk(result.explain, plt),
                 _plot_key(result.explain, plt)]
    if folder:
        os.makedirs(folder, exist_ok=True)
        names = ["Fig_Performance", "Fig_ModalityImportance", "Fig_RiskEffect", "Fig_KeyRegion"]
        for fig, name in zip(figs, names):
            with atomic_open(os.path.join(folder, name + ".png"), "wb") as handle:
                fig.savefig(handle, format="png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)
    return figs


def _plot_performance(result: CVResult, plt):
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    c, n = result.risk_mean.shape
    label = np.zeros((c, n))
    label[result.ydata, np.arange(n)] = 1.0
    score = np.concatenate([p.ravel(order="F") for p in result.risk_test])
    lab = np.tile(label.ravel(order="F"), len(result.risk_test))
    fpr, tpr, rec, prec = _curve(score, lab)
    idx = {name: k for k, name in enumerate(result.metric_name)}
    ax[0].plot(fpr, tpr, color="0.2", lw=1.5)
    ax[0].plot([0, 1], [0, 1], ":", color="0.6")
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate",
              title=f"ROC (micro)  AUROC macro = {result.metric_mean[idx['AUROC']]:.4f}")
    ax[1].plot(rec, prec, color="0.2", lw=1.5)
    ax[1].plot([0, 1], [1.0 / c, 1.0 / c], ":", color="0.6")
    ax[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
              title=f"PR (micro)  AUPRC macro = {result.metric_mean[idx['AUPRC']]:.4f}")
    main = ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1"]
    k = [idx[m] for m in main]
    ax[2].bar(range(6), result.metric_mean[k], 0.6, color="0.35", yerr=result.metric_std[k], capsize=3)
    ax[2].set_xticks(range(6))
    ax[2].set_xticklabels(["AUROC", "AUPRC", "Acc.", "Prec.", "Rec.", "F1"])
    ax[2].set(ylim=(0, 1), title="Mean (+/- SD) over repetitions")
    for a in ax[:2]:
        a.set_aspect("equal")
    fig.tight_layout()
    return fig


def _colors(m: int):
    base = list(MODALITY_COLOR)
    if m <= 3:
        return base[:m]
    import matplotlib.cm as cm
    return base + [cm.tab10(i) for i in range(m - 3)]


def _plot_modality(ex, plt):
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    lp = np.asarray(ex["ModalityImportance"])
    r, m = lp.shape
    color = _colors(m)
    box = ax[0].boxplot([lp[:, i] for i in range(m)], patch_artist=True, showmeans=True)
    for patch, col in zip(box["boxes"], color):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax[0].set_xticks(range(1, m + 1))
    ax[0].set_xticklabels(ex["ModalityName"])
    ax[0].set(ylabel="Modality importance", title="Overall comparison")
    order = np.argsort(lp[:, -1], kind="stable")
    ax[1].stackplot(np.arange(1, r + 1), lp[order].T, colors=color, labels=ex["ModalityName"])
    ax[1].set(xlim=(1, r), ylim=(0, 1), xlabel=f"ROIs sorted by importance of {ex['ModalityName'][-1]}",
              ylabel="Cumulative importance", title="Individual comparison")
    count = np.asarray(ex["DominantCount"])
    keep = np.flatnonzero(count > 0)
    if keep.size == 0:
        ax[2].axis("off")
        ax[2].text(0.5, 0.5, "No dominant modality", ha="center")
    else:
        labels = [f"{ex['ModalityName'][i]}: {count[i]} ({100.0 * count[i] / count.sum():.1f}%)" for i in keep]
        ax[2].pie(count[keep], labels=labels, colors=[color[i] for i in keep])
    ax[2].set_title("# Significant ROIs")
    fig.tight_layout()
    return fig


def _plot_risk(ex, plt):
    theta = np.asarray(ex["RiskEffect"])
    c = theta.shape[1]
    fig, ax = plt.subplots(1, c, figsize=(2.6 * c, 3.0), squeeze=False)
    bound = float(np.max(np.abs(theta))) or 1.0
    edge = np.linspace(-bound, bound, 41)
    for k in range(c):
        a = ax[0, k]
        cnt, _ = np.histogram(theta[:, k], bins=edge)
        center = (edge[:-1] + edge[1:]) / 2
        neg = center < 0
        width = edge[1] - edge[0]
        a.bar(center[neg], cnt[neg], width, color=(0.27, 0.55, 0.85))
        a.bar(center[~neg], cnt[~neg], width, color=(0.90, 0.25, 0.36))
        a.axvline(0, color="k", ls=":")
        a.set(xlim=(-bound, bound), title=f"{ex['ClassName'][k]} (avg {ex['RiskEffectMean'][k]:+.4f})",
              xlabel=f"Theta ({ex['RiskNegativeCount'][k]} neg / {ex['RiskPositiveCount'][k]} pos)")
        if k == 0:
            a.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def _plot_key(ex, plt):
    c = len(ex["ClassName"])
    col = max(3, c)
    fig, ax = plt.subplots(2, col, figsize=(13, 5.6), squeeze=False)
    fin, pin, key = ex["FeatureImportanceNorm"], ex["PermutationImportanceNorm"], ex["IsKey"]
    a = ax[0, 0]
    a.scatter(fin[~key], pin[~key], 16, color="0.75")
    a.scatter(fin[key], pin[key], 24, color="0.25")
    a.axvline(fin.mean(), color="k", ls="--")
    a.axhline(pin.mean(), color="k", ls="--")
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="Feature importance", ylabel="Permutation importance",
          title=f"{int(key.sum())} key ROIs")
    a = ax[0, 1]
    a.bar([1, 2], [ex["GroupLogPKey"], ex["GroupLogPRest"]], 0.6, color="0.3")
    a.set_xticks([1, 2])
    a.set_xticklabels(["Key", "Rest"])
    a.set(ylabel="-log10 P", title="Group comparison")
    a = ax[0, 2]
    keyroi = np.asarray(ex["KeyRoi"], dtype=int)
    if keyroi.size == 0:
        a.axis("off")
        a.text(0.5, 0.5, "No key ROI", ha="center")
    else:
        top = keyroi[np.argsort(ex["CombinedImportance"][keyroi], kind="stable")][-15:]
        a.barh(np.arange(top.size), ex["CombinedImportance"][top], color="0.3")
        a.set_yticks(np.arange(top.size))
        a.set_yticklabels([ex["RoiName"][q] for q in top], fontsize=7)
        a.set(xlabel="Combined importance", title="Top key ROIs")
    for j in range(3, col):
        ax[0, j].axis("off")
    import matplotlib.cm as cm
    for k in range(c):
        a = ax[1, k]
        x = ex["CombinedImportance"][keyroi]
        v = ex["ProbChange"][keyroi, k]
        up = v > 0
        color = cm.tab10(k % 10)
        a.scatter(x[up], v[up], 24, color=color)
        a.scatter(x[~up], v[~up], 24, facecolors="none", edgecolors=color)
        a.axhline(0, color="k", ls=":")
        a.set(xlim=(0, 1), xlabel="Combined importance", title=f"{ex['ClassName'][k]}: {int(up.sum())} ROIs")
        if k == 0:
            a.set_ylabel("Prob. change (%)")
    for j in range(c, col):
        ax[1, j].axis("off")
    fig.tight_layout()
    return fig
