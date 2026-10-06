from itertools import combinations
from pathlib import Path

import numpy as np

from . import cross_validation, read_csv
from ._io import atomic_open

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT.parent / "dataset" / "sample.csv"
OUT_FOLDER = ROOT / "Result"
BASE = {"num_iter": 10, "num_fold": 5, "max_epoch": 500, "learn_rate": 0.005, "reg_coeff": 1e-4,
        "init_steady": 1.0, "num_neighbor": 10, "verbose": 0}
MAIN_METRIC = ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1"]
N_JOBS = 1


def run(dataset, **changes):
    result, _ = cross_validation(dataset, {**BASE, **changes}, n_jobs=N_JOBS)
    return result


def write_table(path, row_label, row_name, col_name, value):
    import csv
    with atomic_open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([row_label] + list(col_name))
        for name, row in zip(row_name, value):
            writer.writerow([name] + [f"{v:.6f}" for v in row])


def evaluate_settings(dataset, settings):

    rows = []
    for setting in settings:
        result = run(dataset, **setting)
        rows.append([result.metric_value(name) for name in MAIN_METRIC])
    return np.asarray(rows)


def main(output_folder="Result") -> None:
    global OUT_FOLDER
    OUT_FOLDER = Path(output_folder)
    dataset = read_csv(str(DATA_FILE), class_name=["SCD", "MCI", "AD", "VCI", "VD"],
                             modality=["sMRI", "fMRI", "PET"])
    OUT_FOLDER.mkdir(exist_ok=True)

    init_steady = [1e-2, 1e-1, 1.0, 1e1, 1e2]
    reg_coeff = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sens = np.full((5, 5), np.nan)
    for i, phi0 in enumerate(init_steady):
        for j, delta in enumerate(reg_coeff):
            res = run(dataset, init_steady=phi0, reg_coeff=delta)
            sens[i, j] = res.metric_value("AUROC")
            print(f"InitSteady {phi0:g}, RegCoeff {delta:g}: AUROC {sens[i, j]:.4f}")
    print("\nAUROC by initial steadiness (rows) and regularization (columns)")
    print(" " * 12 + "".join(f"{d:>14g}" for d in reg_coeff) + f"{'Average':>14s}")
    for i, phi0 in enumerate(init_steady):
        print(f"{phi0:>12g}" + "".join(f"{v:14.4f}" for v in sens[i]) + f"{sens[i].mean():14.4f}")

    variant_name = ["EMGPN", "w/o feature propagation", "w/o modality integration"]
    variant = [{}, {"propagation": "gcn"}, {"integration": "concat"}]
    variant_metric = evaluate_settings(dataset, variant)
    print("\n" + f"{'Variant':<28s}" + "".join(f"{k:>11s}" for k in MAIN_METRIC))
    for name, row in zip(variant_name, variant_metric):
        print(f"{name:<28s}" + "".join(f"{v:11.4f}" for v in row))

    names = dataset.modality_name
    combo = [list(c) for s in range(1, len(names) + 1) for c in combinations(names, s)]
    combo_metric = evaluate_settings(dataset, [{"modality": c} for c in combo])
    print("\n" + f"{'Modalities':<28s}" + "".join(f"{k:>11s}" for k in MAIN_METRIC))
    for c, row in zip(combo, combo_metric):
        print(f"{'+'.join(c):<28s}" + "".join(f"{v:11.4f}" for v in row))

    write_table(OUT_FOLDER / "Ablation_Sensitivity.csv", "InitSteady", [f"{v:g}" for v in init_steady],
                [f"AUROC_RegCoeff_{d:g}" for d in reg_coeff], sens)
    write_table(OUT_FOLDER / "Ablation_Variant.csv", "Variant", variant_name, MAIN_METRIC, variant_metric)
    write_table(OUT_FOLDER / "Ablation_Modality.csv", "Modalities", ["+".join(c) for c in combo], MAIN_METRIC,
                combo_metric)
    print(f"Results saved in {OUT_FOLDER}")


if __name__ == "__main__":
    main()
