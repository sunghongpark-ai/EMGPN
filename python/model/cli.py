from __future__ import annotations

import argparse
import ast
import csv
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .crossval import cross_validation
from .dataset import read_csv, write_csv
from .model import model_fit, risk_predict
from .report import result_export, result_report, result_visualize
from .simulate import data_simulate
from .dataset import EMGPNError
from .model import Model, param_reshape
from ._io import atomic_open

__all__ = ["main", "save_model", "load_model"]


def save_model(model, path: str) -> None:

    if not isinstance(model, Model) or model.best_epoch == 0:
        raise EMGPNError("save_model requires a trained EMGPN Model.")
    with atomic_open(path, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str):

    with open(path, "rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, Model) or model.best_epoch == 0:
        raise EMGPNError("The file does not contain a trained EMGPN Model.")
    return param_reshape(model)


def _parse_params(items: Optional[List[str]]) -> dict:
    params = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise EMGPNError(f"--param expects key=value, got '{item}'")
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


def _read(args, labeled: bool = True):
    return read_csv(args.data, id_column=args.id_column, label_column=args.label_column if labeled else None,
                    class_name=args.class_name, modality=args.modality)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python main.py", description="EMGPN (Park et al., Neural Networks, 2025)")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "dataset" / "sample.csv"), help="wide CSV file (one row per participant)")
        p.add_argument("--id-column", default="SubjectID")
        p.add_argument("--label-column", default="Diagnosis")
        p.add_argument("--class-name", nargs="+", default=["SCD", "MCI", "AD", "VCI", "VD"], help="class order, e.g. SCD MCI AD VCI VD")
        p.add_argument("--modality", nargs="+", default=None, help="modality prefixes, e.g. sMRI fMRI PET")

    cv = sub.add_parser("cv", help="repeated stratified cross-validation, report, CSV tables and figures")
    data_args(cv)
    cv.add_argument("--out", default=None)
    cv.add_argument("--iter", type=int, default=1, help="repetitions (paper: 100)")
    cv.add_argument("--fold", type=int, default=5)
    cv.add_argument("--seed", type=int, default=0)
    cv.add_argument("--jobs", type=int, default=1, help="parallel processes")
    cv.add_argument("--param", nargs="*", help="further parameters key=value, e.g. max_epoch=300")
    cv.add_argument("--figure", action="store_true")
    cv.add_argument("--no-figure", action="store_true")
    cv.add_argument("--final-model", default=None, help="also train on all participants and pickle the model here")

    sim = sub.add_parser("simulate", help="write the synthetic sample dataset")
    sim.add_argument("--out", required=True)
    sim.add_argument("--seed", type=int, default=2025)

    pred = sub.add_parser("predict", help="subtype risks of new participants with a pickled model")
    data_args(pred)
    pred.add_argument("--model", required=True)
    pred.add_argument("--out", required=True)

    items = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(items or ["cv"])
    if args.command == "simulate":
        write_csv(data_simulate(seed=args.seed), args.out)
        print(f"Sample written to {args.out}")
        return 0
    if args.command == "cv":
        dataset = _read(args)
        params = {"num_iter": args.iter, "num_fold": args.fold, "seed": args.seed, "max_epoch": 20, "num_permute": 2, **_parse_params(args.param)}
        result, _ = cross_validation(dataset, params, n_jobs=args.jobs)
        result_report(result)
        files = result_export(result, args.out) if args.out else []
        if args.figure and not args.no_figure:
            try:
                result_visualize(result, args.out)
            except ImportError:
                print("matplotlib not installed: figures skipped", file=sys.stderr)
        print("Saved: " + ", ".join(files))
        if args.final_model:
            final = {**params, "max_epoch": max(1, int(np.floor(result.info["MeanBestEpoch"] + 0.5)))}
            model = model_fit(dataset, final, np.arange(dataset.num_subj))
            save_model(model, args.final_model)
            print(f"Final model saved to {args.final_model}")
        return 0
    model = load_model(args.model)
    dataset = _read(args, labeled=False)
    prob, _ = risk_predict(model, dataset)
    with atomic_open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        names = model.design.class_name
        writer.writerow(["SubjectID"] + [f"Risk_{c}" for c in names] + ["Predicted"])
        for j, sid in enumerate(dataset.subject_id):
            writer.writerow([sid] + [f"{v:.6f}" for v in prob[:, j]] + [names[int(np.argmax(prob[:, j]))]])
    print(f"Risks of {dataset.num_subj} participants written to {args.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:

    try:
        return _main(argv)
    except (EMGPNError, OSError, pickle.UnpicklingError, EOFError) as err:
        print(f"emgpn: {err}", file=sys.stderr)
        return 2
