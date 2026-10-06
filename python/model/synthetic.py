from pathlib import Path

from .dataset import write_csv
from .simulate import data_simulate


def generate_sample(folder=None, seed=20261006):
    folder = Path(folder) if folder is not None else Path(__file__).resolve().parents[2] / "dataset"
    data = data_simulate(seed=seed)
    data.subject_id = [f"SYN_EMGPN_{index + 1:04d}" for index in range(data.num_subj)]
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "sample.csv"
    write_csv(data, file)
    return file
