import json
import random
import re
from pathlib import Path

import tqdm
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def ecg_signal_length(ecg_file: str):
    return sio.loadmat(ecg_file)["val"].shape[1]


DX_PATTERN = re.compile(r"#Dx: ([\w-]+)", re.MULTILINE)


def load_all(path: Path, step: int):
    dataset = []
    stratify_labels = []

    for record in tqdm.tqdm(list(path.glob("*.hea"))):
        with record.open() as header_file:
            dx = DX_PATTERN.search(header_file.read()).group(1)
            label = dx[0] if "AF" in dx or "Normal" in dx else "O"
            stratify_labels.append(label)
        ecg_file = str(record.with_suffix(".mat").absolute())
        num_labels = ecg_signal_length(ecg_file) // step
        dataset.append((ecg_file, [label] * num_labels))

    return dataset, stratify_labels


def make_json(save_path, dataset):
    with open(save_path, "w") as fid:
        for d in dataset:
            json.dump({"ecg": d[0], "labels": d[1]}, fid)
            fid.write("\n")


STEP = 512

if __name__ == "__main__":
    random.seed(2018)
    np.random.seed(2020)

    with open("config.json") as f:
        config = json.load(f)

    data_path = Path("data")
    data, strat_labels = load_all(data_path, config.get("step", STEP))
    train, test, strat_labels, _ = train_test_split(
        data, strat_labels, test_size=0.2, stratify=strat_labels
    )
    train, dev = train_test_split(train, test_size=0.1, stratify=strat_labels)

    make_json("train.json", train)
    make_json("dev.json", dev)
    make_json("test.json", test)
