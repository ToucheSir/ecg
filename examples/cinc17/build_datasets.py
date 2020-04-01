import json
import random
import os

import tqdm
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)["val"].squeeze()


def load_all(path, label_file, step):
    with open(os.path.join(path, label_file), "r") as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] // step
        dataset.append((ecg_file, [label] * num_labels))
    return dataset


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

    data = load_all("data/training2017", "../REFERENCE-v3.csv", config.get("step", STEP))
    train, dev = train_test_split(data, test_size=0.1)
    make_json("train.json", train)
    make_json("dev.json", dev)

    test = load_all(
        "data/sample2017/validation/",
        "../answers.txt",
        config.get("step", STEP),
    )
    make_json("test.json", test)
