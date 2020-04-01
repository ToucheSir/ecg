import json
import os
from typing import List

import tensorflow as tf
import numpy as np
import scipy.io as sio
import tqdm

INPUT_SHAPE = ([None], [None])
STEP = 256


def data_generator(
    batch_size: int,
    preprocessor: "Preprocessor",
    ecgs: List[np.ndarray],
    labels: List[List[str]],
    default_signal: float = 0.0,
    default_label: str = "~",
    prefetch=10,
):
    ds = tf.data.Dataset.from_generator(
        lambda: zip(ecgs, labels), (tf.float32, tf.dtypes.string), INPUT_SHAPE
    )
    ds = ds.repeat().padded_batch(
        batch_size,
        INPUT_SHAPE,
        padding_values=(default_signal, default_label),
        drop_remainder=True,
    )
    return ds.map(preprocessor.process).prefetch(prefetch)


class Preprocessor:
    def __init__(self, ecgs: np.ndarray, labels: List[List[str]]):
        self.mean, self.std = compute_mean_std(ecgs)
        self.classes = sorted(set(l for label in labels for l in label))
        class_index = tf.range(len(self.classes))
        # self.int_to_class = dict(enumerate(self.classes))
        self.int_to_class = create_lookup(class_index, self.classes, "")
        # self.class_to_int = {c: i for i, c in self.int_to_class.items()}
        self.class_to_int = create_lookup(self.classes, class_index, -1)

    def process(self, x, y):
        x = tf.expand_dims((x - self.mean) / self.std, -1)
        y = tf.one_hot(self.class_to_int.lookup(y), len(self.classes))
        return x, y

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["int_to_class"]
        del state["class_to_int"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        class_index = tf.range(len(self.classes))
        self.int_to_class = create_lookup(class_index, self.classes, "")
        self.class_to_int = create_lookup(self.classes, class_index, -1)


def create_lookup(keys, values, default):
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values)),
        default,
    )


def compute_mean_std(x):
    x = np.hstack(x)
    return np.mean(x).astype(np.float32), np.std(x).astype(np.float32)


def load_dataset(data_json: str, step=STEP):
    with open(data_json, "r") as fid:
        data = [json.loads(l) for l in fid]
    labels = []
    ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(np.array(d["labels"]))
        ecgs.append(load_ecg(d["ecg"], step))
    return ecgs, labels


def load_ecg(record: str, step: int):
    ext = os.path.splitext(record)[1]
    if ext == ".npy":
        ecg = np.load(record)
    elif ext == ".mat":
        ecg = sio.loadmat(record)["val"][0]
    else:  # Assumes binary 16 bit integers
        with open(record, "r") as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = len(ecg) // step * step
    return ecg[:trunc_samp]


if __name__ == "__main__":
    train = load_dataset("examples/cinc17/train.json")
    preproc = Preprocessor(*train)
    print(preproc.classes)
    gen = data_generator(32, preproc, *train)
    for x, y in gen.take(10):
        print(x.shape, y.shape)
