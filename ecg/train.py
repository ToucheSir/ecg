import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from tensorflow.keras.callbacks import (
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from . import network, load, util

MAX_EPOCHS = 100


def make_save_dir(root_dir: Path, experiment_name: str):
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = root_dir / experiment_name / start_time
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


file_template = "-".join(
    [
        "{val_loss:.4f}",
        "{val_accuracy:.4f}",
        "{epoch:04d}",
        "{loss:.4f}",
        "{accuracy:.4f}.hdf5",
    ]
)


def get_filename_for_saving(save_dir: Path):
    return str(save_dir / file_template)


def train(args: Any, params: dict):
    step = params.get("step", 256)
    print("Loading training set...")
    train = load.load_dataset(params["train"], step)
    print("Loading dev set...")
    dev = load.load_dataset(params["dev"], step)

    print("Building preprocessor...")
    preprocessor = load.Preprocessor(*train)
    print("Training size:", len(train[0]), "examples.")
    print("Dev size:", len(dev[0]), "examples.")

    save_dir = make_save_dir(Path(params["save_dir"]), args.experiment)
    util.save(preprocessor, save_dir)

    params.update(
        {"input_shape": [None, 1], "num_categories": len(preprocessor.classes)}
    )

    model = network.build_network(**params)
    # model.summary()

    callbacks = [
        TensorBoard(Path("logs/fit") / save_dir.name, histogram_freq=1),
        ModelCheckpoint(get_filename_for_saving(save_dir), save_best_only=True),
        ReduceLROnPlateau(
            factor=0.1, patience=2, min_lr=params["learning_rate"] * 0.001
        ),
        EarlyStopping(patience=8),
    ]
    batch_size = params.get("batch_size", 32)

    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preprocessor, *train)
        dev_gen = load.data_generator(batch_size, preprocessor, *dev)
        model.fit(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=callbacks,
        )
    else:
        train_x, train_y = preprocessor.process(*train)
        dev_x, dev_y = preprocessor.process(*dev)
        model.fit(
            train_x,
            train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=callbacks,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument(
        "--experiment", "-e", help="tag with experiment name", default="default"
    )
    args = parser.parse_args()
    with open(args.config_file, "r") as config_file:
        train(args, json.load(config_file))
