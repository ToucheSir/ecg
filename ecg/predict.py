import argparse
import os

from tensorflow.keras.models import load_model

from . import util, load


def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs = predict(args.data_json, args.model_path)
    print(probs)
