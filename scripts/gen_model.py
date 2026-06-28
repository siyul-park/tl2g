"""Train tiny LightGBM models and dump fixtures for tl2g tests.

Outputs, for each model <name>:
  testdata/<name>.mdl          -- LightGBM text model (Booster.save_model)
  testdata/<name>_golden.json  -- {"X": [[float]], "pred": [float], "raw": [float]}

`pred` is LightGBM's own prediction (the accuracy oracle). `raw` is the
pre-sigmoid margin so the Go side can validate raw output directly.

Run:  python scripts/gen_model.py
Requires: lightgbm, numpy, scikit-learn
"""

import json
import os

import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression, make_classification

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "testdata")

N_TEST = 200
SEED = 42


def dump(name, booster, X_test):
    os.makedirs(TESTDATA, exist_ok=True)
    mdl_path = os.path.join(TESTDATA, f"{name}.mdl")
    booster.save_model(mdl_path)

    pred = booster.predict(X_test)
    raw = booster.predict(X_test, raw_score=True)
    golden = {
        "X": X_test.tolist(),
        "pred": np.asarray(pred).ravel().tolist(),
        "raw": np.asarray(raw).ravel().tolist(),
        "objective": booster.params.get("objective", ""),
    }
    with open(os.path.join(TESTDATA, f"{name}_golden.json"), "w") as f:
        json.dump(golden, f)
    print(f"{name}: trees={booster.num_trees()} feats={X_test.shape[1]} -> {mdl_path}")


def regression():
    X, y = make_regression(n_samples=2000, n_features=10, noise=0.1, random_state=SEED)
    train = lgb.Dataset(X[:-N_TEST], label=y[:-N_TEST])
    params = {"objective": "regression", "num_leaves": 15, "learning_rate": 0.1,
              "verbose": -1, "seed": SEED}
    booster = lgb.train(params, train, num_boost_round=30)
    dump("regression", booster, X[-N_TEST:])


def binary():
    X, y = make_classification(n_samples=2000, n_features=12, n_informative=6,
                               random_state=SEED)
    train = lgb.Dataset(X[:-N_TEST], label=y[:-N_TEST])
    params = {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1,
              "verbose": -1, "seed": SEED}
    booster = lgb.train(params, train, num_boost_round=30)
    dump("binary", booster, X[-N_TEST:])


if __name__ == "__main__":
    regression()
    binary()
