"""Benchmark LightGBM vs TL2cgen prediction on the test fixtures.

Prints per-row latency (ns/row) for batch prediction so the numbers line up
with the Go `go test -bench` output for tl2g. TL2cgen is optional; if treelite
or tl2cgen is missing it is skipped.

Run:  DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python scripts/bench_compare.py
Requires: lightgbm, numpy; optional: treelite, tl2cgen
"""

import json
import os
import tempfile
import time

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "testdata")

REPEAT = 50  # batch repetitions for timing


def load_X(name):
    with open(os.path.join(TESTDATA, f"{name}_golden.json")) as f:
        return np.asarray(json.load(f)["X"], dtype=np.float64)


def bench_lightgbm(name, X):
    booster = lgb.Booster(model_file=os.path.join(TESTDATA, f"{name}.mdl"))
    booster.predict(X)  # warmup
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        booster.predict(X)
        best = min(best, time.perf_counter() - t0)
    return best / len(X) * 1e9


def bench_tl2cgen(name, X):
    import treelite
    import tl2cgen

    mdl = os.path.join(TESTDATA, f"{name}.mdl")
    try:
        model = treelite.Model.load(mdl, model_format="lightgbm")
    except Exception:
        model = treelite.frontend.load_lightgbm_model(mdl)

    with tempfile.TemporaryDirectory() as d:
        libpath = None
        for toolchain in ("gcc", "clang", "cc"):
            cand = os.path.join(d, f"{name}.so")
            try:
                tl2cgen.export_lib(model, toolchain=toolchain, libpath=cand)
                libpath = cand
                break
            except Exception:
                continue
        if not libpath or not os.path.exists(libpath):
            raise RuntimeError("no working toolchain for tl2cgen")

        predictor = tl2cgen.Predictor(libpath)
        dmat = tl2cgen.DMatrix(X)
        predictor.predict(dmat)  # warmup
        best = float("inf")
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            predictor.predict(dmat)
            best = min(best, time.perf_counter() - t0)
    return best / len(X) * 1e9


def main():
    print(f"{'model':<12}{'runtime':<12}{'ns/row':>12}")
    print("-" * 36)
    for name in ("regression", "binary"):
        X = load_X(name)
        print(f"{name:<12}{'lightgbm':<12}{bench_lightgbm(name, X):>12.1f}")
        try:
            print(f"{name:<12}{'tl2cgen':<12}{bench_tl2cgen(name, X):>12.1f}")
        except Exception as e:
            print(f"{name:<12}{'tl2cgen':<12}{'skipped':>12}  ({e})")


if __name__ == "__main__":
    main()
