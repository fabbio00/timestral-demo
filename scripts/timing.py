"""Latency of each model in isolation: median of 3 runs, nothing else running.

Two workloads: one turbine (2,016 steps of history, 288 ahead) and a whole KDD Cup
case (134 turbines in one batch). Chronos-2 is timed on CPU too, the fair
comparison with Timestral, which cannot use Apple's MPS.
"""

import statistics
import time
import warnings

import numpy as np
import pandas as pd

from demo.kddcup import OUTPUT_LEN, arima, clean_context, load_case
from demo.models import Chronos2Model, TimestralModel
from demo.paths import RESULTS_DIR

warnings.filterwarnings("ignore")


def median_seconds(model, context, runs=3):
    model.predict(context, OUTPUT_LEN)  # warm-up: weights, kernels, JIT
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(context, OUTPUT_LEN)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    inputs, _ = load_case(1)
    farm = np.stack([clean_context(df) for df in inputs])
    models = [
        ("Timestral-209M", "CPU", TimestralModel("cpu")),
        ("Chronos-2", "GPU (MPS)", Chronos2Model("mps")),
        ("Chronos-2", "CPU", Chronos2Model("cpu")),
        ("AutoARIMA", "CPU, 10 processi", arima()),
    ]
    rows = []
    for name, device, model in models:
        for workload, context in [("1 turbina", farm[:1]), ("134 turbine", farm)]:
            rows.append({"model": name, "device": device, "workload": workload, "seconds": median_seconds(model, context)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "timing.csv", index=False)


if __name__ == "__main__":
    main()
