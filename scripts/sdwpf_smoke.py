"""Smoke test on the KDD Cup 2022 final-phase test set (SDWPF wind farm).

Each case gives 14 days of 10-min history for 134 turbines; the task is the
next 48 h of active power. Models see only each turbine's own power history
(no weather), zero-shot. Reference scores from the competition: winner 44.92,
GRU baseline 47.85 (lower is better).
"""

import argparse

import numpy as np
import pandas as pd

from demo.kddcup import N_CASES, OUTPUT_LEN, clean_context, load_case, score_case
from demo.models import QUANTILES, Chronos2Model, TimestralModel, timed_predict
from demo.paths import RESULTS_DIR


class Persistence:
    """Last valid observation carried forward: the simplest operational baseline."""

    name = "Persistence"

    def predict(self, context, horizon, known_covariates=None):
        last = pd.DataFrame(context.T).ffill().iloc[-1].fillna(0).values
        point = np.repeat(last[:, None], horizon, axis=1)
        return np.repeat(point[..., None], len(QUANTILES), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10, help=f"number of test cases (max {N_CASES})")
    args = parser.parse_args()

    models = [Persistence(), TimestralModel(), Chronos2Model()]
    cases = np.linspace(1, N_CASES, args.cases).round().astype(int)
    rows = []
    for idx in cases:
        inputs, truths = load_case(idx)
        context = np.stack([clean_context(df) for df in inputs])
        for model in models:
            q, seconds = timed_predict(model, context, OUTPUT_LEN)
            median = np.clip(q[..., QUANTILES.index(0.5)], 0, None)
            case_mae, case_rmse = score_case(median, truths)
            rows.append({"case": idx, "model": model.name, "MAE": case_mae, "RMSE": case_rmse, "seconds": seconds})
        print(pd.DataFrame(rows[-len(models):]).round(2).to_string(index=False, header=idx == cases[0]))

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "sdwpf_smoke.csv", index=False)
    summary = df.groupby("model")[["MAE", "RMSE", "seconds"]].mean()
    summary["score"] = (summary.MAE + summary.RMSE) / 2
    print(f"\nKDD Cup score on {len(cases)}/{N_CASES} cases (winner 44.92, GRU baseline 47.85 on all 142):")
    print(summary.sort_values("score").round(2).to_string())


if __name__ == "__main__":
    main()
