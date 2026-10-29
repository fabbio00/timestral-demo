"""Smoke test on the KDD Cup 2022 final-phase test set (SDWPF wind farm).

Each case gives 14 days of 10-min history for 134 turbines; the task is the
next 48 h of active power. Models see only each turbine's own power history
(no weather), zero-shot. Reference scores from the competition: winner 44.92,
GRU baseline 47.85 (lower is better).

Forecasts of every case run are cached (10/50/90% quantiles), so metrics can be
recomputed and the page rebuilt without re-running models; --showcase-only
refreshes just the page's cases and leaves the metrics untouched.
"""

import argparse

import numpy as np
import pandas as pd

from demo import cache
from demo.kddcup import N_CASES, OUTPUT_LEN, SHOWCASE_CASES, case_row, clean_context, load_case
from demo.models import QUANTILES, Chronos2Model, Persistence, TimestralModel, timed_predict
from demo.paths import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10, help=f"number of test cases (max {N_CASES})")
    parser.add_argument("--showcase-only", action="store_true", help="only refresh the cached showcase forecasts")
    args = parser.parse_args()

    models = [Persistence(), TimestralModel(), Chronos2Model()]
    cases = SHOWCASE_CASES if args.showcase_only else np.linspace(1, N_CASES, args.cases).round().astype(int)
    rows, forecasts = [], {m.name: {} for m in models}
    for idx in cases:
        inputs, truths = load_case(idx)
        context = np.stack([clean_context(df) for df in inputs])
        for model in models:
            q, seconds = timed_predict(model, context, OUTPUT_LEN)
            forecasts[model.name][str(idx)] = q
            rows.append(case_row(idx, model.name, q[..., QUANTILES.index(0.5)], truths, seconds))
        print(pd.DataFrame(rows[-len(models):]).round(2).to_string(index=False, header=idx == cases[0]), flush=True)

    for name, by_case in forecasts.items():
        cache.save("sdwpf", name, list(by_case), np.stack(list(by_case.values())))
    if args.showcase_only:
        return
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "sdwpf_smoke.csv", index=False)
    summary = df.groupby("model")[["MAE", "RMSE", "seconds"]].mean()
    summary["score"] = (summary.MAE + summary.RMSE) / 2
    print(f"\nKDD Cup score on {len(cases)}/{N_CASES} cases (winner 44.92, GRU baseline 47.85 on all 142):")
    print(summary.sort_values("score").round(2).to_string())


if __name__ == "__main__":
    main()
