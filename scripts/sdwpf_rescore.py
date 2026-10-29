"""Recomputes the wind results CSVs from cached forecasts, without running any model.

Use after a change to the scorer: rows keep the runtimes measured by the original
runs (sdwpf_smoke.py, sdwpf_arima.py) and get fresh official and strict scores.
"""

import numpy as np
import pandas as pd

from demo import cache
from demo.kddcup import case_row, load_case
from demo.paths import RESULTS_DIR


def main():
    for name in ("sdwpf_smoke.csv", "sdwpf_arima.csv"):
        old = pd.read_csv(RESULTS_DIR / name)
        stored = {m: cache.load("sdwpf", m) for m in old.model.unique()}
        rows = []
        for idx, group in old.groupby("case"):
            _, truths = load_case(idx)
            for _, r in group.iterrows():
                median = stored[r.model][str(idx)][..., 1]  # cached bands: 10/50/90%
                rows.append(case_row(idx, r.model, median, truths, r.seconds))
        new = pd.DataFrame(rows)
        new.to_csv(RESULTS_DIR / name, index=False)
        summary = new.groupby("model").agg(official=("MAE", "mean"), rmse=("RMSE", "mean"), strict=("MAE_strict", "mean"),
                                           rmse_strict=("RMSE_strict", "mean"), valid=("valid", "mean"))
        summary["official"] = (summary.official + summary.rmse) / 2
        summary["strict"] = (summary.strict + summary.rmse_strict) / 2
        print(f"{name}: {new.case.nunique()} cases")
        print(summary[["official", "strict", "valid"]].round(2).to_string(), "\n")


if __name__ == "__main__":
    main()
