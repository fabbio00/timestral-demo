"""AutoARIMA on a subset of the KDD Cup 2022 final-phase cases.

One AutoARIMA per turbine and case costs ~0.15 s on the last 3 days of history,
so all 142 cases (19k fits) would take ~50 min; the 20 evenly spaced cases in
kddcup.ARIMA_CASES keep the run under 10 min. The page compares ARIMA with the
other models on these same cases, read from results/sdwpf_smoke.csv.

Forecasts of every case run are cached for rescoring and the frontend export;
--showcase-only refreshes just the page's cases and leaves the metrics untouched.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from demo import cache
from demo.kddcup import ARIMA_CASES, OUTPUT_LEN, SHOWCASE_CASES, arima, case_row, clean_context, load_case
from demo.models import QUANTILES
from demo.paths import RESULTS_DIR

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--showcase-only", action="store_true", help="only refresh the cached showcase forecasts")
    args = parser.parse_args()

    model = arima()
    rows, forecasts = [], {}
    for idx in SHOWCASE_CASES if args.showcase_only else ARIMA_CASES:
        inputs, truths = load_case(idx)
        context = np.stack([clean_context(df) for df in inputs])
        start = time.perf_counter()
        q = model.predict(context, OUTPUT_LEN)
        seconds = time.perf_counter() - start
        forecasts[str(idx)] = q
        rows.append(case_row(idx, model.name, q[..., QUANTILES.index(0.5)], truths, seconds))
        print(rows[-1], flush=True)

    cache.save("sdwpf", model.name, list(forecasts), np.stack(list(forecasts.values())))
    if args.showcase_only:
        return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "sdwpf_arima.csv", index=False)
    others = pd.read_csv(RESULTS_DIR / "sdwpf_smoke.csv")
    same = pd.concat([others[others.case.isin(df.case)], df])
    summary = same.groupby("model")[["MAE", "RMSE", "seconds"]].mean()
    summary["score"] = (summary.MAE + summary.RMSE) / 2
    print(f"\nKDD Cup score on the same {len(df)} cases:")
    print(summary.sort_values("score").round(2).to_string())


if __name__ == "__main__":
    main()
