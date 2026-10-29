"""Smoke test on BDG2 (ASHRAE GEPIII data): hourly electricity of industrial-type buildings.

Rolling origins across 2017 (the GEPIII test year), 48 h horizon, context capped
at 4096 h for both models, with and without air temperature as known covariate.
"""

import argparse

import pandas as pd

from demo import cache
from demo.bdg2 import HORIZON, ORIGINS, SEASON, arima, load, window_key, windows
from demo.metrics import coverage, mae, wql
from demo.models import Chronos2Model, SeasonalAverage, SeasonalNaive, TimestralModel, timed_predict
from demo.paths import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    args = parser.parse_args()

    elec, temp, meta = load()
    ctx, cov, y, ids = windows(elec, temp, meta)
    print(f"{elec.shape[1]} buildings, {len(ORIGINS)} origins, {len(y)} windows of {HORIZON} h")

    models = [SeasonalNaive(SEASON), SeasonalAverage(SEASON), TimestralModel(args.device), Chronos2Model(args.device), arima()]
    rows = []
    for model in models:
        for use_cov in ([False, True] if not isinstance(model, (SeasonalNaive, SeasonalAverage)) else [False]):
            q, seconds = timed_predict(model, ctx, HORIZON, cov if use_cov else None)
            name = model.name + (" + temp" if use_cov else "")
            cache.save("bdg2", name, [window_key(b, o) for b, o in ids], q)
            rows.append({
                "model": name,
                "MAE_kWh": mae(y, q),
                "WQL": wql(y, q),
                "cov_80": coverage(y, q),
                "seconds": seconds,
            })
            print(rows[-1])

    table = pd.DataFrame(rows)
    naive = table.loc[table.model == "Seasonal naive"].iloc[0]
    table["MAE_skill_%"] = 100 * (1 - table.MAE_kWh / naive.MAE_kWh)
    table["WQL_skill_%"] = 100 * (1 - table.WQL / naive.WQL)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "bdg2_smoke.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
