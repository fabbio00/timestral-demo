"""Smoke test on BDG2 (ASHRAE GEPIII data): hourly electricity of industrial-type buildings.

Rolling origins across 2017 (the GEPIII test year), 48 h horizon, context capped
at 4096 h for both models. Air temperature is passed as a known covariate using
observed values, i.e. an oracle weather forecast: fine for a smoke test, to be
replaced by real forecasts or perturbed values in the demo.
"""

import argparse

import numpy as np
import pandas as pd

from demo.metrics import coverage, mae, wql
from demo.models import Chronos2Model, SeasonalNaive, TimestralModel, timed_predict
from demo.paths import RAW_DIR, RESULTS_DIR

INDUSTRIAL_USAGES = ["Manufacturing/industrial", "Utility", "Warehouse/storage", "Technology/science"]
# These BDG2 sites ship in Salesforce/GiftEvalPretrain, which Chronos-2 (and likely
# Timestral) trained on; excluding them keeps the comparison zero-shot.
PRETRAIN_SITES = ["Bear", "Fox", "Panther", "Rat"]
HORIZON = 48
CONTEXT = 4096
SEASON = 168


def load():
    raw = RAW_DIR / "bdg2"
    meta = pd.read_csv(raw / "metadata_metadata.csv")
    meta = meta[
        (meta.electricity == "Yes")
        & meta.primaryspaceusage.isin(INDUSTRIAL_USAGES)
        & ~meta.site_id.isin(PRETRAIN_SITES)
    ]
    elec = pd.read_csv(raw / "meters_cleaned_electricity_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp")
    elec = elec[[b for b in meta.building_id if b in elec.columns]]
    # Keep buildings that are mostly observed, otherwise the comparison measures gap handling.
    elec = elec.loc[:, elec.notna().mean() > 0.95]
    weather = pd.read_csv(raw / "weather_weather.csv", parse_dates=["timestamp"])
    temp = weather.pivot_table(index="timestamp", columns="site_id", values="airTemperature").reindex(elec.index)
    temp = temp.interpolate(limit_direction="both")
    site = meta.set_index("building_id").site_id
    return elec, temp, site


def windows(elec, temp, site, origins):
    ctx, cov, y, ids = [], [], [], []
    for origin in origins:
        t = elec.index.get_loc(origin)
        for b in elec.columns:
            target = elec[b].values[t : t + HORIZON]
            if np.isnan(target).any():
                continue
            ctx.append(elec[b].values[t - CONTEXT : t])
            cov.append(temp[site[b]].values[t - CONTEXT : t + HORIZON][None, :])
            y.append(target)
            ids.append((b, origin))
    return np.array(ctx), np.array(cov), np.array(y), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    args = parser.parse_args()

    elec, temp, site = load()
    origins = pd.date_range("2017-02-06", "2017-12-04", freq="5W-MON")
    ctx, cov, y, ids = windows(elec, temp, site, origins)
    print(f"{elec.shape[1]} buildings, {len(origins)} origins, {len(y)} windows of {HORIZON} h")

    models = [SeasonalNaive(SEASON), TimestralModel(args.device), Chronos2Model(args.device)]
    rows = []
    for model in models:
        for use_cov in ([False, True] if not isinstance(model, SeasonalNaive) else [False]):
            q, seconds = timed_predict(model, ctx, HORIZON, cov if use_cov else None)
            rows.append({
                "model": model.name + (" + temp" if use_cov else ""),
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
