"""BDG2 (ASHRAE GEPIII data) loading: hourly electricity of industrial-type buildings.

Air temperature is the known covariate, taken from observed values, i.e. an
oracle weather forecast: fine for a first comparison, to be replaced by real
forecasts or perturbed values in the demo.
"""

import numpy as np
import pandas as pd

from demo.paths import RAW_DIR

INDUSTRIAL_USAGES = ["Manufacturing/industrial", "Utility", "Warehouse/storage", "Technology/science"]
# These BDG2 sites ship in Salesforce/GiftEvalPretrain, which Chronos-2 (and likely
# Timestral) trained on; excluding them keeps the comparison zero-shot.
PRETRAIN_SITES = ["Bear", "Fox", "Panther", "Rat"]
HORIZON = 48
CONTEXT = 4096
SEASON = 168
ORIGINS = pd.date_range("2017-02-06", "2017-12-04", freq="5W-MON")


def arima():
    """AutoARIMA on the last 4 weeks, daily and weekly cycles as 10 Fourier pairs each.

    Fewer terms lose the sharp weekend shutdowns and fall behind seasonal naive.
    """
    from demo.models import AutoARIMAModel

    return AutoARIMAModel(fit_last=672, fourier=((24, 10), (168, 10)))


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
    return elec, temp, meta.set_index("building_id")


def window_key(building, origin) -> str:
    return f"{building}|{origin.isoformat()}"


def windows(elec, temp, meta, origins=ORIGINS):
    ctx, cov, y, ids = [], [], [], []
    typical = elec.quantile(0.95)
    for origin in origins:
        t = elec.index.get_loc(origin)
        for b in elec.columns:
            target = elec[b].values[t : t + HORIZON]
            last_week = elec[b].values[t - SEASON : t]
            # Skip gaps and meters that are mostly switched off: nothing to forecast there.
            off = lambda v: np.mean(np.nan_to_num(v) < 0.01 * typical[b]) > 0.5
            if np.isnan(target).any() or off(target) or off(last_week):
                continue
            ctx.append(elec[b].values[t - CONTEXT : t])
            cov.append(temp[meta.site_id[b]].values[t - CONTEXT : t + HORIZON][None, :])
            y.append(target)
            ids.append((b, origin))
    return np.array(ctx), np.array(cov), np.array(y), ids
