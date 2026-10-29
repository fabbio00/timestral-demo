"""Checks that inputs, model settings and metrics are wired the way the benchmarks assume.

Prints one OK/FAIL line per check. With --official-metrics pointing at the KDD Cup
evaluation code (PaddleSpatial .../kddcup22-sdwpf-evaluation/paddlepaddle/metrics.py),
our scorer is compared with the official one on every cached wind forecast.
"""

import argparse
import importlib.util
import warnings

import numpy as np
import pandas as pd

from demo import bdg2, cache, kddcup
from demo.metrics import coverage, mae, wql
from demo.models import QUANTILES, Chronos2Model, SeasonalAverage, SeasonalNaive, TimestralModel

warnings.filterwarnings("ignore")
failures = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}{' · ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def valid_quantiles(q):
    return bool(np.isfinite(q).all() and (np.diff(q, axis=-1) >= -1e-4).all())


def wind_inputs():
    for idx in (1, 71, 142):
        x = pd.read_csv(kddcup.TEST_DIR / "infile" / f"{idx:04d}in.csv")
        y = pd.read_csv(kddcup.TEST_DIR / "outfile" / f"{idx:04d}out.csv")
        minutes = lambda df: df.Day * 1440 + pd.to_timedelta(df.Tmstamp + ":00").dt.total_seconds() / 60
        last_in = x.assign(t=minutes(x)).groupby("TurbID").t.max()
        first_out = y.assign(t=minutes(y)).groupby("TurbID").t.min()
        check(f"wind case {idx}: 134 turbines x 2016 input / 288 output rows, time-sorted",
              x.TurbID.nunique() == 134 and set(x.groupby("TurbID").size()) == {2016} and set(y.groupby("TurbID").size()) == {288}
              and all(minutes(g).is_monotonic_increasing for _, g in x.groupby("TurbID")))
        check(f"wind case {idx}: forecast starts 10 min after the last input, no overlap",
              bool(((first_out - last_in) == 10).all()), f"gap {set((first_out - last_in).astype(int))} min")
    inputs, _ = kddcup.load_case(1)
    ctx = np.stack([kddcup.clean_context(df) for df in inputs])
    check("wind context: anomalies passed as missing values", True, f"{100 * np.isnan(ctx).mean():.1f}% of points missing or anomalous")


def wind_scorer(official_path):
    spec = importlib.util.spec_from_file_location("official_metrics", official_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    # turbine_scores is the official per-turbine core; regressor_detailed_scores wraps it with
    # submission checks that reject constant forecasts, i.e. the persistence baseline.
    worst = 0.0
    compared = 0
    for model in ("Timestral-209M", "Chronos-2", "AutoARIMA", "Persistence"):
        for key, q in cache.load("sdwpf", model).items():
            _, truths = kddcup.load_case(int(key))
            median = np.clip(q[..., 1], 0, None)
            ours = sum(kddcup.score_case(median, truths)) / 2
            maes, rmses = [], []
            for p, df in zip(median, truths):
                m, r = official.turbine_scores(p[:, None], df.Patv.fillna(0).values[:288, None], df, 288)
                if m == m and r == r and m != -1 and r != -1:
                    maes.append(m)
                    rmses.append(r)
            worst = max(worst, abs(ours - (sum(maes) + sum(rmses)) / 2))
            compared += 1
    check("wind score identical to the official KDD Cup code", worst < 1e-6, f"{compared} model-case pairs, max difference {worst:.2e}")


def wind_models():
    inputs, _ = kddcup.load_case(71)
    ctx = np.stack([kddcup.clean_context(df) for df in inputs])[:8]
    t = TimestralModel()
    q = t.predict(ctx, kddcup.OUTPUT_LEN)
    check("Timestral: runs on CPU, output 8 x 288 x 9 quantiles, finite and ordered",
          t.device == "cpu" and q.shape == (8, 288, len(QUANTILES)) and valid_quantiles(q))
    short = t.predict(ctx[:, -1024:], kddcup.OUTPUT_LEN)
    check("Timestral: reads the whole 14-day history (cutting it changes the forecast)",
          np.abs(short - q).mean() > 1e-3, f"mean change {np.abs(short - q).mean():.1f} kW")
    c_mps, c_cpu = Chronos2Model("mps"), Chronos2Model("cpu")
    qm, qc = c_mps.predict(ctx, kddcup.OUTPUT_LEN), c_cpu.predict(ctx, kddcup.OUTPUT_LEN)
    check("Chronos-2: output 8 x 288 x 9 quantiles, finite and ordered", qm.shape == (8, 288, len(QUANTILES)) and valid_quantiles(qm))
    rel = np.abs(qm - qc).mean() / np.abs(qc).mean()
    check("Chronos-2: GPU (MPS) and CPU give the same forecast", rel < 1e-3, f"relative difference {rel:.1e}")
    short = c_cpu.predict(ctx[:, -1024:], kddcup.OUTPUT_LEN)
    check("Chronos-2: reads the whole 14-day history (cutting it changes the forecast)", np.abs(short - qc).mean() > 1e-3)


def building_inputs():
    elec, temp, meta = bdg2.load()
    ctx, cov, y, ids = bdg2.windows(elec, temp, meta)
    b, origin = ids[0]
    t = elec.index.get_loc(origin)
    check("buildings: 14 buildings, none from the sites in the public pretraining corpus",
          elec.shape[1] == 14 and not meta.loc[elec.columns].site_id.isin(bdg2.PRETRAIN_SITES).any(), f"{len(ids)} windows")
    check("buildings: history ends 1 h before the forecast start, target starts at it",
          np.allclose(ctx[0][-1], elec[b].iloc[t - 1], equal_nan=True) and np.allclose(y[0][0], elec[b].iloc[t])
          and ctx.shape[1] == bdg2.CONTEXT and y.shape[1] == bdg2.HORIZON)
    site = meta.site_id[b]
    check("buildings: temperature aligned with history and with the 48 h forecast",
          np.isclose(cov[0, 0, bdg2.CONTEXT - 1], temp[site].iloc[t - 1]) and np.isclose(cov[0, 0, bdg2.CONTEXT], temp[site].iloc[t])
          and cov.shape[2] == bdg2.CONTEXT + bdg2.HORIZON)
    naive = SeasonalNaive(bdg2.SEASON).predict(ctx[:5], bdg2.HORIZON)[..., 4]
    expected = pd.DataFrame(ctx[:5].T).interpolate(limit_direction="both").T.values[:, -168:-120]
    check("buildings: 'same hour last week' baseline really repeats last week", np.allclose(naive, expected))
    average = SeasonalAverage(bdg2.SEASON).predict(ctx[:5], bdg2.HORIZON)[..., 4]
    filled = pd.DataFrame(ctx[:5].T).interpolate(limit_direction="both").T.values
    expected = np.mean([filled[:, -168 * k : -168 * k + 48] for k in range(1, 5)], axis=0)
    check("buildings: 'last 4 weeks' baseline averages the same hour of those weeks", np.allclose(average, expected, atol=1e-3))
    timestral = TimestralModel()
    plain, with_temp = timestral.predict(ctx[:5], bdg2.HORIZON), timestral.predict(ctx[:5], bdg2.HORIZON, cov[:5])
    check("buildings: temperature actually reaches Timestral (forecast changes with it)", np.abs(plain - with_temp).mean() > 1e-3,
          f"mean change {np.abs(plain - with_temp).mean():.2f} kWh")
    arima = bdg2.arima().predict(ctx[:3], bdg2.HORIZON, cov[:3])
    check("buildings: AutoARIMA with Fourier terms and temperature, finite and ordered quantiles", valid_quantiles(arima))


def metrics():
    y = np.array([[10.0, 20.0]])
    q = np.repeat(np.array([[[12.0], [17.0]]]), len(QUANTILES), axis=-1)  # point forecast 12, 17
    check("metrics: MAE on a hand-computed example", np.isclose(mae(y, q), 2.5))
    # For a point forecast every quantile equals it, so WQL = sum |error| / sum |y|.
    check("metrics: WQL reduces to sum|error| / sum|y| for a point forecast", np.isclose(wql(y, q), 5 / 30))
    bands = np.stack([np.full((1, 2), v) for v in np.linspace(0, 30, len(QUANTILES))], axis=-1)
    check("metrics: band coverage counts actuals inside the 10-90% band", np.isclose(coverage(y, bands), 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-metrics", help="path to the official KDD Cup metrics.py")
    args = parser.parse_args()
    wind_inputs()
    if args.official_metrics:
        wind_scorer(args.official_metrics)
    wind_models()
    building_inputs()
    metrics()
    print(f"\n{'All checks passed' if not failures else f'{len(failures)} checks failed'}")


if __name__ == "__main__":
    main()
