"""Runs the models on a few showcase windows and writes frontend/data.js.

Metrics come from the full runs in results/ (sdwpf_smoke.csv, bdg2_smoke.csv);
this script only adds the series the page plots. The output is a plain script
assigning window.DEMO_DATA, so frontend/index.html opens straight from disk.
"""

import json

import numpy as np
import pandas as pd

from demo import bdg2, kddcup
from demo.models import QUANTILES, Chronos2Model, Persistence, SeasonalNaive, TimestralModel
from demo.paths import RESULTS_DIR, ROOT

SDWPF_CASES = np.linspace(1, kddcup.N_CASES, 6).round().astype(int)
SDWPF_TURBINES = [1, 34, 67, 100, 134]
SDWPF_HISTORY = 432  # 72 h of 10-min steps shown before the forecast
BDG2_HISTORY = 168  # one week of hours shown before the forecast
KDD_REFERENCES = [("Vincitore KDD Cup 2022 (HIK)", 44.92), ("Baseline ufficiale GRU", 47.85)]


def bands(q, scale=1.0, digits=3):
    """q: (H, Q) quantiles -> rounded low / median / high lists."""
    pick = lambda level: as_list(np.clip(q[:, QUANTILES.index(level)], 0, None), scale, digits)
    return {"lo": pick(0.1), "mid": pick(0.5), "hi": pick(0.9)}


def as_list(values, scale=1.0, digits=3):
    return [None if np.isnan(v) else round(float(v) * scale, digits) for v in values]


def export_sdwpf(timestral, chronos):
    models = [timestral, chronos, Persistence()]
    cases = []
    for idx in SDWPF_CASES:
        inputs, truths = kddcup.load_case(idx)
        context = np.stack([kddcup.clean_context(df) for df in inputs])
        actual = np.stack([df.Patv.values[: kddcup.OUTPUT_LEN].astype(float) for df in truths])
        actual[np.stack([kddcup.invalid_mask(df)[: kddcup.OUTPUT_LEN] for df in truths])] = np.nan
        forecasts = {m.name: m.predict(context, kddcup.OUTPUT_LEN) for m in models}

        # Farm-level bands are sums of per-turbine quantiles: indicative, since turbines
        # move together far more than independently.
        series = {"Intero parco": (np.nansum(context, 0), np.nansum(actual, 0),
                                   {n: q.sum(0) for n, q in forecasts.items()}, 2)}
        for t in SDWPF_TURBINES:
            series[f"Turbina {t}"] = (context[t - 1], actual[t - 1], {n: q[t - 1] for n, q in forecasts.items()}, 3)

        cases.append({
            "case": int(idx),
            "series": {
                name: {
                    "history": as_list(hist[-SDWPF_HISTORY:], 1e-3, d),
                    "actual": as_list(act, 1e-3, d),
                    "forecasts": {n: bands(q, 1e-3, d) for n, q in fc.items()},
                }
                for name, (hist, act, fc, d) in series.items()
            },
        })
        print(f"SDWPF case {idx} exported")

    runs = pd.read_csv(RESULTS_DIR / "sdwpf_smoke.csv")
    summary = runs.groupby("model")[["MAE", "RMSE", "seconds"]].mean()
    summary["score"] = (summary.MAE + summary.RMSE) / 2
    metrics = [{"model": m, "score": round(r.score, 2), "mae": round(r.MAE, 2), "rmse": round(r.RMSE, 2),
                "seconds": round(r.seconds, 1), "reference": False} for m, r in summary.iterrows()]
    metrics += [{"model": n, "score": s, "reference": True} for n, s in KDD_REFERENCES]
    return {"stepMinutes": 10, "nCases": int(runs.case.nunique()), "metrics": sorted(metrics, key=lambda r: r["score"]),
            "cases": cases}


def export_bdg2(timestral, chronos):
    elec, temp, meta = bdg2.load()
    ctx, cov, y, ids = bdg2.windows(elec, temp, meta)
    naive = SeasonalNaive(bdg2.SEASON)
    forecasts = {
        timestral.name: timestral.predict(ctx, bdg2.HORIZON, cov),
        chronos.name: chronos.predict(ctx, bdg2.HORIZON, cov),
        naive.name: naive.predict(ctx, bdg2.HORIZON),
    }
    windows = []
    for i, (building, origin) in enumerate(ids):
        windows.append({
            "building": building,
            "origin": origin.isoformat(),
            "history": as_list(ctx[i, -BDG2_HISTORY:], 1, 1),
            "actual": as_list(y[i], 1, 1),
            "forecasts": {n: bands(q[i], 1, 1) for n, q in forecasts.items()},
        })
    print(f"BDG2 {len(windows)} windows exported")

    buildings = [{"id": b, "usage": meta.primaryspaceusage[b], "detail": meta.sub_primaryspaceusage[b],
                  "site": meta.site_id[b], "sqm": float(meta.sqm[b])} for b in elec.columns]
    runs = pd.read_csv(RESULTS_DIR / "bdg2_smoke.csv")
    metrics = [{"model": r.model, "mae": round(r.MAE_kWh, 2), "skill": round(r["MAE_skill_%"], 1),
                "coverage": round(100 * r.cov_80, 1), "seconds": round(r.seconds, 1)} for _, r in runs.iterrows()]
    return {"stepMinutes": 60, "buildings": buildings, "metrics": sorted(metrics, key=lambda r: r["mae"]),
            "windows": windows}


def main():
    timestral, chronos = TimestralModel(), Chronos2Model()
    data = {"sdwpf": export_sdwpf(timestral, chronos), "bdg2": export_bdg2(timestral, chronos)}
    out = ROOT / "frontend" / "data.js"
    out.write_text("window.DEMO_DATA = " + json.dumps(data, separators=(",", ":"), allow_nan=False) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
