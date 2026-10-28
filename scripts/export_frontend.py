"""Builds frontend/data.js from cached forecasts and benchmark metrics: no model runs.

Metrics come from results/ (sdwpf_smoke.csv, sdwpf_arima.csv, bdg2_smoke.csv) and
the plotted forecasts from results/forecasts/, both written by the benchmark
scripts. The output is a plain script assigning window.DEMO_DATA, so
frontend/index.html opens straight from disk.
"""

import json

import numpy as np
import pandas as pd

from demo import bdg2, cache, kddcup
from demo.paths import RESULTS_DIR, ROOT

SDWPF_MODELS = {"Timestral-209M": "sdwpf_smoke.py", "Chronos-2": "sdwpf_smoke.py",
                "AutoARIMA": "sdwpf_arima.py", "Persistence": "sdwpf_smoke.py"}
# The page plots each model with temperature, the same inputs for all of them.
BDG2_MODELS = {"Timestral-209M": "Timestral-209M + temp", "Chronos-2": "Chronos-2 + temp",
               "AutoARIMA": "AutoARIMA + temp", "Seasonal naive": "Seasonal naive"}
SDWPF_TURBINES = [1, 34, 67, 100, 134]
SDWPF_HISTORY = 432  # 72 h of 10-min steps shown before the forecast
BDG2_HISTORY = 168  # one week of hours shown before the forecast
KDD_REFERENCES = [("Vincitore KDD Cup 2022 (HIK)", 44.92), ("Baseline ufficiale GRU", 47.85)]


def cached(dataset, model, keys, script):
    stored = cache.load(dataset, model)
    missing = [k for k in keys if k not in stored]
    if missing:
        raise SystemExit(f"No cached {dataset} forecasts for {model} ({len(missing)} windows): run scripts/{script}")
    return stored


def as_list(values, scale=1.0, digits=3):
    return [None if np.isnan(v) else round(float(v) * scale, digits) for v in values]


def bands(q, scale=1.0, digits=3):
    """q: (H, 3) cached 10/50/90% quantiles -> rounded low / median / high lists."""
    q = np.clip(q, 0, None)
    return {"lo": as_list(q[:, 0], scale, digits), "mid": as_list(q[:, 1], scale, digits), "hi": as_list(q[:, 2], scale, digits)}


def summarize(runs):
    summary = runs.groupby("model")[["MAE", "RMSE", "seconds"]].mean()
    summary["score"] = (summary.MAE + summary.RMSE) / 2
    return summary


def export_sdwpf():
    keys = [str(c) for c in kddcup.SHOWCASE_CASES]
    stored = {m: cached("sdwpf", m, keys, script) for m, script in SDWPF_MODELS.items()}
    cases = []
    for idx in kddcup.SHOWCASE_CASES:
        inputs, truths = kddcup.load_case(idx)
        context = np.stack([kddcup.clean_context(df) for df in inputs])
        actual = np.stack([df.Patv.values[: kddcup.OUTPUT_LEN].astype(float) for df in truths])
        actual[np.stack([kddcup.invalid_mask(df)[: kddcup.OUTPUT_LEN] for df in truths])] = np.nan
        forecasts = {m: s[str(idx)] for m, s in stored.items()}  # (134, H, 3) each

        # The farm total sums per-turbine medians; the page shows no band for it.
        series = {"Intero parco": (np.nansum(context, 0), np.nansum(actual, 0), {n: q.sum(0) for n, q in forecasts.items()}, 2)}
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

    runs = pd.read_csv(RESULTS_DIR / "sdwpf_smoke.csv")
    metrics = [{"model": m, "score": round(r.score, 2), "mae": round(r.MAE, 2), "rmse": round(r.RMSE, 2),
                "seconds": round(r.seconds, 1), "reference": False, "cases": int(runs.case.nunique())}
               for m, r in summarize(runs).iterrows()]
    metrics += [{"model": n, "score": s, "reference": True} for n, s in KDD_REFERENCES]

    # ARIMA only ran on a subset of cases; the page shows every model on that subset next to it.
    arima_runs = pd.read_csv(RESULTS_DIR / "sdwpf_arima.csv")
    subset = summarize(pd.concat([runs[runs.case.isin(arima_runs.case)], arima_runs]))
    r = subset.loc["AutoARIMA"]
    metrics.append({"model": "AutoARIMA", "score": round(r.score, 2), "mae": round(r.MAE, 2), "rmse": round(r.RMSE, 2),
                    "seconds": round(r.seconds, 1), "reference": False, "cases": int(arima_runs.case.nunique())})
    return {"stepMinutes": 10, "nCases": int(runs.case.nunique()), "metrics": sorted(metrics, key=lambda r: r["score"]),
            "arimaCases": int(arima_runs.case.nunique()), "sameCases": {m: round(s, 2) for m, s in subset.score.items()},
            "cases": cases}


def export_bdg2():
    elec, temp, meta = bdg2.load()
    ctx, _, y, ids = bdg2.windows(elec, temp, meta)
    keys = [bdg2.window_key(b, o) for b, o in ids]
    stored = {shown: cached("bdg2", run, keys, "bdg2_smoke.py") for shown, run in BDG2_MODELS.items()}
    windows = [{
        "building": building,
        "origin": origin.isoformat(),
        "history": as_list(ctx[i, -BDG2_HISTORY:], 1, 1),
        "actual": as_list(y[i], 1, 1),
        "forecasts": {m: bands(s[keys[i]], 1, 1) for m, s in stored.items()},
    } for i, (building, origin) in enumerate(ids)]

    buildings = [{"id": b, "usage": meta.primaryspaceusage[b], "detail": meta.sub_primaryspaceusage[b],
                  "site": meta.site_id[b], "sqm": float(meta.sqm[b])} for b in elec.columns]
    runs = pd.read_csv(RESULTS_DIR / "bdg2_smoke.csv")
    metrics = [{"model": r.model, "mae": round(r.MAE_kWh, 2), "skill": round(r["MAE_skill_%"], 1),
                "coverage": round(100 * r.cov_80, 1), "seconds": round(r.seconds, 1)} for _, r in runs.iterrows()]
    return {"stepMinutes": 60, "buildings": buildings, "metrics": sorted(metrics, key=lambda r: r["mae"]),
            "windows": windows}


def main():
    data = {"sdwpf": export_sdwpf(), "bdg2": export_bdg2()}
    out = ROOT / "frontend" / "data.js"
    out.write_text("window.DEMO_DATA = " + json.dumps(data, separators=(",", ":"), allow_nan=False) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
