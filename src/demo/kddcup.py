"""KDD Cup 2022 (SDWPF) final-phase data loading and scoring.

Scoring mirrors the official evaluation in PaddleSpatial
(apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/paddlepaddle): per test case,
MAE and RMSE in MW per turbine on valid points, summed over the 134 turbines;
the final score is (mean MAE + mean RMSE) / 2 over the 142 test cases.
"""

import numpy as np
import pandas as pd

from demo.paths import RAW_DIR

TEST_DIR = RAW_DIR / "sdwpf" / "final_phase_test"
N_TURBINES = 134
OUTPUT_LEN = 288  # 48 h at 10 min
N_CASES = 142
# AutoARIMA only runs on 20 evenly spaced cases (19k fits for all 142 take ~50 min);
# the page plots a subset of those so every model appears in each chart.
ARIMA_CASES = np.linspace(1, N_CASES, 20).round().astype(int)
SHOWCASE_CASES = ARIMA_CASES[::4]


def arima():
    """AutoARIMA on the last 3 days: wind's daily cycle is weak, longer fits cost 10x more."""
    from demo.models import AutoARIMAModel

    return AutoARIMAModel(fit_last=432)


def case_row(idx: int, model: str, median: np.ndarray, truths, seconds: float) -> dict:
    """One results row: official score, strict score and whether the official checks pass."""
    median = np.clip(median, 0, None)
    official, strict = score_case(median, truths), score_case(median, truths, skip_zero=False)
    return {"case": int(idx), "model": model, "MAE": official[0], "RMSE": official[1],
            "MAE_strict": strict[0], "RMSE_strict": strict[1], "valid": passes_official_checks(median), "seconds": seconds}


def invalid_mask(df: pd.DataFrame) -> np.ndarray:
    """Points the official scorer drops: missing, curtailed or physically impossible readings."""
    return (
        df.isna().any(axis=1)
        | (df.Patv < 0)
        | ((df.Patv == 0) & (df.Wspd > 2.5))
        | (df.Pab1 > 89) | (df.Pab2 > 89) | (df.Pab3 > 89)
        | (df.Wdir < -180) | (df.Wdir > 180)
        | (df.Ndir < -720) | (df.Ndir > 720)
    ).values


def load_case(idx: int):
    """Returns per-turbine input frames and ground-truth frames for test case idx (1-based)."""
    x = pd.read_csv(TEST_DIR / "infile" / f"{idx:04d}in.csv")
    y = pd.read_csv(TEST_DIR / "outfile" / f"{idx:04d}out.csv")
    return [g for _, g in x.groupby("TurbID", sort=True)], [g for _, g in y.groupby("TurbID", sort=True)]


def clean_context(df: pd.DataFrame) -> np.ndarray:
    """Patv history cleaned as the SDWPF authors suggest (Scientific Data 2024).

    Slightly negative power is an idle turbine drawing auxiliary power in low wind:
    it becomes 0, not a gap, or the models would never see calm periods. Points are
    unknown (NaN) only when missing, at or below zero with wind above 2.5 m/s, with
    blades pitched past 89 degrees, or with impossible angles. The scorer still uses
    the stricter official invalid_mask.
    """
    patv = np.clip(df.Patv.values.astype(np.float32), 0, None)
    unknown = (
        df.isna().any(axis=1)
        | ((df.Patv <= 0) & (df.Wspd > 2.5))
        | (df.Pab1 > 89) | (df.Pab2 > 89) | (df.Pab3 > 89)
        | (df.Wdir < -180) | (df.Wdir > 180)
        | (df.Ndir < -720) | (df.Ndir > 720)
    ).values
    patv[unknown] = np.nan
    return patv


def passes_official_checks(pred: np.ndarray) -> bool:
    """The official submission checks: at least 90% of turbines with a non-zero forecast
    and at least 10% with a non-constant one. Constant forecasts (persistence) fail."""
    def constant(p):
        return p.min() == p.max() or (p.std() <= 0.1 and len(np.unique(p)) / p.size < 0.1)

    nonzero = np.mean([p.any() for p in pred])
    varying = np.mean([not constant(p) for p in pred])
    return bool(nonzero >= 0.9 and varying >= 0.1)


def score_case(pred: np.ndarray, truth_frames, skip_zero: bool = True) -> tuple[float, float]:
    """pred: (134, 288) in kW. Returns summed MAE and RMSE in MW for one test case.

    skip_zero=True is the official per-turbine rule: a turbine whose valid predictions or
    targets are all zero is skipped, not scored. The official submission checks keep
    that from being gamed with zero forecasts; for forecasts that fail those checks,
    skip_zero=False scores every turbine with a non-zero target instead.
    """
    maes, rmses = [], []
    for p, df in zip(pred, truth_frames):
        gt = df.Patv.fillna(0).values[:OUTPUT_LEN]
        keep = ~invalid_mask(df)[:OUTPUT_LEN]
        if not np.any(gt[keep]) or (skip_zero and not np.any(p[keep])):
            continue
        err = (p[keep] - gt[keep]) / 1000
        maes.append(np.mean(np.abs(err)))
        rmses.append(np.sqrt(np.mean(err**2)))
    return float(np.sum(maes)), float(np.sum(rmses))
