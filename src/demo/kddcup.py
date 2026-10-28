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
    """Patv history with invalid points set to NaN, so models treat them as missing."""
    patv = df.Patv.values.astype(np.float32).copy()
    patv[invalid_mask(df)] = np.nan
    return patv


def score_case(pred: np.ndarray, truth_frames) -> tuple[float, float]:
    """pred: (134, 288) in kW. Returns summed MAE and RMSE in MW for one test case."""
    maes, rmses = [], []
    for p, df in zip(pred, truth_frames):
        gt = df.Patv.fillna(0).values[:OUTPUT_LEN]
        keep = ~invalid_mask(df)[:OUTPUT_LEN]
        if not keep.any():
            continue
        err = (p[keep] - gt[keep]) / 1000
        maes.append(np.mean(np.abs(err)))
        rmses.append(np.sqrt(np.mean(err**2)))
    return float(np.sum(maes)), float(np.sum(rmses))
