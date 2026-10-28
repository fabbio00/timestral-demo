"""Thin wrappers that give Timestral and Chronos-2 the same forecast interface.

Each wrapper takes a batch of contexts (B, T) plus optional known covariates
(B, C, T + H) and returns quantiles shaped (B, H, len(QUANTILES)).
"""

import time

import numpy as np
import torch

from demo.paths import MODELS_DIR

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TimestralModel:
    name = "Timestral-209M"

    def __init__(self, device: str | None = None, path=MODELS_DIR / "Timestral-209M"):
        from timestral import TimestralForecaster

        # Timestral's scaler computes in float64, which Apple MPS does not support.
        self.device = device or ("cpu" if best_device() == "mps" else best_device())
        self.model = TimestralForecaster.from_pretrained(path, device=self.device)

    def predict(self, context, horizon, known_covariates=None):
        forecast = self.model.forecast(
            np.asarray(context, dtype=np.float32),
            prediction_length=horizon,
            known_covariates=None if known_covariates is None else np.asarray(known_covariates, dtype=np.float32),
        )
        # Timestral returns (B, V, H) per quantile; we always pass a single target variate.
        qs = [forecast.quantile(q).detach().cpu().numpy()[:, 0, :] for q in QUANTILES]
        return np.stack(qs, axis=-1)


class Chronos2Model:
    name = "Chronos-2"

    def __init__(self, device: str | None = None, path=MODELS_DIR / "chronos-2"):
        from chronos import Chronos2Pipeline

        self.device = device or best_device()
        self.pipeline = Chronos2Pipeline.from_pretrained(str(path), device_map=self.device)

    def predict(self, context, horizon, known_covariates=None):
        inputs = []
        for i, series in enumerate(np.asarray(context, dtype=np.float32)):
            item = {"target": series}
            if known_covariates is not None:
                cov = np.asarray(known_covariates[i], dtype=np.float32)
                T = len(series)
                item["past_covariates"] = {f"c{j}": cov[j, :T] for j in range(cov.shape[0])}
                item["future_covariates"] = {f"c{j}": cov[j, T:] for j in range(cov.shape[0])}
            inputs.append(item)
        quantiles, _ = self.pipeline.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=QUANTILES)
        return np.stack([q[0].detach().cpu().numpy() for q in quantiles])


class SeasonalNaive:
    """Repeats the last full season; the reference every skill score is measured against."""

    name = "Seasonal naive"

    def __init__(self, season: int):
        self.season = season

    def predict(self, context, horizon, known_covariates=None):
        context = np.asarray(context, dtype=np.float32)
        reps = int(np.ceil(horizon / self.season))
        point = np.tile(context[:, -self.season :], reps)[:, :horizon]
        return np.repeat(point[..., None], len(QUANTILES), axis=-1)


def timed_predict(model, context, horizon, known_covariates=None):
    start = time.perf_counter()
    out = model.predict(context, horizon, known_covariates)
    return out, time.perf_counter() - start
