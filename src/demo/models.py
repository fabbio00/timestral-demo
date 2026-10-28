"""Thin wrappers that give Timestral and Chronos-2 the same forecast interface.

Each wrapper takes a batch of contexts (B, T) plus optional known covariates
(B, C, T + H) and returns quantiles shaped (B, H, len(QUANTILES)).
"""

import time

import numpy as np
import pandas as pd
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


class AutoARIMAModel:
    """Classical benchmark: one AutoARIMA fitted per series on the recent history.

    Long seasonal periods (24 or 168 hourly steps) make seasonal ARIMA's state
    space huge and each fit takes tens of seconds, so seasonality enters as
    Fourier terms on a non-seasonal ARIMA (dynamic harmonic regression).
    Gaps are linearly interpolated (ARIMA cannot skip them) and the quantiles are
    read off the 20/40/60/80% prediction intervals around the mean.
    """

    name = "AutoARIMA"

    def __init__(self, fit_last: int | None = None, fourier: tuple[tuple[int, int], ...] = ()):
        from statsforecast.models import AutoARIMA

        self.fit_last, self.fourier = fit_last, fourier
        self.model = AutoARIMA(season_length=1)

    def _fourier_terms(self, length):
        t = np.arange(length)
        terms = [f(2 * np.pi * k * t / period) for period, K in self.fourier for k in range(1, K + 1) for f in (np.sin, np.cos)]
        return np.array(terms).reshape(-1, length)

    def predict(self, context, horizon, known_covariates=None):
        from statsforecast import StatsForecast

        context = np.asarray(context, dtype=np.float64)
        start = 0 if self.fit_last is None else context.shape[1] - self.fit_last
        frames, futures = [], []
        T = context.shape[1]
        fourier = self._fourier_terms(T + horizon)
        for i, series in enumerate(context):
            y = pd.Series(series[start:]).interpolate(limit_direction="both").fillna(0).values
            frame = pd.DataFrame({"unique_id": i, "ds": np.arange(len(y)), "y": y})
            future = pd.DataFrame({"unique_id": i, "ds": np.arange(len(y), len(y) + horizon)})
            cov = fourier
            if known_covariates is not None:
                cov = np.vstack([fourier, np.asarray(known_covariates[i], dtype=np.float64)])
            for j, row in enumerate(cov):
                frame[f"x{j}"] = row[start:T]
                future[f"x{j}"] = row[T:]
            frames.append(frame)
            futures.append(future)
        x_future = pd.concat(futures) if len(cov) else None
        sf = StatsForecast(models=[self.model], freq=1, n_jobs=-1)
        levels = [20, 40, 60, 80]
        fc = sf.forecast(df=pd.concat(frames), h=horizon, level=levels, X_df=x_future).sort_values(["unique_id", "ds"])

        def column(q):
            if q == 0.5:
                return "AutoARIMA"
            level = round(abs(1 - 2 * q) * 100)
            return f"AutoARIMA-{'lo' if q < 0.5 else 'hi'}-{level}"

        out = np.stack([fc[column(q)].values.reshape(len(context), horizon) for q in QUANTILES], axis=-1)
        return out


class Persistence:
    """Last valid observation carried forward: the simplest operational baseline."""

    name = "Persistence"

    def predict(self, context, horizon, known_covariates=None):
        last = pd.DataFrame(np.asarray(context).T).ffill().iloc[-1].fillna(0).values
        point = np.repeat(last[:, None], horizon, axis=1)
        return np.repeat(point[..., None], len(QUANTILES), axis=-1)


def timed_predict(model, context, horizon, known_covariates=None):
    start = time.perf_counter()
    out = model.predict(context, horizon, known_covariates)
    return out, time.perf_counter() - start
