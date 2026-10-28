"""Forecast cache written by the benchmark scripts and read by the frontend export.

One .npz per dataset and model under results/forecasts/ (git-ignored), holding
the 10/50/90% quantiles keyed by window, so the page can be rebuilt in seconds
without re-running any model.
"""

import numpy as np

from demo.models import QUANTILES
from demo.paths import RESULTS_DIR

CACHE_DIR = RESULTS_DIR / "forecasts"
BAND_LEVELS = [0.1, 0.5, 0.9]


def _path(dataset: str, model: str):
    return CACHE_DIR / f"{dataset}__{model.replace(' ', '_').replace('+', 'plus')}.npz"


def save(dataset: str, model: str, keys: list[str], quantiles: np.ndarray):
    """quantiles: (N, ..., H, len(QUANTILES)) aligned with keys; merged into any existing file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bands = quantiles[..., [QUANTILES.index(q) for q in BAND_LEVELS]].astype(np.float32)
    stored = load(dataset, model)
    stored.update(dict(zip(keys, bands)))
    np.savez_compressed(_path(dataset, model), **stored)


def load(dataset: str, model: str) -> dict[str, np.ndarray]:
    """Returns {key: (..., H, 3) array of 10/50/90% quantiles}, empty if nothing cached."""
    path = _path(dataset, model)
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
