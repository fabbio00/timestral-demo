"""Point and probabilistic metrics, aggregated the way GIFT-Eval / fev-bench do."""

import numpy as np

from demo.models import QUANTILES


def mae(y, q):
    """MAE of the median forecast. y: (B, H), q: (B, H, Q)."""
    median = q[..., QUANTILES.index(0.5)]
    return float(np.nanmean(np.abs(y - median)))


def wql(y, q):
    """Weighted quantile loss: mean pinball loss over quantiles, scaled by sum |y|."""
    losses = []
    for k, level in enumerate(QUANTILES):
        diff = y - q[..., k]
        losses.append(np.nansum(np.maximum(level * diff, (level - 1) * diff)))
    return float(2 * np.mean(losses) / np.nansum(np.abs(y)))


def coverage(y, q, low=0.1, high=0.9):
    """Share of actuals inside the [low, high] quantile band; ideal is high - low."""
    lo, hi = q[..., QUANTILES.index(low)], q[..., QUANTILES.index(high)]
    inside = (y >= lo) & (y <= hi)
    return float(np.nanmean(inside[~np.isnan(y)]))
