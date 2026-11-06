"""Uniform random band selection."""

from __future__ import annotations

import numpy as np


def select_random_bands(n_total: int, n: int, seed: int) -> np.ndarray:
    """Return ``n`` distinct band indices out of ``n_total``, sorted ascending.

    The draw is uniform without replacement and fully determined by ``seed``.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n, replace=False))
