"""The ranking PickyMate performs, as the benchmark needs it.

A candidate is scored by its squared Mahalanobis distance to the confirmed
picks, which is what the application computes when the curator presses Rank by
Embedding Distance with Confirmed as the reference. The relative score subtracts
the distance to the rejected picks, which is the application's second reference,
the one labelled "far from".

The distance itself is the application's own, boxer.ranking.distances -- the
covariance inverted with a pseudo-inverse, which is what allows a ranking to be
formed from a reference set smaller than the dimension of the space -- so the
benchmark cannot drift from what the application computes.
"""
from __future__ import annotations

import numpy as np

from boxer.ranking import MIN_REFERENCE, distances

__all__ = ["MIN_REFERENCE", "distances", "score", "rank"]


def score(vectors, confirmed, rejected, candidates, relative):
    """The score the application would give; lower is better.

    Returns None when the reference sets are too small to form a distribution,
    which is the application's own behaviour: it declines to rank rather than
    ranking from one pick. The caller must then keep the order it already has.
    """
    if len(confirmed) < MIN_REFERENCE:
        return None
    d = distances(vectors[confirmed], vectors[candidates])
    if relative:
        if len(rejected) < MIN_REFERENCE:
            return None
        d = d - distances(vectors[rejected], vectors[candidates])
    return d


def rank(vectors, confirmed, rejected, candidates, relative):
    """The candidates in the order the application would place them.

    Ties keep their incoming order, so a ranking never reshuffles candidates it
    cannot distinguish.
    """
    d = score(vectors, confirmed, rejected, candidates, relative)
    if d is None:
        return None
    return np.asarray(candidates)[np.argsort(d, kind='stable')]
