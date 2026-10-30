"""Ranking by Mahalanobis distance in an embedding space -- the maths
behind "Rank by Embedding Distance", with no window involved, so the
benchmarks in benchmarking/ rank with exactly the code the application
runs.

A reference set (the Confirmed particles, say) defines a distribution:
its mean and the pseudo-inverse of its covariance. Every candidate is
scored by its squared Mahalanobis distance to that; lower is closer. A
"far from" set subtracts a second such distance (see
TomoBoxer.compute_mahalanobis), so particles close to the first set and
far from the second sort first.
"""
from __future__ import annotations

import numpy as np

#: the smallest reference set a distribution can be built from
MIN_REFERENCE = 2


def distribution(vectors):
    """mean + pseudo-inverse covariance of a (N, D) reference array --
    pinv rather than a true inverse since these embeddings are
    typically rank-deficient well before N reaches D (verified
    empirically against real data: covariance rank plateaus far
    below D even with every available particle as reference), so a
    true inverse would simply fail. Building the D x D covariance
    directly (rather than the N x N Gram-matrix/dual approach) is
    deliberate: at D=1024 it's sub-second regardless of N, and the
    dual approach is only cheaper when N << D, which isn't the
    regime this tool runs in."""
    vectors = np.asarray(vectors, dtype=np.float64)
    mu = vectors.mean(axis=0)
    centered = vectors - mu
    cov = (centered.T @ centered) / (len(vectors) - 1)
    covinv = np.linalg.pinv(cov, hermitian=True)
    return mu, covinv


def squared_distances(mu, covinv, targets):
    """Squared Mahalanobis distance of each row in targets to (mu,
    covinv). The matmul-based formulation, not einsum -- verified
    the "obvious" einsum('ij,jk,ik->i', ...) version is over 20x
    slower than this at realistic sizes, a difference real enough
    to matter at a few thousand particles."""
    centered = np.asarray(targets, dtype=np.float64) - mu
    return ((centered @ covinv) * centered).sum(axis=1)


def distances(reference, candidates):
    """Squared Mahalanobis distance of every candidate to a reference set."""
    return squared_distances(*distribution(reference), candidates)


def scores(target_rows, ref_rows, array, leave_one_out=False, extra=None,
           reference_name="the reference"):
    """{index: squared Mahalanobis distance} for every (index, embedding
    row) pair in target_rows, relative to the distribution of the
    reference's own (index, embedding row) pairs in `array`. target_rows
    is computed once by the caller (shared across both terms of a
    "far from" call) so "does this particle have an embedding at all" is
    only ever decided from the target set itself, not re-derived per
    reference. Returns (distances, reference_size); raises ValueError,
    naming the reference, if it has fewer than MIN_REFERENCE embedded
    particles.

    leave_one_out: the target and the reference are the same set, and
    each particle's own row is left out of the reference before scoring
    it -- the exact way, a fresh covariance per particle, which is
    genuinely expensive for anything but a small set. A particle whose
    reference would shrink below MIN_REFERENCE without it gets no score.

    `extra`: an (M, D) array of further reference vectors -- the same
    set's particles in other tomograms of a session -- pooled with this
    tomogram's own. They are never among the particles being scored, so
    leave-one-out only ever removes this tomogram's own."""
    n_extra = 0 if extra is None else len(extra)
    if len(ref_rows) + n_extra < MIN_REFERENCE:
        raise ValueError(
            f"\"{reference_name}\" has only {len(ref_rows) + n_extra} particle(s) with an embedding -- "
            f"need at least {MIN_REFERENCE} to build a distribution.")

    def pooled(rows):
        own = array[rows] if len(rows) else np.empty((0, array.shape[1]))
        return own if extra is None else np.vstack([own, extra])

    result = {}
    if leave_one_out:
        for i, row in target_rows:
            others = [r for j, r in ref_rows if j != i]
            if len(others) + n_extra < MIN_REFERENCE:
                continue
            mu, covinv = distribution(pooled(others))
            result[i] = float(squared_distances(mu, covinv, array[[row]])[0])
    else:
        mu, covinv = distribution(pooled([row for _, row in ref_rows]))
        d2 = squared_distances(mu, covinv, array[[row for _, row in target_rows]])
        for (i, _), d in zip(target_rows, d2):
            result[i] = float(d)
    return result, len(ref_rows) + n_extra
