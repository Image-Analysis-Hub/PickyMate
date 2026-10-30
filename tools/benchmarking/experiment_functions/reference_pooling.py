"""Pooling the decisions of finished tomograms, which is what Experiment 3 measures.

PickyMate can rank a tomogram from the confirmed and rejected picks of the
tomograms already curated, before anything has been decided in the new one; that
is the References from control of the ranking row. This module builds those
pooled reference sets and the orders in which tomograms are accumulated.

Pooling is by pick, not by tomogram: a tomogram that was curated more
thoroughly contributes more references, which is what the application does and
what a curator would experience.
"""
from __future__ import annotations

import numpy as np

from boxer import metadata


def _rows_of(path):
    records = metadata._parse_particle_file(str(path))
    return np.array([r['embedding_row'] for r in records
                     if r['embedding_row'] is not None], dtype=np.int64)


def curated_sets(working_directory):
    """The confirmed and rejected picks a finished tomogram leaves behind."""
    confirmed = _rows_of(working_directory / 'Confirmed.txt')
    bin_path = working_directory / 'Recycle Bin.txt'
    rejected = _rows_of(bin_path) if bin_path.is_file() else np.empty(0, dtype=np.int64)
    return confirmed, rejected


def accumulation_orders(sources, count, seed=1000):
    """The orders in which the finished tomograms are accumulated.

    The order is arbitrary, so the experiment repeats it: the first order is the
    order the tomograms are given in, and the rest are drawn from a generator
    seeded per order, so that a run can be repeated exactly.
    """
    orders = [list(sources)]
    for offset in range(count - 1):
        orders.append(list(np.random.default_rng(seed + offset).permutation(sources)))
    return orders


def pooled_references(tomograms, space, used, confirmed_rows, rejected_rows):
    """The reference vectors of every tomogram used, stacked into one set."""
    confirmed = np.concatenate([tomograms[name].spaces[space][1][confirmed_rows[name]]
                                for name in used])
    rejected = [tomograms[name].spaces[space][1][rejected_rows[name]]
                for name in used if len(rejected_rows[name])]
    rejected = (np.concatenate(rejected) if rejected
                else np.empty((0, confirmed.shape[1])))
    return confirmed, rejected
