"""Splitting a candidate list into the half that is curated and the half that is scored.

A ranking must not be judged on the candidates it was built from. The list is
therefore dealt alternately into two halves: the candidates at the first, third,
fifth position and so on form the labelling half, in which picks are confirmed
and rejected, and the remaining candidates form the evaluation half, which is
re-ranked and on which precision is measured.

Dealing alternately rather than cutting the list in two matters: the list arrives
in descending template-matching score order, so an alternating deal gives both
halves the same range of scores, and leaves the template-matching order of the
evaluation half a fair baseline to compare against.
"""
from __future__ import annotations

import numpy as np


def deal_alternately(n):
    """Return (labelling, evaluation) as positions within a list of n candidates."""
    index = np.arange(n)
    return index[index % 2 == 0], index[index % 2 == 1]


def check_split(labelling, evaluation, n):
    """The two halves must be disjoint and cover the list."""
    assert len(np.intersect1d(labelling, evaluation)) == 0, 'the halves overlap'
    assert len(labelling) + len(evaluation) == n, 'the halves do not cover the list'
    return True
