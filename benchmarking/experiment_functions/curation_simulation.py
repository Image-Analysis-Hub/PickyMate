"""Simulating what a curator does, so that a ranking can be measured.

Two kinds of curation are simulated.

`examine_until_confirmed` walks the labelling half from the top, confirming the
candidates that match a deposited particle and rejecting the others, and stops
when the chosen number of picks has been confirmed. It records how many
candidates had to be examined to get there, which is the cost the curator pays.

`reject_region` is the third use of the region mask: instead of removing the
candidates outside the mask from the list, they are rejected in one action, the
way a curator clears an area with the Eraser or the Select tool. They then serve
as rejected picks, which is what lets the relative distance be formed before the
curator has rejected anything by hand.

Neither simulation consults a candidate below the point reached, so every
ranking the benchmark measures is one a curator could have asked for at that
moment.
"""
from __future__ import annotations

import numpy as np


def examine_until_confirmed(labels, wanted):
    """Examine from the top until `wanted` picks are confirmed.

    Returns (confirmed, rejected, examined) as positions within `labels`, or
    None when the list does not contain that many true picks.
    """
    confirmed, rejected = [], []
    for position, is_true in enumerate(labels):
        (confirmed if is_true else rejected).append(position)
        if len(confirmed) == wanted:
            # the dtype is fixed: an empty rejected list must still index an array
            return (np.array(confirmed, dtype=np.int64),
                    np.array(rejected, dtype=np.int64), position + 1)
    return None


def reject_region(inside_region):
    """The candidates a curator clears in one action: those outside the region."""
    return np.flatnonzero(~np.asarray(inside_region, dtype=bool))


def first_run_length(labels):
    """The length of the leading stretch of candidates that share one label."""
    labels = np.asarray(labels, dtype=bool)
    if labels.size == 0:
        return 0
    change = np.flatnonzero(np.diff(labels))
    return int(change[0] + 1) if change.size else int(labels.size)
