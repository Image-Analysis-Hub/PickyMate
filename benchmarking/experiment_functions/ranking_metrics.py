"""What a ranking is worth: precision, average precision, and decisions.

Average precision summarises an order: it is the area under the precision-recall
curve of the list read from the top, computed without interpolation. Recall is
counted against the true picks present in the list being ranked, so the measure
describes the order of that list and not the fraction of the particles of the
tomogram the list happens to contain.

Decisions count effort differently. Walking a ranked list, a stretch of
consecutive true picks is accepted in one action and a stretch of consecutive
other candidates is rejected in one action, so the number of such stretches is
the number of decisions the curator pays. A better order produces longer
stretches and therefore fewer decisions.
"""
from __future__ import annotations

import numpy as np


def precision_recall_curve(labels, particle_count):
    """Precision, recall, F1 and the fraction of the list read, at every depth."""
    hit = np.asarray(labels, dtype=np.float64)
    depth = np.arange(1, len(hit) + 1, dtype=np.float64)
    found = np.cumsum(hit)
    precision = found / depth
    recall = found / particle_count if particle_count else np.zeros_like(depth)
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, depth / len(hit)


def average_precision(precision, recall):
    """The area under the precision-recall curve, without interpolation."""
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def average_precision_of(labels, particle_count):
    precision, recall, _, _ = precision_recall_curve(labels, particle_count)
    return average_precision(precision, recall)


def stretches(labels):
    """Every maximal stretch of candidates sharing a label: (label, length)."""
    labels = np.asarray(labels, dtype=bool)
    if labels.size == 0:
        return np.empty(0, dtype=bool), np.empty(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [labels.size])))
    return labels[starts], lengths


def decisions_for_whole_list(labels):
    """How many decisions it costs to triage an order from end to end."""
    return int(len(stretches(labels)[1]))


def decisions_to_recall(labels, particle_count, target=0.9):
    """Decisions spent by the time `target` of the true picks have been accepted.

    Returns None when the order never reaches the target, which happens when the
    list does not contain that many true picks.
    """
    need = target * particle_count
    values, lengths = stretches(labels)
    spent = accepted = 0
    for is_true, length in zip(values, lengths):
        spent += 1
        if is_true:
            accepted += length
            if accepted >= need:
                return spent
    return None


def improvement(value, baseline):
    """The percentage change of a value against the order it is compared with."""
    return 100.0 * (value - baseline) / baseline if baseline else float('nan')
