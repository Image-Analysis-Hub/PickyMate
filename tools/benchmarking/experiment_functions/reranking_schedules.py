"""Walking a list while re-ranking it, which is what Experiment 2 measures.

The curator does not rank once and then read to the end. He confirms a few
picks, ranks, acts on what rises, confirms more, and ranks again. Each ranking
is itself a decision, and so is sorting back to the template-matching order when
a ranking fails to put anything acceptable at the front. A schedule is therefore
worth its cost only if the stretches it creates save more decisions than the
rankings spend.

Schedules
---------
never        keep the template-matching order throughout; the baseline
once         rank once, as soon as the first picks have been confirmed
every-K      rank again after every K further confirmed picks
before-every-run  rank before every stretch; the over-eager extreme

A ranking that does not place at least `bulk` acceptable candidates at the front
is treated as a failure: it is reverted at the cost of one further decision and
the walk continues in the order it had.
"""
from __future__ import annotations

import numpy as np

from . import mahalanobis_ranking as ranking
from .curation_simulation import first_run_length


def walk(labels, vectors, schedule, every_k=None, relative=False,
         min_confirmed=10, bulk=2, record_trace=True):
    """Walk one list under one schedule and count what it costs.

    Parameters
    ----------
    labels : (n,) boolean, true picks of the list, in the order it arrives
    vectors : (n, d) embeddings of the same candidates, or None for `never`
    schedule : 'never', 'once', 'every' or 'before_every_run'
    every_k : the interval, in accepted picks, when the schedule is 'every'

    Returns a dictionary with the decisions spent, the picks accepted, the
    number of rankings and reversions, and the trace of (decisions, accepted)
    from which any recall target can be read.
    """
    labels = np.asarray(labels, dtype=bool)
    remaining = np.arange(len(labels))
    confirmed, rejected = [], []
    decisions = accepted = rankings = reversions = since_ranking = 0
    ranked_now = False
    trace = [(0, 0)]

    while remaining.size:
        due = (schedule != 'never' and vectors is not None
               and len(confirmed) >= min_confirmed
               and (schedule == 'before_every_run'
                    or (schedule == 'once' and rankings == 0)
                    or (schedule == 'every' and since_ranking >= every_k)))
        order = remaining
        if due:
            scores = ranking.score(vectors, confirmed, rejected, remaining, relative)
            if scores is not None:
                decisions += 1
                rankings += 1
                since_ranking = 0
                proposed = remaining[np.argsort(scores, kind='stable')]
                if labels[proposed[0]] and first_run_length(labels[proposed]) >= bulk:
                    order, ranked_now = proposed, True
                else:
                    # nothing acceptable in bulk at the front: the curator sorts
                    # back to the template-matching order, which costs a decision
                    decisions += 1
                    reversions += 1
                    ranked_now = False
                    order = np.sort(remaining)
        elif ranked_now:
            order = remaining

        length = first_run_length(labels[order])
        stretch = order[:length]
        decisions += 1
        if labels[stretch[0]]:
            confirmed.extend(stretch.tolist())
            accepted += length
            since_ranking += length
        else:
            rejected.extend(stretch.tolist())
        keep = np.ones(len(order), dtype=bool)
        keep[:length] = False
        remaining = order[keep]
        if record_trace:
            trace.append((decisions, accepted))

    return {'decisions': decisions, 'accepted': accepted, 'rankings': rankings,
            'reversions': reversions, 'trace': np.array(trace),
            'true_picks': int(labels.sum()), 'candidates': int(labels.size)}


def decisions_at_recall(trace, true_picks, target=0.9):
    """Decisions spent by the time `target` of the true picks were accepted."""
    need = target * true_picks
    reached = np.flatnonzero(trace[:, 1] >= need)
    return int(trace[reached[0], 0]) if reached.size else None


def saving(decisions, baseline_decisions):
    """The percentage of decisions a schedule saves against never re-ranking."""
    if not baseline_decisions:
        return float('nan')
    return 100.0 * (baseline_decisions - decisions) / baseline_decisions
