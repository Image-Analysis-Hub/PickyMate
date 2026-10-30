#!/usr/bin/env python3
"""Experiment 1: how many confirmed picks one re-ranking needs to improve a list.

The candidate list of a tomogram is kept in its template-matching score order
and dealt alternately into two halves. The labelling half is examined from the
top until a chosen number of picks has been confirmed; the evaluation half is
then re-ranked once from those picks and scored. Repeating that for two, five,
ten and up to three hundred confirmed picks says how much curation a single
ranking needs before it is worth asking for.

The cytosol mask enters in three ways, each of which is measured separately:

    all candidates   the mask is not used
    cytosol region   the candidates outside the mask are removed before anything
                     is scored
    region rejected  the complete list is kept and the candidates of the
                     labelling half outside the mask are rejected in one action,
                     so they serve as rejected picks from the first ranking on

Each distance is measured in each space given with --spaces: the full embeddings
of a model and the two-dimensional UMAP supplied with it.

    python3 experiment_1_confirmed_picks.py --data DIR --out DIR
    python3 experiment_1_confirmed_picks.py --data DIR --tomograms TS_026 TS_027

Checks performed before anything is written: the two halves are disjoint and
cover the list, the reference set holds exactly the requested number of
confirmed picks and never touches the evaluation half, scores are finite, and a
condition that cannot be formed is recorded as absent rather than replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiment_functions import (curation_simulation, evaluation_split,
                                  mahalanobis_ranking, ranking_metrics,
                                  run_report, tomogram_data)

CONFIRMED_PICKS = [2, 5, 10, 25, 50, 100, 200, 300]
CONDITIONS = [('all candidates', False, False),
              ('cytosol region', True, False),
              ('region rejected', False, True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='the dataset root')
    parser.add_argument('--out', default='results/experiment_1', help='where to write')
    parser.add_argument('--particle', default='ribosome', choices=sorted(tomogram_data.LAYOUT))
    parser.add_argument('--tomograms', nargs='+', default=tomogram_data.TOMOGRAMS)
    parser.add_argument('--spaces', nargs='+', default=['tomotwin', 'cryosiam'],
                        choices=list(tomogram_data.SPACES))
    parser.add_argument('--confirmed-picks', nargs='+', type=int, default=CONFIRMED_PICKS)
    parser.add_argument('--radius', type=float, default=10.0,
                        help='how close a candidate must sit to count as a true pick')
    args = parser.parse_args()

    run = run_report.Run('experiment 1, confirmed picks', Path(args.out), vars(args))
    for name in args.tomograms:
        tomogram = tomogram_data.Tomogram(args.data, args.particle, name,
                                          args.radius, tuple(args.spaces))
        print(tomogram.describe(), flush=True)
        for condition, use_region, reject_region in CONDITIONS:
            kept = np.flatnonzero(tomogram.kept(use_region))
            labels = tomogram.is_true_pick[kept]
            labelling, evaluation = evaluation_split.deal_alternately(len(kept))
            run.check('the halves are disjoint and cover the list',
                      evaluation_split.check_split(labelling, evaluation, len(kept)))
            evaluation_labels = labels[evaluation]
            particles = int(evaluation_labels.sum())
            baseline = ranking_metrics.average_precision_of(evaluation_labels, particles)
            common = dict(particle=args.particle, tomogram=name, condition=condition,
                          candidates=len(kept), evaluation_candidates=len(evaluation),
                          true_picks_in_evaluation=particles)
            run.add(**common, space='template matching', distance='original',
                    confirmed_picks=0, examined_candidates=0, rejected_picks=0,
                    average_precision=round(baseline, 6))

            cleared = (curation_simulation.reject_region(tomogram.inside_region[kept][labelling])
                       if reject_region else np.empty(0, dtype=np.int64))
            for space in args.spaces:
                vectors = tomogram.spaces[space][1][kept]
                for wanted in args.confirmed_picks:
                    curated = curation_simulation.examine_until_confirmed(
                        labels[labelling], wanted)
                    if curated is None:
                        continue              # the half holds too few true picks
                    confirmed, rejected, examined = curated
                    run.check('the reference holds the requested picks', len(confirmed) == wanted)
                    confirmed = labelling[confirmed]
                    rejected = np.concatenate([labelling[rejected], labelling[cleared]]) \
                        if reject_region else labelling[rejected]
                    rejected = np.unique(rejected)
                    rejected = rejected[~np.isin(rejected, confirmed)]
                    run.check('the reference never touches the evaluation half',
                              not np.intersect1d(np.concatenate([confirmed, rejected]),
                                                 evaluation).size)
                    for distance, relative in (('single', False), ('relative', True)):
                        if reject_region and not relative:
                            continue          # rejecting a region only feeds the relative distance
                        scores = mahalanobis_ranking.score(vectors, confirmed, rejected,
                                                           evaluation, relative)
                        if scores is None:
                            continue          # the distance cannot be formed yet
                        run.note_finite(scores)
                        order = np.argsort(scores, kind='stable')
                        value = ranking_metrics.average_precision_of(
                            evaluation_labels[order], particles)
                        run.add(**common, space=space, distance=distance,
                                confirmed_picks=wanted, examined_candidates=examined,
                                rejected_picks=len(rejected),
                                average_precision=round(value, 6),
                                improvement_percent=round(
                                    ranking_metrics.improvement(value, baseline), 2))
    run.write('experiment_1_confirmed_picks.csv')


if __name__ == '__main__':
    main()
