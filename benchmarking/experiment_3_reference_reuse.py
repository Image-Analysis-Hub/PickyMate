#!/usr/bin/env python3
"""Experiment 3: ranking a tomogram from the picks of the tomograms already curated.

A dataset is curated one tomogram after another, and each finished tomogram
leaves behind a set of confirmed picks and a set of rejected ones. PickyMate can
pool those sets and rank a fresh tomogram from them before anything has been
decided in it, which is the References from control of the ranking row. This
experiment measures what that is worth.

Because no reference comes from the target, its list is not divided: the whole
candidate list is ranked and scored, once over all candidates and once over the
candidates inside the cytosol region. The curated tomograms are accumulated one
at a time, and because the order in which they are added is arbitrary the
accumulation is repeated over several orders and averaged within each target.

    python3 experiment_3_reference_reuse.py --data DIR --curated DIR --out DIR
    python3 experiment_3_reference_reuse.py --data DIR --curated DIR --orders 5

Checks performed: a target never contributes to its own reference set, the
confirmed and rejected sets are disjoint, scores are finite, and the
template-matching order of each target is scored by the same code as the
rankings, so the comparison is internal to the run.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from experiment_functions import (mahalanobis_ranking, ranking_metrics,
                                  reference_pooling, run_report, tomogram_data)

RECALL_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='the dataset root')
    parser.add_argument('--curated', required=True,
                        help='the folder of curated working directories, one per tomogram')
    parser.add_argument('--out', default='results/experiment_3')
    parser.add_argument('--particle', default='ribosome', choices=sorted(tomogram_data.LAYOUT))
    parser.add_argument('--tomograms', nargs='+', default=tomogram_data.TOMOGRAMS)
    parser.add_argument('--spaces', nargs='+', default=['tomotwin', 'cryosiam'],
                        choices=list(tomogram_data.SPACES))
    parser.add_argument('--orders', type=int, default=5,
                        help='how many accumulation orders to average')
    parser.add_argument('--recall-levels', nargs='+', type=float, default=RECALL_LEVELS)
    parser.add_argument('--radius', type=float, default=10.0)
    args = parser.parse_args()

    run = run_report.Run('experiment 3, reuse of curated picks', Path(args.out), vars(args))
    curated_root = Path(args.curated)
    tomograms, confirmed_rows, rejected_rows = {}, {}, {}
    for name in args.tomograms:
        tomograms[name] = tomogram_data.Tomogram(args.data, args.particle, name,
                                                 args.radius, tuple(args.spaces))
        confirmed_rows[name], rejected_rows[name] = reference_pooling.curated_sets(
            curated_root / f'{name}_picks')
        run.check('the curated sets are disjoint',
                  not np.intersect1d(confirmed_rows[name], rejected_rows[name]).size)
        print(f'{tomograms[name].describe()}  confirmed {len(confirmed_rows[name]):>6} '
              f'rejected {len(rejected_rows[name]):>6}', flush=True)

    for target in args.tomograms:
        tomogram = tomograms[target]
        for region in ('all candidates', 'cytosol region'):
            use_region = region == 'cytosol region'
            kept = np.flatnonzero(tomogram.kept(use_region))
            labels = tomogram.is_true_pick[kept]
            particles = int(labels.sum())
            common = dict(particle=args.particle, target=target, region=region,
                          candidates=len(kept), true_picks=particles)
            baseline_ap = ranking_metrics.average_precision_of(labels, particles)
            baseline = {level: ranking_metrics.decisions_to_recall(labels, particles, level)
                        for level in args.recall_levels}
            run.add(**common, space='template matching', distance='original',
                    curated_tomograms=0, accumulation_order=0, reference_picks=0,
                    average_precision=round(baseline_ap, 6),
                    **{f'decisions_to_{int(level * 100)}pct': baseline[level]
                       for level in args.recall_levels})

            others = [name for name in args.tomograms if name != target]
            orders = reference_pooling.accumulation_orders(others, args.orders)
            for index, order in enumerate(orders):
                for count in range(1, len(order) + 1):
                    used = order[:count]
                    run.check('the target never ranks itself', target not in used)
                    for space in args.spaces:
                        vectors = tomogram.spaces[space][1]
                        confirmed, rejected = reference_pooling.pooled_references(
                            tomograms, space, used, confirmed_rows, rejected_rows)
                        if len(confirmed) < mahalanobis_ranking.MIN_REFERENCE:
                            continue
                        for distance, relative in (('single', False), ('relative', True)):
                            if relative and len(rejected) < mahalanobis_ranking.MIN_REFERENCE:
                                continue
                            scores = mahalanobis_ranking.distances(confirmed, vectors[kept])
                            if relative:
                                scores = scores - mahalanobis_ranking.distances(
                                    rejected, vectors[kept])
                            run.note_finite(scores)
                            ranked = labels[np.argsort(scores, kind='stable')]
                            decisions = {level: ranking_metrics.decisions_to_recall(
                                ranked, particles, level) for level in args.recall_levels}
                            run.check('reaching more true picks never costs fewer decisions',
                                      all(a is None or b is None or a <= b for a, b in
                                          zip([decisions[l] for l in args.recall_levels],
                                              [decisions[l] for l in args.recall_levels][1:])))
                            run.add(**common, space=space, distance=distance,
                                    curated_tomograms=count, accumulation_order=index,
                                    reference_picks=len(confirmed) + (len(rejected) if relative else 0),
                                    average_precision=round(
                                        ranking_metrics.average_precision_of(ranked, particles), 6),
                                    **{f'decisions_to_{int(level * 100)}pct': decisions[level]
                                       for level in args.recall_levels})
        print(f'{target} done, {len(run.rows)} rows', flush=True)
    run.write('experiment_3_reference_reuse.csv')


if __name__ == '__main__':
    main()
