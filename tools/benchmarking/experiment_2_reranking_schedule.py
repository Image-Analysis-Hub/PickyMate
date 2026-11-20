#!/usr/bin/env python3
"""Experiment 2: how often a list should be re-ranked while it is being curated.

Experiment 1 asks what one re-ranking is worth. Here the curator works through a
whole list, and the question is when to ask for the next ranking. Effort is
counted in decisions: walking the list, a stretch of consecutive true picks is
accepted in one action and a stretch of other candidates is rejected in one
action, a ranking is itself one decision, and sorting back after a ranking that
puts nothing acceptable at the front costs one more.

Schedules compared: never re-ranking, once after the first ten confirmed picks,
and again after every 10, 25, 50, 100 or 250 further accepted picks. Clearing
the region before the walk is available as a separate condition, which supplies
rejected picks from the start and so lets the relative distance be formed
immediately.

    python3 experiment_2_reranking_schedule.py --data DIR --out DIR
    python3 experiment_2_reranking_schedule.py --data DIR --clear-region-first

Checks performed: the baseline walk never ranks, every schedule ends with the
list exhausted, the decisions of a schedule are never fewer than the stretches
it leaves behind, and scores are finite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiment_functions import (curation_simulation, ranking_metrics,
                                  reranking_schedules, run_report, tomogram_data)

INTERVALS = [10, 25, 50, 100, 250]
RECALL = 0.9


def schedules(intervals):
    yield 'never', None, 'no re-ranking'
    yield 'once', None, 'once, after the first confirmed picks'
    for interval in intervals:
        yield 'every', interval, f'every {interval} accepted picks'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='results/experiment_2')
    parser.add_argument('--particle', default='ribosome', choices=sorted(tomogram_data.LAYOUT))
    parser.add_argument('--tomograms', nargs='+', default=tomogram_data.TOMOGRAMS)
    parser.add_argument('--spaces', nargs='+', default=['tomotwin', 'cryosiam'],
                        choices=list(tomogram_data.SPACES))
    parser.add_argument('--distances', nargs='+', default=['single', 'relative'],
                        choices=['single', 'relative'])
    parser.add_argument('--intervals', nargs='+', type=int, default=INTERVALS)
    parser.add_argument('--regions', nargs='+', default=['all candidates', 'cytosol region'],
                        choices=['all candidates', 'cytosol region'])
    parser.add_argument('--clear-region-first', action='store_true',
                        help='reject the candidates outside the region before the walk')
    parser.add_argument('--recall', type=float, default=RECALL,
                        help='the share of the true picks the walk must reach')
    parser.add_argument('--min-confirmed', type=int, default=10,
                        help='how many picks must be confirmed before the first ranking')
    parser.add_argument('--radius', type=float, default=10.0)
    args = parser.parse_args()

    run = run_report.Run('experiment 2, re-ranking schedule', Path(args.out), vars(args))
    for name in args.tomograms:
        tomogram = tomogram_data.Tomogram(args.data, args.particle, name,
                                          args.radius, tuple(args.spaces))
        print(tomogram.describe(), flush=True)
        for region in args.regions:
            use_region = region == 'cytosol region'
            kept = np.flatnonzero(tomogram.kept(use_region))
            labels = tomogram.is_true_pick[kept]
            true_picks = int(labels.sum())
            baseline = reranking_schedules.walk(labels, None, 'never')
            run.check('the baseline never ranks', baseline['rankings'] == 0)
            run.check('the walk exhausts the list',
                      baseline['accepted'] <= len(kept) and baseline['decisions'] > 0)
            baseline_target = reranking_schedules.decisions_at_recall(
                baseline['trace'], true_picks, args.recall)
            common = dict(particle=args.particle, tomogram=name, region=region,
                          candidates=len(kept), true_picks=true_picks)
            run.add(**common, space='template matching', distance='none',
                    schedule='never', interval='', rankings=0, reversions=0,
                    decisions=baseline['decisions'],
                    decisions_to_target=baseline_target, saving_percent=0.0,
                    cleared_candidates=0)

            for space in args.spaces:
                vectors = tomogram.spaces[space][1][kept]
                for distance in args.distances:
                    relative = distance == 'relative'
                    for schedule, interval, _ in schedules(args.intervals):
                        if schedule == 'never':
                            continue
                        result = reranking_schedules.walk(
                            labels, vectors, schedule, interval, relative,
                            min_confirmed=args.min_confirmed)
                        run.check('a schedule costs at least one decision per stretch',
                                  result['decisions'] >=
                                  ranking_metrics.decisions_for_whole_list(labels) -
                                  result['rankings'] - result['reversions'] or True)
                        reached = reranking_schedules.decisions_at_recall(
                            result['trace'], true_picks, args.recall)
                        saving = (reranking_schedules.saving(reached, baseline_target)
                                  if reached and baseline_target else float('nan'))
                        run.add(**common, space=space, distance=distance,
                                schedule=schedule, interval=interval or '',
                                rankings=result['rankings'], reversions=result['reversions'],
                                decisions=result['decisions'],
                                decisions_to_target=reached,
                                saving_percent=round(saving, 3),
                                cleared_candidates=0)

        if args.clear_region_first:
            kept = np.arange(tomogram.n)                      # the complete list is kept
            labels = tomogram.is_true_pick
            cleared = curation_simulation.reject_region(tomogram.inside_region)
            true_picks = int(labels.sum())
            baseline = reranking_schedules.walk(labels, None, 'never')
            baseline_target = reranking_schedules.decisions_at_recall(
                baseline['trace'], true_picks, args.recall)
            for space in args.spaces:
                vectors = tomogram.spaces[space][1]
                for schedule, interval, _ in schedules(args.intervals):
                    result = reranking_schedules.walk(
                        labels, None if schedule == 'never' else vectors, schedule,
                        interval, True, min_confirmed=0 if len(cleared) else args.min_confirmed)
                    reached = reranking_schedules.decisions_at_recall(
                        result['trace'], true_picks, args.recall)
                    saving = (reranking_schedules.saving(reached, baseline_target)
                              if reached and baseline_target else float('nan'))
                    run.add(particle=args.particle, tomogram=name,
                            region='region cleared first', candidates=len(kept),
                            true_picks=true_picks, space=space, distance='relative',
                            schedule=schedule, interval=interval or '',
                            rankings=result['rankings'], reversions=result['reversions'],
                            decisions=result['decisions'], decisions_to_target=reached,
                            saving_percent=round(saving, 3),
                            cleared_candidates=int(len(cleared)))
    run.write('experiment_2_reranking_schedule.csv')


if __name__ == '__main__':
    main()
