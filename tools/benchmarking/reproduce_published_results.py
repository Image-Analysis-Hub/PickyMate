#!/usr/bin/env python3
"""Run the three experiments on a subset and compare with the published results.

The three tables of Supplementary Note 5 were produced by these experiments. This
script runs each of them on a few tomograms and compares every value it obtains
with the published one, so that a change to the benchmark that alters a result
is caught rather than discovered later.

    python3 reproduce_published_results.py --data DATA --published DIR

`--published` is the folder holding the result files of the published runs. The
script reports how many values are identical and stops with a non-zero status if
any differ.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-6


def read(path):
    return list(csv.DictReader(Path(path).open()))


def run(script, arguments):
    command = [sys.executable, str(HERE / script)] + arguments
    print('    ' + ' '.join(command[1:]), flush=True)
    subprocess.run(command, check=True, cwd=HERE,
                   stdout=subprocess.DEVNULL, stderr=sys.stderr)


def compare(label, pairs):
    same = different = 0
    for key, (got, want) in pairs.items():
        if want is None:
            continue
        if isinstance(got, float) and abs(got - want) < TOLERANCE or got == want:
            same += 1
        else:
            different += 1
            print(f'      differs at {key}: {got} against {want}')
    print(f'  {label}: {same} identical, {different} different')
    return different


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--curated', help='the curated working directories, for experiment 3')
    parser.add_argument('--published', required=True,
                        help='the folder holding the published result files')
    parser.add_argument('--tomograms', nargs='+', default=['TS_026', 'TS_027'])
    args = parser.parse_args()
    published = Path(args.published)
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print('experiment 1')
        run('experiment_1_confirmed_picks.py',
            ['--data', args.data, '--out', str(tmp / 'e1'), '--spaces', 'tomotwin',
             '--tomograms', *args.tomograms])
        ladder = {(r['tomogram'], r['mask'], r['scoring'], r['rung']):
                  round(float(r['average_precision']), 6)
                  for r in read(published / 'summary_confirmed_ladder.csv')
                  if r['space'] in ('tomotwin', 'template_matching')
                  and r['average_precision'] not in ('', 'None')}
        rejected = {(r['tomogram'], 'rej', 'relative', r['rung']):
                    round(float(r['average_precision']), 6)
                    for r in read(published / 'summary_region_rejected.csv')
                    if r['space'] == 'tomotwin' and r['average_precision'] not in ('', 'None')}
        pairs = {}
        for row in read(tmp / 'e1/experiment_1_confirmed_picks.csv'):
            mask = {'all candidates': '0', 'cytosol region': '1'}.get(row['condition'], 'rej')
            scoring = 'original' if row['space'] == 'template matching' else row['distance']
            key = (row['tomogram'], mask, scoring, row['confirmed_picks'])
            want = ladder.get(key, rejected.get(key))
            pairs[key] = (round(float(row['average_precision']), 6), want)
        failures += compare('experiment 1, average precision', pairs)

        print('experiment 2')
        run('experiment_2_reranking_schedule.py',
            ['--data', args.data, '--out', str(tmp / 'e2'), '--spaces', 'tomotwin',
             '--distances', 'single', '--tomograms', *args.tomograms])
        walk = {}
        for r in read(published / 'task3_summary_ribosome_tomotwin_positive.csv'):
            region = 'all candidates' if r['mask'] == '0' else 'cytosol region'
            schedule = r['schedule']
            key = ((r['tomogram'], region, 'never', '') if schedule == 'never' else
                   (r['tomogram'], region, 'once', '') if schedule == 'once' else
                   (r['tomogram'], region, 'every', schedule.split('-')[1]))
            walk[key] = r['decisions_to_90pct']
        pairs = {}
        for row in read(tmp / 'e2/experiment_2_reranking_schedule.csv'):
            key = (row['tomogram'], row['region'], row['schedule'], str(row['interval']))
            pairs[key] = (str(row['decisions_to_target']), walk.get(key))
        failures += compare('experiment 2, decisions at ninety per cent', pairs)

        if args.curated:
            print('experiment 3')
            run('experiment_3_reference_reuse.py',
                ['--data', args.data, '--curated', args.curated, '--out', str(tmp / 'e3'),
                 '--spaces', 'tomotwin', '--orders', '1', '--tomograms', *args.tomograms])
            transfer = {(r['target'], r['mask'], r['scoring'], r['n_curated']):
                        round(float(r['average_precision']), 6)
                        for r in read(published / 'transfer_full_tomotwin.csv')
                        if r['permutation'] == '0'}
            pairs = {}
            for row in read(tmp / 'e3/experiment_3_reference_reuse.csv'):
                if row['curated_tomograms'] not in ('0', '1'):
                    continue
                mask = '0' if row['region'] == 'all candidates' else '1'
                scoring = 'original' if row['space'] == 'template matching' else row['distance']
                key = (row['target'], mask, scoring, row['curated_tomograms'] or '0')
                pairs[key] = (round(float(row['average_precision']), 6), transfer.get(key))
            failures += compare('experiment 3, average precision', pairs)

    print('\n' + ('every compared value is identical to the published result'
                  if not failures else f'{failures} values differ'))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
