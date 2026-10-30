"""Writing results, and the checks a benchmark run makes on itself.

Every experiment writes one table of results and one report. The report records
what was run, how long it took, and the checks that passed, so that a table can
always be traced back to the run that produced it.

The checks are part of the benchmark rather than an afterthought: a run that
cannot reproduce its own baseline, or that produces a score that is not finite,
stops instead of writing a result.
"""
from __future__ import annotations

import csv
import json
import platform
import time

import numpy as np


class Run:
    """Collects rows, counts checks, and writes the table and the report."""

    def __init__(self, name, directory, settings):
        self.name = name
        self.directory = directory
        self.settings = dict(settings)
        self.rows = []
        self.checks = {}
        self.started = time.perf_counter()
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, **row):
        self.rows.append(row)

    def check(self, label, condition, detail=''):
        """Record a check; stop the run if it fails."""
        if not condition:
            raise AssertionError(f'{self.name}: {label} failed {detail}')
        self.checks[label] = self.checks.get(label, 0) + 1

    def note_finite(self, scores):
        self.check('scores are finite', bool(np.all(np.isfinite(scores))))

    def write(self, table_name='results.csv', report_name='run_report.json'):
        if not self.rows:
            raise SystemExit(f'{self.name}: nothing to write')
        fields = sorted({key for row in self.rows for key in row})
        table = self.directory / table_name
        with table.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        report = self.directory / report_name
        report.write_text(json.dumps({
            'experiment': self.name,
            'settings': self.settings,
            'rows': len(self.rows),
            'checks': self.checks,
            'minutes': round((time.perf_counter() - self.started) / 60, 2),
            'python': platform.python_version(),
            'numpy': np.__version__,
        }, indent=2))
        print(f'wrote {table} ({len(self.rows)} rows) and {report}')
        return table
