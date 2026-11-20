"""The three original end-to-end tests, each a script on the real
EMPIAR-10988 data (see their docstrings), run as they were written: as a
process of their own, since each builds its QApplication and temporary
working directories at import. They print "SKIPPED" and exit 0 when the
dataset is absent, which becomes a pytest skip here."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["test_session.py", "test_session_map.py", "test_umap_selection.py"]


@pytest.mark.dataset
@pytest.mark.parametrize("script", SCRIPTS)
def test_script(script):
    run = subprocess.run([sys.executable, os.path.join(HERE, script)],
                         capture_output=True, text=True, timeout=1800,
                         env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    if run.stdout.startswith("SKIPPED"):
        pytest.skip(run.stdout.strip())
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
