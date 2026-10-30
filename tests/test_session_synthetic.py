"""Switching tomograms in a session, on synthetic data: the first open
imports, a reopen does not, and the closed tomogram is really released --
its windows deleted and its volume unreferenced -- not merely hidden."""
import csv
import gc
import weakref

import shiboken6
from PySide6 import QtWidgets

from boxer import categories
from boxer.session import Session
from boxer.session_window import SessionController
from conftest import options, write_tomogram


def test_switch_releases_previous_tomogram(tmp_path, qapp):
    rows = [write_tomogram(tmp_path, "TS_A", seed=0), write_tomogram(tmp_path, "TS_B", seed=1)]
    manifest = tmp_path / "session.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ctl = SessionController(Session.from_manifest(str(manifest)), options())
    ctl.start("TS_A")
    win = ctl.current
    assert win.session_name == "TS_A" and len(win.boxes) == 20
    volume = weakref.ref(win.data)
    gallery = win.boxesviewer

    ctl.open("TS_B")
    qapp.processEvents()
    gc.collect()
    assert ctl.current.session_name == "TS_B" and len(ctl.current.boxes) == 20
    assert not shiboken6.isValid(win) and not shiboken6.isValid(gallery), \
        "the previous tomogram's windows must be deleted, not hidden"
    assert volume() is None, "the previous tomogram's volume is still referenced"

    ctl.open("TS_A")
    normal = [c for c in ctl.current.sets if c not in categories.FIXED_CLASS_IDS]
    assert len(normal) == 1, "reopening a tomogram must not import its particle list again"
    shown = [w for w in QtWidgets.QApplication.topLevelWidgets()
             if w.isVisible() and type(w).__name__ in ("TomoBoxer", "ParticleListPanel")]
    assert len(shown) == 2, [type(w).__name__ for w in shown]

    cur = ctl.current
    cur._confirm_close = lambda: True
    cur.close()
    qapp.processEvents()
    assert ctl.current is None and not ctl.navigator.isVisible()
