"""The session map: one UMAP over every tomogram of a session.

Computes a real map in the background on copies of three EMPIAR-10988
working directories, then checks cross-tomogram selection, the navigator's
counts, the view-only browser, Escape, switching tomograms (including in
the middle of a computation) and out-of-date detection. Needs umap-learn in
the running environment (requirements.txt / environment.yml); skipped if it
is missing or the dataset is absent.

    QT_QPA_PLATFORM=offscreen python3 tests/test_session_map.py
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

DATA = "/media/mohamad/LinuxStorage/Ali_ParticlePicking/PickyMate_EMPIAR10988"
if not os.path.isdir(DATA):
    print(f"SKIPPED: {DATA} not found")
    sys.exit(0)
if subprocess.run([sys.executable, "-c", "import umap"], capture_output=True).returncode != 0:
    print(f"SKIPPED: umap-learn is not installed in {sys.executable} "
          f"(conda env create -f environment.yml)")
    sys.exit(0)

TMP = tempfile.mkdtemp(prefix="pickymate_session_map_test_")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from boxer import metadata
from boxer.session import Session
from boxer.session_browser import MAX_THUMBNAILS
from boxer.session_window import SessionController

NAMES = ["TS_026", "TS_037", "TS_043"]      # TS_043 records another machine's paths
for t in NAMES:
    shutil.copytree(f"{DATA}/Ribosomes_PickyMate/{t}_picks", f"{TMP}/{t}_picks")
manifest = os.path.join(TMP, "session.csv")
with open(manifest, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tomogram", "working_dir", "cryosiam_embeddings", "cryosiam_umap",
                "tomotwin_embeddings", "tomotwin_umap"])
    for t in NAMES:
        w.writerow([f"{DATA}/Tomograms/{t}.rec", f"{t}_picks",
                    f"{DATA}/CryoSiam/Ribosomes_embeds/{t}_embeds.h5",
                    f"{DATA}/CryoSiam/Ribosomes_embeds/vis/{t}_embeds_umap_data.csv",
                    f"{DATA}/TomoTwin/Ribosomes_embeddings/{t}.temb",
                    f"{DATA}/TomoTwin/Ribosomes_embeddings/{t}.tumap"])
base = argparse.Namespace(mode="3D", label=None, clean=False, apix=None, working_dir=None,
                          particle_lists=None, native_memmap=False, companion_tomogram=None,
                          embedded_particle_list=None, cryosiam_embeddings=None,
                          cryosiam_umap=None, tomotwin_embeddings=None, tomotwin_umap=None)


def wait_for(cond, seconds=600):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > seconds:
            raise AssertionError("timed out")
    app.processEvents()


def drag(canvas, pts, mod=Qt.NoModifier):
    QTest.mousePress(canvas, Qt.LeftButton, mod, pts[0])
    for p in pts[1:]:
        QtWidgets.QApplication.sendEvent(canvas, QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(p), QtCore.QPointF(canvas.mapToGlobal(p)),
            Qt.NoButton, Qt.LeftButton, mod))
    QTest.mouseRelease(canvas, Qt.LeftButton, mod, pts[-1])
    app.processEvents()


def square(x0, y0, x1, y1):
    return [QPoint(x0, y0), QPoint(x1, y0), QPoint(x1, y1), QPoint(x0, y1), QPoint(x0, y0)]


try:
    ctl = SessionController(Session.from_manifest(manifest), base)
    ctl.start("TS_026")
    win = ctl.current
    win.open_umap_window()
    app.processEvents()
    uw = win.umap_window

    # -- 1. offered, not yet computed ------------------------------------------------
    keys = [uw.space_combo.itemData(i) for i in range(uw.space_combo.count())]
    assert keys == ["tomotwin_umap", "cryosiam_umap", "session:tomotwin", "session:cryosiam"], keys
    uw.space_combo.setCurrentIndex(keys.index("session:tomotwin"))
    app.processEvents()
    assert uw.session_row.isVisibleTo(uw) and uw.compute_button.text() == "Compute"
    assert len(uw.canvas._xy) == 0 and "not been computed" in uw.canvas.empty_message
    print("OK: the map list offers both session maps after this tomogram's own; "
          "not computed yet, it says so and offers Compute")

    # -- 2. compute in the background ---------------------------------------------------
    progress = []
    ctl.session_map_job.progress.connect(lambda n, t: progress.append((n, t)))
    uw._compute()
    assert ctl.session_map_job.running() and uw.cancel_button.isVisibleTo(uw)
    wait_for(lambda: not ctl.session_map_job.running())
    m, status = ctl.session_map.load("tomotwin")
    assert status == "ok" and [n for n, _ in m.layout] == NAMES, (status, m and m.layout)
    assert any(t for _, t in progress) and progress[-1][0] == progress[-1][1]
    tables = {n: ctl.session.particle_table(n) for n in NAMES[1:]}
    live = sum(1 for b in win.boxes if b[7] is not None)
    assert len(uw.canvas._xy) == live + sum(len(t["uids"]) for t in tables.values())
    assert uw.compute_button.text() == "Recompute" and not uw.progress_bar.isVisibleTo(uw)
    print(f"OK: computed in the background ({len(progress)} progress updates) -- "
          f"{len(uw.canvas._xy)} particles from all 3 tomograms on one map")

    # -- 3. one lasso selects across tomograms --------------------------------------------
    c = uw.canvas
    W, H = c.width(), c.height()
    box = square(int(W * .25), int(H * .25), int(W * .75), int(H * .75))
    drag(c, box)
    sx, sy = c._to_screen(c._xy)
    inside = (sx >= box[0].x()) & (sx <= box[2].x()) & (sy >= box[0].y()) & (sy <= box[2].y())
    tomo, uids, names = uw._session_points
    expect = {n: set(uids[inside & (tomo == t)].tolist()) for t, n in enumerate(names)}
    assert all(expect[n] for n in NAMES), {n: len(v) for n, v in expect.items()}
    assert win.umap_selection == expect["TS_026"]
    for n in NAMES[1:]:
        assert ctl.session.selection_uids(n) == expect[n], n
        on_disk = {r["uid"] for r in metadata.load_umap_selection(f"{TMP}/{n}_picks")}
        assert on_disk == expect[n]
    nav = {ctl.navigator.table.item(r, 0).text().strip(" ▶"): ctl.navigator.table.item(r, 5).text()
           for r in range(len(NAMES))}
    assert nav == {n: str(len(expect[n])) for n in NAMES}, nav
    print("OK: one lasso selects in every tomogram -- " +
          ", ".join(f"{n}: {len(expect[n])}" for n in NAMES) +
          "; this tomogram's go to its tab, the others' to their working directories, "
          "and the navigator counts them")

    # -- 4. colour by tomogram ---------------------------------------------------------------
    uw.colour_combo.setCurrentIndex(uw.colour_combo.findData("tomogram"))
    app.processEvents()
    assert all(n in uw.legend.text() for n in NAMES)
    assert len(set((c._classes % len(NAMES)).tolist())) == len(NAMES)
    print("OK: colouring by tomogram gives each tomogram its own colour, with a legend")

    # -- 5. browse the other tomograms, view only ------------------------------------------------
    uw.open_browser()
    app.processEvents()
    br = uw._browser
    tabs = [br.tabs.tabText(i) for i in range(br.tabs.count())]
    assert tabs == [f"{n} ({len(expect[n])})" for n in NAMES[1:]], tabs
    for i, n in enumerate(NAMES[1:]):
        br.tabs.setCurrentIndex(i)
        app.processEvents()
        grid = br._tabs[n]["grid"]
        assert set(grid._images) == set(sorted(expect[n])[:MAX_THUMBNAILS]), n
        img = next(iter(grid._images.values()))
        assert img.ndim == 2 and img.shape[0] == img.shape[1] and np.isfinite(img).all()
    print("OK: Browse Other Tomograms shows exactly each other tomogram's selection as "
          "thumbnails cut from its file on disk")

    # -- 6. Escape on the session map clears every tomogram ------------------------------------
    QtWidgets.QApplication.setActiveWindow(uw)
    win._global_escape()
    app.processEvents()
    assert win.umap_selection == set()
    assert all(not ctl.session.selection_uids(n) for n in NAMES[1:])
    assert all(not os.path.isfile(metadata.umap_selection_path(f"{TMP}/{n}_picks")) for n in NAMES)
    print("OK: Escape on the session map clears the selection of every tomogram")

    # -- 7. switch: the other tomogram opens with its selection, the map is still there -----------
    drag(c, box)
    ctl.open("TS_037")
    app.processEvents()
    w2 = ctl.current
    assert w2.umap_selection == expect["TS_037"]
    assert w2.boxesviewer.tabs.indexOf(w2.boxesviewer.umap_grid) == 1
    w2.open_umap_window()
    app.processEvents()
    uw2 = w2.umap_window
    uw2.space_combo.setCurrentIndex(uw2.space_combo.findData("session:tomotwin"))
    app.processEvents()
    assert uw2.compute_button.text() == "Recompute" and len(uw2.canvas._xy) == len(c._xy)
    print("OK: the tomogram selected on the map opens with its UMAP Selection tab, and the "
          "session map is kept -- no recomputation after switching")

    # -- 8. a computation survives switching tomograms ---------------------------------------------
    before = json.loads(str(np.load(ctl.session_map.path("tomotwin"))["meta"]))["computed"]
    time.sleep(1.1)
    uw2._compute()
    ctl.open("TS_043")                     # mid-computation
    app.processEvents()
    assert ctl.session_map_job.running(), "switching must not stop the computation"
    wait_for(lambda: not ctl.session_map_job.running())
    after = json.loads(str(np.load(ctl.session_map.path("tomotwin"))["meta"]))["computed"]
    assert after != before
    assert ctl.current.umap_selection == expect["TS_043"]
    print("OK: a computation started in one tomogram finishes after switching to another")

    # -- 9. out of date, and unusable ---------------------------------------------------------------
    store = ctl.session_map
    m, _ = store.load("tomotwin")
    fp = [list(x) for x in m.fingerprint]
    fp[0][2] += 1                                      # the file's size changed
    store.save("tomotwin", m.coords, [list(x) for x in m.layout], fp, m.params)
    assert store.load("tomotwin")[1] == "stale"
    # fitted on a different number of rows than the files now hold, as after
    # the embeddings were recomputed for another pick list
    lay = [list(x) for x in m.layout]
    lay[0][1] += 5
    fp = [list(x) for x in m.fingerprint]
    fp[0][1] += 5
    store.save("tomotwin", np.vstack([m.coords, np.zeros((5, 2))]), lay, fp, m.params)
    assert store.load("tomotwin")[1] == "incompatible"
    print("OK: a map from changed embedding files is flagged out of date; one whose row "
          "counts no longer match is refused")

    cur = ctl.current
    cur._confirm_close = lambda: True
    cur.close()
    app.processEvents()
    print("\nALL SESSION MAP TESTS PASSED")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
