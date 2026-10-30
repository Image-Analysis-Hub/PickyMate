"""The UMAP map window and its selection tab.

Drives the window with real mouse events -- lasso, Ctrl+lasso, click,
Ctrl+click -- on a copy of a real EMPIAR-10988 working directory, and
checks the selection tab, "Show Active Set Only", Escape, persistence and
the interplay with edits. Skipped if the dataset is absent.

    QT_QPA_PLATFORM=offscreen python3 tests/test_umap_selection.py
"""
import argparse
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

DATA = "/media/mohamad/LinuxStorage/Ali_ParticlePicking/PickyMate_EMPIAR10988"
if not os.path.isdir(DATA):
    print(f"SKIPPED: {DATA} not found")
    sys.exit(0)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from boxer import categories, metadata
from boxer.main_window import TomoBoxer
from boxer.umap_view import points_in_polygon

TMP = tempfile.mkdtemp(prefix="pickymate_umap_test_")
T = "TS_026"
WD = f"{TMP}/{T}_picks"
shutil.copytree(f"{DATA}/Ribosomes_PickyMate/{T}_picks", WD)


def opts():
    return argparse.Namespace(mode="3D", label=None, clean=False, apix=None, working_dir=WD,
                              particle_lists=None, native_memmap=False, companion_tomogram=None,
                              embedded_particle_list=None, cryosiam_embeddings=None,
                              cryosiam_umap=None, tomotwin_embeddings=None, tomotwin_umap=None)


def close(w):
    w._confirm_close = lambda: True
    w.close()
    app.processEvents()


def drag(canvas, points, modifier=Qt.NoModifier):
    """A real drag: press, moves with the button held (Qt drops button-less
    moves on a widget without mouse tracking, as a real drag never is),
    release -- here always back at the start, as a lasso is drawn."""
    QTest.mousePress(canvas, Qt.LeftButton, modifier, points[0])
    for p in points[1:]:
        ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(p),
                               QtCore.QPointF(canvas.mapToGlobal(p)),
                               Qt.NoButton, Qt.LeftButton, modifier)
        QtWidgets.QApplication.sendEvent(canvas, ev)
    QTest.mouseRelease(canvas, Qt.LeftButton, modifier, points[-1])
    app.processEvents()


try:
    # -- 1. point-in-polygon, against cases with a known answer --------------
    px = np.array([1., 5., 9., 5., 2., 8.])
    py = np.array([1., 5., 9., 2., 8., 8.])
    square = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], float)
    assert points_in_polygon(px, py, square).tolist() == [True, True, False, True, False, False]
    ell = np.array([[0, 0], [10, 0], [10, 3], [3, 3], [3, 10], [0, 10]], float)   # an L
    assert points_in_polygon(px, py, ell).tolist() == [True, False, False, True, True, False]
    print("OK: lasso containment is right for a square and a concave (L-shaped) lasso")

    # -- 2. the window ---------------------------------------------------------
    win = TomoBoxer(opts(), f"{DATA}/Tomograms/{T}.rec")
    win.show()
    app.processEvents()
    assert win.umap_action in win.boxesviewer.view_toolbar.actions(), "no UMAP button in the particle list"
    win.open_umap_window()
    app.processEvents()
    uw, canvas = win.umap_window, win.umap_window.canvas
    spaces = [uw.space_combo.itemData(i) for i in range(uw.space_combo.count())]
    assert spaces == ["tomotwin_umap", "cryosiam_umap"], spaces
    embedded = [b for b in win.boxes if b.embedding_row is not None]
    assert len(canvas._xy) == len(embedded) > 0
    print(f"OK: map shows all {len(embedded)} embedded particles; TomoTwin UMAP is offered first")

    sets_before = {cid: sum(1 for b in win.boxes if b.classid == cid) for cid in win.sets}

    # -- 3. a real lasso -------------------------------------------------------
    w, h = canvas.width(), canvas.height()
    box = [QPoint(int(w * .30), int(h * .30)), QPoint(int(w * .70), int(h * .30)),
           QPoint(int(w * .70), int(h * .70)), QPoint(int(w * .30), int(h * .70)),
           QPoint(int(w * .30), int(h * .30))]
    drag(canvas, box)
    sx, sy = canvas._to_screen(canvas._xy)

    def inside(corners):
        # from the exact pixel corners the mouse visited, not the fractions
        # they were computed from -- they differ by up to a pixel
        x0, y0, x1, y1 = corners[0].x(), corners[0].y(), corners[2].x(), corners[2].y()
        return set(uw._uids[(sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)].tolist())

    expect = inside(box)
    assert win.umap_selection == expect and len(expect) > 0, (len(win.umap_selection), len(expect))
    print(f"OK: a lasso drawn with the mouse selects exactly the {len(expect)} particles inside it")

    bv = win.boxesviewer
    idx = bv.tabs.indexOf(bv.umap_grid)
    assert idx == 1, f"the UMAP Selection tab should sit right after All, found at {idx}"
    assert bv.tabs.tabText(idx) == f"UMAP Selection ({len(expect)})"
    assert bv.is_umap_selection_active(), "a lasso should bring its tab to the front"
    assert bv._is_tab_locked(idx), "the selection tab must not be draggable"
    assert set(bv.umap_grid._images) == expect
    sets_after = {cid: sum(1 for b in win.boxes if b.classid == cid) for cid in win.sets}
    assert sets_after == sets_before, "selecting must not move particles between sets"
    assert metadata.discover_classes(WD) == sorted(metadata.discover_classes(WD))
    assert "UMAP Selection" not in metadata.discover_classes(WD), "the selection became a set"
    assert os.path.isfile(metadata.umap_selection_path(WD))
    print("OK: the \"UMAP Selection\" tab appears after All with the count, shows exactly the "
          "selection, moves nothing, and is saved without becoming a set")

    # -- 4. Ctrl+lasso adds; click replaces; Ctrl+click toggles ----------------
    # around a particle that the first lasso did not take, so the added
    # region is guaranteed to contain something new
    out = [j for j in range(len(sx)) if int(uw._uids[j]) not in expect
           and 20 < sx[j] < w - 20 and 20 < sy[j] < h - 20]
    j = out[len(out) // 2]
    cx_, cy_ = int(sx[j]), int(sy[j])
    corner = [QPoint(cx_ - 15, cy_ - 15), QPoint(cx_ + 15, cy_ - 15), QPoint(cx_ + 15, cy_ + 15),
              QPoint(cx_ - 15, cy_ + 15), QPoint(cx_ - 15, cy_ - 15)]
    extra = inside(corner)
    assert extra - expect, "precondition: the second lasso must reach unselected particles"
    drag(canvas, corner, Qt.ControlModifier)
    assert win.umap_selection == expect | extra
    print(f"OK: Ctrl+lasso adds {len(extra - expect)} new particles and keeps the "
          f"{len(expect)} already selected")
    shot = os.environ.get("UMAP_SCREENSHOT")
    if shot:
        uw.grab().save(shot)

    i = int(np.argmin((sx - w / 2) ** 2 + (sy - h / 2) ** 2))
    at = QPoint(int(round(sx[i])), int(round(sy[i])))
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, at)
    app.processEvents()
    target_uid = int(uw._uids[canvas._nearest(QtCore.QPointF(at))])
    assert win.umap_selection == {target_uid}, win.umap_selection
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.ControlModifier, at)
    app.processEvents()
    assert win.umap_selection == set(), "Ctrl+click on a selected particle should remove it"
    assert bv.tabs.indexOf(bv.umap_grid) < 0, "an empty selection should remove its tab"
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.ControlModifier, at)
    app.processEvents()
    assert win.umap_selection == {target_uid}
    print("OK: a click selects one particle; Ctrl+click removes it and adds it back; "
          "the tab disappears while nothing is selected")

    # -- 5. edits keep particles in the selection, with their new set ------------
    drag(canvas, box)
    sel = set(win.umap_selection)
    moved_uid = next(iter(sel))
    moved = [j for j, b in enumerate(win.boxes) if b.uid == moved_uid]
    win.currentset = win.boxes[moved[0]].classid
    win._reassign_boxes(moved, categories.RECYCLE_BIN_ID)
    app.processEvents()
    assert win.umap_selection == sel, "an edit changed the selection"
    k = int(np.flatnonzero(uw._uids == moved_uid)[0])
    assert canvas._classes[k] == categories.RECYCLE_BIN_ID, "the map did not follow the edit"
    print("OK: recycling a selected particle keeps it in the selection, and the map "
          "recolours it immediately")

    # -- 6. Show Active Set Only on the selection tab ----------------------------
    bv.activate_umap_tab()
    win.view_only_selected_action.setChecked(True)
    app.processEvents()
    shown = {b.uid for b in win.boxes if win._is_visible_particle(b)}
    assert shown == sel, (len(shown), len(sel))
    recycled = next(b for b in win.boxes if b.uid == moved_uid)
    assert win._is_visible_particle(recycled), "a selected particle in the Recycle Bin must still show"
    bv.tabs.setCurrentWidget(bv.grid)          # to "All", solo still on
    app.processEvents()
    shown_all = {b.uid for b in win.boxes if win._is_visible_particle(b)}
    assert shown_all > sel, "back on All, every set should show again"
    win.view_only_selected_action.setChecked(False)
    app.processEvents()
    print(f"OK: Show Active Set Only on the selection tab shows only its {len(sel)} particles "
          f"(Recycle Bin ones included); switching to All shows everything again")

    # -- 7. a deleted particle leaves the saved file; undo brings it back ---------
    gone = next(u for u in sel if u != moved_uid)
    win.del_box([j for j, b in enumerate(win.boxes) if b.uid == gone])
    app.processEvents()
    on_disk = {r["uid"] for r in metadata.load_umap_selection(WD)}
    assert gone not in on_disk and len(on_disk) == len(sel) - 1
    win.undo()
    app.processEvents()
    on_disk = {r["uid"] for r in metadata.load_umap_selection(WD)}
    assert gone in on_disk, "undoing the delete should restore it to the selection"
    print("OK: deleting a selected particle drops it from the saved selection; undo restores it")

    # -- 8. it survives a restart ------------------------------------------------
    close(win)
    win = TomoBoxer(opts(), f"{DATA}/Tomograms/{T}.rec")
    win.show()
    app.processEvents()
    assert win.umap_selection == sel, (len(win.umap_selection), len(sel))
    bv = win.boxesviewer
    assert bv.tabs.indexOf(bv.umap_grid) == 1
    print("OK: the selection and its tab are restored when the tomogram is reopened")

    # -- 9. Escape in the map window clears it; closing the window changes nothing
    win.open_umap_window()
    app.processEvents()
    win.umap_window.close()
    app.processEvents()
    assert win.umap_selection == sel, "closing the map window must not touch the selection"
    win.open_umap_window()
    app.processEvents()
    win.umap_window.activateWindow()
    QtWidgets.QApplication.setActiveWindow(win.umap_window)
    app.processEvents()
    win._global_escape()
    app.processEvents()
    assert win.umap_selection == set() and not os.path.isfile(metadata.umap_selection_path(WD))
    assert bv.tabs.indexOf(bv.umap_grid) < 0
    print("OK: closing the map changes nothing; Escape in the map clears the selection, "
          "its tab and its file")

    # -- 10. clearing while Show Active Set Only is on the selection tab ---------
    # Removing the selection tab switched tabs, which (with solo on) redrew
    # everything -- re-entering the half-finished tab removal, which then took
    # out the tab at the old position: the Recycle Bin. Every later redraw of
    # the grouped views failed after that.
    # errors raised inside Qt's event handlers are collected here; the hook is
    # put back however this section ends, so a crash is reported, not swallowed
    errors = []
    old_hook = sys.excepthook
    sys.excepthook = lambda t, v, tb: errors.append(f"{t.__name__}: {v}")
    try:
        uw, canvas = win.umap_window, win.umap_window.canvas
        app.processEvents()
        w, h = canvas.width(), canvas.height()
        sx, sy = canvas._to_screen(canvas._xy)
        box = [QPoint(int(w * .30), int(h * .30)), QPoint(int(w * .70), int(h * .30)),
               QPoint(int(w * .70), int(h * .70)), QPoint(int(w * .30), int(h * .70)),
               QPoint(int(w * .30), int(h * .30))]
        drag(canvas, box)
        chosen = set(win.umap_selection)
        assert chosen and bv.is_umap_selection_active()
        win.view_only_selected_action.setChecked(True)
        app.processEvents()
        tabs_before = [bv.tabs.tabText(i) for i in range(bv.tabs.count())]

        # a lasso around nothing leaves the selection alone
        gx, gy = next((x, y) for x in range(10, w - 30, 10) for y in range(10, h - 30, 10)
                      if not ((sx > x - 5) & (sx < x + 25) & (sy > y - 5) & (sy < y + 25)).any())
        drag(canvas, [QPoint(gx, gy), QPoint(gx + 20, gy), QPoint(gx + 20, gy + 20),
                      QPoint(gx, gy + 20), QPoint(gx, gy)])
        assert win.umap_selection == chosen, "a lasso around nothing must not change the selection"

        QtWidgets.QApplication.setActiveWindow(uw)
        win._global_escape()
        app.processEvents()
        tabs_after = [bv.tabs.tabText(i) for i in range(bv.tabs.count())]
        assert tabs_after == [t for t in tabs_before if not t.startswith("UMAP Selection")], \
            (tabs_before, tabs_after)
        assert not errors, errors
        assert sum(win._is_visible_particle(b) for b in win.boxes) == len(win.boxes)
        drag(canvas, box)
        bv.tabs.setCurrentWidget(bv._class_grids[categories.RECYCLE_BIN_ID])
        app.processEvents()
        assert win.umap_selection == chosen and not errors, errors
        win.view_only_selected_action.setChecked(False)
    finally:
        sys.excepthook = old_hook
    print("OK: a lasso around nothing changes nothing; Escape with Show Active Set Only on "
          "removes only the selection tab, keeps the Recycle Bin, and selecting still works")

    close(win)
    print("\nALL UMAP SELECTION TESTS PASSED")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
