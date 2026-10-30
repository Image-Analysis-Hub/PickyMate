"""The safety net for refactoring: what a tomogram's working directory
holds after the core gestures, through the same methods the GUI calls, on
the synthetic tomogram from conftest.py. Runs in seconds on any machine.

The golden screenshots (tests/golden/*.png) are written on the first run
and compared pixel for pixel afterwards; a pure restructuring of the window
code must leave them identical. Set PICKYMATE_UPDATE_GOLDEN=1 to accept a
deliberate change of appearance.
"""
import json
import os

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from boxer import categories, metadata
from conftest import CLUSTER_A, CLUSTER_B, EMPTY_SPOT, close_window

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


def particles(win):
    """{uid: (x, y, z, set name, embedding_row, score)} -- what must survive a
    save and a reload, keyed by uid so the order boxes are held in is free
    to change."""
    return {b.uid: (b.x, b.y, b.z, win.sets[b.classid], b.embedding_row, b.score) for b in win.boxes}


def files(pdir):
    return {fn: open(os.path.join(pdir, fn), "rb").read() for fn in sorted(os.listdir(pdir))}


def target_set(win):
    return next(cid for cid in win.sets if cid not in categories.FIXED_CLASS_IDS)


def indices(win, rows):
    """Box indices of the particles imported at the given embedding rows."""
    return [i for i, b in enumerate(win.boxes) if b.embedding_row in rows]


def in_set(win, name):
    return {uid for uid, p in particles(win).items() if p[3] == name}


def click(qapp, view, x, y, modifier=Qt.NoModifier):
    """A left click on image coordinate (x, y) of a slice view."""
    sx, sy = view.img_to_scr(x, y)
    pos = QtCore.QPoint(int(sx), int(sy))
    assert view.viewport().rect().contains(pos), (pos, view.viewport().rect())
    QTest.mouseClick(view.viewport(), Qt.LeftButton, modifier, pos)
    qapp.processEvents()


def drag(qapp, view, x0, y0, x1, y1):
    """A left drag from image (x0, y0) to (x1, y1) in a slice view: press,
    a move with the button held, release."""
    def at(x, y):
        sx, sy = view.img_to_scr(x, y)
        return QtCore.QPoint(int(sx), int(sy))
    QTest.mousePress(view.viewport(), Qt.LeftButton, Qt.NoModifier, at(x0, y0))
    p = at(x1, y1)
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(p),
                           QtCore.QPointF(view.viewport().mapToGlobal(p)),
                           Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(view.viewport(), ev)
    QTest.mouseRelease(view.viewport(), Qt.LeftButton, Qt.NoModifier, p)
    qapp.processEvents()


def test_roundtrip_is_byte_identical(tomo, open_window):
    win = open_window(tomo)
    pdir = tomo["picks_dir"]
    assert sorted(win.sets.values()) == ["Confirmed", "Recycle Bin", "TS_A"]
    assert len(win.boxes) == 20 and all(b.embedding_row is not None for b in win.boxes)
    assert win.ordered_space_names() == ["tomotwin", "tomotwin_umap"]

    target = target_set(win)
    win.send_to_confirmed(indices(win, CLUSTER_A[:3]))
    win.send_to_recycle_bin(indices(win, CLUSTER_B[:3]))
    win.compute_mahalanobis(categories.CONFIRMED_ID, target, "tomotwin")
    win.save_picks()
    before = particles(win)
    on_disk = files(pdir)
    assert set(on_disk) == {"TS_A.txt", "Confirmed.txt", "Recycle Bin.txt", "embeddings.settings"}
    close_window(win)

    win = open_window(tomo, first_open=False)
    assert particles(win) == before
    assert win.ordered_space_names() == ["tomotwin", "tomotwin_umap"]
    win.save_picks()
    assert files(pdir) == on_disk


def test_ranking_from_confirmed_picks(tomo, open_window):
    win = open_window(tomo)
    target = target_set(win)
    win.send_to_confirmed(indices(win, CLUSTER_A[:3]))
    res = win.compute_mahalanobis(categories.CONFIRMED_ID, target, "tomotwin")
    assert res["reference_size"] == 3 and res["scored"] == 17
    scored = [(b.score, b.embedding_row) for b in win.boxes if b.classid == target]
    assert all(s is not None for s, _ in scored)
    ranked = [row for _, row in sorted(scored)]
    assert set(ranked[:7]) == set(CLUSTER_A[3:]), "lookalikes of the confirmed picks must rank first"
    # the same numbers, recomputed independently
    arr = win.embedding_spaces["tomotwin"]["array"].astype(np.float64)
    ref = arr[CLUSTER_A[:3]]
    mu = ref.mean(0)
    cen = ref - mu
    ci = np.linalg.pinv(cen.T @ cen / (len(ref) - 1), hermitian=True)
    d = arr[[row for _, row in scored]] - mu
    assert np.allclose(((d @ ci) * d).sum(1), [s for s, _ in scored], rtol=1e-9, atol=1e-9)


def test_undo_redo(tomo, open_window):
    win = open_window(tomo)
    pdir = tomo["picks_dir"]
    before = particles(win)
    moved = indices(win, [0, 1])
    uids = {win.boxes[i].uid for i in moved}

    win.send_to_confirmed(moved)
    assert in_set(win, "Confirmed") == uids
    win.undo()
    assert particles(win) == before
    win.redo()
    assert in_set(win, "Confirmed") == uids
    assert {r["uid"] for r in metadata.load_class_records(pdir, "Confirmed")} == uids
    after_confirm = particles(win)

    gone = indices(win, [5])[0]
    gone_uid = win.boxes[gone].uid
    win.del_box(gone)
    assert gone_uid not in particles(win) and len(win.boxes) == 19
    assert gone_uid not in {r["uid"] for r in metadata.load_class_records(pdir, "TS_A")}
    win.undo()
    assert particles(win) == after_confirm
    assert gone_uid in {r["uid"] for r in metadata.load_class_records(pdir, "TS_A")}
    win.redo()
    assert gone_uid not in particles(win)


def test_sets_rename_and_delete(tomo, open_window):
    win = open_window(tomo)
    pdir = tomo["picks_dir"]
    target = target_set(win)
    junk = win.new_set("junk")
    assert win.sets[junk] == "junk" and junk in win.sets_visible

    win.rename_set(f"{target}::TS_A", "ribosomes")
    assert win.sets[target] == "ribosomes"
    assert os.path.isfile(os.path.join(pdir, "ribosomes.txt"))
    assert not os.path.exists(os.path.join(pdir, "TS_A.txt"))

    win.delete_set(f"{target}::ribosomes")
    assert target not in win.sets
    assert not os.path.exists(os.path.join(pdir, "ribosomes.txt"))
    assert len(in_set(win, "Recycle Bin")) == 20, "a deleted set's particles go to the Recycle Bin"
    assert win.currentset == junk


def test_pick_and_erase_by_click(tomo, open_window, qapp):
    win = open_window(tomo)
    view = win.xyview
    assert win.tool_pick_action.isChecked()

    x, y, z = EMPTY_SPOT
    win.z_loc = z
    click(qapp, view, x, y)
    assert len(win.boxes) == 21
    new = win.boxes[-1]
    assert abs(new.x - x) < 1 and abs(new.y - y) < 1 and new.z == z
    assert new.classid == win.currentset and new.embedding_row is None, "a hand pick has no embedding"
    win.undo()
    assert len(win.boxes) == 20
    win.redo()
    assert len(win.boxes) == 21

    win.tool_eraser_action.trigger()
    assert win.tool_eraser_action.isChecked()
    win.eraser_radius.setValue(8)
    first = win.boxes[0]
    x, y, z = first.x, first.y, first.z
    win.z_loc = z
    click(qapp, view, x, y)
    binned = [(b.x, b.y, b.z) for b in win.boxes if b.classid == categories.RECYCLE_BIN_ID]
    assert binned == [(x, y, z)], binned


def test_side_views_pick_and_drag(tomo, open_window, qapp):
    """The XZ and ZY views map the mouse onto their own two axes, the third
    coming from the slice shown."""
    win = open_window(tomo)
    x, y, z = EMPTY_SPOT
    win.y_loc = y
    click(qapp, win.xzview, x, z)           # XZ view: (x, z) under the mouse
    assert len(win.boxes) == 21
    new = win.boxes[-1]
    assert abs(new.x - x) < 1 and new.y == y and abs(new.z - z) < 1
    win.undo()
    win.x_loc = x
    click(qapp, win.zyview, z, y)           # ZY view: (z, y) under the mouse
    new = win.boxes[-1]
    assert new.x == x and abs(new.y - y) < 1 and abs(new.z - z) < 1
    # dragging it in the XZ view moves it in x and z, never in y
    y_before = new.y
    drag(qapp, win.xzview, x, z, x + 10, z - 5)
    assert abs(new.x - (x + 10)) < 1 and new.y == y_before and abs(new.z - (z - 5)) < 1


def test_golden_actions(tomo, open_window):
    """Every action's text, tooltip, shortcuts and flags, as a golden: a
    change to how actions are built must leave all of them the same."""
    win = open_window(tomo)
    found = {}
    for owner in (win, win.boxesviewer, win.boxesviewer.ranking):
        for a in owner.findChildren(QtGui.QAction):
            found[f"{type(owner).__name__}/{a.text()}"] = {
                "tooltip": a.toolTip(), "shortcuts": [s.toString() for s in a.shortcuts()],
                "context": str(a.shortcutContext()), "checkable": a.isCheckable(),
                "checked": a.isChecked(), "enabled": a.isEnabled(), "visible": a.isVisible()}
    path = os.path.join(GOLDEN, "actions.json")
    if os.environ.get("PICKYMATE_UPDATE_GOLDEN") or not os.path.isfile(path):
        os.makedirs(GOLDEN, exist_ok=True)
        with open(path, "w") as f:
            json.dump(found, f, indent=1, sort_keys=True)
        pytest.skip(f"golden written: {path}")
    with open(path) as f:
        golden = json.load(f)
    assert found == golden


@pytest.mark.parametrize("which", ["main", "particle_list"])
def test_golden_screenshot(tomo, open_window, qapp, which):
    win = open_window(tomo)
    widget = win if which == "main" else win.boxesviewer
    widget.resize(800, 600)
    qapp.processEvents()
    image = widget.grab().toImage()
    path = os.path.join(GOLDEN, f"{which}.png")
    if os.environ.get("PICKYMATE_UPDATE_GOLDEN") or not os.path.isfile(path):
        os.makedirs(GOLDEN, exist_ok=True)
        assert image.save(path)
        pytest.skip(f"golden written: {path}")
    golden = QtGui.QImage(path)
    if image.convertToFormat(golden.format()) != golden:
        actual = os.path.join(GOLDEN, f"{which}.actual.png")
        image.save(actual)
        pytest.fail(f"{which} renders differently from {path}; see {actual}. "
                    f"PICKYMATE_UPDATE_GOLDEN=1 accepts the new look.")
