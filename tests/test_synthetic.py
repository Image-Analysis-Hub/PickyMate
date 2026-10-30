"""The safety net for refactoring: what a tomogram's working directory
holds after the core gestures, through the same methods the GUI calls, on
the synthetic tomogram from conftest.py. Runs in seconds on any machine.

The golden screenshots (tests/golden/*.png) are written on the first run
and compared pixel for pixel afterwards; a pure restructuring of the window
code must leave them identical. Set PICKYMATE_UPDATE_GOLDEN=1 to accept a
deliberate change of appearance.
"""
import os

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
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
