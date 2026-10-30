"""Multi-tomogram sessions: manifest, navigator, switching, and pooling
references from other tomograms when ranking.

Runs on copies of real EMPIAR-10988 working directories in a temporary
folder, never on the originals, and is skipped if the dataset is absent.

    QT_QPA_PLATFORM=offscreen python3 tests/test_session.py
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
from PySide6 import QtWidgets

DATA = "/media/mohamad/LinuxStorage/Ali_ParticlePicking/PickyMate_EMPIAR10988"
if not os.path.isdir(DATA):
    print(f"SKIPPED: {DATA} not found")
    sys.exit(0)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from boxer import categories, metadata
from boxer.session import Session
from boxer.session_window import SessionController

TMP = tempfile.mkdtemp(prefix="pickymate_session_test_")
OPENED = ["TS_026", "TS_037"]         # initialised: copies of real working dirs
FRESH = "TS_043"                      # never opened: exercises the first-open import
MOVED = "TS_045"                      # initialised on another machine: its recorded paths are stale
for t in OPENED + [MOVED]:
    shutil.copytree(f"{DATA}/Ribosomes_PickyMate/{t}_picks", f"{TMP}/{t}_picks")
# MOVED must look like it was initialised elsewhere whatever state the
# dataset's own copy is in (opening it once repairs the paths for good):
# point every recorded embedding file at a folder that doesn't exist here
moved_dir = f"{TMP}/{MOVED}_picks"
metadata.save_embeddings_settings(moved_dir, [
    dict(sp, path=f"/another/machine/{os.path.basename(sp['path'])}")
    for sp in metadata.load_embeddings_settings(moved_dir, include_missing=True)])

manifest = os.path.join(TMP, "session.csv")
with open(manifest, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tomogram", "working_dir", "embedded_particle_list", "cryosiam_embeddings",
                "cryosiam_umap", "tomotwin_embeddings", "tomotwin_umap", "companion_tomogram"])
    for t in OPENED + [FRESH, MOVED]:
        w.writerow([f"{DATA}/Tomograms/{t}.rec", f"{t}_picks",
                    f"{DATA}/TemplateMatching_Ribosomes_PicksOnly/{t}.txt",
                    f"{DATA}/CryoSiam/Ribosomes_embeds/{t}_embeds.h5",
                    f"{DATA}/CryoSiam/Ribosomes_embeds/vis/{t}_embeds_umap_data.csv",
                    f"{DATA}/TomoTwin/Ribosomes_embeddings/{t}.temb",
                    f"{DATA}/TomoTwin/Ribosomes_embeddings/{t}.tumap", ""])

base = argparse.Namespace(mode="3D", label=None, clean=False, apix=None, working_dir=None,
                          particle_lists=None, native_memmap=False, companion_tomogram=None,
                          embedded_particle_list=None, cryosiam_embeddings=None,
                          cryosiam_umap=None, tomotwin_embeddings=None, tomotwin_umap=None)

try:
    # -- 1. manifest -----------------------------------------------------------
    session = Session.from_manifest(manifest)
    assert session.names == OPENED + [FRESH, MOVED], session.names
    assert session.entry("TS_026").working_dir == f"{TMP}/TS_026_picks"
    o = session.options_for("TS_026", base)
    assert o.embedded_particle_list is None and o.tomotwin_embeddings is None, \
        "an initialised tomogram must not be re-imported (it would duplicate its set)"
    o = session.options_for(FRESH, base)
    assert o.embedded_particle_list and o.tomotwin_embeddings, "first open must import"
    print("OK: relative paths resolve against the manifest; import flags only on first open")

    c = session.entry("TS_026").counts()
    n_conf = len(metadata.load_class_records(f"{TMP}/TS_026_picks", "Confirmed"))
    n_bin = len(metadata.load_class_records(f"{TMP}/TS_026_picks", "Recycle Bin"))
    assert c["confirmed"] == n_conf and c["recycle"] == n_bin, (c, n_conf, n_bin)
    print(f"OK: navigator counts read from disk match the files ({c})")

    # -- 2. references from other tomograms --------------------------------------
    vec, used, skipped = session.reference_vectors("tomotwin", "Confirmed", ["TS_037"], 32)
    # only embedded particles can be references: a hand pick in Confirmed has no row
    rows = [r["embedding_row"] for r in metadata.load_class_records(f"{TMP}/TS_037_picks", "Confirmed")
            if r["embedding_row"] is not None]
    assert used == ["TS_037"] and vec.shape == (len(rows), 32), (used, vec.shape)
    _, used, skipped = session.reference_vectors("tomotwin", "Confirmed", [FRESH], 32)
    assert used == [] and skipped and skipped[0][0] == FRESH, skipped
    _, used, skipped = session.reference_vectors("tomotwin", "Confirmed", ["TS_037"], 99)
    assert used == [] and "dimensions" in skipped[0][1]
    print("OK: other tomograms' Confirmed vectors gathered; missing sets and "
          "mismatched dimensions are skipped with a reason, not mixed in")

    # -- 3. open, rank, switch -----------------------------------------------------
    ctl = SessionController(session, base)
    ctl.start("TS_026")
    win = ctl.current
    assert win is not None and win.session_name == "TS_026"
    assert win.ordered_space_names()[0] == "tomotwin", win.ordered_space_names()
    combo = win.boxesviewer.ranking.compute_distance_space_combo
    shown = [combo.itemData(i) for i in range(combo.count())]
    assert shown == ["tomotwin", "tomotwin_umap", "cryosiam", "cryosiam_umap"], shown
    assert win.boxesviewer.ranking.compute_distance_scope_row.isVisibleTo(win.boxesviewer)
    assert win.boxesviewer.ranking.reference_tomograms() is None, "default must be this tomogram only"
    print("OK: space dropdown is TomoTwin full, TomoTwin UMAP, CryoSIAM full, CryoSIAM UMAP; "
          "references default to this tomogram")

    target = [cid for cid in win.sets if cid not in categories.FIXED_CLASS_IDS][0]
    res_own = win.compute_mahalanobis(categories.CONFIRMED_ID, target, "tomotwin")
    own_scores = {b.uid: b.score for b in win.boxes if b.classid == target and b.score is not None}
    res_all = win.compute_mahalanobis(categories.CONFIRMED_ID, target, "tomotwin",
                                      reference_tomograms=session.names)
    assert res_all["other_tomograms_used"] == ["TS_037", MOVED], res_all
    assert isinstance(res_all["skipped_no_embedding"], int), res_all
    assert any(FRESH in n for n in res_all["notes"]), res_all["notes"]
    assert res_all["reference_size"] == res_own["reference_size"] + res_all["extra_reference_size"]

    # recompute the pooled ranking independently and demand exact agreement
    arr = win.embedding_spaces["tomotwin"]["array"].astype(np.float64)
    own_rows = [b.embedding_row for b in win.boxes if b.classid == categories.CONFIRMED_ID and b.embedding_row is not None]
    other, _, _ = session.reference_vectors("tomotwin", "Confirmed", ["TS_037", MOVED], 32)
    ref = np.vstack([arr[own_rows], other])
    mu = ref.mean(0); cen = ref - mu
    ci = np.linalg.pinv(cen.T @ cen / (len(ref) - 1), hermitian=True)
    tgt = [(b.uid, b.embedding_row) for b in win.boxes if b.classid == target and b.embedding_row is not None]
    d = arr[[r for _, r in tgt]] - mu
    mine = ((d @ ci) * d).sum(1)
    theirs = np.array([next(b.score for b in win.boxes if b.uid == u) for u, _ in tgt])
    assert np.allclose(mine, theirs, rtol=1e-9, atol=1e-9), np.abs(mine - theirs).max()
    assert any(abs(own_scores[u] - s) > 1e-9 for (u, _), s in zip(tgt, theirs)), \
        "pooling other tomograms should change the scores"
    print(f"OK: pooled rank uses {res_all['extra_reference_size']} particles from TS_037 and "
          f"{MOVED} and is bit-identical to an independent recomputation")

    # contrastive with pooled Recycle Bins: both terms pool, nothing crashes
    res_c = win.compute_mahalanobis(categories.CONFIRMED_ID, target, "tomotwin",
                                    negative_classid=categories.RECYCLE_BIN_ID,
                                    reference_tomograms=session.names)
    assert res_c["extra_negative_size"] > 0 and isinstance(res_c["skipped_no_embedding"], int)
    print(f"OK: contrastive rank pools {res_c['extra_negative_size']} Recycle Bin particles "
          f"from the other tomograms")

    # an edit made before switching must be on disk afterwards
    moved = [i for i, b in enumerate(win.boxes) if b.classid == target][:3]
    moved_uids = {win.boxes[i].uid for i in moved}
    win.currentset = target
    win._reassign_boxes(moved, categories.RECYCLE_BIN_ID)
    categories.set_manual_color(0, categories.QtGui.QColor("orange"))
    # a geometry that fits the headless test screen: the default layout makes
    # the gallery full-height, and Qt rightly pulls a full-height window back
    # up if it is moved down, which would make the comparison meaningless
    win.boxesviewer.setGeometry(130, 60, 520, 420)
    win.splitter_top.setSizes([120, 220])      # the user's own panel proportions
    top_before = win.splitter_top.sizes()
    old_gallery = win.boxesviewer
    app.processEvents()
    # read after the event pass: the particle list's combo boxes size
    # themselves to their contents on first show, which raises its minimum
    # width, so a geometry read before that pass is not what gets saved
    gallery_before = win.boxesviewer.geometry()
    win.open_umap_window()                      # an open map must go with its tomogram
    old_map = win.umap_window
    app.processEvents()

    ctl.open("TS_037")
    app.processEvents()
    assert ctl.current.session_name == "TS_037" and ctl.current is not win
    import shiboken6
    assert not shiboken6.isValid(win) and not shiboken6.isValid(old_gallery), \
        "the previous tomogram's windows must be deleted, not just hidden"
    assert not shiboken6.isValid(old_map), "the previous tomogram's UMAP window was left behind"
    # TS_037's working directory defines no colour overrides, so the orange
    # set on TS_026's set 0 must be gone rather than repainting TS_037's
    assert not os.path.isfile(f"{TMP}/TS_037_picks/config.settings")
    assert not categories.has_manual_color(0), "a class colour leaked across the switch"
    on_disk = {r["uid"] for r in metadata.load_class_records(f"{TMP}/TS_026_picks", "Recycle Bin")}
    assert moved_uids <= on_disk, "the edit made before the switch was not saved"
    assert ctl.current.boxesviewer.geometry() == gallery_before, \
        (ctl.current.boxesviewer.geometry(), gallery_before)
    # The slice views are panels inside splitters, not windows. Restoring a
    # window geometry onto them once shifted them down by a title bar's
    # height, leaving the tomogram view blank until the window was maximised
    # and restored; they must sit exactly where the splitter puts them.
    new = ctl.current
    for split in (new.splitter_top, new.splitter_bottom):
        for i in range(split.count()):
            panel = split.widget(i)
            assert panel.y() == 0 and panel.height() == split.height(), \
                (panel, panel.geometry(), split.geometry())
    assert new.splitter_top.sizes() == top_before, (new.splitter_top.sizes(), top_before)
    print("OK: switching saves the previous tomogram, closes its windows, clears its class "
          "colours, and reopens at the same window positions with the slice views laid out "
          "correctly and the panel proportions kept")

    # Only the open tomogram may stay in memory. Each switch once leaked the
    # whole previous tomogram (about 1 GB): the particle list is its own
    # top-level window and was never deleted, and its signals kept the old
    # main window -- volume included -- alive through a cycle in Qt.
    for t in ("TS_026", "TS_037", "TS_026", "TS_037"):
        ctl.open(t)
    app.processEvents()
    alive = {name: [w for w in QtWidgets.QApplication.topLevelWidgets() if type(w).__name__ == name]
             for name in ("TomoBoxer", "ParticleListPanel")}
    assert all(len(v) == 1 for v in alive.values()), {k: len(v) for k, v in alive.items()}
    assert win.data is None and not win.embedding_spaces and not win.boxesimgs
    print("OK: after repeated switches exactly one tomogram window and one particle list "
          "exist, and the closed tomogram's volume, embeddings and thumbnails are released")

    # the navigator shows the new state, and the scope survives the switch
    ctl.reference_scope = ("all", [])
    ctl.open("TS_026")
    assert ctl.current.boxesviewer.ranking.reference_tomograms() == session.names
    ctl.navigator.refresh()
    assert ctl.navigator.table.item(0, 0).text().strip().startswith("▶")
    print("OK: navigator marks the open tomogram; the references choice survives switching")

    # -- 4. first open of a fresh tomogram imports it once --------------------------
    ctl.open(FRESH)
    fresh = ctl.current
    n_fresh_spaces = len(fresh.embedding_spaces)   # read now: switching away releases them
    assert fresh.session_name == FRESH and n_fresh_spaces == 4
    n_sets = len([c for c in fresh.sets if c not in categories.FIXED_CLASS_IDS])
    ctl.open("TS_026")
    ctl.open(FRESH)
    assert len([c for c in ctl.current.sets if c not in categories.FIXED_CLASS_IDS]) == n_sets, \
        "reopening a tomogram must not import its particle list again"
    print(f"OK: a never-opened tomogram is imported on first open "
          f"({n_fresh_spaces} spaces) and not duplicated when reopened")

    # -- 5. a working directory whose recorded embedding paths moved -------------
    recorded = metadata.load_embeddings_settings(f"{TMP}/{MOVED}_picks", include_missing=True)
    assert recorded and not any(r["exists"] for r in recorded), \
        "precondition: this copy should still record another machine's paths"
    fixes = session.entry(MOVED).embedding_path_fixes()
    assert set(fixes) == {"cryosiam", "cryosiam_umap", "tomotwin", "tomotwin_umap"}, fixes
    vec, used, _ = session.reference_vectors("tomotwin", "Confirmed", [MOVED], 32)
    assert used == [MOVED] and len(vec) > 0
    # a recorded space the manifest cannot fix must survive the rewrite
    with open(metadata.embeddings_settings_path(f"{TMP}/{MOVED}_picks"), "a") as f:
        f.write("space extra_space tomotwin embeddings 15375 /nonexistent/elsewhere/extra.temb\n")
    ctl.open(MOVED)
    assert len(ctl.current.embedding_spaces) == 4, list(ctl.current.embedding_spaces)
    rewritten = metadata.load_embeddings_settings(f"{TMP}/{MOVED}_picks", include_missing=True)
    kept = [r for r in rewritten if r["name"] == "extra_space"]
    assert kept and kept[0]["path"] == "/nonexistent/elsewhere/extra.temb", \
        "the repair erased a record it could not fix"
    rewritten = [r for r in rewritten if r["name"] != "extra_space"]
    assert all(r["exists"] and r["path"].startswith(DATA) for r in rewritten), rewritten
    assert session.entry(MOVED).embedding_path_fixes() == {}
    print(f"OK: stale embedding paths from another machine are replaced by the manifest's, "
          f"the tomogram opens with all 4 spaces, and the corrected paths are saved")

    # never substitute a file with a different name: it could be other data
    e = session.entry(MOVED)
    saved = e.import_flags["tomotwin_embeddings"]
    e.import_flags["tomotwin_embeddings"] = f"{DATA}/TomoTwin/Ribosomes_embeddings/TS_026.temb"
    with open(metadata.embeddings_settings_path(e.working_dir), "a") as f:
        f.write("space tomotwin tomotwin embeddings 15375 /nonexistent/elsewhere/TS_045.temb\n")
    assert "tomotwin" not in e.embedding_path_fixes()
    e.import_flags["tomotwin_embeddings"] = saved
    print("OK: a manifest file with a different name is never substituted for a missing one")

    # -- 6. closing the tomogram by hand ends the session ----------------------------
    cur = ctl.current
    cur._confirm_close = lambda: True
    cur.close()
    app.processEvents()
    assert ctl.current is None and not ctl.navigator.isVisible()
    print("OK: closing the open tomogram's window ends the session")
    print("\nALL SESSION TESTS PASSED")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
