#!/usr/bin/env python3
"""Regenerate every image in docs/images/ from real data.

    python docs/make_figures.py [--data DIR] [--out docs/images]

Screenshots are taken from the app itself, rendered offscreen, on a
temporary copy of a real curated working directory (EMPIAR-10988 TS_026,
ribosomes) -- never on the original. The README overview figure and the
composed panels are drawn with matplotlib from the same data. Rerun this
whenever the interface changes so the documentation cannot drift away from
what the app actually looks like.

Needs the pickymate environment (PySide6 + matplotlib + the data).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PySide6 import QtCore, QtGui, QtWidgets

DATA = "/media/mohamad/LinuxStorage/Ali_ParticlePicking/PickyMate_EMPIAR10988"
ACCENT = QtGui.QColor("#e67e22")          # annotation badges
TEXT = "#333333"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def annotate(pixmap, badges, radius=16):
    """Draw numbered badges onto a screenshot; badges: [(x, y, "1"), ...]."""
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    font = QtGui.QFont()
    font.setBold(True)
    font.setPixelSize(radius + 4)
    p.setFont(font)
    for x, y, label in badges:
        p.setBrush(ACCENT)
        p.setPen(QtGui.QPen(QtGui.QColor("white"), 2))
        p.drawEllipse(QtCore.QPointF(x, y), radius, radius)
        p.drawText(QtCore.QRectF(x - radius, y - radius, 2 * radius, 2 * radius),
                   QtCore.Qt.AlignCenter, label)
    p.end()
    return pixmap


def grab(widget):
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    return widget.grab()


def save(pixmap, out, name):
    path = os.path.join(out, name)
    pixmap.save(path)
    print(f"wrote {path}  ({pixmap.width()}x{pixmap.height()})")


def to_array(pixmap):
    img = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=h * img.bytesPerLine())
    return buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()


def norm(img):
    """Percentile display range, not mean +- 3 sigma: one gold bead in the
    field is enough to blow up the standard deviation and wash everything
    else out to a flat grey."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, (1.0, 99.0))
    return np.clip((img - lo) / ((hi - lo) or 1.0), 0, 1)


def grid_head(grid, height=560):
    """The head of a gallery grid -- its first slots. Slots fill bottom-left
    first, so the beginning of the list is at the bottom of the scene."""
    bar = grid.verticalScrollBar()
    bar.setValue(bar.maximum())
    pm = grab(grid)
    w = pm.width() - 18          # drop the scrollbar
    return pm.copy(0, pm.height() - height, w, height)


def crop_at(view, size=380):
    """A size x size crop from the centre of a view widget."""
    pm = grab(view)
    cx, cy = pm.width() // 2, pm.height() // 2
    return pm.copy(cx - size // 2, cy - size // 2, size, size)


# ---------------------------------------------------------------------------
# the app, opened on a temporary copy of the curated TS_026
# ---------------------------------------------------------------------------
def gt_matcher(data):
    """is_particle(x, y, z): does a pick land on a deposited ground-truth
    ribosome (within one particle radius)? Used only to colour-code figure
    panels -- the app itself never sees the ground truth."""
    from boxer import metadata
    gt = np.array(metadata.load_particle_list(
        os.path.join(data, "Ribosomes_GroundTruth/TS_026.txt")), dtype=float)

    def is_particle(x, y, z):
        return bool((((gt - [x, y, z]) ** 2).sum(1) < 100).any())
    return is_particle


def mark_grid(grid, pixmap, y_offset, is_ok):
    """Colour-code a gallery grab against the ground truth: a dot in each
    visible thumbnail's corner, green = a real particle, red = junk.
    is_ok: uid -> bool."""
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    for uid, item in grid._items.items():
        v = grid.mapFromScene(item.sceneBoundingRect().topRight())
        x, y = v.x() - 12, v.y() - y_offset + 12
        if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
            p.setBrush(QtGui.QColor("#2ecc71" if is_ok(uid) else "#e74c3c"))
            p.setPen(QtGui.QPen(QtGui.QColor("white"), 1))
            p.drawEllipse(QtCore.QPointF(x, y), 6, 6)
    p.end()
    return pixmap


def open_curated(data, tmp):
    from boxer.main_window import TomoBoxer
    wd = os.path.join(tmp, "TS_026_picks")
    if not os.path.isdir(wd):
        shutil.copytree(os.path.join(data, "Ribosomes_PickyMate/TS_026_picks"), wd)
        for extra in ("UMAP Selection.csv",):
            try:
                os.remove(os.path.join(wd, extra))
            except OSError:
                pass
    o = argparse.Namespace(mode="3D", working_dir=wd, particle_lists=None,
                           native_memmap=False, companion_tomogram=None,
                           embedded_particle_list=None, cryosiam_embeddings=None,
                           cryosiam_umap=None, tomotwin_embeddings=None, tomotwin_umap=None)
    win = TomoBoxer(o, os.path.join(data, "Tomograms/TS_026.rec"))
    win.show()
    win.setMinimumSize(900, 860)
    win.boxesviewer.setMinimumSize(1020, 860)
    QtWidgets.QApplication.processEvents()
    return win


def close(win):
    win._confirm_close = lambda: True
    win.close()
    QtWidgets.QApplication.processEvents()


def shot_main_window(win, out):
    win.scroll_to(480, 460, 250)
    pm = grab(win)
    # badges: tools toolbar, XY view, ZY view, XZ view, selection toolbar
    tools = win.tools_toolbar if hasattr(win, "tools_toolbar") else None
    def centre(w, dx=0, dy=0):
        c = w.mapTo(win, QtCore.QPoint(w.width() // 2 + dx, w.height() // 2 + dy))
        return c.x(), c.y()
    badges = []
    if tools is not None:
        c = tools.mapTo(win, QtCore.QPoint(90, tools.height() // 2))
        badges.append((c.x(), c.y(), "1"))
    for i, w in enumerate((win.xyview, win.zyview, win.xzview), start=2):
        x, y = centre(w)
        badges.append((x, y - (w.height() // 2 - 30), str(i)))
    if hasattr(win, "selection_toolbar"):
        x, y = centre(win.selection_toolbar)
        badges.append((x, y, "5"))
    save(annotate(pm, badges), out, "main-window-overview.png")


def shot_particle_list(win, out):
    bv = win.boxesviewer
    cid = next(c for c, n in win.sets.items() if n == "TS_026")
    bv.tabs.setCurrentWidget(bv.grid)
    grid = bv.grid
    grab(bv)   # let the deferred grouped rebuild finish first
    from boxer import categories
    # centre the view on an early Confirmed particle: setting the scrollbar
    # is undone by the grouped view's own rebuild cycles, but centering on a
    # concrete item survives them, and the red/green group boundary is what
    # makes this screenshot informative
    conf_uids = [b[6] for b in win.boxes if b[5] == categories.CONFIRMED_ID]
    item = grid._items.get(conf_uids[0]) if conf_uids else None
    if item is not None:
        grid.centerOn(item)
    QtWidgets.QApplication.processEvents()
    if item is not None:
        grid.centerOn(item)
    pm = bv.grab()
    bar = bv.tabs.tabBar()
    tab_idx = next(i for i in range(bv.tabs.count()) if "TS_026" in bv.tabs.tabText(i))
    r = bar.tabRect(tab_idx)
    p1 = bar.mapTo(bv, QtCore.QPoint(r.right() + 20, r.center().y()))
    grid_c = grid.mapTo(bv, QtCore.QPoint(80, 80))
    rank_c = bv.compute_distance_group.mapTo(
        bv, QtCore.QPoint(bv.compute_distance_group.width() - 40,
                          bv.compute_distance_group.height() - 60))
    tb_c = bv.gallery_toolbar.mapTo(
        bv, QtCore.QPoint(bv.gallery_toolbar.width() - 40,
                          bv.gallery_toolbar.height() // 2))
    save(annotate(pm, [(p1.x(), p1.y(), "1"), (grid_c.x(), grid_c.y(), "2"),
                       (rank_c.x(), rank_c.y(), "3"), (tb_c.x(), tb_c.y(), "4")]),
         out, "particle-list-overview.png")


def shot_umap(win, out):
    from boxer import categories
    win.open_umap_window()
    uw = win.umap_window
    uw.setMinimumSize(760, 700)
    QtWidgets.QApplication.processEvents()
    uw.refresh(refit=True)
    QtWidgets.QApplication.processEvents()
    # lasso-like selection: everything within an ellipse of the confirmed cloud
    c = uw.canvas
    conf = c._classes == categories.CONFIRMED_ID
    if conf.any():
        cx, cy = np.median(c._xy[conf], axis=0)
        r = 0.9 * np.median(np.abs(c._xy[conf] - [cx, cy]).max(axis=1))
        inside = np.abs(c._xy - [cx, cy]).max(axis=1) < r
        win.set_umap_selection(uw._uids[inside].tolist(), "replace")
        QtWidgets.QApplication.processEvents()
        uw.refresh()
    save(grab(uw), out, "umap-map.png")
    win.clear_umap_selection()
    uw.close()


def shot_tools(win, out, tmp):
    """Eraser before/after and Select-tool crops from the XY view, on the
    fresh-import state (thousands of raw picks in one set)."""
    from boxer import categories
    cid = next(c for c in win.sets if c not in categories.FIXED_CLASS_IDS)
    # a crowded neighbourhood: the densest picked slice
    zs = [b[2] for b in win.boxes if b[5] == cid]
    z = int(np.bincount(zs).argmax())
    near = [b for b in win.boxes if abs(b[2] - z) < 4 and b[5] == cid]
    xs, ys = np.array([b[0] for b in near]), np.array([b[1] for b in near])
    # centre the demo on the pick with the most neighbours within the demo
    # radius (a histogram bin can be won by duplicates stacked on one gold
    # bead), and stay away from the volume edge so no black padding shows
    ok = ((xs > 260) & (xs < win.data.nx - 260) & (ys > 260) & (ys < win.data.ny - 260))
    xs, ys = xs[ok], ys[ok]
    counts = ((xs[:, None] - xs) ** 2 + (ys[:, None] - ys) ** 2 < 80 ** 2).sum(1)
    best = int(np.argmax(counts))
    cx, cy = int(xs[best]), int(ys[best])
    # scroll_to only pans; the viewing depth is the slider's
    win.wdepth.setValue(z)
    QtWidgets.QApplication.processEvents()
    win.scroll_to(cx, cy, z)

    before = crop_at(win.xyview, 420)
    # the eraser cursor: a dashed circle around the point it is about to clear
    p = QtGui.QPainter(before)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(ACCENT, 3, QtCore.Qt.DashLine)
    p.setPen(pen)
    p.drawEllipse(QtCore.QPointF(210, 210), 80, 80)
    p.end()
    before.save(os.path.join(tmp, "erase_before.png"))

    # erase: send everything within that radius (in view pixels ~ voxels here)
    victims = [i for i, b in enumerate(win.boxes)
               if b[5] == cid and abs(b[2] - z) < win.get_boxsize(b[5]) // 2
               and (b[0] - cx) ** 2 + (b[1] - cy) ** 2 < 80 ** 2]
    if victims:
        win.currentset = win.boxes[victims[0]][5]
        win._reassign_boxes(victims, categories.RECYCLE_BIN_ID)
    crop_at(win.xyview, 420).save(os.path.join(tmp, "erase_after.png"))
    win.undo()
    QtWidgets.QApplication.processEvents()

    # select: highlight everything inside a circle, plus the shape outline
    chosen = [b[6] for b in win.boxes
              if abs(b[2] - z) < win.get_boxsize(b[5]) // 2
              and (b[0] - cx) ** 2 + (b[1] - cy) ** 2 < 90 ** 2
              and b[5] == cid]
    win.boxesviewer.replace_selection(chosen)
    win.update_all(reslice=False)
    QtWidgets.QApplication.processEvents()
    sel = crop_at(win.xyview, 420)
    p = QtGui.QPainter(sel)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor("white"), 2, QtCore.Qt.DashLine))
    p.drawEllipse(QtCore.QPointF(210, 210), 90, 90)
    p.end()
    sel.save(os.path.join(tmp, "select.png"))
    win.boxesviewer.clear_selection()

    # compose: before | after | select
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for ax, name, title in (
            (axes[0], "erase_before.png", "Eraser: drag over junk..."),
            (axes[1], "erase_after.png", "...it goes to the Recycle Bin"),
            (axes[2], "select.png", "Select: everything in a shape,\nthen act on it in bulk")):
        ax.imshow(plt.imread(os.path.join(tmp, name)))
        ax.set_title(title, fontsize=11, color=TEXT)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tools-erase-select.png"), dpi=110,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/tools-erase-select.png")


# ---------------------------------------------------------------------------
# a mid-triage state: fresh import, a few confirmations, then rank
# ---------------------------------------------------------------------------
def shot_ranking(data, out, tmp):
    from boxer import categories, metadata
    from boxer.main_window import TomoBoxer
    wd = os.path.join(tmp, "fresh_picks")
    o = argparse.Namespace(
        mode="3D", working_dir=wd, particle_lists=None, native_memmap=False,
        companion_tomogram=None,
        embedded_particle_list=os.path.join(data, "TemplateMatching_Ribosomes_PicksOnly/TS_026.txt"),
        cryosiam_embeddings=None, cryosiam_umap=None,
        tomotwin_embeddings=os.path.join(data, "TomoTwin/Ribosomes_embeddings/TS_026.temb"),
        tomotwin_umap=os.path.join(data, "TomoTwin/Ribosomes_embeddings/TS_026.tumap"))
    win = TomoBoxer(o, os.path.join(data, "Tomograms/TS_026.rec"))
    win.show()
    win.setMinimumSize(980, 900)
    win.boxesviewer.setMinimumSize(980, 760)
    QtWidgets.QApplication.processEvents()

    # confirm the first 60 particles the real session confirmed
    real = metadata.load_class_records(
        os.path.join(data, "Ribosomes_PickyMate/TS_026_picks"), "Confirmed")
    rows = {r["embedding_row"] for r in real if r["embedding_row"] is not None}
    idx = [i for i, b in enumerate(win.boxes) if b[7] in rows][:60]
    cid = next(c for c in win.sets if c not in categories.FIXED_CLASS_IDS)
    win.currentset = cid
    win._reassign_boxes(idx, categories.CONFIRMED_ID)

    shot_tools(win, out, tmp)

    bv = win.boxesviewer
    bv.activate_class_tab(cid)
    QtWidgets.QApplication.processEvents()
    is_particle = gt_matcher(data)
    coords = {b[6]: (b[0], b[1], b[2]) for b in win.boxes}
    is_ok = lambda uid: is_particle(*coords[uid])
    grid = bv.tabs.currentWidget()

    def head_marked(name):
        bar = grid.verticalScrollBar()
        bar.setValue(bar.maximum())
        pm = grab(grid)
        y_off = pm.height() - 560
        pm = pm.copy(0, y_off, pm.width() - 18, 560)
        mark_grid(grid, pm, y_off, is_ok).save(os.path.join(tmp, name))

    head_marked("rank_before.png")
    bar0 = grid.verticalScrollBar()
    bar0.setValue(bar0.maximum())
    grab(grid).copy(0, grab(grid).height() - 560, grid.width() - 18, 560).save(
        os.path.join(tmp, "gif_before.png"))
    grab(bv).copy(0, bv.height() - 190, bv.width(), 190).save(
        os.path.join(tmp, "gif_rank.png"))

    win.run_compute_distance(target_classid=cid, reference_classid=categories.CONFIRMED_ID,
                             space="tomotwin", leave_one_out=False,
                             negative_classid=None)
    for w in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(w, QtWidgets.QMessageBox):
            w.close()
    QtWidgets.QApplication.processEvents()
    head_marked("rank_after.png")
    bar0.setValue(bar0.maximum())
    grab(grid).copy(0, grab(grid).height() - 560, grid.width() - 18, 560).save(
        os.path.join(tmp, "gif_after.png"))
    ranking_gif(out, tmp)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    for ax, name, title in (
            (axes[0], "rank_before.png",
             "Template-matching order: junk and particles mixed"),
            (axes[1], "rank_after.png",
             "Ranked close to Confirmed: review from the top, decide in bulk")):
        ax.imshow(plt.imread(os.path.join(tmp, name)))
        ax.set_title(title, fontsize=11, color=TEXT)
        ax.axis("off")
    fig.text(0.5, 0.015, "dots: ground truth for this figure only — green = a real ribosome, "
                          "red = junk (the app never sees this)",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "ranking-before-after.png"), dpi=110,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/ranking-before-after.png")

    close(win)
    shutil.rmtree(wd, ignore_errors=True)


def write_shortcut_table():
    """Regenerate the shortcut table in docs/keyboard-shortcuts.md from the
    one canonical list in boxer/shortcuts.py."""
    from boxer.shortcuts import reference_markdown
    path = os.path.join(HERE, "docs", "keyboard-shortcuts.md")
    text = open(path).read()
    begin = text.index("<!-- shortcuts:begin")
    begin = text.index("-->", begin) + 4
    end = text.index("<!-- shortcuts:end -->")
    open(path, "w").write(text[:begin] + reference_markdown() + text[end:])
    print(f"wrote the shortcut table into {path}")


def caption_frame(png_path, text, width=880):
    """One GIF frame: the image scaled to `width`, a caption bar above."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(png_path).convert("RGB")
    img = img.resize((width, int(img.height * width / img.width)), Image.LANCZOS)
    bar = 44
    frame = Image.new("RGB", (width, img.height + bar), "white")
    frame.paste(img, (0, bar))
    draw = ImageDraw.Draw(frame)
    font_path = os.path.join(os.path.dirname(matplotlib.__file__),
                             "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
    draw.text((14, 10), text, fill=(30, 30, 30),
              font=ImageFont.truetype(font_path, 21))
    return frame


def ranking_gif(out, tmp):
    """README animation: the gallery reordering the moment picks are ranked.
    Every frame shares one canvas (GIF frames must); the middle frame is the
    dimmed before-grid with the ranking row overlaid, as if just clicked."""
    from PIL import Image
    before = caption_frame(os.path.join(tmp, "gif_before.png"),
                           "Template-matching picks, in pick order — junk and particles mixed")
    after = caption_frame(os.path.join(tmp, "gif_after.png"),
                          "Reordered: closest to your Confirmed first — review from the top, decide in bulk")
    rank = caption_frame(os.path.join(tmp, "gif_before.png"),
                         "You confirm a few good particles...  then ▶ Rank by Embedding Distance")
    rank = Image.blend(rank, Image.new("RGB", rank.size, "white"), 0.55)
    panel = Image.open(os.path.join(tmp, "gif_rank.png")).convert("RGB")
    panel = panel.resize((rank.width, int(panel.height * rank.width / panel.width)),
                         Image.LANCZOS)
    rank.paste(panel, (0, (rank.height - panel.height) // 2))
    frames = [before, rank] + [Image.blend(rank, after, t) for t in (0.25, 0.5, 0.75)] + [after]
    durations = [2600, 2000, 120, 120, 120, 3800]
    path = os.path.join(out, "ranking.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {path}  ({os.path.getsize(path) // 1024} KB)")


# ---------------------------------------------------------------------------
# session navigator
# ---------------------------------------------------------------------------
def shot_session(data, out, tmp):
    import csv
    from boxer.session import Session
    from boxer.session_window import SessionController
    for t in ("TS_026", "TS_037", "TS_043"):
        dst = os.path.join(tmp, f"{t}_picks")
        if not os.path.isdir(dst):
            shutil.copytree(os.path.join(data, f"Ribosomes_PickyMate/{t}_picks"), dst)
    manifest = os.path.join(tmp, "session.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tomogram", "working_dir"])
        for t in ("TS_026", "TS_037", "TS_043"):
            w.writerow([os.path.join(data, f"Tomograms/{t}.rec"), f"{t}_picks"])
    base = argparse.Namespace(mode="3D", working_dir=None, particle_lists=None,
                              native_memmap=False, companion_tomogram=None,
                              embedded_particle_list=None, cryosiam_embeddings=None,
                              cryosiam_umap=None, tomotwin_embeddings=None,
                              tomotwin_umap=None)
    ctl = SessionController(Session.from_manifest(manifest), base)
    ctl.start("TS_026")
    save(grab(ctl.navigator), out, "session-navigator.png")
    panel = ctl.current.boxesviewer.compute_distance_group
    save(grab(panel), out, "ranking-panel.png")
    win = ctl.current
    win._confirm_close = lambda: True
    win.close()
    QtWidgets.QApplication.processEvents()


# ---------------------------------------------------------------------------
# the README overview figure -- pure matplotlib, straight from the data
# ---------------------------------------------------------------------------
def hero(data, out):
    from boxer import embeddings_io, metadata
    from boxer.volume import Volume

    picks = np.array(metadata.load_particle_list(
        os.path.join(data, "TemplateMatching_Ribosomes_PicksOnly/TS_026.txt")))
    emb = embeddings_io.load_tomotwin_embeddings(
        os.path.join(data, "TomoTwin/Ribosomes_embeddings/TS_026.temb"))["embeddings"]
    umap = embeddings_io.load_tomotwin_umap(
        os.path.join(data, "TomoTwin/Ribosomes_embeddings/TS_026.tumap"))["umap_xy"]
    wd = os.path.join(data, "Ribosomes_PickyMate/TS_026_picks")
    conf_rows = np.array([r["embedding_row"] for r in
                          metadata.load_class_records(wd, "Confirmed")
                          if r["embedding_row"] is not None])
    junk_rows = np.array([r["embedding_row"] for r in
                          metadata.load_class_records(wd, "Recycle Bin")
                          if r["embedding_row"] is not None])
    vol = Volume.from_file(os.path.join(data, "Tomograms/TS_026.rec"), native_memmap=True)

    # a handful of early decisions, as at the start of a session
    few_conf, few_junk = conf_rows[:40], junk_rows[:60]

    # rank everything else by Mahalanobis distance to those few confirmed
    ref = np.asarray(emb[few_conf], dtype=np.float64)
    mu = ref.mean(0)
    c = ref - mu
    covinv = np.linalg.pinv(c.T @ c / (len(ref) - 1), hermitian=True)
    d = np.asarray(emb, dtype=np.float64) - mu
    score = ((d @ covinv) * d).sum(1)
    undecided = np.setdiff1d(np.arange(len(picks)), np.concatenate([few_conf, few_junk]))
    order = undecided[np.argsort(score[undecided])]

    def thumb(row):
        x, y, z = picks[row]
        cube = vol.get_clip_centered(x, y, z, 44, 44, 7)
        return norm(cube.mean(axis=0, dtype=np.float32))

    is_particle = gt_matcher(data)

    def block(rows, n_cols=6, n_rows=2):
        """Thumbnails as RGB, each with a thin band along the top: green if
        the pick is a real ribosome by the ground truth, red if junk."""
        rows = list(rows[: n_cols * n_rows])
        cells = []
        for r in rows:
            t = np.repeat(thumb(r)[:, :, None], 3, axis=2)
            colour = (0.18, 0.80, 0.44) if is_particle(*picks[r]) else (0.91, 0.30, 0.24)
            t[:4, :] = colour
            cells.append(np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=1.0))
        return np.vstack([np.hstack(cells[i * n_cols:(i + 1) * n_cols])
                          for i in range(n_rows)])

    z = int(np.bincount(picks[:, 2].astype(int)).argmax())
    near = np.abs(picks[:, 2] - z) < 8
    cx, cy = np.median(picks[near, 0]), np.median(picks[near, 1])
    full = vol.slab_projection("z", z, 2)
    # clamp: a negative start index would wrap around and slice nothing
    x0 = int(np.clip(cx - 240, 0, full.shape[1] - 480))
    y0 = int(np.clip(cy - 240, 0, full.shape[0] - 480))
    sl = norm(full[y0:y0 + 480, x0:x0 + 480])

    fig = plt.figure(figsize=(15.5, 4.4), facecolor="white")
    gs = fig.add_gridspec(1, 4, wspace=0.14, left=0.015, right=0.985, top=0.82, bottom=0.10)

    ax = fig.add_subplot(gs[0])
    ax.imshow(sl, cmap="gray", origin="lower")
    ax.scatter(picks[near, 0] - x0, picks[near, 1] - y0, s=60, facecolors="none",
               edgecolors="#39c0ed", linewidths=1.0)
    ax.set_xlim(0, 480)
    ax.set_ylim(0, 480)
    ax.set_title("Template matching over-picks:\nthousands of hits, most junk",
                 fontsize=11, color=TEXT)

    ax = fig.add_subplot(gs[1])
    ax.scatter(umap[:, 0], umap[:, 1], s=2, c="0.75", linewidths=0)
    ax.set_title("A foundation model embeds\nevery pick", fontsize=11, color=TEXT)

    ax = fig.add_subplot(gs[2])
    ax.scatter(umap[:, 0], umap[:, 1], s=2, c="0.82", linewidths=0)
    ax.scatter(umap[few_junk, 0], umap[few_junk, 1], s=14, c="crimson", linewidths=0,
               label="you reject a few")
    ax.scatter(umap[few_conf, 0], umap[few_conf, 1], s=14, c="limegreen", linewidths=0,
               label="you confirm a few")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("You make a handful\nof decisions", fontsize=11, color=TEXT)

    ax = fig.add_subplot(gs[3])
    ranked = block(order)
    unranked = block(undecided)
    gap = np.ones((10, ranked.shape[1], 3))
    ax.imshow(np.vstack([unranked, gap, ranked]))
    ax.text(-0.02, 0.76, "pick\norder", transform=ax.transAxes, fontsize=10,
            ha="right", va="center", color=TEXT)
    ax.text(-0.02, 0.26, "after\nranking", transform=ax.transAxes, fontsize=10,
            ha="right", va="center", color="#1e8449")
    ax.set_title("PickyMate reorders the rest,\nlive — decide in bulk",
                 fontsize=11, color=TEXT)

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.85")
    for x in (0.253, 0.498, 0.743):
        fig.text(x, 0.47, "→", fontsize=26, color=ACCENT.name(), ha="center", va="center")
    fig.text(0.878, 0.028, "bands: ground truth, figure only\n"
                           "green = real particle · red = junk",
             fontsize=8.5, color="0.4", ha="center", va="center")

    fig.savefig(os.path.join(out, "overview.png"), dpi=115)
    plt.close(fig)
    print(f"wrote {out}/overview.png")
    owner = getattr(vol, "_mmap_owner", None)
    if owner is not None:
        owner.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=os.path.join(HERE, "docs", "images"))
    ap.add_argument("--only", nargs="+", default=None,
                    help="regenerate a subset: hero, main, gallery, umap, tools, ranking, "
                         "session, shortcuts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = lambda k: a.only is None or k in a.only

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tmp = tempfile.mkdtemp(prefix="pickymate_figures_")
    try:
        if want("shortcuts"):
            write_shortcut_table()
        if want("hero"):
            hero(a.data, a.out)
        if want("ranking") or want("tools"):
            shot_ranking(a.data, a.out, tmp)
        if want("session"):
            shot_session(a.data, a.out, tmp)
        if any(want(k) for k in ("main", "gallery", "umap")):
            win = open_curated(a.data, tmp)
            if want("main"):
                shot_main_window(win, a.out)
            if want("gallery"):
                shot_particle_list(win, a.out)
            if want("umap"):
                shot_umap(win, a.out)
            close(win)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
