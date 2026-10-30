#!/usr/bin/env python3
"""Regenerate every image in docs/images/ from real data.

    python docs/make_figures.py [--data DIR] [--out docs/images]

Screenshots are taken from the app itself, rendered offscreen, on a
temporary copy of a real curated working directory (EMPIAR-10988 TS_026,
ribosomes) -- never on the original. The composed panels are drawn with
matplotlib from the same data. Rerun this
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
    conf_uids = [b.uid for b in win.boxes if b.classid == categories.CONFIRMED_ID]
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
    rank_c = bv.ranking.mapTo(
        bv, QtCore.QPoint(bv.ranking.width() - 40,
                          bv.ranking.height() - 60))
    tb_c = bv.gallery_toolbar.mapTo(
        bv, QtCore.QPoint(bv.gallery_toolbar.width() - 40,
                          bv.gallery_toolbar.height() // 2))
    save(annotate(pm, [(p1.x(), p1.y(), "1"), (grid_c.x(), grid_c.y(), "2"),
                       (rank_c.x(), rank_c.y(), "3"), (tb_c.x(), tb_c.y(), "4")]),
         out, "particle-list-overview.png")


def shot_umap(win, out, tmp):
    """The UMAP-selection GIF: the map, a lasso being drawn, the selection
    lighting up, and the UMAP Selection tab it becomes in the Particle
    List."""
    from boxer import categories
    from boxer.umap_view import points_in_polygon
    win.open_umap_window()
    uw = win.umap_window
    uw.setMinimumSize(760, 700)
    QtWidgets.QApplication.processEvents()
    uw.refresh(refit=True)
    QtWidgets.QApplication.processEvents()
    c = uw.canvas
    grab(uw).save(os.path.join(tmp, "umap_plain.png"))

    # a hand-drawn-looking lasso around the confirmed cloud, in data coords
    conf = c._classes == categories.CONFIRMED_ID
    cx, cy = np.median(c._xy[conf], axis=0)
    rx = 1.25 * np.percentile(np.abs(c._xy[conf, 0] - cx), 92)
    ry = 1.25 * np.percentile(np.abs(c._xy[conf, 1] - cy), 92)
    th = np.linspace(0, 2 * np.pi, 80)
    poly = np.c_[cx + rx * (1 + 0.10 * np.sin(3 * th)) * np.cos(th),
                 cy + ry * (1 + 0.10 * np.cos(2 * th)) * np.sin(th)]
    lasso = grab(uw)
    sx, sy = c._to_screen(poly)
    o = c.mapTo(uw, QtCore.QPoint(0, 0))
    p = QtGui.QPainter(lasso)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtGui.QPen(ACCENT, 3, QtCore.Qt.DashLine))
    p.drawPolyline(QtGui.QPolygonF(
        [QtCore.QPointF(x + o.x(), y + o.y()) for x, y in zip(sx, sy)]))
    p.end()
    lasso.save(os.path.join(tmp, "umap_lasso.png"))

    inside = points_in_polygon(c._xy[:, 0], c._xy[:, 1], poly)
    win.set_umap_selection(uw._uids[inside].tolist(), "replace")
    QtWidgets.QApplication.processEvents()
    uw.refresh()
    grab(uw).save(os.path.join(tmp, "umap_selected.png"))
    print(f"umap gif: lassoed {int(inside.sum())} picks")

    bv = win.boxesviewer
    bv.tabs.setCurrentWidget(bv.umap_grid)
    grab(bv)   # let the freshly shown tab's deferred sync finish
    # centre the gallery on a Confirmed member of the selection, so the
    # frame shows the colour-bordered groups rather than one red block
    cls = {b.uid: b.classid for b in win.boxes}
    conf_sel = [u for u in uw._uids[inside].tolist()
                if cls.get(u) == categories.CONFIRMED_ID]
    item = bv.umap_grid._items.get(conf_sel[0]) if conf_sel else None
    if item is not None:
        bv.umap_grid.centerOn(item)
        QtWidgets.QApplication.processEvents()
        bv.umap_grid.centerOn(item)
    grab(bv).save(os.path.join(tmp, "umap_tab.png"))

    from PIL import Image
    frames = [caption_frame(os.path.join(tmp, f), text) for f, text in (
        ("umap_plain.png", "The UMAP map: every embedded pick, colored by set, live"),
        ("umap_lasso.png", "Drag to lasso a region..."),
        ("umap_selected.png", "...it becomes the selection — picks stay in their sets"),
        ("umap_tab.png", "...and appears as the UMAP Selection tab in the Particle List"))]
    w, h = max(f.width for f in frames), max(f.height for f in frames)
    canvas = []
    for f in frames:
        page = Image.new("RGB", (w, h), "white")
        page.paste(f, ((w - f.width) // 2, (h - f.height) // 2))
        canvas.append(page)
    path = os.path.join(out, "umap-selection.gif")
    canvas[0].save(path, save_all=True, append_images=canvas[1:],
                   duration=[2800, 2200, 2200, 4200], loop=0, optimize=True)
    print(f"wrote {path}  ({os.path.getsize(path) // 1024} KB)")
    win.clear_umap_selection()
    uw.close()


def shot_tools(win, out, tmp):
    """Eraser before/after and Select-tool crops from the XY view, on the
    fresh-import state (thousands of raw picks in one set)."""
    from boxer import categories
    cid = next(c for c in win.sets if c not in categories.FIXED_CLASS_IDS)
    # a crowded neighbourhood: the densest picked slice
    zs = [b.z for b in win.boxes if b.classid == cid]
    z = int(np.bincount(zs).argmax())
    near = [b for b in win.boxes if abs(b.z - z) < 4 and b.classid == cid]
    xs, ys = np.array([b.x for b in near]), np.array([b.y for b in near])
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
               if b.classid == cid and abs(b.z - z) < win.get_boxsize(b.classid) // 2
               and (b.x - cx) ** 2 + (b.y - cy) ** 2 < 80 ** 2]
    if victims:
        win.currentset = win.boxes[victims[0]].classid
        win._reassign_boxes(victims, categories.RECYCLE_BIN_ID)
    crop_at(win.xyview, 420).save(os.path.join(tmp, "erase_after.png"))
    win.undo()
    QtWidgets.QApplication.processEvents()

    # select: highlight everything inside a circle, plus the shape outline
    chosen = [b.uid for b in win.boxes
              if abs(b.z - z) < win.get_boxsize(b.classid) // 2
              and (b.x - cx) ** 2 + (b.y - cy) ** 2 < 90 ** 2
              and b.classid == cid]
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
    idx = [i for i, b in enumerate(win.boxes) if b.embedding_row in rows][:60]
    cid = next(c for c in win.sets if c not in categories.FIXED_CLASS_IDS)

    # ground-truth colour-coding for the static panels
    is_particle = gt_matcher(data)
    coords = {b.uid: (b.x, b.y, b.z) for b in win.boxes}
    is_ok = lambda uid: is_particle(*coords[uid])
    bv = win.boxesviewer
    bv.activate_class_tab(cid)
    QtWidgets.QApplication.processEvents()
    grid = bv.tabs.currentWidget()

    def head_marked(name):
        bar = grid.verticalScrollBar()
        bar.setValue(bar.maximum())
        pm = grab(grid)
        y_off = pm.height() - 560
        pm = pm.copy(0, y_off, pm.width() - 18, 560)
        mark_grid(grid, pm, y_off, is_ok).save(os.path.join(tmp, name))

    # the pristine import is the "before" everywhere: removing particles
    # from a live tab leaves holes in its layout (sync_particles keeps
    # slots), so the untouched template-matching order must be grabbed
    # before anything is confirmed
    head_marked("rank_before.png")
    bar = grid.verticalScrollBar()
    bar.setValue(bar.maximum())
    pm = grab(grid)
    y_off = pm.height() - 560
    pm = pm.copy(0, y_off, pm.width() - 18, 560)
    pm.save(os.path.join(tmp, "gif_pick.png"))
    # ...and the same view with the picks about to be confirmed outlined:
    # the GIF must show where the confirmed particles come from
    conf_uids = {win.boxes[i].uid for i in idx}
    p = QtGui.QPainter(pm)
    p.setPen(QtGui.QPen(QtGui.QColor("#2ecc71"), 5))
    shown = 0
    for uid in conf_uids:
        item = grid._items.get(uid)
        if item is None:
            continue
        r = grid.mapFromScene(item.sceneBoundingRect()).boundingRect()
        r.translate(0, -y_off)
        if r.top() >= 0 and r.bottom() < pm.height() and r.right() < pm.width():
            p.drawRect(r.adjusted(3, 3, -3, -3))
            shown += 1
    p.end()
    pm.save(os.path.join(tmp, "gif_confirm.png"))
    print(f"ranking gif: {shown} of the confirmed picks visible in the head")

    win.currentset = cid
    win._reassign_boxes(idx, categories.CONFIRMED_ID)

    shot_tools(win, out, tmp)

    bv.activate_class_tab(cid)
    QtWidgets.QApplication.processEvents()
    grab(bv).copy(0, bv.height() - 190, bv.width(), 190).save(
        os.path.join(tmp, "gif_rank.png"))

    def rank():
        win.run_compute_distance(target_classid=cid,
                                 reference_classid=categories.CONFIRMED_ID,
                                 space="tomotwin", leave_one_out=False,
                                 negative_classid=None)
        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, QtWidgets.QMessageBox):
                w.close()
        QtWidgets.QApplication.processEvents()

    rank()
    bar0 = grid.verticalScrollBar()
    bar0.setValue(bar0.maximum())
    grab(grid).copy(0, grab(grid).height() - 560, grid.width() - 18, 560).save(
        os.path.join(tmp, "gif_after.png"))
    ranking_gif(out, tmp)

    # For the static "after" panel only: its point is "review from the top",
    # so recycle the ground-truth junk that ranked into the visible head and
    # re-rank -- which also re-compacts the layout -- until the head is
    # clean. Pure illustration; the GIF above is the unretouched reorder.
    for _ in range(8):
        bar0.setValue(bar0.maximum())
        grab(grid)
        h = grid.viewport().height()
        junk = []
        for uid, item in grid._items.items():
            v = grid.mapFromScene(item.sceneBoundingRect()).boundingRect()
            if v.bottom() > h - 566 and v.top() < h and not is_ok(uid):
                junk.append(uid)
        if not junk:
            break
        win._reassign_boxes([i for i, b in enumerate(win.boxes) if b.uid in set(junk)],
                            categories.RECYCLE_BIN_ID)
        rank()
    head_marked("rank_after.png")

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
    """README animation: the pick-order gallery, the picks the user confirms
    outlined in green, the rank click, and the reorder. Every frame shares
    one canvas (GIF frames must); the rank frame is the dimmed gallery with
    the ranking row overlaid, as if just clicked."""
    from PIL import Image
    pick = caption_frame(os.path.join(tmp, "gif_pick.png"),
                         "Template-matching picks, in pick order — junk and particles mixed")
    confirm = caption_frame(os.path.join(tmp, "gif_confirm.png"),
                            "You confirm good particles as you meet them (green)...")
    after = caption_frame(os.path.join(tmp, "gif_after.png"),
                          "Reordered: closest to your Confirmed first — review from the top, decide in bulk")
    rank = caption_frame(os.path.join(tmp, "gif_confirm.png"),
                         "...then ▶ Rank by Embedding Distance")
    rank = Image.blend(rank, Image.new("RGB", rank.size, "white"), 0.55)
    panel = Image.open(os.path.join(tmp, "gif_rank.png")).convert("RGB")
    panel = panel.resize((rank.width, int(panel.height * rank.width / panel.width)),
                         Image.LANCZOS)
    rank.paste(panel, (0, (rank.height - panel.height) // 2))
    frames = [pick, confirm, rank] + [Image.blend(rank, after, t) for t in (0.25, 0.5, 0.75)] + [after]
    durations = [2600, 2600, 1800, 120, 120, 120, 3800]
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
    panel = ctl.current.boxesviewer.ranking
    save(grab(panel), out, "ranking-panel.png")
    win = ctl.current
    win._confirm_close = lambda: True
    win.close()
    QtWidgets.QApplication.processEvents()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=os.path.join(HERE, "docs", "images"))
    ap.add_argument("--only", nargs="+", default=None,
                    help="regenerate a subset: main, gallery, umap, tools, ranking, "
                         "session, shortcuts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = lambda k: a.only is None or k in a.only

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tmp = tempfile.mkdtemp(prefix="pickymate_figures_")
    try:
        if want("shortcuts"):
            write_shortcut_table()
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
                shot_umap(win, a.out, tmp)
            close(win)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
