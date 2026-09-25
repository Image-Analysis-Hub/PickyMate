"""The UMAP map window: every embedded particle of the open tomogram at its
2-D UMAP coordinate, coloured by the set it is in, with lasso selection.

What is selected here becomes the "UMAP Selection" tab in the particle
list -- a view onto those particles, like "All", that never moves them out
of their own sets. Selecting:

    drag                    lasso: the particles inside become the selection
                            (a lasso around nothing changes nothing)
    Ctrl+drag               lasso: add the particles inside
    click on a particle     that particle alone becomes the selection
    Ctrl+click              add or remove that one particle
    Escape                  clear the selection (or cancel a lasso in progress)
    wheel / right-drag      zoom / pan;  double-click or Fit: show everything

Ctrl+click and Escape mean the same here as everywhere else in PickyMate.

Points are rasterised with numpy into an image rather than drawn as one Qt
item each. The map has to stay responsive at the ~100k particles of a
whole session (Task 3's combined map), where per-point graphics items are
far too slow; a raster costs the same no matter how the points overlap.
The rendered image is cached and only rebuilt when the points, their
colours, the selection or the view actually change -- not while a lasso is
being drawn over it.
"""
from __future__ import annotations

import weakref

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import categories

BACKGROUND = (30, 30, 30)
HALO = (255, 255, 0)        # the app's selection colour (categories.SELECTED_COLOR)
CLICK_RADIUS = 6            # screen pixels within which a click picks a particle


def _disk(radius):
    r = int(np.ceil(radius))
    return [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
            if dx * dx + dy * dy <= radius * radius + 0.5]


def points_in_polygon(px, py, poly):
    """Boolean mask: which (px, py) lie inside the closed polygon `poly`
    (an (M, 2) array), by even-odd ray casting, vectorised over the points.
    Points outside the polygon's bounding box are rejected first, so a
    small lasso over a large map costs little."""
    inside = np.zeros(len(px), dtype=bool)
    if len(poly) < 3 or len(px) == 0:
        return inside
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    cand = np.flatnonzero((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1))
    if cand.size == 0:
        return inside
    cx, cy = px[cand], py[cand]
    hit = np.zeros(cand.size, dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = (yi > cy) != (yj > cy)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (cy - yi) / (yj - yi) + xi
        hit ^= crosses & (cx < xint)
        xj, yj = xi, yi
    inside[cand] = hit
    return inside


class UmapCanvas(QtWidgets.QWidget):
    # (indices into the points last given to set_points, mode) --
    # mode is "replace", "add" or "toggle"
    selection_made = QtCore.Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setFocusPolicy(Qt.StrongFocus)
        self._xy = np.empty((0, 2))
        self._classes = np.empty(0, dtype=np.int64)
        self._selected = np.empty(0, dtype=bool)
        self._colors = {}
        self._layers = []
        self._view = None           # (centre x, centre y, pixels per data unit)
        self._image = None          # cached QImage, None = must re-render
        self._buffer = None         # keeps the QImage's pixel data alive
        self._lasso = None          # [QPointF, ...] while a lasso is being drawn
        self._press = None
        self._pan_from = None
        self.point_radius = 2.0
        self.empty_message = ""
        self._user_view = False     # True once zoomed/panned by hand

    # -- data --------------------------------------------------------------
    def set_points(self, xy, classes, colors, layers, selected, refit=False):
        """xy: (N, 2) map coordinates. classes: (N,) set id of each point.
        colors: {set id: (r, g, b)}. layers: set ids in drawing order,
        first drawn first (so later ones sit on top). selected: (N,) bool.
        Only invalidates the rendered image when something drawn changed."""
        xy = np.asarray(xy, dtype=np.float64)
        classes = np.asarray(classes)
        selected = np.asarray(selected, dtype=bool)
        changed = (xy.shape != self._xy.shape or not np.array_equal(xy, self._xy)
                   or not np.array_equal(classes, self._classes)
                   or not np.array_equal(selected, self._selected)
                   or colors != self._colors or layers != self._layers)
        self._xy, self._classes, self._selected = xy, classes, selected
        self._colors, self._layers = dict(colors), list(layers)
        if refit or self._view is None:
            self.fit()
        elif changed:
            self._invalidate()

    def fit(self):
        if len(self._xy) == 0 or self.width() < 2:
            self._view = None
            self._invalidate()
            return
        lo, hi = self._xy.min(0), self._xy.max(0)
        span = np.maximum(hi - lo, 1e-9)
        scale = 0.92 * min(self.width() / span[0], self.height() / span[1])
        self._view = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, scale)
        self._user_view = False
        self._invalidate()

    def _invalidate(self):
        self._image = None
        self.update()

    # -- geometry ----------------------------------------------------------
    def _to_screen(self, xy):
        cx, cy, s = self._view
        return (xy[:, 0] - cx) * s + self.width() / 2, self.height() / 2 - (xy[:, 1] - cy) * s

    def _to_data(self, x, y):
        cx, cy, s = self._view
        return (x - self.width() / 2) / s + cx, (self.height() / 2 - y) / s + cy

    # -- rendering -----------------------------------------------------------
    def _render(self):
        w, h = max(self.width(), 1), max(self.height(), 1)
        buf = np.empty((h, w, 4), dtype=np.uint8)
        buf[..., :3] = BACKGROUND
        buf[..., 3] = 255
        if self._view is not None and len(self._xy):
            sx, sy = self._to_screen(self._xy)
            px, py = np.rint(sx).astype(np.int64), np.rint(sy).astype(np.int64)
            flat = buf.reshape(-1, 4)
            normal = _disk(self.point_radius)
            halo = _disk(self.point_radius + 1.6)

            def stamp(mask, color, offsets):
                xs, ys = px[mask], py[mask]
                for dx, dy in offsets:
                    x, y = xs + dx, ys + dy
                    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    flat[y[ok] * w + x[ok], :3] = color

            # every set as its own layer, so overlapping points of one set
            # never depend on drawing order; the selection last, as a
            # yellow halo with the particle's own colour on top of it
            for cid in self._layers:
                stamp((self._classes == cid) & ~self._selected, self._colors[cid], normal)
            if self._selected.any():
                stamp(self._selected, HALO, halo)
                for cid in self._layers:
                    stamp((self._classes == cid) & self._selected, self._colors[cid], normal)
        self._buffer = buf
        self._image = QtGui.QImage(buf.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)

    def paintEvent(self, event):
        if self._image is None or self._image.width() != self.width() \
                or self._image.height() != self.height():
            self._render()
        p = QtGui.QPainter(self)
        p.drawImage(0, 0, self._image)
        if not len(self._xy) and self.empty_message:
            p.setPen(QtGui.QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.empty_message)
        if self._lasso and len(self._lasso) > 1:
            pen = QtGui.QPen(QtGui.QColor(*HALO), 1.5, Qt.DashLine)
            p.setPen(pen)
            p.drawPolyline(QtGui.QPolygonF(self._lasso + [self._lasso[0]]))
        p.end()

    def resizeEvent(self, event):
        # stay fitted to the window until the user takes over the view
        super().resizeEvent(event)
        if self._view is None or not self._user_view:
            self.fit()
        else:
            self._invalidate()

    # -- interaction -----------------------------------------------------------
    def cancel_lasso(self):
        """Drop a lasso in progress; True if there was one."""
        if self._lasso is None:
            return False
        self._lasso = None
        self._press = None
        self.update()
        return True

    def _nearest(self, pos):
        if self._view is None or not len(self._xy):
            return None
        sx, sy = self._to_screen(self._xy)
        d2 = (sx - pos.x()) ** 2 + (sy - pos.y()) ** 2
        i = int(np.argmin(d2))
        return i if d2[i] <= CLICK_RADIUS ** 2 else None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._view is not None:
            self._press = event.position()
            self._lasso = [event.position()]
        elif event.button() in (Qt.RightButton, Qt.MiddleButton) and self._view is not None:
            self._pan_from = event.position()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._lasso is not None:
            last = self._lasso[-1]
            if abs(pos.x() - last.x()) + abs(pos.y() - last.y()) >= 2:
                self._lasso.append(pos)
                self.update()
        elif self._pan_from is not None:
            cx, cy, s = self._view
            dx, dy = pos.x() - self._pan_from.x(), pos.y() - self._pan_from.y()
            self._view = (cx - dx / s, cy + dy / s, s)
            self._pan_from = pos
            self._user_view = True
            self._invalidate()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan_from = None
            return
        if event.button() != Qt.LeftButton or self._lasso is None:
            return
        ctrl = bool(event.modifiers() & Qt.ControlModifier)
        lasso, press = self._lasso, self._press
        self._lasso = self._press = None
        pos = event.position()
        # Click or lasso is decided by how far the drag reached, not where
        # it ended: a lasso drawn as a closed loop finishes back at its
        # starting point, and must not be mistaken for a click there.
        reach = max(abs(q.x() - press.x()) + abs(q.y() - press.y()) for q in lasso + [pos])
        if reach < 4 or len(lasso) < 3:
            i = self._nearest(pos)
            if i is not None:
                self.selection_made.emit(np.array([i]), "toggle" if ctrl else "replace")
            self.update()
            return
        sx, sy = self._to_screen(self._xy)
        poly = np.array([[q.x(), q.y()] for q in lasso])
        inside = np.flatnonzero(points_in_polygon(sx, sy, poly))
        self.update()
        # A lasso around nothing leaves the selection as it is, the same as a
        # click on empty space: wiping a large selection because a lasso
        # missed would be easy to do and tedious to redo. Escape clears.
        if inside.size:
            self.selection_made.emit(inside, "add" if ctrl else "replace")

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.fit()

    def wheelEvent(self, event):
        if self._view is None:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        pos = event.position()
        # keep the map point under the cursor where it is
        mx, my = self._to_data(pos.x(), pos.y())
        cx, cy, s = self._view
        s2 = s * (1.2 ** steps)
        self._view = (mx - (pos.x() - self.width() / 2) / s2,
                      my + (pos.y() - self.height() / 2) / s2, s2)
        self._user_view = True
        self._invalidate()


class UmapWindow(QtWidgets.QWidget):
    """The map for one open tomogram; see the module docstring."""

    def __init__(self, target):
        super().__init__()
        self._target = weakref.ref(target)
        self._uids = np.empty(0, dtype=np.int64)
        self._space = None
        self.setWindowTitle(f"PickyMate — UMAP — {target.session_name}")
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Map:"))
        self.space_combo = QtWidgets.QComboBox()
        self.space_combo.setToolTip("Which 2-D UMAP to show.")
        self.space_combo.currentIndexChanged.connect(lambda _i: self.refresh(refit=True))
        top.addWidget(self.space_combo)
        top.addWidget(QtWidgets.QLabel("Point size:"))
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(1, 8)
        self.size_spin.setValue(2)
        self.size_spin.valueChanged.connect(self._on_size)
        top.addWidget(self.size_spin)
        fit = QtWidgets.QPushButton("Fit")
        fit.setToolTip("Show every particle (or double-click the map).")
        fit.clicked.connect(lambda: self.canvas.fit())
        top.addWidget(fit)
        top.addStretch(1)
        self.count_label = QtWidgets.QLabel()
        top.addWidget(self.count_label)
        layout.addLayout(top)

        self.canvas = UmapCanvas()
        self.canvas.selection_made.connect(self._on_selection)
        layout.addWidget(self.canvas, 1)

        # Refreshes after edits are coalesced and deferred to the next pass
        # of the event loop: the edit itself -- slice views, gallery -- is
        # drawn first, and a burst of edits costs one map refresh, not one
        # each. (~45 ms per refresh at 17k particles with half selected.)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self.refresh)

        hint = QtWidgets.QLabel(
            "Drag to lasso · Ctrl+drag adds · Ctrl+click adds or removes one · "
            "Escape clears · wheel zooms · right-drag pans · double-click fits. "
            "The selection appears as the \"UMAP Selection\" tab of the particle list; "
            "particles stay in their own sets.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(hint)
        self.resize(720, 760)

    def _map_spaces(self, target):
        return [n for n in target.ordered_space_names()
                if target.embedding_spaces[n]["array"].ndim == 2
                and target.embedding_spaces[n]["array"].shape[1] == 2]

    def refresh(self, refit=False):
        """Re-read positions, sets and the selection from the tomogram.
        Called after every edit while the window is open (see
        TomoBoxer.update_boximgs), so colours follow particles as they
        move between sets."""
        target = self._target()
        if target is None:
            return
        spaces = self._map_spaces(target)
        current = self.space_combo.currentData()
        if [self.space_combo.itemData(i) for i in range(self.space_combo.count())] != spaces:
            self.space_combo.blockSignals(True)
            self.space_combo.clear()
            for n in spaces:
                self.space_combo.addItem(target.embedding_spaces[n]["label"], n)
            idx = self.space_combo.findData(current)
            self.space_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.space_combo.blockSignals(False)
        space = self.space_combo.currentData()
        if space is None:
            self.canvas.empty_message = ("No 2-D UMAP is attached to this tomogram's "
                                         "particles, so there is no map to show.")
            self._uids = np.empty(0, dtype=np.int64)
            self.canvas.set_points(np.empty((0, 2)), [], {}, [], [], refit=True)
            self.count_label.setText("")
            return
        refit = refit or space != self._space
        self._space = space
        array = target.embedding_spaces[space]["array"]
        rows, uids, classes = [], [], []
        missing = 0
        for b in target.boxes:
            if b[7] is None or not (0 <= b[7] < len(array)):
                missing += 1
                continue
            rows.append(b[7])
            uids.append(b[6])
            classes.append(b[5])
        self._uids = np.asarray(uids, dtype=np.int64)
        classes = np.asarray(classes, dtype=np.int64)
        selection = target.umap_selection
        selected = (np.isin(self._uids, np.fromiter(selection, dtype=np.int64))
                    if selection else np.zeros(len(self._uids), dtype=bool))
        present = set(classes.tolist())
        colors = {cid: categories.class_color(cid).getRgb()[:3] for cid in present}
        # Recycle Bin underneath, then the ordinary sets, Confirmed on top
        order = [categories.RECYCLE_BIN_ID] + [
            c for c in categories.category_order(target.sets)
            if c not in categories.FIXED_CLASS_IDS] + [categories.CONFIRMED_ID]
        layers = [c for c in order if c in present] + sorted(present - set(order))
        self.canvas.set_points(array[np.asarray(rows, dtype=np.int64)] if rows else np.empty((0, 2)),
                               classes, colors, layers, selected, refit=refit)
        text = f"{len(uids)} particles · {int(selected.sum())} selected"
        if missing:
            text += f" · {missing} without an embedding (not on the map)"
        self.count_label.setText(text)

    def schedule_refresh(self):
        self._refresh_timer.start()

    def _on_size(self, value):
        self.canvas.point_radius = float(value)
        self.canvas._invalidate()

    def _on_selection(self, indices, mode):
        target = self._target()
        if target is None:
            return
        target.set_umap_selection(self._uids[np.asarray(indices, dtype=np.int64)].tolist(),
                                  mode, from_map=True)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
