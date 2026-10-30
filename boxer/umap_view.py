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
from .session_browser import OtherTomogramsWindow
from .session_map import MAPPABLE

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


# Session map colours: by category (the same red/green as the Recycle Bin
# and Confirmed everywhere else, blue for particles still in an ordinary set)
# or by tomogram (a qualitative palette, cycling past ten)
CATEGORY_COLORS = {0: (100, 150, 230), 1: QtGui.QColor("crimson").getRgb()[:3],
                   2: QtGui.QColor("limegreen").getRgb()[:3]}
CATEGORY_NAMES = {0: "in a set", 1: "Recycle Bin", 2: "Confirmed"}
TOMOGRAM_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                   (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
_BUCKETS = 5   # tomogram colouring interleaves each tomogram over this many drawing layers


class UmapWindow(QtWidgets.QWidget):
    """The map window of the open tomogram; see the module docstring.

    In a multi-tomogram session the "Map" list also offers maps of *all*
    the session's tomograms together (boxer/session_map.py), computed on
    demand in the background. On such a map a selection spans every
    tomogram: the open one's selected particles go to its "UMAP Selection"
    tab as usual, the others' are written to their own working directories
    -- each tomogram's tab shows them the next time it is opened, and the
    navigator counts them -- and Browse Other Tomograms shows them now,
    view only. A lasso there replaces the selection of every tomogram,
    and Escape clears them all; on a single tomogram's map both only ever
    touch that tomogram."""

    def __init__(self, target):
        super().__init__()
        self._target = weakref.ref(target)
        controller = getattr(target, "session_controller", None)
        self._controller = weakref.ref(controller) if controller is not None else None
        self._uids = np.empty(0, dtype=np.int64)
        self._session_points = None     # (tomogram index per point, uid per point, names)
        self._key = None
        self._browser = None
        self.setWindowTitle(f"PickyMate — UMAP — {target.session_name}")
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Map:"))
        self.space_combo = QtWidgets.QComboBox()
        self.space_combo.setToolTip("Which 2-D map to show: this tomogram's own UMAPs, or -- in a "
                                    "session -- a map of all its tomograms together.")
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
        layout.addLayout(top)
        # below the map with the legend, not in the top row, where a long map
        # name ("All 10 tomograms -- ...") left it too little room
        self.count_label = QtWidgets.QLabel()

        # -- session map controls, shown only while a session map is chosen -----
        self.session_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.session_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("Colour by:"))
        self.colour_combo = QtWidgets.QComboBox()
        self.colour_combo.addItem("set", "category")
        self.colour_combo.addItem("tomogram", "tomogram")
        self.colour_combo.setToolTip(
            "Set: Confirmed green, Recycle Bin red, still in a set blue. Tomogram: one colour "
            "per tomogram -- well-mixed colours mean particles from different tomograms "
            "land in the same places.")
        self.colour_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        row.addWidget(self.colour_combo)
        self.compute_button = QtWidgets.QPushButton("Compute")
        self.compute_button.setToolTip("Fit one UMAP on the full embeddings of every tomogram in "
                                       "the session. Runs in the background.")
        self.compute_button.clicked.connect(self._compute)
        row.addWidget(self.compute_button)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(180)
        self.progress_bar.setVisible(False)
        row.addWidget(self.progress_bar)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel)
        row.addWidget(self.cancel_button)
        row.addStretch(1)
        self.browse_button = QtWidgets.QPushButton("Browse Other Tomograms...")
        self.browse_button.setToolTip("See the particles selected on this map in the other "
                                      "tomograms, view only -- open a tomogram to edit its own.")
        self.browse_button.clicked.connect(self.open_browser)
        row.addWidget(self.browse_button)
        layout.addWidget(self.session_row)
        self.session_status = QtWidgets.QLabel()
        self.session_status.setWordWrap(True)
        layout.addWidget(self.session_status)
        self.session_row.setVisible(False)
        self.session_status.setVisible(False)

        self.canvas = UmapCanvas()
        self.canvas.selection_made.connect(self._on_selection)
        layout.addWidget(self.canvas, 1)

        self.legend = QtWidgets.QLabel()
        self.legend.setWordWrap(True)
        self.legend.setVisible(False)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.legend, 1)
        bottom.addWidget(self.count_label, 0, Qt.AlignRight)
        layout.addLayout(bottom)

        # Refreshes after edits are coalesced and deferred to the next pass
        # of the event loop: the edit itself -- slice views, gallery -- is
        # drawn first, and a burst of edits costs one map refresh, not one
        # each. (~45 ms per refresh at 17k particles with half selected.)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self.refresh)

        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.hint)
        self.resize(760, 800)

        if controller is not None:
            job = controller.session_map_job
            job.status.connect(self._on_job_status)
            job.progress.connect(self._on_job_progress)
            job.finished.connect(self._on_job_finished)

    # -- which map --------------------------------------------------------------
    def _map_entries(self, target):
        entries = [(n, target.embedding_spaces[n]["label"]) for n in target.ordered_space_names()
                   if target.embedding_spaces[n]["array"].ndim == 2
                   and target.embedding_spaces[n]["array"].shape[1] == 2]
        ctl = self._controller() if self._controller else None
        if ctl is not None:
            n = len(ctl.session.names)
            for space, label in MAPPABLE.items():
                if space in target.embedding_spaces:
                    entries.append((f"session:{space}", f"All {n} tomograms — {label}"))
        return entries

    def _session_space(self):
        key = self.space_combo.currentData()
        return key.split(":", 1)[1] if isinstance(key, str) and key.startswith("session:") else None

    def refresh(self, refit=False):
        """Re-read positions, sets and the selection. Called after every edit
        while the window is open (see TomoBoxer.update_boximgs), so colours
        follow particles as they move between sets."""
        target = self._target()
        if target is None:
            return
        entries = self._map_entries(target)
        current = self.space_combo.currentData()
        if [self.space_combo.itemData(i) for i in range(self.space_combo.count())] != [k for k, _ in entries]:
            self.space_combo.blockSignals(True)
            self.space_combo.clear()
            for key, label in entries:
                self.space_combo.addItem(label, key)
            idx = self.space_combo.findData(current)
            self.space_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.space_combo.blockSignals(False)
        key = self.space_combo.currentData()
        refit = refit or key != self._key
        self._key = key
        session_space = self._session_space()
        self.session_row.setVisible(session_space is not None)
        self.session_status.setVisible(session_space is not None)
        self.legend.setVisible(session_space is not None)
        if session_space is not None:
            self.hint.setText(
                "Map of every tomogram in the session. Drag to lasso · Ctrl+drag adds · "
                "Ctrl+click adds or removes one · Escape clears — here all of these apply to "
                "every tomogram at once. This tomogram's particles appear in its \"UMAP "
                "Selection\" tab; the others' are saved to their working directories.")
            return self._refresh_session(target, session_space, refit)
        self._session_points = None
        self.hint.setText(
            "Drag to lasso · Ctrl+drag adds · Ctrl+click adds or removes one · "
            "Escape clears · wheel zooms · right-drag pans · double-click fits. "
            "The selection appears as the \"UMAP Selection\" tab of the particle list; "
            "particles stay in their own sets.")
        if key is None:
            self.canvas.empty_message = ("No 2-D UMAP is attached to this tomogram's "
                                         "particles, so there is no map to show.")
            self._uids = np.empty(0, dtype=np.int64)
            self.canvas.set_points(np.empty((0, 2)), [], {}, [], [], refit=True)
            self.count_label.setText("")
            return
        array = target.embedding_spaces[key]["array"]
        rows, uids, classes = [], [], []
        missing = 0
        for b in target.boxes:
            if b.embedding_row is None or not (0 <= b.embedding_row < len(array)):
                missing += 1
                continue
            rows.append(b.embedding_row)
            uids.append(b.uid)
            classes.append(b.classid)
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

    # -- the session map --------------------------------------------------------
    def _refresh_session(self, target, space, refit):
        ctl = self._controller()
        store, session, job = ctl.session_map, ctl.session, ctl.session_map_job
        running = job.running()
        self.compute_button.setEnabled(not running)
        self.progress_bar.setVisible(running)
        self.cancel_button.setVisible(running)
        m, status = store.load(space)
        self.compute_button.setText("Recompute" if m is not None else "Compute")
        if not running:
            self.session_status.setText(self._describe(m, status))
        if m is None or status == "incompatible":
            n = len(session.names)
            self.canvas.empty_message = (
                f"The map of all {n} tomograms has not been computed yet. Press Compute: it "
                f"runs in the background, typically well under a minute for ~100k particles "
                f"with TomoTwin." if m is None else
                "This map no longer matches the tomograms' embeddings (their particle counts "
                "changed). Press Recompute.")
            self._session_points = None
            self.canvas.set_points(np.empty((0, 2)), [], {}, [], [], refit=True)
            self.count_label.setText("")
            self.legend.setText("")
            return

        names = [name for name, _n in m.layout]
        xy, cats, tomo, uids, sel = [], [], [], [], []
        here = target.session_name
        for t, (name, n) in enumerate(m.layout):
            coords = m.coords_for(name)
            if name == here:
                emb = [b for b in target.boxes if b.embedding_row is not None]
                rows = np.asarray([b.embedding_row for b in emb], dtype=np.int64)
                u = np.asarray([b.uid for b in emb], dtype=np.int64)
                c = np.asarray([2 if b.classid == categories.CONFIRMED_ID else
                                1 if b.classid == categories.RECYCLE_BIN_ID else 0 for b in emb],
                               dtype=np.int8)
                chosen = target.umap_selection
            else:
                table = session.particle_table(name)
                rows, u, c = table["rows"], table["uids"], table["cats"]
                chosen = session.selection_uids(name)
            ok = (rows >= 0) & (rows < n)
            rows, u, c = rows[ok], u[ok], c[ok]
            xy.append(coords[rows])
            cats.append(c)
            uids.append(u)
            tomo.append(np.full(len(u), t, dtype=np.int32))
            sel.append(np.isin(u, np.fromiter(chosen, dtype=np.int64)) if chosen
                       else np.zeros(len(u), dtype=bool))
        xy = np.vstack(xy) if xy else np.empty((0, 2))
        cats, uids, tomo, sel = (np.concatenate(a) if a else np.empty(0) for a in (cats, uids, tomo, sel))
        sel = sel.astype(bool)
        self._session_points = (tomo, uids.astype(np.int64), names)

        if self.colour_combo.currentData() == "tomogram":
            # each tomogram spread over several interleaved layers, so no one
            # tomogram is always drawn on top of the others
            bucket = (uids.astype(np.int64) * 2654435761) % _BUCKETS
            classes = bucket * len(names) + tomo
            colors = {int(b * len(names) + t): TOMOGRAM_COLORS[t % len(TOMOGRAM_COLORS)]
                      for b in range(_BUCKETS) for t in range(len(names))}
            layers = [int(b * len(names) + t) for b in range(_BUCKETS) for t in range(len(names))]
            self.legend.setText("  ".join(
                f'<span style="color: rgb{TOMOGRAM_COLORS[t % len(TOMOGRAM_COLORS)]};">●</span> {name}'
                for t, name in enumerate(names)))
        else:
            classes = cats.astype(np.int64)
            colors = dict(CATEGORY_COLORS)
            layers = [1, 0, 2]      # Recycle Bin underneath, Confirmed on top
            self.legend.setText("  ".join(
                f'<span style="color: rgb{CATEGORY_COLORS[k]};">●</span> {CATEGORY_NAMES[k]}'
                for k in (2, 0, 1)))
        self.canvas.set_points(xy, classes, colors, layers, sel, refit=refit)
        n_here = int(sel[tomo == names.index(here)].sum()) if here in names else 0
        self.count_label.setText(f"{len(uids)} particles in {len(names)} tomograms · "
                                 f"{int(sel.sum())} selected ({n_here} here)")

    @staticmethod
    def _describe(m, status):
        if m is None:
            return "Not computed yet."
        p = m.params
        how = f"{p.get('input_dims', '?')} dimensions"
        if p.get("pca_dims"):
            how += (f" → {p['pca_dims']} by PCA ({100 * p.get('pca_variance_kept', 1):.2f}% of "
                    f"the variance kept)")
        text = (f"{sum(n for _, n in m.layout)} particles from {len(m.layout)} tomograms, {how}, "
                f"UMAP with {p.get('neighbors')} neighbours, min_dist {p.get('min_dist')}.")
        if status == "stale":
            text = ("⚠ Computed from embedding files that have changed since; still usable, but "
                    "Recompute for a current map. " + text)
        return text

    # -- computing it -------------------------------------------------------------
    def _compute(self):
        space = self._session_space()
        ctl = self._controller() if self._controller else None
        if space is None or ctl is None or ctl.session_map_job.running():
            return
        self.compute_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        ctl.session_map_job.start(space)

    def _cancel(self):
        ctl = self._controller() if self._controller else None
        if ctl is not None:
            self.session_status.setText("Cancelling...")
            ctl.session_map_job.cancel()

    def _on_job_status(self, text):
        self.session_status.setText(text)

    def _on_job_progress(self, done, total):
        if total:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        else:
            self.progress_bar.setRange(0, 0)    # busy, no measure of how far

    def _on_job_finished(self, ok, space, message):
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.compute_button.setEnabled(True)
        if not ok and message.startswith("umap-learn is not installed"):
            QtWidgets.QMessageBox.warning(self, "UMAP not available", message)
        self.session_status.setText(message)
        if ok:
            self.refresh(refit=True)

    # -- selecting ----------------------------------------------------------------
    def _on_selection(self, indices, mode):
        target = self._target()
        if target is None:
            return
        indices = np.asarray(indices, dtype=np.int64)
        if self._session_points is not None and self._session_space() is not None:
            tomo, uids, names = self._session_points
            picked = {}
            for t, u in zip(tomo[indices].tolist(), uids[indices].tolist()):
                picked.setdefault(names[t], set()).add(u)
            self._controller().apply_session_selection(picked, mode)
            return
        target.set_umap_selection(self._uids[indices].tolist(), mode, from_map=True)

    def clear_selection(self):
        """Escape: on a session map, every tomogram's selection; on this
        tomogram's own map, only its own."""
        target = self._target()
        if target is None:
            return
        if self._session_space() is not None and self._controller is not None:
            self._controller().apply_session_selection({}, "replace")
        else:
            target.clear_umap_selection()

    def open_browser(self):
        target = self._target()
        if target is None or self._controller is None:
            return
        if self._browser is None:
            self._browser = OtherTomogramsWindow(self._controller(), target)
        self._browser.refresh()
        self._browser.show()
        self._browser.raise_()

    # -- misc -------------------------------------------------------------------
    def schedule_refresh(self):
        self._refresh_timer.start()
        if self._browser is not None and self._browser.isVisible():
            self._browser.schedule_refresh()

    def _on_size(self, value):
        self.canvas.point_radius = float(value)
        self.canvas._invalidate()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def closeEvent(self, event):
        if self._browser is not None:
            self._browser.close()
        super().closeEvent(event)

    def dispose(self):
        """Delete this window and the browser, a separate top-level window
        that would otherwise outlive it -- used when a session switches
        tomograms."""
        if self._browser is not None:
            self._browser.close()
            self._browser.deleteLater()
            self._browser = None
        self.close()
        self.deleteLater()

