"""A thumbnail grid for the boxed-particle list, replacing the subset of
EMAN2's OpenGL EMImageMXWidget used by e2spt_boxer.py.

Unlike a plain auto-flowing QListWidget, this grid keeps each particle's
thumbnail pinned to a stable "slot" that is assigned once (in bottom-left-
first order) and never reassigned just because another particle elsewhere
was added or removed -- deleting a particle leaves a gap. The "Organize
Grid" button (see ParticleListPanel) explicitly compacts the grid, removing
gaps while preserving relative order.

Particles are tracked by a caller-supplied unique id (uid), not by list
position, so slot assignment survives additions/removals anywhere else in
the particle list. A uid that reappears after being removed (only possible
via undo, since uids are never reused) returns to its old slot and is
shown selected/highlighted, rather than being appended at the end.

The "All" tab is the one exception to the stable-slot rule: it groups
particles by class (Recycle Bin and Confirmed first, then alphabetically),
via ParticleGrid.sync_particles_grouped, which fully recomputes slots on
every call so classes stay visually clustered as particles move between
them. Per-class tabs are unaffected and keep the stable/gap-preserving
behavior described above.
"""
from __future__ import annotations

import weakref

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import categories
from .widgets import space_key_state

THUMB_SIZE = 96
MIN_THUMB = 24
MAX_THUMB = 256
SPACING = 4
UID_ROLE = 0


class ParticleGrid(QtWidgets.QGraphicsView):
    """Grid view: mx_image_selected(event, [uid]) is emitted on a plain
    click (navigates to that particle, as before). selection_changed([uids])
    is emitted whenever the multi-selection changes (plain click -> single
    item selection; Ctrl+click -> toggle; Shift+click -> range select)."""

    mx_image_selected = QtCore.Signal(object, object)
    selection_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        self._thumb_size = THUMB_SIZE
        self._mouse_mode = "App"
        self._items = {}        # uid -> QGraphicsPixmapItem
        self._images = {}       # uid -> raw numpy image (kept for re-scaling on zoom)
        self._slot_of_uid = {}  # uid -> stable slot index
        self._known_slot = {}   # uid -> last slot it ever occupied (survives removal,
                                 # so a particle brought back by undo returns to the
                                 # same spot instead of being appended at the end)
        self._next_slot = 0
        self._selected = []      # ordered list of selected uids
        self._selection_rects = {}  # uid -> QGraphicsRectItem
        self._last_clicked_uid = None
        self._labels = {}        # uid -> (bg QGraphicsRectItem, text QGraphicsSimpleTextItem)
        self._border_rects = {}  # uid -> QGraphicsRectItem (class-color border, grouped views only)
        self._border_colors = {}  # uid -> QColor
        self._pending_restore = []  # uids re-added this sync that were previously removed
        self._pan_last = None    # last screen pos while right-button panning
        self._levels = None      # (lo, hi) contrast override; None = per-thumbnail autocontrast

        self._update_scene_rect()

    def set_mouse_mode(self, mode):
        self._mouse_mode = mode

    def set_levels(self, lo, hi):
        """Apply an explicit (lo, hi) contrast range to every thumbnail
        here, overriding the default per-thumbnail mean/std autocontrast --
        mirrors ImageView2D.set_manual_levels so the main window's contrast
        slider affects the gallery identically."""
        self._levels = (float(lo), float(hi))
        self._refresh_all_pixmaps()

    def clear_levels(self):
        self._levels = None
        self._refresh_all_pixmaps()

    def _refresh_all_pixmaps(self):
        for uid, img in self._images.items():
            item = self._items.get(uid)
            if item is not None:
                item.setPixmap(self._make_pixmap(img))

    def selected_uids(self):
        return list(self._selected)

    def clear_selection(self, emit=True):
        for uid in list(self._selection_rects.keys()):
            self._hide_selection(uid)
        self._selected = []
        self._last_clicked_uid = None
        if emit:
            self.selection_changed.emit([])


    # -- syncing particles -----------------------------------------------
    def sync_particles(self, uid_img_pairs):
        """Reconcile displayed items with an iterable of (uid, image) pairs.
        New uids are appended to the next free slot (bottom-left first);
        uids no longer present are removed, leaving their slot empty;
        uids already shown just get their thumbnail image refreshed. A uid
        that reappears after having been removed (only possible via undo,
        since uids are never reused) returns to its old slot and gets
        highlighted -- see _add_item."""
        wanted = dict(uid_img_pairs)
        for uid in list(self._items.keys()):
            if uid not in wanted:
                self._remove_item(uid)
        self._pending_restore = []
        for uid, img in uid_img_pairs:
            if uid in self._items:
                self._update_pixmap(uid, img)
            else:
                self._add_item(uid, img)
        self._update_scene_rect()
        self._finish_restore_highlight()

    def sync_particles_grouped(self, entries, class_order):
        """Like sync_particles, but groups items by class: entries is an
        iterable of (uid, image, classid) and class_order lists classids in
        the order their groups should appear (see categories.category_order).
        Unlike the plain per-class grids, slots are fully recomputed on
        every call -- that's unavoidable to keep classes clustered together
        as particles move between them -- so this is only used for the
        "All" tab. Each item also gets a border in its class's color so
        categories stay visually distinct even without column headers."""
        entries = list(entries)
        wanted = {uid for uid, _img, _cid in entries}
        for uid in list(self._items.keys()):
            if uid not in wanted:
                self._remove_item(uid)

        by_class = {}
        for uid, img, cid in entries:
            by_class.setdefault(cid, []).append((uid, img))

        self._pending_restore = []
        self._slot_of_uid = {}
        cols = self._cols()
        slot = 0
        for cid in class_order:
            group = by_class.get(cid)
            if not group:
                continue
            if slot % cols != 0:
                slot += cols - (slot % cols)
            color = categories.class_color(cid)
            for uid, img in group:
                if uid in self._items:
                    self._update_pixmap(uid, img)
                    self._slot_of_uid[uid] = slot
                    self._known_slot[uid] = slot
                else:
                    self._add_item(uid, img, slot=slot)
                self.set_border_color(uid, color)
                slot += 1
        self._next_slot = slot
        self._relayout_all()
        self._update_scene_rect()
        self._finish_restore_highlight()

    def _finish_restore_highlight(self):
        if self._pending_restore:
            self._set_selection(self._pending_restore)
        self._pending_restore = []

    def reorganize(self):
        """Compact the grid: drop gaps left by deleted particles while
        keeping the relative (bottom-left-first) order of what remains."""
        ordered = sorted(self._slot_of_uid.keys(), key=lambda u: self._slot_of_uid[u])
        self._slot_of_uid = {}
        self._next_slot = 0
        for uid in ordered:
            self._slot_of_uid[uid] = self._next_slot
            self._known_slot[uid] = self._next_slot
            self._next_slot += 1
        for uid in ordered:
            self._position_item(uid)
        self._update_scene_rect()

    # -- internal item management -----------------------------------------
    def _add_item(self, uid, img, slot=None):
        self._images[uid] = img
        restored = uid in self._known_slot
        if slot is None:
            if restored and self._known_slot[uid] not in self._slot_of_uid.values():
                slot = self._known_slot[uid]
            else:
                slot = self._next_slot
                self._next_slot += 1
        self._known_slot[uid] = slot
        self._slot_of_uid[uid] = slot
        item = QtWidgets.QGraphicsPixmapItem(self._make_pixmap(img))
        item.setData(UID_ROLE, uid)
        self._scene.addItem(item)
        self._items[uid] = item
        self._add_label(uid)
        self._position_item(uid)
        if restored:
            self._pending_restore.append(uid)

    def _remove_item(self, uid):
        item = self._items.pop(uid, None)
        if item is not None:
            self._scene.removeItem(item)
        self._images.pop(uid, None)
        self._slot_of_uid.pop(uid, None)
        self._hide_selection(uid)
        self._hide_label(uid)
        self._border_colors.pop(uid, None)
        rect = self._border_rects.pop(uid, None)
        if rect is not None:
            self._scene.removeItem(rect)
        if uid in self._selected:
            self._selected.remove(uid)

    def _update_pixmap(self, uid, img):
        self._images[uid] = img
        item = self._items.get(uid)
        if item is not None:
            item.setPixmap(self._make_pixmap(img))

    def _make_pixmap(self, img):
        pix = _thumbnail_pixmap(img, levels=self._levels)
        return pix.scaled(self._thumb_size, self._thumb_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def _cols(self):
        w = max(self.viewport().width(), self._thumb_size + SPACING)
        return max(1, (w - SPACING) // (self._thumb_size + SPACING))

    def _position_item(self, uid):
        item = self._items.get(uid)
        if item is None:
            return
        slot = self._slot_of_uid[uid]
        cols = self._cols()
        row, col = divmod(slot, cols)
        step = self._thumb_size + SPACING
        # Row 0 sits at the bottom (y in [-thumb, 0]); higher rows get more
        # negative y, i.e. they grow upward, so the first particle's slot
        # never moves as later rows are added above it.
        x = SPACING + col * step
        y = -SPACING - row * step - self._thumb_size
        item.setPos(x, y)
        rect = self._selection_rects.get(uid)
        if rect is not None:
            rect.setRect(x - 2, y - 2, self._thumb_size + 4, self._thumb_size + 4)
        self._position_label(uid)
        self._update_border(uid)

    # -- particle number-id label -------------------------------------------
    def _add_label(self, uid):
        bg = self._scene.addRect(0, 0, 0, 0, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor(0, 0, 0, 160)))
        bg.setZValue(6)
        bg.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        text = self._scene.addSimpleText(str(uid))
        text.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        font = text.font()
        font.setPointSize(8)
        text.setFont(font)
        text.setZValue(7)
        text.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self._labels[uid] = (bg, text)
        self._position_label(uid)

    def _position_label(self, uid):
        item = self._items.get(uid)
        lbl = self._labels.get(uid)
        if item is None or lbl is None:
            return
        bg, text = lbl
        pos = item.pos()
        text.setPos(pos.x() + 2, pos.y() + 1)
        br = text.boundingRect()
        bg.setRect(pos.x() + 1, pos.y() + 1, br.width() + 3, br.height())

    def _hide_label(self, uid):
        lbl = self._labels.pop(uid, None)
        if lbl is not None:
            bg, text = lbl
            self._scene.removeItem(bg)
            self._scene.removeItem(text)

    # -- per-item class-color border (used by the grouped "All" view) -------
    def set_border_color(self, uid, color):
        self._border_colors[uid] = color
        self._update_border(uid)

    def _update_border(self, uid):
        color = self._border_colors.get(uid)
        item = self._items.get(uid)
        if item is None or color is None:
            return
        rect = self._border_rects.get(uid)
        if rect is None:
            pen = QtGui.QPen(color)
            pen.setWidth(2)
            rect = self._scene.addRect(0, 0, 0, 0, pen)
            rect.setZValue(1)
            rect.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self._border_rects[uid] = rect
        else:
            pen = rect.pen()
            pen.setColor(color)
            rect.setPen(pen)
        pos = item.pos()
        rect.setRect(pos.x() - 1, pos.y() - 1, self._thumb_size + 2, self._thumb_size + 2)

    def _relayout_all(self):
        for uid in self._items:
            self._position_item(uid)
        self._update_scene_rect()

    def _update_scene_rect(self):
        if self._items:
            rect = self._scene.itemsBoundingRect().adjusted(-SPACING, -SPACING, SPACING, SPACING)
        else:
            rect = QtCore.QRectF(0, -self._thumb_size - 2 * SPACING, self._thumb_size + 2 * SPACING, self._thumb_size + 2 * SPACING)
        self._scene.setSceneRect(rect)

    def _show_selection(self, uid):
        item = self._items.get(uid)
        if item is None:
            return
        rect = self._selection_rects.get(uid)
        if rect is None:
            pen = QtGui.QPen(QtGui.QColor("yellow"))
            pen.setWidth(3)
            rect = self._scene.addRect(0, 0, 0, 0, pen)
            rect.setZValue(10)
            # The highlight sits visually on top of the thumbnail, but must
            # not intercept mouse clicks -- otherwise itemAt() returns this
            # rect (which has no UID_ROLE data) instead of the pixmap item
            # underneath once a particle is selected, silently breaking
            # every click gesture (including Shift-click delete) on already
            # -selected particles.
            rect.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            rect.setAcceptHoverEvents(False)
            self._selection_rects[uid] = rect
        pos = item.pos()
        rect.setRect(pos.x() - 2, pos.y() - 2, self._thumb_size + 4, self._thumb_size + 4)

    def _hide_selection(self, uid):
        rect = self._selection_rects.pop(uid, None)
        if rect is not None:
            self._scene.removeItem(rect)

    def _set_selection(self, uids, emit=True):
        uids = list(dict.fromkeys(uids))  # de-dup, preserve order
        for uid in list(self._selection_rects.keys()):
            if uid not in uids:
                self._hide_selection(uid)
        for uid in uids:
            self._show_selection(uid)
        self._selected = uids
        if emit:
            self.selection_changed.emit(list(uids))

    # -- zoom (mouse wheel) -----------------------------------------------
    def zoom_in(self):
        self._thumb_size = min(MAX_THUMB, int(round(self._thumb_size * 1.15)))
        self._rescale_all()

    def zoom_out(self):
        self._thumb_size = max(MIN_THUMB, int(round(self._thumb_size / 1.15)))
        self._rescale_all()

    def _rescale_all(self):
        for uid, img in self._images.items():
            self._items[uid].setPixmap(self._make_pixmap(img))
        self._relayout_all()

    # -- events -------------------------------------------------------------
    def wheelEvent(self, event):
        # Plain wheel zooms the thumbnails (matches the tomogram views);
        # Shift+wheel falls back to normal scrolling.
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_all()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            # Grab-and-pan: right-click and drag to scroll the grid, same
            # gesture as the tomogram views (ImageView2D). A plain
            # right-click with no drag does nothing else -- it does NOT
            # select, since that made right-click redundant with (and
            # easily confused for) a left-click.
            self._pan_last = (event.position().x(), event.position().y())
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            # itemAt() returns the strict topmost item regardless of
            # acceptedMouseButtons/acceptHoverEvents, so a selection
            # highlight rect (also drawn at this position, on top so it's
            # visible) would shadow the pixmap item underneath once a
            # particle is selected. Walk all items at this position instead
            # and use the first one that actually carries a UID.
            uid = None
            for candidate in self.items(event.pos()):
                cand_uid = candidate.data(UID_ROLE)
                if cand_uid is not None:
                    uid = cand_uid
                    break
            if uid is not None:
                mods = event.modifiers()
                # Shift(+Ctrl)+click sends this particle (or it and
                # everything after it in the list) to the Recycle Bin;
                # Space+click sends it to Confirmed -- both handled by
                # TomoBoxer.img_selected. Bypass multi-select entirely and
                # forward the click as-is.
                if mods & QtCore.Qt.ShiftModifier or space_key_state().held:
                    self.mx_image_selected.emit(event, [uid])
                    event.accept()
                    return
                # Multi-select uses Ctrl (toggle one item) and Alt
                # (range-select from the last clicked item), since Shift
                # and Space are reserved for the gestures above.
                if mods & QtCore.Qt.AltModifier and self._last_clicked_uid is not None:
                    self._range_select(self._last_clicked_uid, uid)
                elif mods & QtCore.Qt.ControlModifier:
                    sel = list(self._selected)
                    if uid in sel:
                        sel.remove(uid)
                    else:
                        sel.append(uid)
                    self._last_clicked_uid = uid
                    self._set_selection(sel)
                else:
                    self._last_clicked_uid = uid
                    self._set_selection([uid])
                    self.mx_image_selected.emit(event, [uid])
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None and (event.buttons() & QtCore.Qt.RightButton):
            x, y = event.position().x(), event.position().y()
            dx, dy = x - self._pan_last[0], y - self._pan_last[1]
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(dx))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(dy))
            self._pan_last = (x, y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self._pan_last = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _range_select(self, from_uid, to_uid):
        if from_uid not in self._slot_of_uid or to_uid not in self._slot_of_uid:
            self._set_selection([to_uid])
            return
        lo, hi = sorted((self._slot_of_uid[from_uid], self._slot_of_uid[to_uid]))
        uids = [u for u, s in self._slot_of_uid.items() if lo <= s <= hi]
        uids.sort(key=lambda u: self._slot_of_uid[u])
        self._last_clicked_uid = to_uid
        self._set_selection(uids)


def _thumbnail_pixmap(img: np.ndarray, levels=None) -> QtGui.QPixmap:
    """levels, when given, is an explicit (lo, hi) display range (the
    contrast slider) used instead of this thumbnail's own mean/std -- so
    manual contrast applies identically here and in the tomogram views."""
    img = np.asarray(img, dtype=np.float32)
    if levels is not None:
        lo, hi = levels
    else:
        mean, std = float(img.mean()), float(img.std())
        lo, hi = (mean - 3 * std, mean + 3 * std) if std else (mean - 1, mean + 1)
    if hi <= lo:
        hi = lo + 1
    buf = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    buf = np.ascontiguousarray(buf)
    h, w = buf.shape
    qimg = QtGui.QImage(buf.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8).copy()
    return QtGui.QPixmap.fromImage(qimg)


class DetachedGridWindow(QtWidgets.QWidget):
    """A floating top-level window holding a single class's ParticleGrid
    that has been "detached" out of the tabbed panel. Closing it (via the
    window's [x] or the explicit Reattach button) sends the grid back to
    its tab in the parent ParticleListPanel."""

    closed = QtCore.Signal(int)  # classid

    def __init__(self, classid, grid, title):
        super().__init__()
        self.classid = classid
        self.grid = grid
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(grid)
        # QTabWidget/QStackedWidget explicitly hide()s a page when it is
        # removed (removeTab), and that explicit-hidden flag survives being
        # reparented into this window's layout -- so force it visible again.
        grid.setVisible(True)
        self.reattach_button = QtWidgets.QPushButton("Reattach")
        layout.addWidget(self.reattach_button)
        self.reattach_button.clicked.connect(self.close)
        self.resize(420, 480)

    def closeEvent(self, event):
        self.closed.emit(self.classid)
        super().closeEvent(event)


class ParticleListPanel(QtWidgets.QWidget):
    """Wraps a tabbed set of ParticleGrids: an "All" tab showing every
    visible particle, plus one tab per class (rebuilt via rebuild_tabs()).
    Any class tab can be "detached" into its own floating window with the
    Detach button, and reattached by closing that window."""

    mx_image_selected = QtCore.Signal(object, object)
    selection_changed = QtCore.Signal(object)  # list of uids currently selected

    def __init__(self, target=None, parent=None):
        super().__init__(parent)
        self._target = weakref.ref(target) if target is not None else None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.grid = ParticleGrid()
        self.tabs.addTab(self.grid, "All")
        self._connect_grid(self.grid)

        self._class_grids = {}     # classid -> ParticleGrid (docked, in a tab)
        self._class_names = {}     # classid -> name currently shown
        self._detached = {}        # classid -> DetachedGridWindow
        self._all_class_order = []  # classids, fixed categories first then alphabetical
        self._levels_state = None  # (lo, hi) currently applied, so a newly created grid gets it too

        btnrow = QtWidgets.QHBoxLayout()
        self.organize_button = QtWidgets.QPushButton("Organize Grid")
        btnrow.addWidget(self.organize_button)
        self.detach_button = QtWidgets.QPushButton("Detach Tab")
        btnrow.addWidget(self.detach_button)
        layout.addLayout(btnrow)

        self.organize_button.clicked.connect(lambda: self._active_grid().reorganize())
        self.detach_button.clicked.connect(self._detach_current_tab)

    def _connect_grid(self, grid):
        grid.mx_image_selected.connect(self.mx_image_selected)
        grid.selection_changed.connect(self.selection_changed)

    def _active_grid(self):
        w = self.tabs.currentWidget()
        return w if isinstance(w, ParticleGrid) else self.grid

    def set_mouse_mode(self, mode):
        self.grid.set_mouse_mode(mode)
        for g in self._class_grids.values():
            g.set_mouse_mode(mode)

    def _all_grids(self):
        yield self.grid
        yield from self._class_grids.values()
        for win in self._detached.values():
            yield win.grid

    def set_levels(self, lo, hi):
        self._levels_state = (lo, hi)
        for g in self._all_grids():
            g.set_levels(lo, hi)

    def clear_levels(self):
        self._levels_state = None
        for g in self._all_grids():
            g.clear_levels()

    # -- tabs / classes -----------------------------------------------------
    def rebuild_tabs(self, sets: dict):
        """sets: {classid: name}. Adds/removes/renames class tabs to match,
        ordered with the fixed Recycle Bin / Confirmed categories first and
        everything else alphabetically after (see categories.category_order).
        Detached windows are left alone (they aren't tabs), just their title
        is refreshed on rename."""
        wanted = dict(sets)
        self._all_class_order = categories.category_order(wanted)

        for classid in list(self._class_grids.keys()):
            if classid not in wanted:
                grid = self._class_grids.pop(classid)
                self._class_names.pop(classid, None)
                idx = self.tabs.indexOf(grid)
                if idx >= 0:
                    self.tabs.removeTab(idx)

        for classid in list(self._detached.keys()):
            if classid not in wanted:
                win = self._detached.pop(classid)
                win.close()

        for classid in self._all_class_order:
            name = wanted[classid]
            if classid in self._detached:
                win = self._detached[classid]
                win.setWindowTitle(name)
                self._class_names[classid] = name
                continue
            if classid in self._class_grids:
                grid = self._class_grids[classid]
                if self._class_names.get(classid) != name:
                    idx = self.tabs.indexOf(grid)
                    if idx >= 0:
                        self.tabs.setTabText(idx, name)
                    self._class_names[classid] = name
                continue
            grid = ParticleGrid()
            self._connect_grid(grid)
            if self._levels_state is not None:
                grid.set_levels(*self._levels_state)
            self._class_grids[classid] = grid
            self._class_names[classid] = name
            self.tabs.addTab(grid, name)

        self._reorder_tabs()

    def _reorder_tabs(self):
        """Move docked class tabs into _all_class_order (tab 0 is always
        "All"). Needed both after rebuild_tabs adds/removes tabs and after
        reattaching a detached tab, since insertTab there just appends."""
        bar = self.tabs.tabBar()
        for pos, classid in enumerate(self._all_class_order, start=1):
            grid = self._class_grids.get(classid)
            if grid is None:
                continue
            cur = self.tabs.indexOf(grid)
            if cur >= 0 and cur != pos:
                bar.moveTab(cur, pos)

    def _detach_current_tab(self):
        w = self.tabs.currentWidget()
        classid = None
        for cid, grid in self._class_grids.items():
            if grid is w:
                classid = cid
                break
        if classid is None:
            return  # "All" tab can't be detached
        grid = self._class_grids.pop(classid)
        name = self._class_names.get(classid, str(classid))
        idx = self.tabs.indexOf(grid)
        if idx >= 0:
            self.tabs.removeTab(idx)
        win = DetachedGridWindow(classid, grid, name)
        win.closed.connect(self._reattach)
        self._detached[classid] = win
        win.show()

    def _reattach(self, classid):
        win = self._detached.pop(classid, None)
        if win is None:
            return
        grid = win.grid
        name = self._class_names.get(classid, str(classid))
        self._class_grids[classid] = grid
        grid.setVisible(True)
        self.tabs.addTab(grid, name)
        self._reorder_tabs()
        self.tabs.setCurrentWidget(grid)

    # -- particle sync --------------------------------------------------------
    def sync_particles(self, entries):
        """entries: iterable of (uid, img, classid)."""
        entries = list(entries)
        self.grid.sync_particles_grouped(entries, self._all_class_order)
        by_class = {}
        for uid, img, cid in entries:
            by_class.setdefault(cid, []).append((uid, img))
        for classid, grid in self._class_grids.items():
            grid.sync_particles(by_class.get(classid, []))
        for classid, win in self._detached.items():
            win.grid.sync_particles(by_class.get(classid, []))

    def closeEvent(self, event):
        target = self._target() if self._target is not None else None
        if target is not None and not getattr(target, "_closing", False):
            event.ignore()
            target.request_close_all()
            return
        for win in list(self._detached.values()):
            win.close()
        event.accept()

