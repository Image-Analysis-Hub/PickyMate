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
the particle list. A uid that reappears after being removed -- via undo,
or because its set's visibility checkbox was toggled off and back on, or
detach/reattach -- returns to its old slot rather than being appended at
the end. It is deliberately *not* auto-selected just for reappearing here
(a uid reappearing isn't always an undo -- see the point above); undoing a
delete instead has TomoBoxer explicitly select the particles it actually
restored, in _reinsert_boxes.

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

_label_style = None  # lazily-built (bg_brush, no_pen, text_brush, font) shared by every
                      # particle's number-id label -- see _get_label_style()


def _get_label_style():
    """The pen/brushes/font used for every particle's number-id label are
    identical every time -- build them once and share the same instances
    (safe: QPen/QBrush/QFont are implicitly-shared value types that every
    setter below copies from) instead of constructing fresh ones per
    particle, which added up when populating a grid with many thousands of
    thumbnails at once. Built lazily rather than at import time since these
    need at least a QGuiApplication to exist."""
    global _label_style
    if _label_style is None:
        bg_brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 160))
        no_pen = QtGui.QPen(QtCore.Qt.NoPen)
        text_brush = QtGui.QBrush(QtGui.QColor("white"))
        font = QtGui.QFont()
        font.setPointSize(8)
        _label_style = (bg_brush, no_pen, text_brush, font)
    return _label_style


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
        self.setFocusPolicy(QtCore.Qt.StrongFocus)  # so Escape (clear selection) reaches keyPressEvent below

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
                item.setPixmap(self._render_pixmap(img))
                self._apply_item_scale(uid)

    def selected_uids(self):
        return list(self._selected)

    def clear_selection(self, emit=True):
        for uid in list(self._selection_rects.keys()):
            self._hide_selection(uid)
        self._selected = []
        self._last_clicked_uid = None
        if emit:
            self.selection_changed.emit([])

    def toggle_uid_selection(self, uid):
        """Toggle one uid in/out of the selection, as if Ctrl+clicked here --
        used by TomoBoxer for Ctrl+click in the main tomogram views, so
        both windows always share the exact same selection list."""
        sel = list(self._selected)
        if uid in sel:
            sel.remove(uid)
        else:
            sel.append(uid)
        self._last_clicked_uid = uid
        self._set_selection(sel)

    def replace_selection(self, uids):
        """Replace the selection outright -- used by TomoBoxer's shape-based
        select tool (rectangle/circle/freeform) in the main tomogram views."""
        uids = list(uids)
        self._last_clicked_uid = uids[-1] if uids else None
        self._set_selection(uids)


    # -- syncing particles -----------------------------------------------
    def sync_particles(self, uid_img_pairs):
        """Reconcile displayed items with an iterable of (uid, image) pairs.
        New uids are appended to the next free slot (bottom-left first);
        uids no longer present are removed, leaving their slot empty;
        uids already shown just get their thumbnail image refreshed. A uid
        that reappears after having been removed returns to its old slot --
        see _add_item."""
        wanted = dict(uid_img_pairs)
        for uid in list(self._items.keys()):
            if uid not in wanted:
                self._remove_item(uid)
        for uid, img in uid_img_pairs:
            if uid in self._items:
                self._update_pixmap(uid, img)
            else:
                self._add_item(uid, img)
        self._update_scene_rect()

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
        if slot is None:
            known = self._known_slot.get(uid)
            if known is not None and known not in self._slot_of_uid.values():
                slot = known
            else:
                slot = self._next_slot
                self._next_slot += 1
        self._known_slot[uid] = slot
        self._slot_of_uid[uid] = slot
        item = QtWidgets.QGraphicsPixmapItem(self._render_pixmap(img))
        item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        item.setData(UID_ROLE, uid)
        self._scene.addItem(item)
        self._items[uid] = item
        self._add_label(uid)
        self._position_item(uid)

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
            item.setPixmap(self._render_pixmap(img))
            self._apply_item_scale(uid)

    def _render_pixmap(self, img):
        """The one place that actually touches pixel data: the contrast
        stretch from the raw particle crop, at its native (box-size)
        resolution -- no resize here. Zoom is a display-only concern (see
        _apply_item_scale) so it never needs to re-run this."""
        return _thumbnail_pixmap(img, levels=self._levels)

    def _apply_item_scale(self, uid):
        """Scale the item to the current thumbnail size via a transform
        (with SmoothTransformation set on it in _add_item, the same
        bilinear-quality resampling a manual .scaled() call would do) rather
        than regenerating its pixmap -- this is what makes zoom cheap: it
        no longer touches pixel data at all, just this one number."""
        item = self._items.get(uid)
        if item is None:
            return
        pix = item.pixmap()
        native = max(pix.width(), pix.height())
        if native > 0:
            item.setScale(self._thumb_size / native)

    def _cols(self):
        w = max(self.viewport().width(), self._thumb_size + SPACING)
        return max(1, (w - SPACING) // (self._thumb_size + SPACING))

    def _position_item(self, uid):
        item = self._items.get(uid)
        if item is None:
            return
        self._apply_item_scale(uid)
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
        bg_brush, no_pen, text_brush, font = _get_label_style()
        bg = self._scene.addRect(0, 0, 0, 0, no_pen, bg_brush)
        bg.setZValue(6)
        bg.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        text = self._scene.addSimpleText(str(uid))
        text.setBrush(text_brush)
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
        self._relayout_all()

    def zoom_out(self):
        self._thumb_size = max(MIN_THUMB, int(round(self._thumb_size / 1.15)))
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
        self.setFocus(QtCore.Qt.MouseFocusReason)  # so Escape works right after clicking here
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

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.clear_selection()
            event.accept()
            return
        super().keyPressEvent(event)

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

    def __init__(self, classid, grid, title, visibility_checkbox):
        super().__init__()
        self.classid = classid
        self.grid = grid
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # The docked tab's visibility checkbox goes away with the tab when
        # detaching (it's embedded in the tab bar) -- without a replacement
        # here, there'd be no way to show/hide this set's particles in the
        # tomogram views while its window is floating.
        self.visibility_checkbox = visibility_checkbox
        self.visibility_checkbox.setText("Visible in tomogram view")
        layout.addWidget(self.visibility_checkbox)

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
    visible particle, one tab per class (rebuilt via rebuild_tabs()), and a
    permanent "+" tab at the end that prompts for a name and creates a new
    set instead of ever actually being shown. Any class tab can be
    "detached" into its own floating window with the Detach button, and
    reattached by closing that window.

    This is also where Sets management now lives, replacing the standalone
    Options window's Sets panel: New/Rename/Delete/visibility all act on
    tabs directly (a new tab, double-click to rename, a Delete Set button,
    a per-tab visibility checkbox), rather than duplicating that as a
    separate list somewhere else."""

    mx_image_selected = QtCore.Signal(object, object)
    selection_changed = QtCore.Signal(object)  # list of uids currently selected
    class_activated = QtCore.Signal(int)  # classid, when a normal class's tab becomes current

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
        self._pending_entries = {}  # classid -> [(uid, img), ...] not yet applied to a
                                     # not-currently-shown docked tab's grid (see sync_particles)
        self._visibility_checkboxes = {}  # classid -> QCheckBox, embedded in that tab

        self._plus_placeholder = QtWidgets.QWidget()
        self.tabs.addTab(self._plus_placeholder, "+")
        self.tabs.setTabToolTip(self.tabs.indexOf(self._plus_placeholder), "Create a new set")
        self._last_tab_index = 0

        btnrow = QtWidgets.QHBoxLayout()
        self.organize_button = QtWidgets.QPushButton("Organize Grid")
        self.organize_button.setToolTip("Compact the grid, removing gaps left by deleted particles.")
        btnrow.addWidget(self.organize_button)
        self.detach_button = QtWidgets.QPushButton("Detach Tab")
        self.detach_button.setToolTip("Pop the current tab's grid into its own floating window.")
        self.detach_button.setEnabled(False)  # starts on "All", which can't be detached
        btnrow.addWidget(self.detach_button)
        self.delete_set_button = QtWidgets.QPushButton("Delete Set")
        self.delete_set_button.setToolTip("Delete this set, sending every particle in it to the Recycle Bin.")
        self.delete_set_button.setEnabled(False)
        btnrow.addWidget(self.delete_set_button)
        self.empty_recycle_button = QtWidgets.QPushButton("Empty Recycle Bin")
        self.empty_recycle_button.setToolTip(
            "Permanently delete everything currently in the Recycle Bin "
            "(undoable with Ctrl+Z right after).")
        self.empty_recycle_button.setVisible(False)  # only shown while viewing the Recycle Bin tab
        btnrow.addWidget(self.empty_recycle_button)
        layout.addLayout(btnrow)

        self.organize_button.clicked.connect(lambda: self._active_grid().reorganize())
        self.detach_button.clicked.connect(self._detach_current_tab)
        self.delete_set_button.clicked.connect(self._delete_current_set)
        self.empty_recycle_button.clicked.connect(self._empty_recycle_bin)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tabBar().tabBarDoubleClicked.connect(self._on_tab_double_clicked)

    def _target_or_none(self):
        return self._target() if self._target is not None else None

    def _active_classid(self):
        w = self.tabs.currentWidget()
        for cid, grid in self._class_grids.items():
            if grid is w:
                return cid
        return None

    def _on_tab_changed(self, index):
        widget = self.tabs.widget(index)
        if widget is self._plus_placeholder:
            self._prompt_new_set()
            return
        self._last_tab_index = index

        cid = self._active_classid()
        is_fixed = cid in categories.FIXED_CLASS_IDS if cid is not None else False
        self.delete_set_button.setEnabled(cid is not None and not is_fixed)
        self.detach_button.setEnabled(cid is not None)  # "All" can't be detached
        self.empty_recycle_button.setVisible(cid == categories.RECYCLE_BIN_ID)
        if cid is not None and not is_fixed:
            self.class_activated.emit(cid)

        for classid, grid in self._class_grids.items():
            if grid is widget:
                pending = self._pending_entries.pop(classid, None)
                if pending is not None:
                    grid.sync_particles(pending)
                break

    # -- sets management (New/Rename/Delete/visibility/Save/Empty Bin) ------
    def _prompt_new_set(self):
        target = self._target_or_none()
        self.tabs.setCurrentIndex(self._last_tab_index)  # never linger on "+"
        if target is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New Set", "Enter a name for the new set:")
        if not ok:
            return
        name = str(name).strip()
        if not name:
            return
        if name in target.sets.values():
            QtWidgets.QMessageBox.information(self, "New Set", "A set with that name already exists.")
            return
        cid = target.new_set(name)
        self.rebuild_tabs(target.sets)
        grid = self._class_grids.get(cid)
        if grid is not None:
            self.tabs.setCurrentWidget(grid)

    def _on_tab_double_clicked(self, index):
        widget = self.tabs.widget(index)
        if widget is self.grid or widget is self._plus_placeholder:
            return  # "All" and "+" aren't renamable
        target = self._target_or_none()
        classid = None
        for cid, grid in self._class_grids.items():
            if grid is widget:
                classid = cid
                break
        if target is None or classid is None:
            return
        if classid in categories.FIXED_CLASS_IDS:
            QtWidgets.QMessageBox.information(
                self, "Rename Set", "Recycle Bin and Confirmed are fixed categories and can't be renamed.")
            return
        oldname = target.sets[classid]
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename Set", "Enter a new name:", text=oldname)
        if not ok:
            return
        name = str(name).strip()
        if not name or name == oldname:
            return
        if name in target.sets.values():
            QtWidgets.QMessageBox.information(self, "Rename Set", "A set with that name already exists.")
            return
        target.rename_set(f"{classid}::{oldname}", name)
        self.rebuild_tabs(target.sets)

    def _delete_current_set(self):
        target = self._target_or_none()
        cid = self._active_classid()
        if target is None or cid is None or cid in categories.FIXED_CLASS_IDS:
            return
        name = target.sets.get(cid)
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Set",
            f'Are you sure you want to delete "{name}"? Every particle in it will be sent to the Recycle Bin.',
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            # Switch away from the tab being removed *before* removing it --
            # otherwise Qt shifts the current tab onto whatever ends up in
            # its place, which (when deleting the last real class) is the
            # "+" placeholder, immediately firing an unwanted "create a new
            # set" prompt as a side effect of deleting one.
            self.tabs.setCurrentWidget(self.grid)
            target.delete_set(f"{cid}::{name}")
            self.rebuild_tabs(target.sets)

    def _empty_recycle_bin(self):
        target = self._target_or_none()
        if target is None:
            return
        count = sum(1 for b in target.boxes if b[5] == categories.RECYCLE_BIN_ID)
        if count == 0:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Empty Recycle Bin",
            f"Permanently delete {count} particle(s) in the Recycle Bin? (Undo-able with Ctrl+Z right after.)",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            target.empty_recycle_bin()

    def _make_visibility_checkbox(self, classid):
        cb = QtWidgets.QCheckBox()
        cb.setToolTip("Show/hide this set's particles in the tomogram views.")
        target = self._target_or_none()
        cb.setChecked(target is not None and classid in target.sets_visible)
        cb.toggled.connect(lambda checked, cid=classid: self._set_visibility(cid, checked))
        return cb

    def _set_visibility(self, classid, checked):
        target = self._target_or_none()
        if target is None:
            return
        if checked:
            target.show_set(f"{classid}::")
        else:
            target.hide_set(f"{classid}::")

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
                self._pending_entries.pop(classid, None)
                self._visibility_checkboxes.pop(classid, None)
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
            # Insert before "+" (which addTab would otherwise land after,
            # since it's always the last tab) -- _reorder_tabs fixes up the
            # rest of the ordering below regardless.
            self.tabs.insertTab(self.tabs.indexOf(self._plus_placeholder), grid, name)
            cb = self._make_visibility_checkbox(classid)
            self._visibility_checkboxes[classid] = cb
            self.tabs.tabBar().setTabButton(self.tabs.indexOf(grid), QtWidgets.QTabBar.LeftSide, cb)

        self._reorder_tabs()

    def _reorder_tabs(self):
        """Move docked class tabs into _all_class_order (tab 0 is always
        "All"), and pin "+" to the very last position. Needed both after
        rebuild_tabs adds/removes tabs and after reattaching a detached
        tab, since insertTab there just appends."""
        bar = self.tabs.tabBar()
        for pos, classid in enumerate(self._all_class_order, start=1):
            grid = self._class_grids.get(classid)
            if grid is None:
                continue
            cur = self.tabs.indexOf(grid)
            if cur >= 0 and cur != pos:
                bar.moveTab(cur, pos)
        plus_idx = self.tabs.indexOf(self._plus_placeholder)
        last = self.tabs.count() - 1
        if plus_idx >= 0 and plus_idx != last:
            bar.moveTab(plus_idx, last)

    def _detach_current_tab(self):
        classid = self._active_classid()
        if classid is None:
            return  # "All"/"+" can't be detached
        grid = self._class_grids.pop(classid)
        name = self._class_names.get(classid, str(classid))
        # Popped (not reused) here: removeTab below deletes a tab's corner
        # widget as a side effect of removing the tab, so this Python
        # reference could already be pointing at a destroyed C++ object by
        # the time DetachedGridWindow might otherwise try to reuse it. A
        # fresh one is made just below instead (same as _reattach does).
        self._visibility_checkboxes.pop(classid, None)
        # Switch away from the tab being removed *first* -- otherwise, when
        # it's the last real class tab, Qt shifts the current tab onto
        # whatever ends up in its place, which is the "+" placeholder,
        # immediately (and confusingly) firing the "create a new set"
        # prompt as a side effect of detaching one (same issue as deleting
        # the last real set -- see _delete_current_set).
        self.tabs.setCurrentWidget(self.grid)
        idx = self.tabs.indexOf(grid)
        if idx >= 0:
            self.tabs.removeTab(idx)
        win = DetachedGridWindow(classid, grid, name, self._make_visibility_checkbox(classid))
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
        cb = self._make_visibility_checkbox(classid)
        self._visibility_checkboxes[classid] = cb
        self.tabs.tabBar().setTabButton(self.tabs.indexOf(grid), QtWidgets.QTabBar.LeftSide, cb)
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
        current_widget = self.tabs.currentWidget()
        for classid, grid in self._class_grids.items():
            class_entries = by_class.get(classid, [])
            if grid is current_widget:
                grid.sync_particles(class_entries)
                self._pending_entries.pop(classid, None)
            else:
                # Not the tab on screen right now -- its thumbnails aren't
                # visible either way, so defer the (potentially expensive,
                # many-thumbnail) sync until it's actually switched to (see
                # _on_tab_changed) instead of eagerly maintaining every
                # class's full grid on every single particle change,
                # including ones nobody is currently looking at. This is
                # what made opening a tomogram with many pre-existing
                # particles across several classes slow: it was building
                # every tab's entire thumbnail grid up front, most of which
                # wouldn't be looked at before the next edit rebuilt them
                # all again anyway.
                self._pending_entries[classid] = class_entries
        for classid, win in self._detached.items():
            # Detached windows are independent, possibly-visible-right-now
            # top-level windows, not tab pages Qt can tell us are hidden --
            # always keep them eagerly in sync.
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

