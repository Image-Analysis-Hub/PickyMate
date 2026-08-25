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
the particle list.
"""
from __future__ import annotations

import weakref

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

THUMB_SIZE = 96
MIN_THUMB = 24
MAX_THUMB = 256
SPACING = 4
UID_ROLE = 0


class ParticleGrid(QtWidgets.QGraphicsView):
    """Grid view: mx_image_selected(event, [uid]) is emitted on click."""

    mx_image_selected = QtCore.Signal(object, object)

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
        self._next_slot = 0
        self._selection = None  # QGraphicsRectItem highlighting the current pick

        self._update_scene_rect()

    def set_mouse_mode(self, mode):
        self._mouse_mode = mode

    # -- syncing particles -----------------------------------------------
    def sync_particles(self, uid_img_pairs):
        """Reconcile displayed items with an iterable of (uid, image) pairs.
        New uids are appended to the next free slot (bottom-left first);
        uids no longer present are removed, leaving their slot empty;
        uids already shown just get their thumbnail image refreshed."""
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

    def reorganize(self):
        """Compact the grid: drop gaps left by deleted particles while
        keeping the relative (bottom-left-first) order of what remains."""
        ordered = sorted(self._slot_of_uid.keys(), key=lambda u: self._slot_of_uid[u])
        self._slot_of_uid = {}
        self._next_slot = 0
        for uid in ordered:
            self._slot_of_uid[uid] = self._next_slot
            self._next_slot += 1
        for uid in ordered:
            self._position_item(uid)
        self._update_scene_rect()

    # -- internal item management -----------------------------------------
    def _add_item(self, uid, img):
        self._images[uid] = img
        slot = self._next_slot
        self._next_slot += 1
        self._slot_of_uid[uid] = slot
        item = QtWidgets.QGraphicsPixmapItem(self._make_pixmap(img))
        item.setData(UID_ROLE, uid)
        self._scene.addItem(item)
        self._items[uid] = item
        self._position_item(uid)

    def _remove_item(self, uid):
        item = self._items.pop(uid, None)
        if item is not None:
            self._scene.removeItem(item)
        self._images.pop(uid, None)
        self._slot_of_uid.pop(uid, None)
        if self._selection is not None and self._selection.data(UID_ROLE) == uid:
            self._scene.removeItem(self._selection)
            self._selection = None

    def _update_pixmap(self, uid, img):
        self._images[uid] = img
        item = self._items.get(uid)
        if item is not None:
            item.setPixmap(self._make_pixmap(img))

    def _make_pixmap(self, img):
        pix = _thumbnail_pixmap(img)
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
        if self._selection is not None and self._selection.data(UID_ROLE) == uid:
            self._selection.setRect(x - 2, y - 2, self._thumb_size + 4, self._thumb_size + 4)

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
        if self._selection is None:
            pen = QtGui.QPen(QtGui.QColor("yellow"))
            pen.setWidth(3)
            self._selection = self._scene.addRect(0, 0, 0, 0, pen)
        self._selection.setData(UID_ROLE, uid)
        pos = item.pos()
        self._selection.setRect(pos.x() - 2, pos.y() - 2, self._thumb_size + 4, self._thumb_size + 4)
        self._selection.setZValue(10)

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
        if event.button() == QtCore.Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item is not None:
                uid = item.data(UID_ROLE)
                if uid is not None:
                    self._show_selection(uid)
                    self.mx_image_selected.emit(event, [uid])
                    event.accept()
                    return
        super().mousePressEvent(event)


def _thumbnail_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    img = np.asarray(img, dtype=np.float32)
    mean, std = float(img.mean()), float(img.std())
    lo, hi = (mean - 3 * std, mean + 3 * std) if std else (mean - 1, mean + 1)
    if hi <= lo:
        hi = lo + 1
    buf = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    buf = np.ascontiguousarray(buf)
    h, w = buf.shape
    qimg = QtGui.QImage(buf.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8).copy()
    return QtGui.QPixmap.fromImage(qimg)


class ParticleListPanel(QtWidgets.QWidget):
    """Wraps the ParticleGrid with an "Organize Grid" button beneath it."""

    mx_image_selected = QtCore.Signal(object, object)

    def __init__(self, target=None, parent=None):
        super().__init__(parent)
        self._target = weakref.ref(target) if target is not None else None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.grid = ParticleGrid()
        layout.addWidget(self.grid)
        self.organize_button = QtWidgets.QPushButton("Organize Grid")
        layout.addWidget(self.organize_button)
        self.organize_button.clicked.connect(self.grid.reorganize)
        self.grid.mx_image_selected.connect(self.mx_image_selected)

    def set_mouse_mode(self, mode):
        self.grid.set_mouse_mode(mode)

    def sync_particles(self, uid_img_pairs):
        self.grid.sync_particles(uid_img_pairs)

    def closeEvent(self, event):
        target = self._target() if self._target is not None else None
        if target is not None and not getattr(target, "_closing", False):
            event.ignore()
            target.request_close_all()
            return
        event.accept()
