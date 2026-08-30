"""A QGraphicsView-based 2D image view: pan/zoom, auto-contrast,
shape-overlay annotation, and mouse/keyboard signal plumbing.

This replaces the subset of EMAN2's OpenGL-based EMImage2DWidget that
e2spt_boxer.py actually relies on (set_data/add_shape/get_shapes/del_shape,
scr_to_img, scroll_to, get_origin/set_origin, set_scale, and the
mousedown/mousedrag/mouseup/mousewheel/mousemove/keypress signals).

Particle shapes are persistent QGraphicsItems (one per key) rather than
being redrawn from scratch on every repaint: Qt's own scene graph tracks
which items actually changed and only repaints those regions, instead of
every visible shape being walked and redrawn on every single change to any
one particle. This mirrors the approach thumb_grid.ParticleGrid already
uses for the particle gallery. The pan/zoom coordinate system is kept as
plain (_origin, _scale) bookkeeping identical to before -- rather than
Qt's native scrollbar-based view scrolling -- specifically so scr_to_img/
img_to_scr keep their exact original formulas and precision (including
sub-pixel origins), applied here as the view's transform instead of a
per-paint QPainter transform.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .shapes import Shape


class _ShapeItem(QtWidgets.QGraphicsItem):
    """Renders one Shape (particle outline, crosshair line, eraser preview)
    exactly as the old ImageView2D._draw_shape did. shape_ref is read live
    in boundingRect()/paint(), so code that mutates a Shape's fields
    directly (TomoBoxer.update_sliceview does, for linewidth/geom/kind) is
    reflected correctly the next time refresh() is called -- refresh() is
    already required after such a mutation today, via sync_shape()."""

    def __init__(self, shape: Shape, get_scale):
        super().__init__()
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self._shape = shape
        self._last_kind = shape.kind
        self._get_scale = get_scale
        self._pen_key = None  # (color, width) the cached pen below was built for
        self._pen = None
        self.setVisible(shape.kind != "hidden")

    def set_shape(self, shape: Shape):
        self._shape = shape
        self._last_kind = None  # force refresh() below regardless of the new shape's kind
        self.refresh()

    def refresh(self):
        self.prepareGeometryChange()
        self.setVisible(self._shape.kind != "hidden")
        self.update()

    def _pen_width(self):
        scale = self._get_scale()
        lw = max(self._shape.linewidth, 1)
        return lw / scale if scale else float(lw)

    def boundingRect(self):
        shape = self._shape
        g = shape.geom
        pad = self._pen_width() / 2 + 1
        if shape.kind == "circle" and len(g) >= 3:
            x, y, r = g[0], g[1], g[2]
            return QtCore.QRectF(x - r - pad, y - r - pad, 2 * (r + pad), 2 * (r + pad))
        if shape.kind in ("rect", "line") and len(g) >= 4:
            xlo, xhi = min(g[0], g[2]), max(g[0], g[2])
            ylo, yhi = min(g[1], g[3]), max(g[1], g[3])
            return QtCore.QRectF(xlo - pad, ylo - pad, xhi - xlo + 2 * pad, yhi - ylo + 2 * pad)
        if shape.kind == "polygon" and len(g) >= 4:
            xs, ys = g[0::2], g[1::2]
            xlo, xhi = min(xs), max(xs)
            ylo, yhi = min(ys), max(ys)
            return QtCore.QRectF(xlo - pad, ylo - pad, xhi - xlo + 2 * pad, yhi - ylo + 2 * pad)
        return QtCore.QRectF()

    def paint(self, painter, option, widget=None):
        shape = self._shape
        if shape.kind == "hidden":
            return
        width = self._pen_width()
        key = (shape.color, width)
        if key != self._pen_key:
            r, g, b = shape.color
            color = QtGui.QColor.fromRgbF(max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
            pen = QtGui.QPen(color)
            pen.setWidthF(width)
            self._pen = pen
            self._pen_key = key
        painter.setPen(self._pen)
        painter.setBrush(QtCore.Qt.NoBrush)

        geo = shape.geom
        if shape.kind == "circle" and len(geo) >= 3:
            painter.drawEllipse(QtCore.QPointF(geo[0], geo[1]), geo[2], geo[2])
        elif shape.kind == "rect" and len(geo) >= 4:
            painter.drawRect(QtCore.QRectF(QtCore.QPointF(geo[0], geo[1]), QtCore.QPointF(geo[2], geo[3])))
        elif shape.kind == "line" and len(geo) >= 4:
            painter.drawLine(QtCore.QPointF(geo[0], geo[1]), QtCore.QPointF(geo[2], geo[3]))
        elif shape.kind == "polygon" and len(geo) >= 4:
            pts = [QtCore.QPointF(geo[i], geo[i + 1]) for i in range(0, len(geo) - 1, 2)]
            # drawPolygon always closes back to the first point -- exactly
            # the "live preview closes back to the start" look the
            # click-to-place freeform selector wants while it's mid-placement.
            painter.drawPolygon(QtGui.QPolygonF(pts))


class ImageView2D(QtWidgets.QGraphicsView):
    mousedown = QtCore.Signal(object)
    mousedrag = QtCore.Signal(object)
    mouseup = QtCore.Signal(object)
    mousewheel = QtCore.Signal(object)
    mousemove = QtCore.Signal(object)
    mousedoubleclick = QtCore.Signal(object)
    signal_set_scale = QtCore.Signal(float)
    origin_update = QtCore.Signal(tuple)
    keypress = QtCore.Signal(object)

    def __init__(self, sizehint=(512, 512), parent=None):
        super().__init__(parent)
        self._sizehint = sizehint
        self.setMinimumSize(64, 64)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # This view's own interactivity (item selection/dragging, rubber-
        # band, native scrollbar panning) is entirely disabled -- every
        # mouse/keyboard gesture below is handled by hand, exactly as it
        # was for the plain QWidget this used to be. The scene exists
        # purely to hold persistent shape items Qt can redraw efficiently.
        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setSceneRect(-1.0e6, -1.0e6, 2.0e6, 2.0e6)
        self.setScene(self._scene)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setInteractive(False)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        # QGraphicsView keeps its own scrollbar position independent of
        # transform() (by default it auto-adjusts on setTransform()/resize
        # to keep some anchor point fixed) -- disable that anchoring and
        # pin the scrollbars at 0 in _apply_view_transform() below instead,
        # so transform() alone is the complete, exact scene->viewport
        # mapping (matching the hand-computed img_to_scr formula).
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        self._img = np.zeros((4, 4), dtype=np.float32)
        self._qimage = QtGui.QImage(4, 4, QtGui.QImage.Format_Grayscale8)
        self._qimage.fill(0)
        self._levels = (0.0, 1.0)
        self._manual_levels = None  # (lo, hi) override set via the contrast slider; None = autocontrast
        self._scale = 1.0
        self._origin = (0.0, 0.0)  # image coord shown at widget top-left
        self._center = None        # image coord to keep centered across resizes/zooms
        self.mag = 1.1              # zoom-in factor per wheel notch (matches EMAN2)
        self.invmag = 1.0 / self.mag
        self._pan_last = None       # last screen pos while middle-button panning

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._pixmap_item.setPos(0, 0)
        self._pixmap_item.setZValue(-10)
        self._scene.addItem(self._pixmap_item)
        self._rebuild_qimage()

        self.shapes = {}
        self._items = {}            # key -> _ShapeItem, mirrors self.shapes 1:1
        self._visible_keys = set()  # keys of self.shapes whose kind != "hidden"
        self.shapechange = 0
        self._eraser_shape = None
        self._eraser_item = None
        self._select_preview = None
        self._select_preview_item = None

        self._apply_view_transform()

    def sizeHint(self):
        return QtCore.QSize(*self._sizehint)

    # -- data -----------------------------------------------------------
    def set_data(self, img: np.ndarray, keepcontrast: bool = False):
        self._img = np.asarray(img, dtype=np.float32)
        if self._manual_levels is not None:
            self._levels = self._manual_levels
        elif not keepcontrast or self._levels is None:
            mean = float(self._img.mean())
            std = float(self._img.std())
            if std == 0:
                lo, hi = mean - 1, mean + 1
            else:
                lo, hi = mean - 3 * std, mean + 3 * std
            self._levels = (lo, hi)
        self._rebuild_qimage()

    def set_manual_levels(self, lo, hi):
        """Set an explicit (lo, hi) display range, overriding autocontrast
        until clear_manual_levels() is called -- used by the contrast
        slider."""
        self._manual_levels = (float(lo), float(hi))
        self._levels = self._manual_levels
        self._rebuild_qimage()

    def clear_manual_levels(self):
        self._manual_levels = None

    def _rebuild_qimage(self):
        lo, hi = self._levels
        if hi <= lo:
            hi = lo + 1
        h, w = self._img.shape
        scaled = (self._img - lo) / (hi - lo) * 255.0
        buf = np.clip(scaled, 0, 255).astype(np.uint8)
        buf = np.ascontiguousarray(buf)
        qimg = QtGui.QImage(buf.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8)
        self._qimage = qimg.copy()
        self._pixmap_item.setPixmap(QtGui.QPixmap.fromImage(self._qimage))

    # -- shapes -----------------------------------------------------------
    def get_shapes(self):
        return self.shapes

    def add_shape(self, key, shape: Shape):
        self.shapes[key] = shape
        item = self._items.get(key)
        if item is None:
            item = _ShapeItem(shape, lambda: self._scale)
            self._scene.addItem(item)
            self._items[key] = item
        else:
            item.set_shape(shape)
        self._update_visible(key, shape.kind)

    def add_shapes(self, shape_dict):
        for key, shape in shape_dict.items():
            self.add_shape(key, shape)

    def del_shape(self, key):
        self.shapes.pop(key, None)
        item = self._items.pop(key, None)
        if item is not None:
            self._scene.removeItem(item)
        self._visible_keys.discard(key)

    def sync_shape(self, key):
        """Call after directly mutating self.shapes[key]'s kind (and
        possibly its linewidth/geom) in place -- TomoBoxer.update_sliceview
        does this to toggle a particle's visibility/geometry for the
        current slice -- to sync the on-screen item and the visible-keys
        index without a full add_shape() replacement.

        update_sliceview re-derives every box's kind on most calls (see its
        only_indices docstring), so this runs once per box per recompute --
        usually thousands of times for boxes that are (and remain) off the
        current slice. refresh() calls prepareGeometryChange(), which
        touches the scene's spatial index and is genuinely expensive;
        skipping it whenever a shape was already hidden and stays hidden
        turns a full recompute from O(every box) expensive Qt calls into
        O(visible boxes), usually a small fraction. A hidden shape's
        geometry/linewidth are never drawn, so nothing is lost by not
        refreshing them until they actually become visible again (kind
        changes). item._shape is the same object as self.shapes[key] (see
        add_shape), so no separate shapes-dict lookup is needed here, and
        when kind hasn't actually changed, visible-keys membership can't
        have changed either -- so that's skipped too in the common case."""
        item = self._items.get(key)
        if item is None:
            return
        kind = item._shape.kind
        if item._last_kind == kind:
            if kind != "hidden":
                item.refresh()
            return
        item._last_kind = kind
        item.refresh()
        self._update_visible(key, kind)

    def remap_shape_keys(self, key_order):
        """Renumber self.shapes/self._items (and the visible-keys index) so
        the shape currently at old key key_order[new_key] becomes the shape
        at new_key -- used by TomoBoxer.del_box, where removing some
        particles shifts every later one's index.

        Any old key *not* present in key_order belongs to a particle that
        was actually deleted (not just renumbered), and its graphics item
        needs removing from the scene here -- otherwise it stays in the
        scene forever under no key at all, frozen at whatever kind/geometry
        it last had (never hidden again, since nothing iterates it anymore),
        showing on every slice and unclickable/undeletable since no box
        index maps to it any more."""
        old_shapes, old_items, old_visible = self.shapes, self._items, self._visible_keys
        kept_old_keys = set(key_order)
        for old_key, item in old_items.items():
            if old_key not in kept_old_keys:
                self._scene.removeItem(item)
        self.shapes = {}
        self._items = {}
        self._visible_keys = set()
        for new_key, old_key in enumerate(key_order):
            shape = old_shapes.get(old_key)
            if shape is None:
                continue
            self.shapes[new_key] = shape
            item = old_items.get(old_key)
            if item is not None:
                self._items[new_key] = item
            if old_key in old_visible:
                self._visible_keys.add(new_key)

    def _update_visible(self, key, kind):
        if kind == "hidden":
            self._visible_keys.discard(key)
        else:
            self._visible_keys.add(key)

    # -- eraser preview (plain attribute in callers: TomoBoxer.mouse_move
    # assigns view.eraser_shape = Shape(...) / None directly) -------------
    @property
    def eraser_shape(self):
        return self._eraser_shape

    @eraser_shape.setter
    def eraser_shape(self, shape):
        self._eraser_shape = shape
        if shape is None:
            if self._eraser_item is not None:
                self._eraser_item.setVisible(False)
            return
        if self._eraser_item is None:
            self._eraser_item = _ShapeItem(shape, lambda: self._scale)
            self._eraser_item.setZValue(10)  # always drawn above particle shapes, as before
            self._scene.addItem(self._eraser_item)
        else:
            self._eraser_item.set_shape(shape)
        self._eraser_item.setVisible(True)
        self._eraser_item.refresh()

    # -- shape-select tool preview (same pattern as eraser_shape above; a
    # separate property since the two are shown at mutually exclusive times
    # but there's no reason to make one reuse/clobber the other's item) ----
    @property
    def select_preview(self):
        return self._select_preview

    @select_preview.setter
    def select_preview(self, shape):
        self._select_preview = shape
        if shape is None:
            if self._select_preview_item is not None:
                self._select_preview_item.setVisible(False)
            return
        if self._select_preview_item is None:
            self._select_preview_item = _ShapeItem(shape, lambda: self._scale)
            self._select_preview_item.setZValue(10)
            self._scene.addItem(self._select_preview_item)
        else:
            self._select_preview_item.set_shape(shape)
        self._select_preview_item.setVisible(True)
        self._select_preview_item.refresh()

    # -- view transform -----------------------------------------------------
    def get_origin(self):
        return self._origin

    def set_origin(self, x, y, quiet=False):
        self._origin = (x, y)
        self._apply_view_transform()
        if not quiet:
            self.origin_update.emit(self._origin)

    def set_scale(self, scale, quiet=False):
        self._scale = max(0.05, float(scale))
        self._recenter(quiet=True)
        if not quiet:
            self.signal_set_scale.emit(self._scale)
        self._apply_view_transform()

    def scroll_to(self, x, y, quiet=False):
        """Center the view on image coordinate (x, y)."""
        self._center = (x, y)
        self._recenter(quiet=quiet)

    def _recenter(self, quiet=False):
        """Recompute the origin from the last requested center point and the
        widget's current size. Called on resize and after zooming so the
        view stays centered on the same content regardless of when Qt
        finalizes the widget's layout size."""
        if self._center is None:
            return
        x, y = self._center
        w, h = self.viewport().width(), self.viewport().height()
        ox = x - (w / 2.0) / self._scale
        oy = y - (h / 2.0) / self._scale
        self.set_origin(ox, oy, quiet=quiet)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recenter(quiet=True)

    def scr_to_img(self, pos):
        sx, sy = pos
        ox, oy = self._origin
        return (ox + sx / self._scale, oy + sy / self._scale)

    def img_to_scr(self, x, y):
        ox, oy = self._origin
        return ((x - ox) * self._scale, (y - oy) * self._scale)

    def _apply_view_transform(self):
        # Same composition as the old painter.scale(); painter.translate()
        # pair in paintEvent, just applied as the view's persistent
        # transform instead of redone by hand on every paint -- so it maps
        # scene (image) coordinates to viewport (screen) pixels exactly as
        # img_to_scr() computes by hand, keeping every hit-test that relies
        # on scr_to_img/img_to_scr pixel-identical to before.
        ox, oy = self._origin
        t = QtGui.QTransform()
        t.scale(self._scale, self._scale)
        t.translate(-ox, -oy)
        self.setTransform(t)
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)
        self.viewport().update()

    # -- mouse / keyboard -----------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            # Grab-and-pan: right-click and drag to shift the view.
            self._pan_last = (event.position().x(), event.position().y())
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            self.mousedown.emit(event)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Qt delivers a double click as press/release/DOUBLE-CLICK/release --
        # not a second ordinary press -- so this doesn't also add a
        # duplicate vertex through mousePressEvent's mousedown above; it's
        # purely the "finish the freeform selection" signal.
        if event.button() == QtCore.Qt.LeftButton:
            self.mousedoubleclick.emit(event)
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None and (event.buttons() & QtCore.Qt.RightButton):
            x, y = event.position().x(), event.position().y()
            dx, dy = x - self._pan_last[0], y - self._pan_last[1]
            ox, oy = self._origin
            nox, noy = ox - dx / self._scale, oy - dy / self._scale
            w, h = self.viewport().width(), self.viewport().height()
            self._center = (nox + (w / 2.0) / self._scale, noy + (h / 2.0) / self._scale)
            self.set_origin(nox, noy)
            self._pan_last = (x, y)
            event.accept()
            return
        if event.buttons() & QtCore.Qt.LeftButton:
            self.mousedrag.emit(event)
        else:
            self.mousemove.emit(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self._pan_last = None
            self.unsetCursor()
            event.accept()
            return
        self.mouseup.emit(event)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Plain wheel zooms in/out centered on the mouse cursor (so zooming
        # happens right where you're picking); Shift+wheel is passed through
        # as a "mousewheel" event (used here for depth/slice navigation).
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            self.mousewheel.emit(event)
        else:
            delta = event.angleDelta().y()
            pos = event.position()
            if delta > 0:
                self._zoom_at(pos.x(), pos.y(), self._scale * self.mag)
            elif delta < 0:
                self._zoom_at(pos.x(), pos.y(), self._scale * self.invmag)
        event.accept()

    def _zoom_at(self, scr_x, scr_y, new_scale):
        """Zoom so the image point currently under (scr_x, scr_y) stays fixed
        on screen, i.e. zoom in/out centered on the mouse cursor."""
        new_scale = max(0.05, float(new_scale))
        img_x, img_y = self.scr_to_img((scr_x, scr_y))
        self._scale = new_scale
        ox = img_x - scr_x / new_scale
        oy = img_y - scr_y / new_scale
        w, h = self.viewport().width(), self.viewport().height()
        self._center = (ox + (w / 2.0) / new_scale, oy + (h / 2.0) / new_scale)
        self.set_origin(ox, oy, quiet=False)
        self.signal_set_scale.emit(self._scale)

    def keyPressEvent(self, event):
        self.keypress.emit(event)
        super().keyPressEvent(event)
