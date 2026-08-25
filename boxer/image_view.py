"""A lightweight QPainter-based 2D image view: pan/zoom, auto-contrast,
shape-overlay annotation, and mouse/keyboard signal plumbing.

This replaces the subset of EMAN2's OpenGL-based EMImage2DWidget that
e2spt_boxer.py actually relies on (set_data/add_shape/get_shapes/del_shape,
scr_to_img, scroll_to, get_origin/set_origin, set_scale, and the
mousedown/mousedrag/mouseup/mousewheel/mousemove/keypress signals).
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .shapes import Shape


class ImageView2D(QtWidgets.QWidget):
    mousedown = QtCore.Signal(object)
    mousedrag = QtCore.Signal(object)
    mouseup = QtCore.Signal(object)
    mousewheel = QtCore.Signal(object)
    mousemove = QtCore.Signal(object)
    signal_set_scale = QtCore.Signal(float)
    origin_update = QtCore.Signal(tuple)
    keypress = QtCore.Signal(object)

    def __init__(self, sizehint=(512, 512), parent=None):
        super().__init__(parent)
        self._sizehint = sizehint
        self.setMinimumSize(64, 64)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._img = np.zeros((4, 4), dtype=np.float32)
        self._qimage = QtGui.QImage(4, 4, QtGui.QImage.Format_Grayscale8)
        self._qimage.fill(0)
        self._levels = (0.0, 1.0)
        self._scale = 1.0
        self._origin = (0.0, 0.0)  # image coord shown at widget top-left
        self._center = None        # image coord to keep centered across resizes/zooms
        self.mag = 1.1              # zoom-in factor per wheel notch (matches EMAN2)
        self.invmag = 1.0 / self.mag
        self._pan_last = None       # last screen pos while middle-button panning

        self.shapes = {}
        self.shapechange = 0
        self.eraser_shape = None

    def sizeHint(self):
        return QtCore.QSize(*self._sizehint)

    # -- data -----------------------------------------------------------
    def set_data(self, img: np.ndarray, keepcontrast: bool = False):
        self._img = np.asarray(img, dtype=np.float32)
        if not keepcontrast or self._levels is None:
            mean = float(self._img.mean())
            std = float(self._img.std())
            if std == 0:
                lo, hi = mean - 1, mean + 1
            else:
                lo, hi = mean - 3 * std, mean + 3 * std
            self._levels = (lo, hi)
        self._rebuild_qimage()
        self.update()

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

    # -- shapes -----------------------------------------------------------
    def get_shapes(self):
        return self.shapes

    def add_shape(self, key, shape: Shape):
        self.shapes[key] = shape

    def add_shapes(self, shape_dict):
        self.shapes.update(shape_dict)

    def del_shape(self, key):
        self.shapes.pop(key, None)

    # -- view transform -----------------------------------------------------
    def get_origin(self):
        return self._origin

    def set_origin(self, x, y, quiet=False):
        self._origin = (x, y)
        if not quiet:
            self.origin_update.emit(self._origin)
        self.update()

    def set_scale(self, scale, quiet=False):
        self._scale = max(0.05, float(scale))
        self._recenter(quiet=True)
        if not quiet:
            self.signal_set_scale.emit(self._scale)
        self.update()

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
        w, h = self.width(), self.height()
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

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        painter.save()
        ox, oy = self._origin
        painter.scale(self._scale, self._scale)
        painter.translate(-ox, -oy)
        painter.drawImage(0, 0, self._qimage)

        for shape in list(self.shapes.values()) + ([self.eraser_shape] if self.eraser_shape else []):
            self._draw_shape(painter, shape)
        painter.restore()

    def _draw_shape(self, painter: QtGui.QPainter, shape: Shape):
        if shape is None or shape.kind == "hidden":
            return
        r, g, b = shape.color
        color = QtGui.QColor.fromRgbF(max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
        pen = QtGui.QPen(color)
        pen.setWidthF(max(shape.linewidth, 1) / self._scale)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)

        g_ = shape.geom
        if shape.kind == "circle" and len(g_) >= 3:
            x, y, rad = g_[0], g_[1], g_[2]
            painter.drawEllipse(QtCore.QPointF(x, y), rad, rad)
        elif shape.kind == "rect" and len(g_) >= 4:
            x0, y0, x1, y1 = g_[0], g_[1], g_[2], g_[3]
            painter.drawRect(QtCore.QRectF(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1)))
        elif shape.kind == "line" and len(g_) >= 4:
            painter.drawLine(QtCore.QPointF(g_[0], g_[1]), QtCore.QPointF(g_[2], g_[3]))

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

    def mouseMoveEvent(self, event):
        if self._pan_last is not None and (event.buttons() & QtCore.Qt.RightButton):
            x, y = event.position().x(), event.position().y()
            dx, dy = x - self._pan_last[0], y - self._pan_last[1]
            ox, oy = self._origin
            nox, noy = ox - dx / self._scale, oy - dy / self._scale
            self._center = (nox + (self.width() / 2.0) / self._scale, noy + (self.height() / 2.0) / self._scale)
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
        self._center = (ox + (self.width() / 2.0) / new_scale, oy + (self.height() / 2.0) / new_scale)
        self.set_origin(ox, oy, quiet=False)
        self.signal_set_scale.emit(self._scale)
        self.update()

    def keyPressEvent(self, event):
        self.keypress.emit(event)
        super().keyPressEvent(event)
