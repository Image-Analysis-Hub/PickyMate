"""Small labeled-value widgets for the app's handful of numeric controls
(box size, eraser radius, layer thickness)."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

# A QAction-based toolbar button (QToolButton, auto-raise) is flat and
# borderless until hovered by default -- that's the standard toolbar
# convention, but next to a real QPushButton like AutoContrast it can read
# as "not actually a button" rather than deliberately minimal. Rather than
# give every toolbar button AutoContrast's raised 3D chrome (which looks
# dated for a row with this many actions), this gives them a flat "chip"
# look that's still clearly bounded at rest, with a hover tint, a pressed
# state, and -- most importantly for the exclusive Pick/Eraser/Select
# tools, which had no strong visual cue for which one was active -- a
# solid accent fill while checked. palette(...) roles keep it adapting to
# whatever light/dark theme is active, rather than hardcoding colors that
# could look wrong on either.
#
# Shared here (rather than defined once inline) because it's applied to
# two separate top-level windows -- the tomogram window and the particle
# list -- which don't share a widget hierarchy for a stylesheet set on one
# to cascade to the other.
TOOLBAR_BUTTON_STYLE = """
    QToolBar QToolButton {
        background-color: palette(button);
        border: 1px solid palette(mid);
        border-radius: 4px;
        padding: 3px 9px;
        margin: 1px;
    }
    QToolBar QToolButton:hover {
        background-color: palette(light);
        border-color: palette(dark);
    }
    QToolBar QToolButton:pressed {
        background-color: palette(mid);
    }
    QToolBar QToolButton:checked {
        background-color: palette(highlight);
        color: palette(highlighted-text);
        border-color: palette(highlight);
    }
    QToolBar QToolButton:disabled {
        background-color: palette(window);
        border-color: palette(midlight);
        color: palette(mid);
    }
    QToolBar QToolButton#destructiveAction {
        color: #c62828;
        border-color: #c62828;
    }
    QToolBar QToolButton#destructiveAction:hover {
        background-color: #ffebee;
        border-color: #b71c1c;
    }
    QToolBar QToolButton#destructiveAction:pressed {
        background-color: #ffcdd2;
    }
    QToolBar QToolButton#destructiveAction:disabled {
        color: palette(mid);
        border-color: palette(midlight);
    }
    QToolBar QToolButton#pulsingAttention {
        background-color: #ff9800;
        color: white;
        border-color: #e65100;
    }
"""


def mark_destructive(widget):
    """Tag a toolbar button as a destructive/irreversible-feeling action
    (Delete Set, Empty Recycle Bin, ...) so it picks up the red accent
    above instead of the default neutral chip look -- the one other color
    used anywhere in the app besides the green "primary action" button, so
    a user learns two consistent meanings (green = do this, red = this
    removes something) instead of every button looking equally weighted."""
    if widget is not None:
        widget.setObjectName("destructiveAction")


class RangeSlider(QtWidgets.QWidget):
    """A single horizontal slider with two draggable handles marking a
    (low, high) sub-range of [minimum, maximum] -- used for the manual
    contrast control, where the two handles are the black/white display
    levels rather than two independent values."""

    rangeChanged = QtCore.Signal(float, float)
    HANDLE_RADIUS = 7

    def __init__(self, minimum=0.0, maximum=1.0, low=0.0, high=1.0, parent=None):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._low = float(low)
        self._high = float(high)
        self._dragging = None  # "low" | "high" | None
        self.setMinimumHeight(20)
        self.setMinimumWidth(80)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_bounds(self, minimum, maximum):
        self._min, self._max = float(minimum), float(maximum)
        self._low = max(self._min, min(self._low, self._max))
        self._high = max(self._min, min(self._high, self._max))
        self.update()

    def set_values(self, low, high, emit=True):
        self._low = max(self._min, min(float(low), self._max))
        self._high = max(self._min, min(float(high), self._max))
        if self._low > self._high:
            self._low, self._high = self._high, self._low
        self.update()
        if emit:
            self.rangeChanged.emit(self._low, self._high)

    def values(self):
        return self._low, self._high

    def _value_to_x(self, value):
        w = max(1, self.width() - 2 * self.HANDLE_RADIUS)
        span = self._max - self._min
        frac = 0.0 if span <= 0 else (value - self._min) / span
        return self.HANDLE_RADIUS + frac * w

    def _x_to_value(self, x):
        w = max(1, self.width() - 2 * self.HANDLE_RADIUS)
        frac = max(0.0, min(1.0, (x - self.HANDLE_RADIUS) / w))
        return self._min + frac * (self._max - self._min)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        mid_y = self.height() // 2

        pen = QtGui.QPen(QtGui.QColor(140, 140, 140))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(self.HANDLE_RADIUS, mid_y, self.width() - self.HANDLE_RADIUS, mid_y)

        lo_x, hi_x = self._value_to_x(self._low), self._value_to_x(self._high)
        pen.setColor(QtGui.QColor(90, 160, 250))
        painter.setPen(pen)
        painter.drawLine(int(lo_x), mid_y, int(hi_x), mid_y)

        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60)))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(235, 235, 235)))
        for x in (lo_x, hi_x):
            painter.drawEllipse(QtCore.QPointF(x, mid_y), self.HANDLE_RADIUS, self.HANDLE_RADIUS)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        x = event.position().x()
        lo_x, hi_x = self._value_to_x(self._low), self._value_to_x(self._high)
        self._dragging = "low" if abs(x - lo_x) <= abs(x - hi_x) else "high"
        self._drag_to(x)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_to(event.position().x())
            event.accept()

    def mouseReleaseEvent(self, event):
        # The handle itself tracks the cursor live (via _drag_to's repaint)
        # for smooth visual feedback, but rangeChanged -- which drives the
        # actual (expensive: re-slicing 3 views plus every gallery
        # thumbnail) contrast update -- only fires once, here, on release.
        # Applying it on every mouse-move made dragging feel laggy.
        was_dragging = self._dragging is not None
        self._dragging = None
        if was_dragging:
            self.rangeChanged.emit(self._low, self._high)

    def _drag_to(self, x):
        value = self._x_to_value(x)
        if self._dragging == "low":
            self._low = min(value, self._high)
        else:
            self._high = max(value, self._low)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


class _SpaceKeyState(QtCore.QObject):
    """Tracks whether the space bar is currently held down, application-
    wide. Installed once as an event filter on the QApplication instance so
    every view (tomogram slices, particle gallery) can cheaply check
    space_key_state().held from within a mouse-press handler to implement
    the "space+click = confirm this particle" gesture, without each widget
    needing its own keyboard focus/tracking logic.

    Every Space key event is consumed here (returning True) rather than
    left to propagate: Space is a real key with its own default behavior on
    many widgets (toggling a focused checkable list item, activating a
    focused button, ...), and letting it through caused exactly that --
    e.g. holding Space to confirm a particle could silently toggle whatever
    checkbox happened to have keyboard focus in the Sets panel, hiding
    classes as a side effect. Since nothing else in this app uses Space,
    swallowing it globally is safe and removes that whole bug class."""

    def __init__(self):
        super().__init__()
        self.held = False

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Space:
            if not event.isAutoRepeat():
                self.held = True
            return True
        if etype == QtCore.QEvent.KeyRelease and event.key() == QtCore.Qt.Key_Space:
            if not event.isAutoRepeat():
                self.held = False
            return True
        return False


_space_key_state = None


def space_key_state() -> _SpaceKeyState:
    """Returns the shared space-bar tracker, installing it on first call.
    Call this once eagerly (TomoBoxer does, at construction) rather than
    relying on the first gesture to trigger it lazily -- installing it only
    when a click handler first asks for .held means the very first Space
    keypress of the session (the one the user is holding down for that
    first click) fires before the filter exists to observe it, so that
    first "space+click" silently falls through as a plain click instead of
    a confirm."""
    global _space_key_state
    if _space_key_state is None:
        _space_key_state = _SpaceKeyState()
        QtWidgets.QApplication.instance().installEventFilter(_space_key_state)
    return _space_key_state


class LabeledSpinBox(QtWidgets.QWidget):
    """Label + integer spin box, used for the app's integer-valued controls
    (box size, eraser radius).

    Keyboard tracking is off: typing a new value (e.g. box size, which can
    trigger rewriting every particle's shape/thumbnail) doesn't apply on
    every keystroke, only once you press Enter, click elsewhere, or use the
    up/down arrows/spin buttons."""

    valueChanged = QtCore.Signal(object)

    def __init__(self, label="", value=0, minimum=0, maximum=100000, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = None
        if label:
            self.label = QtWidgets.QLabel(label)
            layout.addWidget(self.label)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setKeyboardTracking(False)
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(int(value))
        layout.addWidget(self.spin)
        self.spin.valueChanged.connect(lambda v: self.valueChanged.emit(v))

    def getValue(self):
        return self.spin.value()

    def setValue(self, value):
        self.spin.blockSignals(True)
        self.spin.setValue(int(value))
        self.spin.blockSignals(False)

    def setToolTip(self, text):
        # A tooltip set on this composite widget wouldn't otherwise be seen
        # while hovering the label or spin box specifically -- Qt shows a
        # widget's own tooltip for whichever child is actually under the
        # cursor, not a parent's, so it has to be applied to both by hand.
        super().setToolTip(text)
        self.spin.setToolTip(text)
        if self.label is not None:
            self.label.setToolTip(text)
