"""Small labeled-value widgets, replacing EMAN2's ValBox/ValSlider for the
handful of numeric controls e2spt_boxer.py uses (box size, eraser radius,
layer thickness, lowpass filter strength)."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


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
    """Label + integer spin box. Drop-in equivalent of ValBox for the
    integer-valued controls used here (box size, eraser radius)."""

    valueChanged = QtCore.Signal(object)

    def __init__(self, label="", value=0, minimum=0, maximum=100000, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if label:
            lbl = QtWidgets.QLabel(label)
            layout.addWidget(lbl)
        self.spin = QtWidgets.QSpinBox()
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


class LabeledSlider(QtWidgets.QWidget):
    """Label + slider + spin box. Drop-in equivalent of ValSlider for the
    "Filt" lowpass control."""

    valueChanged = QtCore.Signal(object)

    def __init__(self, rng=(0, 150), label="", value=0, parent=None):
        super().__init__(parent)
        self._lo, self._hi = rng
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if label:
            layout.addWidget(QtWidgets.QLabel(label))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(int(self._lo), int(self._hi))
        self.slider.setValue(int(value))
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(int(self._lo), int(self._hi))
        self.spin.setValue(int(value))
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, v):
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _on_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def getValue(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(value))
        self.spin.setValue(int(value))
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
