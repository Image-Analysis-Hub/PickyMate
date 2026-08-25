"""Small labeled-value widgets, replacing EMAN2's ValBox/ValSlider for the
handful of numeric controls e2spt_boxer.py uses (box size, eraser radius,
layer thickness, lowpass filter strength)."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


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
