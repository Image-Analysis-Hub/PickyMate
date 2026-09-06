"""Settings > Customize Keyboard Shortcuts... -- lets the user rebind any
of the app's registered keyboard shortcuts (see TomoBoxer's
_shortcut_registry, built in __init__) and saves the result via
shortcuts.save_overrides.

This is a separate, editable dialog from Help > Keyboard Shortcuts (a
read-only reference/cheat-sheet) -- the two intentionally don't share
code, since one only displays fixed text and the other edits live state.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class KeyboardShortcutsDialog(QtWidgets.QDialog):
    def __init__(self, registry, parent=None):
        """registry: TomoBoxer._shortcut_registry -- a list of dicts with
        'id', 'label', 'get' (() -> QKeySequence), 'set'
        (QKeySequence -> None), and 'default' (QKeySequence, captured
        once at registration time). Only read here (via 'get'/'default');
        nothing is applied live until values() is read back by the
        caller on accept()."""
        super().__init__(parent)
        self.setWindowTitle("Customize Keyboard Shortcuts")
        self.resize(520, 440)
        self._registry = registry
        self._edits = {}  # id -> QKeySequenceEdit

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Click a shortcut field and press the new key combination. "
            "Clear a field (Backspace/Delete while it's focused) to remove "
            "that shortcut entirely.")
        note.setWordWrap(True)
        layout.addWidget(note)

        table = QtWidgets.QTableWidget(len(registry), 3, self)
        table.setHorizontalHeaderLabels(["Action", "Shortcut", ""])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        for row, entry in enumerate(registry):
            item = QtWidgets.QTableWidgetItem(entry["label"])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, 0, item)

            edit = QtWidgets.QKeySequenceEdit(entry["get"](), table)
            edit.keySequenceChanged.connect(self._check_conflicts)
            table.setCellWidget(row, 1, edit)
            self._edits[entry["id"]] = edit

            default_text = entry["default"].toString() or "(none)"
            reset_button = QtWidgets.QPushButton("Reset", table)
            reset_button.setToolTip(f"Restore the default: {default_text}")
            reset_button.clicked.connect(
                lambda checked=False, e=edit, d=entry["default"]: e.setKeySequence(d))
            table.setCellWidget(row, 2, reset_button)

        table.resizeColumnToContents(2)
        layout.addWidget(table)
        self._table = table

        self._conflict_label = QtWidgets.QLabel("")
        self._conflict_label.setStyleSheet("color: #c62828;")
        self._conflict_label.setWordWrap(True)
        layout.addWidget(self._conflict_label)

        button_row = QtWidgets.QHBoxLayout()
        restore_all = QtWidgets.QPushButton("Restore All Defaults")
        restore_all.clicked.connect(self._restore_all_defaults)
        button_row.addWidget(restore_all)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._check_conflicts()

    def _restore_all_defaults(self):
        for entry in self._registry:
            self._edits[entry["id"]].setKeySequence(entry["default"])

    def _conflicting_ids(self):
        seen = {}
        conflicts = set()
        for entry in self._registry:
            seq = self._edits[entry["id"]].keySequence()
            if seq.isEmpty():
                continue
            text = seq.toString()
            if text in seen:
                conflicts.add(seen[text])
                conflicts.add(entry["id"])
            else:
                seen[text] = entry["id"]
        return conflicts

    def _check_conflicts(self):
        conflicts = self._conflicting_ids()
        for entry in self._registry:
            edit = self._edits[entry["id"]]
            edit.setStyleSheet("background-color: #ffebee;" if entry["id"] in conflicts else "")
        if conflicts:
            names = ", ".join(e["label"] for e in self._registry if e["id"] in conflicts)
            self._conflict_label.setText(f"These share the same shortcut: {names}")
        else:
            self._conflict_label.setText("")

    def accept(self):
        conflicts = self._conflicting_ids()
        if conflicts:
            reply = QtWidgets.QMessageBox.question(
                self, "Customize Keyboard Shortcuts",
                "Some shortcuts are assigned to more than one action, so only one of "
                "them will actually respond to that key press. Save anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        super().accept()

    def values(self):
        """{id: QKeySequence} -- the (possibly edited) value of every
        registered shortcut, whether changed or not; the caller decides
        what counts as "different from default" when persisting."""
        return {entry["id"]: self._edits[entry["id"]].keySequence() for entry in self._registry}
