"""The multi-tomogram session: navigator window and switching controller.

The navigator lists every tomogram in the session with its progress, read
from the working directories on disk, and opens one on double-click or
with Next / Previous. Opening a tomogram closes the current one first
(saving it, without asking) and builds a fresh set of windows for the new
one, at the positions the previous set was left in.

Rebuilding the windows rather than swapping the tomogram inside them is
deliberate. Everything about a tomogram -- volume, particles, sets, uid
counter, undo history, thumbnails, embeddings -- is built into the window
as it opens, so reloading in place would mean resetting every one of those
by hand, and missing one would carry state from one tomogram into the next
(a wrong colour, a uid collision, an undo that edits the wrong tomogram).
A rebuild cannot leak anything. It costs a few seconds per switch -- about
half of that is reading the volume from disk, which is hidden by reading
the next tomogram in the list on a background thread while you work.
"""
from __future__ import annotations

import gc
import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import categories


class SessionController(QtCore.QObject):
    def __init__(self, session, base_options):
        super().__init__()
        self.session = session
        self.base_options = base_options
        self.current = None
        # ("this" | "all" | "choose", [names]) -- which tomograms' references
        # ranking pools; kept here so it survives switching tomograms.
        self.reference_scope = ("this", [])
        self._geometry = {}
        self._layout = {}
        # The session map and its background computation belong to the
        # session, not to whichever tomogram is open, so switching tomograms
        # neither drops a computed map nor kills one being computed.
        from .session_map import SessionMapJob, SessionMapStore
        self.session_map = SessionMapStore(session)
        self.session_map_job = SessionMapJob(self.session_map)
        self.navigator = SessionNavigator(self)

    def start(self, name=None):
        self.navigator.show()
        QtWidgets.QApplication.processEvents()
        self.open(name or self.session.names[0])

    @property
    def current_name(self):
        return self.current.session_name if self.current is not None else None

    def open(self, name):
        if self.current is not None and self.current.session_name == name:
            for w in self.current.session_windows().values():
                w.raise_()
            return
        from .main_window import TomoBoxer   # here, not at module level: main_window imports us

        self.navigator.set_busy(f"Opening {name}...")
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            if self.current is not None:
                old = self.current
                self._geometry = {k: w.saveGeometry() for k, w in old.session_windows().items()}
                self._layout = old.session_layout()
                self.current = None
                old.close_for_switch()
                old.deleteLater()
                del old
                QtWidgets.QApplication.processEvents()
                # deleteLater only takes effect back in the event loop, after
                # this method returns; flush it now so the old windows are
                # really gone before the next tomogram's volume is read, and
                # collect the Python side of the cycle they formed
                QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
                gc.collect()
            categories.clear_all_manual_colors()
            options = self.session.options_for(name, self.base_options)
            options.session_controller = self
            try:
                win = TomoBoxer(options, datafile=options.tomogram)
            except Exception as e:  # noqa: BLE001 -- report and stay in the session
                QtWidgets.QMessageBox.critical(
                    self.navigator, "Open tomogram", f"Could not open {name}:\n{e}")
                return
            win.show()
            for key, w in win.session_windows().items():
                if key in self._geometry:
                    w.restoreGeometry(self._geometry[key])
            if self._layout:
                win.restore_session_layout(self._layout)
            self.current = win
            self.navigator.refresh()
            idx = self.session.index_of(name)
            if idx + 1 < len(self.session.names):
                self.session.prefetch(self.session.names[idx + 1],
                                      bool(getattr(self.base_options, "native_memmap", False)))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.navigator.set_busy(None)

    def apply_session_selection(self, picked, mode):
        """A selection made on the session map, applied to every tomogram.
        picked: {tomogram name: set of uids}; mode as for
        TomoBoxer.set_umap_selection. "replace" gives every tomogram exactly
        what was picked in it -- nothing, for those the lasso missed. The
        open tomogram's selection is set live (its "UMAP Selection" tab);
        every other tomogram's is written to its working directory, which
        it loads as that tab the next time it is opened."""
        here = self.current_name
        new_here = None
        for name in self.session.names:
            pick = set(picked.get(name, ()))
            current = (set(self.current.umap_selection) if name == here
                       else self.session.selection_uids(name))
            if mode == "add":
                new = current | pick
            elif mode == "toggle":
                new = current ^ pick
            else:
                new = pick
            if name == here:
                new_here = new
            elif new != current:
                self.session.write_selection(name, new)
        if self.current is not None and new_here is not None:
            # other tomograms first, so the map refresh this triggers reads them
            self.current.set_umap_selection(new_here, "replace", from_map=bool(new_here))
            if self.current.umap_window is not None:
                self.current.umap_window.schedule_refresh()
        self.navigator.refresh()

    def step(self, delta):
        names = self.session.names
        if self.current_name is None:
            self.open(names[0])
            return
        idx = self.session.index_of(self.current_name) + delta
        if 0 <= idx < len(names):
            self.open(names[idx])

    def tomogram_closed_by_user(self, win):
        """The open tomogram's windows were closed by hand (not by a
        switch): end the session, the same way closing the windows ends
        the program in single-tomogram mode."""
        if win is self.current:
            self.current = None
        self.navigator.end_session()


class SessionNavigator(QtWidgets.QWidget):
    COLUMNS = ("Tomogram", "Confirmed", "Recycle Bin", "Unlabelled", "Reviewed", "UMAP selected")

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._ending = False
        session = controller.session
        self.setWindowTitle("PickyMate — Session")
        layout = QtWidgets.QVBoxLayout(self)

        where = os.path.basename(session.manifest_path) if session.manifest_path else "session"
        layout.addWidget(QtWidgets.QLabel(
            f"<b>{where}</b> — {len(session.names)} tomograms. "
            f"Double-click one to open it; only the open one is in memory."))

        self.table = QtWidgets.QTableWidget(len(session.names), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda row, _col: self.controller.open(session.names[row]))
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("◀  Previous")
        self.prev_button.setToolTip("Open the previous tomogram (Ctrl+Page Up)")
        self.prev_button.clicked.connect(lambda: self.controller.step(-1))
        self.next_button = QtWidgets.QPushButton("Next  ▶")
        self.next_button.setToolTip("Open the next tomogram (Ctrl+Page Down)")
        self.next_button.clicked.connect(lambda: self.controller.step(+1))
        self.open_button = QtWidgets.QPushButton("Open Selected")
        self.open_button.clicked.connect(self._open_selected)
        buttons.addWidget(self.prev_button)
        buttons.addWidget(self.next_button)
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.status)

        # Application-wide, so Next / Previous work from whichever of the
        # tomogram's windows has focus, not only from this one.
        for keys, delta in (("Ctrl+PgDown", +1), ("Ctrl+PgUp", -1)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda d=delta: self.controller.step(d))

        self.resize(560, 120 + 28 * len(session.names))
        self.refresh()

    def _open_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.controller.open(self.controller.session.names[rows[0].row()])

    def refresh(self):
        """Re-read every tomogram's counts from disk -- cheap, and the only
        way to see a tomogram's progress without opening it."""
        current = self.controller.current_name
        bold = QtGui.QFont()
        bold.setBold(True)
        for row, entry in enumerate(self.controller.session.entries):
            c = entry.counts()
            total = c["confirmed"] + c["recycle"] + c["unlabelled"]
            reviewed = f"{100 * (c['confirmed'] + c['recycle']) / total:.0f}%" if total else "—"
            name = ("▶ " if entry.name == current else "   ") + entry.name
            if not os.path.isdir(entry.working_dir):
                reviewed = "not started"
            selected = len(self.controller.session.selection_uids(entry.name))
            values = (name, c["confirmed"], c["recycle"], c["unlabelled"], reviewed,
                      selected or "")
            for col, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(v))
                if col:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if entry.name == current:
                    item.setFont(bold)
                self.table.setItem(row, col, item)
        if current is not None:
            self.table.selectRow(self.controller.session.index_of(current))
        names = self.controller.session.names
        idx = names.index(current) if current in names else -1
        self.prev_button.setEnabled(idx > 0)
        self.next_button.setEnabled(idx < len(names) - 1)

    def set_busy(self, text):
        self.status.setText(text or "")
        for b in (self.prev_button, self.next_button, self.open_button):
            b.setEnabled(text is None)
        self.table.setEnabled(text is None)
        if text is None:
            self.refresh()

    def end_session(self):
        self._ending = True
        self.close()

    def closeEvent(self, event):
        current = self.controller.current
        if self._ending or current is None:
            event.accept()
            return
        # Ask through the open tomogram, which saves it and -- on a yes --
        # reports back through tomogram_closed_by_user, ending the session.
        event.ignore()
        current.request_close_all()


class ChooseTomogramsDialog(QtWidgets.QDialog):
    """Pick which other tomograms' references to pool when ranking."""

    def __init__(self, names, chosen, parent=None):
        super().__init__(parent)
        self.setWindowTitle("References From")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Pool this tomogram's references with those of:"))
        self.list = QtWidgets.QListWidget()
        for n in names:
            item = QtWidgets.QListWidgetItem(n)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if n in chosen else Qt.Unchecked)
            self.list.addItem(item)
        layout.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        for label, state in (("Select All", Qt.Checked), ("Select None", Qt.Unchecked)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, s=state: self._set_all(s))
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _set_all(self, state):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)

    def chosen(self):
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]
