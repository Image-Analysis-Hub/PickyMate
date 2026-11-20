"""View-only browser of the particles selected on a session map in the
*other* tomograms of a session -- the ones that are not open.

One tab per tomogram with a selection, its thumbnails cut straight from
that tomogram's file through a memory map, so nothing is loaded whole and
only the few megabytes around the selected particles are ever read. Each
thumbnail is bordered in its category's colour -- Confirmed, Recycle Bin,
or still in an ordinary set. Nothing here can be edited: particles belong
to their own tomogram, and are confirmed or recycled by opening it, where
the same selection appears as its "UMAP Selection" tab.

A tab's thumbnails are only cut when the tab is first shown, and again
only if that tomogram's selection has changed since.
"""
from __future__ import annotations

import weakref

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from . import categories, metadata
from .session import CAT_CONFIRMED, CAT_RECYCLE
from .thumb_grid import ParticleGrid
from .volume import Volume

MAX_THUMBNAILS = 3000   # per tomogram; beyond this the tab says how many it left out
_CATEGORY_CLASS = {CAT_CONFIRMED: categories.CONFIRMED_ID, CAT_RECYCLE: categories.RECYCLE_BIN_ID}
_ORDER = [categories.CONFIRMED_ID, 0, categories.RECYCLE_BIN_ID]


class OtherTomogramsWindow(QtWidgets.QWidget):
    def __init__(self, controller, target):
        super().__init__()
        self._controller = weakref.ref(controller)
        self._target = weakref.ref(target)
        self._tabs = {}          # name -> {"grid", "built_for": frozenset or None}
        self.setWindowTitle("PickyMate — Selected in Other Tomograms (view only)")
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Particles selected on the session map in the tomograms that are not open. View "
            "only -- open a tomogram (navigator) to confirm or recycle them; there they are its "
            "\"UMAP Selection\" tab. Borders: green Confirmed, red Recycle Bin, other colours "
            "still in a set.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(lambda _i: self._build_current())
        layout.addWidget(self.tabs, 1)
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.status)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.refresh)
        self.resize(820, 640)

    def schedule_refresh(self):
        self._timer.start()

    def refresh(self):
        """One tab per not-open tomogram with a selection, in session order."""
        ctl, target = self._controller(), self._target()
        if ctl is None or target is None:
            return
        session = ctl.session
        wanted = [(n, session.selection_uids(n)) for n in session.names if n != target.session_name]
        wanted = [(n, u) for n, u in wanted if u]
        current = self.tabs.tabText(self.tabs.currentIndex()).split(" (")[0] if self.tabs.count() else None
        for name in list(self._tabs):
            if name not in dict(wanted):
                self.tabs.removeTab(self.tabs.indexOf(self._tabs.pop(name)["grid"]))
        for i, (name, uids) in enumerate(wanted):
            entry = self._tabs.get(name)
            if entry is None:
                grid = ParticleGrid()
                entry = self._tabs[name] = {"grid": grid, "built_for": None}
                self.tabs.insertTab(i, grid, "")
            self.tabs.setTabText(self.tabs.indexOf(entry["grid"]), f"{name} ({len(uids)})")
        if current:
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i).split(" (")[0] == current:
                    self.tabs.setCurrentIndex(i)
        self.status.setText("" if wanted else "Nothing is selected in the other tomograms.")
        self._build_current()

    def _build_current(self):
        ctl, target = self._controller(), self._target()
        if ctl is None or target is None or not self.tabs.count():
            return
        name = self.tabs.tabText(self.tabs.currentIndex()).split(" (")[0]
        entry = self._tabs.get(name)
        if entry is None:
            return
        uids = frozenset(ctl.session.selection_uids(name))
        if entry["built_for"] == uids:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries, skipped = thumbnails(ctl.session, name, sorted(uids), target.nlayers())
            self.status.setText(
                f"{name}: {len(entries)} particles" +
                (f" (the first {MAX_THUMBNAILS} of {len(uids)} are shown)" if skipped else ""))
            entry["grid"].sync_particles_grouped(entries, _ORDER)
            entry["built_for"] = uids
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()


def thumbnails(session, name, uids, nlayers):
    """[(uid, image, category class id)] for the selected particles of a
    tomogram that is not open, cut from its file through a memory map at
    that tomogram's own box sizes, averaged over the same number of slices
    as the open tomogram's thumbnails. Returns (entries, skipped)."""
    e = session.entry(name)
    table = session.particle_table(name)
    settings = metadata.load_class_settings(e.working_dir)
    vol = Volume.from_file(e.tomogram, native_memmap=True)
    half = max(int(nlayers) - 1, 0)
    out = []
    try:
        for uid in uids[:MAX_THUMBNAILS]:
            r = table["records"].get(uid)
            if r is None:
                continue
            bs = int(settings.get(r["set"], {}).get("boxsize", metadata.DEFAULT_BOXSIZE))
            cube = vol.get_clip_centered(r["x"], r["y"], r["z"], bs, bs, 2 * half + 1)
            img = cube.mean(axis=0, dtype=np.float32)
            cat = (CAT_CONFIRMED if r["set"] == categories.CONFIRMED_NAME else
                   CAT_RECYCLE if r["set"] == categories.RECYCLE_BIN_NAME else None)
            out.append((uid, img, _CATEGORY_CLASS.get(cat, 0)))
    finally:
        owner = getattr(vol, "_mmap_owner", None)
        if owner is not None:
            owner.close()
    return out, max(len(uids) - MAX_THUMBNAILS, 0)
