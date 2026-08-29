"""Standalone tomogram particle boxer main window.

This is a functional port of EMAN2's e2spt_boxer.py (EMTomoBoxer) that runs
outside the EMAN2 environment, built on plain PySide6 + numpy + mrcfile.

Feature-complete for v1: orthogonal XY/XZ/ZY slice viewing, 2D or 3D particle
picking with mouse (move/drag/shift-delete), an eraser tool, multiple
picking "sets" (classes) with visibility toggles, per-class box size, slab
thickness averaging, saving/loading picks to a JSON sidecar next to the
tomogram, saving picks as a plain text coordinate file, and saving
extracted particles as an MRC stack (2D) or a folder of MRC volumes (3D).

Deferred to a later pass (noted in the code where relevant): PCA-based
tomogram "flatten" re-orientation, GIF export, and window-geometry
persistence.
"""
from __future__ import annotations

import os
import weakref

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .image_view import ImageView2D
from .thumb_grid import ParticleListPanel
from .widgets import LabeledSpinBox, RangeSlider, space_key_state
from . import categories
from . import metadata
from .metadata import base_name
from .shapes import Shape
from .volume import Volume, normalize, ParticleStackWriter


def parse_setname(name):
    p0 = name.find("::")
    if p0 > 0:
        try:
            return int(name[:p0])
        except ValueError:
            pass
    return -1


class TomoBoxer(QtWidgets.QMainWindow):
    """Main window: orthogonal slice views + picking controls."""

    keyPress = QtCore.Signal(QtGui.QKeyEvent)

    def __init__(self, options, datafile):
        super().__init__()
        self.initialized = False
        self._closing = False
        self.options = options
        self.boxshape = "circle" if options.mode == "3D" else "rect"
        self.setWindowTitle("Tomogram Boxer (standalone)")

        # Install the space-bar tracker now rather than waiting for the
        # first space+click gesture to trigger it lazily -- see
        # space_key_state()'s docstring for why that races the very first
        # keypress of the session.
        space_key_state()

        # -- menu ---------------------------------------------------------
        self.mfile = self.menuBar().addMenu("File")
        self.mfile_read_boxloc = self.mfile.addAction("Read Box Coord")
        self.mfile_save_boxloc = self.mfile.addAction("Save Box Coord")
        self.mfile_save_boxes_stack = self.mfile.addAction("Save Boxes as Stack")

        # -- central layout: 3 orthogonal views + control panel ------------
        self.setCentralWidget(QtWidgets.QWidget())
        self.gbl = QtWidgets.QGridLayout(self.centralWidget())
        gblh = QtWidgets.QHBoxLayout()
        self.gbl.addLayout(gblh, 0, 0, 2, 2)

        self.splitter_top = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter_bottom = QtWidgets.QSplitter(Qt.Horizontal)
        splitter_wrapper = QtWidgets.QSplitter(Qt.Vertical)

        self.xyview = ImageView2D(sizehint=(1024, 1024))
        self.xzview = ImageView2D(sizehint=(1024, 256))
        self.zyview = ImageView2D(sizehint=(256, 1024))

        self.wdepth = QtWidgets.QSlider()
        self.gbl.addWidget(self.wdepth, 0, 2)

        self.splitter_top.addWidget(self.zyview)
        self.splitter_top.addWidget(self.xyview)
        splitter_wrapper.addWidget(self.splitter_top)
        splitter_wrapper.addWidget(self.splitter_bottom)
        gblh.addWidget(splitter_wrapper)

        self.gbl2 = QtWidgets.QGridLayout()
        grid_widget = QtWidgets.QWidget()
        grid_widget.setLayout(self.gbl2)
        self.splitter_bottom.addWidget(grid_widget)
        self.splitter_bottom.addWidget(self.xzview)

        self.splitter_top.splitterMoved.connect(self._on_top_splitter_moved)
        self.splitter_bottom.splitterMoved.connect(self._on_bottom_splitter_moved)

        # -- control panel --------------------------------------------------
        gbl3 = QtWidgets.QHBoxLayout()
        self.gbl2.addLayout(gbl3, 0, 0, 1, 2)

        self.wcoords = QtWidgets.QLabel("")
        gbl3.addWidget(self.wcoords)

        self.wbautoc = QtWidgets.QPushButton("AutoContrast")
        self.wbautoc.setCheckable(True)
        self.wbautoc.setChecked(True)
        gbl3.addWidget(self.wbautoc)

        self.wcontrast = RangeSlider(minimum=0.0, maximum=1.0, low=0.0, high=1.0)
        self.gbl2.addWidget(self.wcontrast, 1, 0, 1, 2)

        self.wboxsize = LabeledSpinBox(label="Box Size:", value=0, minimum=1, maximum=8192)
        self.gbl2.addWidget(self.wboxsize, 2, 0)

        self.wboxsize_allsets = QtWidgets.QCheckBox("Apply to All Sets")
        self.wboxsize_allsets.setChecked(True)
        self.gbl2.addWidget(self.wboxsize_allsets, 2, 1)

        label0 = QtWidgets.QLabel("Thickness")
        self.gbl2.addWidget(label0, 3, 0)
        self.wnlayers = QtWidgets.QSpinBox()
        self.wnlayers.setRange(1, 256)
        self.wnlayers.setValue(1)
        self.gbl2.addWidget(self.wnlayers, 3, 1)

        self.curbox = -1
        self.boxes = []       # [x, y, z, method, score, class_id, uid]
        self.boxesimgs = []
        self.dragging = -1
        self._next_uid = 0
        self._undo_stack = []
        self._redo_stack = []

        # -- wiring -----------------------------------------------------------
        self.mfile_read_boxloc.triggered.connect(self.menu_file_read_boxloc)
        self.mfile_save_boxloc.triggered.connect(self.menu_file_save_boxloc)
        self.mfile_save_boxes_stack.triggered.connect(self.save_boxes)

        self.wdepth.valueChanged[int].connect(self.event_depth)
        self.wnlayers.valueChanged[int].connect(self.event_nlayers)
        self.wboxsize.valueChanged.connect(self.event_boxsize)
        self.wboxsize_allsets.toggled.connect(self.event_boxsize_allsets_toggled)
        self.wbautoc.toggled.connect(self.event_autocontrast)
        self.wcontrast.rangeChanged.connect(self.event_contrast_slider)

        self.xyview.mousemove.connect(self.xy_move)
        self.xyview.mousedown.connect(self.xy_down)
        self.xyview.mousedrag.connect(self.xy_drag)
        self.xyview.mouseup.connect(self.mouse_up)
        self.xyview.mousewheel.connect(self.xy_wheel)
        self.xyview.signal_set_scale.connect(self.event_scale)
        self.xyview.origin_update.connect(self.xy_origin)

        self.xzview.mousedown.connect(self.xz_down)
        self.xzview.mousedrag.connect(self.xz_drag)
        self.xzview.mouseup.connect(self.mouse_up)
        self.xzview.mousewheel.connect(self.xz_wheel)
        self.xzview.signal_set_scale.connect(self.event_scale)
        self.xzview.origin_update.connect(self.xz_origin)
        self.xzview.mousemove.connect(self.xz_move)

        self.zyview.mousedown.connect(self.zy_down)
        self.zyview.mousedrag.connect(self.zy_drag)
        self.zyview.mouseup.connect(self.mouse_up)
        self.zyview.mousewheel.connect(self.zy_wheel)
        self.zyview.signal_set_scale.connect(self.event_scale)
        self.zyview.origin_update.connect(self.zy_origin)
        self.zyview.mousemove.connect(self.zy_move)

        self.xyview.keypress.connect(self.key_press)

        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Undo, self)
        self.undo_shortcut.setContext(Qt.ApplicationShortcut)
        self.undo_shortcut.activated.connect(self.undo)

        # QKeySequence.Redo resolves to whatever the desktop theme defines
        # as "the" redo key -- on Linux (GNOME/KDE) that's normally
        # Ctrl+Shift+Z, not Ctrl+Y (the Windows convention), so a plain
        # QShortcut(QKeySequence.Redo, self) left Ctrl+Y doing nothing.
        # Merge Ctrl+Y into the same shortcut's key list (rather than
        # registering it as a second separate QShortcut) so it works
        # everywhere without ever colliding with the native binding as an
        # "ambiguous shortcut" on platforms where they're already the same.
        redo_keys = QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.Redo)
        ctrl_y = QtGui.QKeySequence("Ctrl+Y")
        if ctrl_y not in redo_keys:
            redo_keys.append(ctrl_y)
        self.redo_shortcut = QtGui.QShortcut(self)
        self.redo_shortcut.setKeys(redo_keys)
        self.redo_shortcut.setContext(Qt.ApplicationShortcut)
        self.redo_shortcut.activated.connect(self.redo)

        # -- load data ---------------------------------------------------------
        self.datafilename = datafile
        self.basename = base_name(datafile)
        p0 = datafile.find("__")
        if p0 > 0:
            p1 = datafile.rfind(".")
            self.filetag = datafile[p0:p1]
            if self.filetag[-1] != "_":
                self.filetag += "_"
        else:
            self.filetag = "__"

        data = Volume.from_file(datafile)
        if options.apix:
            data.apix = options.apix
        self.data = None
        self.set_data(data)

        # Per-class plain-text pick files live in "<basename>_picks/".
        self.picksdir = metadata.picks_dir(datafile)

        self.boxesviewer = ParticleListPanel(self)
        self.boxesviewer.setWindowTitle("Particle List")
        self.boxesviewer.set_mouse_mode("App")
        self.boxesviewer.show()

        self.setspanel = TomoSetsPanel(self)
        self.selectionpanel = SelectionPanel(self)
        self.optionviewer = TomoBoxerOptions(self)
        self.optionviewer.add_panel(self.setspanel, "Sets")
        self.optionviewer.add_panel(self.selectionpanel, "Selection")
        self.optionviewer.show()

        self.boxesviewer.mx_image_selected.connect(self.img_selected)
        self.boxesviewer.selection_changed.connect(self.selectionpanel.set_selection)
        self.boxesviewer.selection_changed.connect(self.set_selected_uids)

        self._arrange_windows()

        # -- metadata (picks + classes) ------------------------------------------
        # self.sets: class_id (int) -> class name (str), the name doubling
        # as the "<name>.txt" filename inside self.picksdir.
        self.sets = {}
        self.boxsize = {}

        boxsizes_by_name = metadata.load_boxsizes(self.picksdir)
        classnames = metadata.discover_classes(self.picksdir)
        normal_names = [n for n in classnames if n not in categories.FIXED_CLASS_NAMES]

        self.sets_visible = set()

        for i, name in enumerate(normal_names):
            self.sets[i] = name
            self.boxsize[i] = boxsizes_by_name.get(name, metadata.DEFAULT_BOXSIZE)
            for x, y, z in metadata.load_class_coords(self.picksdir, name):
                bdf = [x, y, z, "manual", 0.0, i, self._new_uid()]
                self.boxes.append(bdf)

        # The Recycle Bin and Confirmed categories always exist, whether or
        # not their pick files were present on disk yet.
        for name, cid in categories.FIXED_CLASS_NAMES.items():
            self.sets[cid] = name
            self.boxsize[cid] = boxsizes_by_name.get(name, metadata.DEFAULT_BOXSIZE)
            for x, y, z in metadata.load_class_coords(self.picksdir, name):
                bdf = [x, y, z, "manual", 0.0, cid, self._new_uid()]
                self.boxes.append(bdf)

        shape = [self.data.nx, self.data.ny, self.data.nz]
        for b in self.boxes:
            for i in range(3):
                b[i] = int(np.clip(round(b[i]), 1, shape[i] - 1))

        if not normal_names:
            self.new_set("particles_00")

        self.currentset = sorted(cid for cid in self.sets if cid not in categories.FIXED_CLASS_IDS)[0]
        if options.label:
            sid = [i for i in self.sets if self.sets[i] == options.label]
            if sid:
                self.currentset = sid[0]
            else:
                print("cannot find specified label")

        # All classes are visible by default when the tomogram is opened,
        # except the Recycle Bin -- rejected particles are more useful
        # tucked out of the way until you deliberately want to review them.
        self.sets_visible.update(self.sets.keys())
        self.sets_visible.discard(categories.RECYCLE_BIN_ID)
        self.setspanel.update_sets()
        self.wboxsize.setValue(self.get_boxsize())

        if options.clean:
            self.optionviewer.erasercheckbox.setChecked(True)

        for i in range(len(self.boxes)):
            self.update_box(i)

        self.update_all()
        self.initialized = True

    def showEvent(self, event):
        super().showEvent(event)
        self._arrange_windows()
        self.sync_splitters(self.splitter_top, self.splitter_bottom)
        if self.data is not None:
            self.scroll_to(self.data.nx // 2, self.data.ny // 2, self.data.nz // 2)

    def _arrange_windows(self):
        """Split the screen evenly between this window (tomogram) and the
        particle list, and put the options window on top, centered, so it
        can be minimized out of the way without affecting the other two."""
        # self.screen() is meaningless before the window has ever been
        # shown (Qt just returns some default/primary screen), which is
        # why everything used to always land on the primary monitor. Use
        # whichever screen the mouse cursor is currently on instead, which
        # reflects where the user is actually working; only fall back to
        # self.screen()/primaryScreen() once the window is already mapped
        # to a real screen.
        screen = None
        if self.isVisible():
            screen = self.screen()
        if screen is None:
            screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        half_w = avail.width() // 2

        self.setGeometry(avail.x(), avail.y(), half_w, avail.height())
        self.boxesviewer.setGeometry(avail.x() + half_w, avail.y(), avail.width() - half_w, avail.height())

        opt_w = min(420, avail.width() // 3)
        opt_h = min(640, avail.height())
        opt_x = avail.x() + (avail.width() - opt_w) // 2
        opt_y = avail.y() + (avail.height() - opt_h) // 2
        self.optionviewer.setGeometry(opt_x, opt_y, opt_w, opt_h)
        self.optionviewer.raise_()
        self.optionviewer.activateWindow()

    def sync_splitters(self, source, target):
        target.blockSignals(True)
        target.setSizes(source.sizes())
        target.blockSignals(False)

    def _on_top_splitter_moved(self, pos, index):
        self.sync_splitters(self.splitter_top, self.splitter_bottom)

    def _on_bottom_splitter_moved(self, pos, index):
        self.sync_splitters(self.splitter_bottom, self.splitter_top)

    # -- data / geometry -----------------------------------------------------------
    def set_data(self, data: Volume):
        self.data = data
        self.apix = data.apix
        self.datasize = (data.nx, data.ny, data.nz)
        self.x_loc, self.y_loc, self.z_loc = data.nx // 2, data.ny // 2, data.nz // 2

        self.wdepth.setRange(0, data.nz - 1)
        self.wdepth.setValue(data.nz // 2)
        self.boxes = []
        self.curbox = -1
        self._next_uid = 0
        self._undo_stack = []
        self._redo_stack = []
        self._selected_uids = set()

        self._init_contrast_slider(data)

        if self.initialized:
            self.update_all()

    def _init_contrast_slider(self, data: Volume):
        """Set the contrast slider's range and default handle positions
        from the volume's overall intensity distribution, so it starts
        somewhere sensible instead of at (0, 1). A full-volume mean/std is
        too slow for a real tomogram (multiple seconds), so this samples
        every 4th voxel along each axis instead -- close enough for a
        contrast range and orders of magnitude faster."""
        sample = data.data[::4, ::4, ::4]
        mean, std = float(sample.mean()), float(sample.std())
        if std == 0:
            std = 1.0
        self._contrast_defaults = (mean - 3 * std, mean + 3 * std)
        self.wcontrast.set_bounds(mean - 6 * std, mean + 6 * std)
        self.wcontrast.set_values(*self._contrast_defaults, emit=False)

    def eraser_width(self):
        return int(self.optionviewer.eraser_radius.getValue())

    def get_cube(self, x, y, z, centerslice=False, boxsz=-1):
        bs = self.get_boxsize() if boxsz < 0 else boxsz
        bz = 1 if centerslice else bs
        return self.data.get_clip_centered(x, y, z, bs, bs, bz)

    def get_slice(self, idx, thk=1, axis="z"):
        return self.data.slab_projection(axis, idx, int(thk - 1))

    # -- box size / thickness events -----------------------------------
    def event_boxsize(self):
        new_size = int(self.wboxsize.getValue())
        if self.wboxsize_allsets.isChecked():
            self.apply_boxsize_to_all_sets(new_size)
        else:
            if self.get_boxsize() == new_size:
                return
            self.boxsize[self.currentset] = new_size
            self._resize_boxes(lambda b: b[5] == self.currentset)

    def event_boxsize_allsets_toggled(self, checked):
        # Checking the box is itself an action ("make everything match the
        # currently displayed size"), not just a mode switch for the next
        # edit -- otherwise checking it while the number happens to already
        # equal the current set's size silently did nothing.
        if checked:
            self.apply_boxsize_to_all_sets(int(self.wboxsize.getValue()))

    def apply_boxsize_to_all_sets(self, new_size):
        changed_classes = {cid for cid in self.sets if self.boxsize.get(cid) != new_size}
        if not changed_classes:
            return
        for cid in changed_classes:
            self.boxsize[cid] = new_size
        self._resize_boxes(lambda b: b[5] in changed_classes)

    def _resize_boxes(self, affects):
        self.initialized = False
        for i in range(len(self.boxes)):
            if affects(self.boxes[i]):
                self.update_box(i)
        self.initialized = True
        self.update_all()

    def event_scale(self, newscale):
        # quiet=True: the view that originated the zoom already emitted
        # signal_set_scale once; re-emitting here would recurse.
        self.xyview.set_scale(newscale, quiet=True)
        self.xzview.set_scale(newscale, quiet=True)
        self.zyview.set_scale(newscale, quiet=True)

    def event_depth(self):
        if self.z_loc != self.wdepth.value():
            self.z_loc = self.wdepth.value()
        if self.initialized:
            self.update_sliceview()

    def event_nlayers(self):
        self.update_all()

    # -- contrast -----------------------------------------------------------
    def event_autocontrast(self, checked):
        if checked:
            for v in (self.xyview, self.xzview, self.zyview):
                v.clear_manual_levels()
            self.boxesviewer.clear_levels()
            self.wcontrast.set_values(*self._contrast_defaults, emit=False)
            self.update_all()
        else:
            self._apply_manual_contrast(*self.wcontrast.values())

    def event_contrast_slider(self, lo, hi):
        if self.wbautoc.isChecked():
            # Dragging the slider is a deliberate manual override -- switch
            # off AutoContrast, which (via event_autocontrast above) reads
            # the slider's own just-updated values and applies them.
            self.wbautoc.setChecked(False)
        else:
            self._apply_manual_contrast(lo, hi)

    def _apply_manual_contrast(self, lo, hi):
        for v in (self.xyview, self.xzview, self.zyview):
            v.set_manual_levels(lo, hi)
        self.boxesviewer.set_levels(lo, hi)

    def get_boxsize(self, clsid=None):
        # clsid=None means "the current set". -1 used to double as that
        # sentinel, but Recycle Bin/Confirmed are real classids that are
        # themselves negative (see categories.py), so a caller asking for
        # one of those got back whatever the *currently selected* set's box
        # size was instead of its own -- every particle sitting in Recycle
        # Bin/Confirmed would visibly resize as you switched sets.
        if clsid is None:
            clsid = self.currentset
        try:
            return int(self.boxsize[clsid])
        except KeyError:
            print(f"No box size saved for {clsid}..")
            return 32

    def nlayers(self):
        return int(self.wnlayers.value())

    # -- file menu -----------------------------------------------------------
    def menu_file_read_boxloc(self):
        fsp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select box coordinate text file")
        if not fsp or not os.path.isfile(fsp):
            return
        with open(fsp) as f:
            for line in f:
                parts = line.split()[:3]
                if len(parts) < 3:
                    continue
                b2 = [int(float(v)) for v in parts]
                bdf = [0, 0, 0, "manual", 0.0, self.currentset]
                for j in range(len(b2)):
                    bdf[j] = b2[j]
                bdf.append(self._new_uid())
                self.boxes.append(bdf)
                self.update_box(len(self.boxes) - 1)

    def menu_file_save_boxloc(self):
        fsp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output text file")
        if not fsp:
            return
        clsid = list(self.sets_visible)
        if not clsid:
            print("No visible particles to save")
            return
        self._write_coords(fsp, [b for b in self.boxes if int(b[5]) in clsid])

    def _write_coords(self, path, boxes):
        with open(path, "w") as out:
            for b in boxes:
                out.write(f"{b[0]} {b[1]} {b[2]}\n")

    def save_picks(self, only=None):
        """Write every class's particles to its own "<name>.txt" file in
        self.picksdir, and update config.settings with any non-default box
        sizes. This is the sole persistence mechanism (no JSON).

        `only`, when given, restricts the file writes to those classids --
        a single reclassification only changes the source and destination
        classes' contents, so rewriting every other class's file too (one
        open+write per class, every single click) was pure waste. Box sizes
        aren't affected by reclassifying particles, so that part is skipped
        for a scoped save; it still gets written on any full (only=None)
        save, e.g. at close."""
        classids = self.sets.keys() if only is None else (cid for cid in only if cid in self.sets)
        for clsid in classids:
            name = self.sets[clsid]
            coords = [(b[0], b[1], b[2]) for b in self.boxes if b[5] == clsid]
            metadata.save_class_coords(self.picksdir, name, coords)
        if only is None:
            boxsizes_by_name = {self.sets[k]: self.boxsize[k] for k in self.sets}
            metadata.save_boxsizes(self.picksdir, boxsizes_by_name)

    def save_boxes(self, clsid=None):
        clsid = clsid or []
        defaultname = "ptcls.mrcs" if not clsid else "_".join(self.sets[i] for i in clsid) + ".mrcs"
        name, ok = QtWidgets.QInputDialog.getText(self, "Save particles", "Filename suffix:", text=defaultname)
        if not ok:
            return
        name = self.filetag + str(name)

        is2d = self.options.mode != "3D"
        dr = "particles" if is2d else "particles3d"
        os.makedirs(dr, exist_ok=True)

        if is2d:
            if not name.lower().endswith((".mrc", ".mrcs")):
                name += ".mrcs"
            fsp = os.path.join(dr, self.basename) + name
        else:
            fsp = os.path.join(dr, self.basename) + os.path.splitext(name)[0]

        print(f"Saving {self.options.mode} particles to {fsp}")

        progress = QtWidgets.QProgressDialog("Saving", "Abort", 0, len(self.boxes), None)
        writer = ParticleStackWriter(fsp, is3d=not is2d, apix=self.apix)

        boxsz = -1
        sz = [s // 2 for s in self.datasize]
        for i, b in enumerate(self.boxes):
            if clsid and int(b[5]) not in clsid:
                continue
            bs = self.get_boxsize(b[5])
            if boxsz < 0:
                boxsz = bs
            elif boxsz != bs:
                print(f"Inconsistent box size in the particles to save.. Using {boxsz}..")
                bs = boxsz

            img = self.get_cube(b[0], b[1], b[2], centerslice=is2d, boxsz=bs)
            if not is2d:
                img = normalize(img)
                img = img * -1
            writer.add(img[0] if is2d else img)

            progress.setValue(i + 1)
            if progress.wasCanceled():
                break

        writer.close()

    # -- slice view update ----------------------------------------------------------
    def update_sliceview(self, axis=("x", "y", "z"), reslice=True, only_indices=None):
        """Recompute box-shape visibility/geometry for the given axes and
        repaint those views. When reslice is True (the default) each view's
        background is also re-fetched from the volume and rebuilt -- that's
        the expensive part (a full mean-projection over the volume plus an
        8-bit image rebuild per view) and is only needed when the viewing
        position/thickness/filter actually changed. Purely
        adding/deleting/reclassifying particles never moves the viewing
        cursor, so callers for those (see update_all's reslice param) pass
        reslice=False to just refresh the box overlays instead -- this is
        what made every recycle/confirm/delete click needlessly re-slice
        and repaint the whole tomogram.

        only_indices restricts the shape-visibility recompute below to just
        those box indices, instead of every box in the tomogram. A pure
        reclassification only changes the inplane/color/visibility outcome
        for the reassigned particle(s) -- every other box's position, class
        and the viewing cursor are all unchanged, so its outcome can't have
        changed either, and re-deriving it from scratch every time was pure
        O(particle count) waste on every single click. Pass an empty list
        when nothing needs recomputing at all (e.g. a plain deletion, where
        surviving boxes' shapes are untouched and only need repainting).
        None (the default) recomputes every box, which is still required
        whenever the viewing cursor, visible-set membership generally, or
        box sizes could have changed for boxes other than a known few."""
        boxes = self.boxes

        pms = {
            "z": (2, self.xyview, self.z_loc),
            "y": (1, self.xzview, self.y_loc),
            "x": (0, self.zyview, self.x_loc),
        }

        lwi = 2 if self.boxshape == "circle" else 3  # index into geom for linewidth handled separately

        for v in (self.xyview, self.xzview, self.zyview):
            v.del_shape("xline")
            v.del_shape("yline")

        if only_indices is None:
            items = list(enumerate(boxes))
        else:
            items = [(i, boxes[i]) for i in only_indices if 0 <= i < len(boxes)]

        for ax in axis:
            ia, view, loc = pms[ax]
            for i, b in items:
                bs = self.get_boxsize(b[5])
                dst = abs(b[ia] - loc)
                inplane = dst < bs // 2
                rad = bs // 2 - dst

                if ax == "z" and self.options.mode == "2D":
                    inplane = dst < 1

                shp = view.get_shapes().get(i)
                if shp is None:
                    continue
                if inplane and (b[5] in self.sets_visible):
                    shp.linewidth = 3 if self.curbox == i else 2
                    if self.options.mode == "3D":
                        shp.geom[2] = rad
                    shp.kind = self.boxshape
                else:
                    shp.kind = "hidden"
                view.sync_shape(i)

            if reslice:
                img = self.get_slice(loc, self.nlayers(), ax)
                view.set_data(img, keepcontrast=not self.wbautoc.isChecked())
            else:
                view.update()

        self.update_coords()

    def update_all(self, reslice=True, only_indices=None):
        if self.data is None:
            return
        self.update_sliceview(reslice=reslice, only_indices=only_indices)
        self.update_boximgs()

    def update_coords(self):
        self.wcoords.setText(f"X: {int(self.x_loc)}\tY: {int(self.y_loc)}\tZ: {int(self.z_loc)}")

    def inside_box(self, n, x=-1, y=-1, z=-1):
        box = self.boxes[n]
        if box[5] not in self.sets_visible:
            return False
        bs = self.get_boxsize(box[5]) / 2
        if self.options.mode == "3D":
            rr = (x >= 0) * (box[0] - x) ** 2 + (y >= 0) * (box[1] - y) ** 2 + (z >= 0) * (box[2] - z) ** 2
        else:
            rr = (x >= 0) * (box[0] - x) ** 2 + (y >= 0) * (box[1] - y) ** 2 + (z >= 0) * (box[2] != z) * (1e3 * bs ** 2)
        return rr <= bs ** 2

    def del_box(self, delids, record_undo=True):
        if not isinstance(delids, list):
            delids = [delids]
        delids = sorted({i for i in delids if 0 <= i < len(self.boxes)})
        if not delids:
            return
        affected_classes = {self.boxes[i][5] for i in delids}
        if record_undo:
            removed = [(i, list(self.boxes[i]), self.boxesimgs[i] if i < len(self.boxesimgs) else None) for i in delids]
            self._push_undo({"type": "delete", "removed": removed})
        kpids = [i for i in range(len(self.boxes)) if i not in delids]
        self.boxes = [self.boxes[i] for i in kpids]
        self.boxesimgs = [self.boxesimgs[i] for i in kpids if i < len(self.boxesimgs)]
        for view in (self.xyview, self.xzview, self.zyview):
            view.remap_shape_keys(kpids)
        self.curbox = -1
        if self.initialized:
            self.save_json(only=affected_classes)
        # Deleting never moves the viewing cursor, and surviving boxes'
        # shapes are untouched (just renumbered above) -- nothing needs
        # recomputing, only repainting with the now-smaller shape dicts.
        self.update_all(reslice=False, only_indices=[])

    def send_to_recycle_bin(self, indices):
        """Reassign particles to the Recycle Bin instead of deleting them
        outright, so a cleanup pass (shift+click) stays reversible. A
        particle that is already in the Recycle Bin is permanently deleted
        instead -- that's how a specific particle gets deleted out of the
        bin without emptying the whole thing (see also empty_recycle_bin,
        which purges everything in it at once)."""
        if not isinstance(indices, list):
            indices = [indices]
        idx = sorted({i for i in indices if 0 <= i < len(self.boxes)})
        already_binned = [i for i in idx if self.boxes[i][5] == categories.RECYCLE_BIN_ID]
        rest = [i for i in idx if self.boxes[i][5] != categories.RECYCLE_BIN_ID]
        if rest:
            self._reassign_boxes(rest, categories.RECYCLE_BIN_ID)
        if already_binned:
            self.del_box(already_binned)

    def send_to_confirmed(self, indices):
        """Reassign particles to the Confirmed category (space+click)."""
        self._reassign_boxes(indices, categories.CONFIRMED_ID)

    def _reassign_boxes(self, indices, target_id):
        if not isinstance(indices, list):
            indices = [indices]
        idx = sorted({i for i in indices if 0 <= i < len(self.boxes)})
        changed = [(i, self.boxes[i][5]) for i in idx if self.boxes[i][5] != target_id]
        if not changed:
            return
        for i, _prev_class in changed:
            self.boxes[i][5] = target_id
            self.update_box_shape(i, self.boxes[i])
        self._push_undo({"type": "reassign", "changes": changed, "new_class": target_id})
        # Deliberately not forcing target_id into sets_visible here: if the
        # user has Recycle Bin/Confirmed (or any class) hidden, sending
        # more particles into it should not silently pop it back into view.
        self.curbox = -1
        affected_classes = {target_id} | {prev for _, prev in changed}
        if self.initialized:
            self.save_json(only=affected_classes)
        # Reclassifying never moves the viewing cursor, and every other
        # box's inplane/visibility outcome is unchanged, so only the
        # reassigned indices need their shapes recomputed.
        self.update_all(reslice=False, only_indices=[i for i, _ in changed])

    def empty_recycle_bin(self):
        """Permanently purge every particle currently in the Recycle Bin."""
        delids = [i for i, b in enumerate(self.boxes) if b[5] == categories.RECYCLE_BIN_ID]
        self.del_box(delids)

    # -- particle uids / undo -----------------------------------------------
    def _new_uid(self):
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def _index_for_uid(self, uid):
        for i, b in enumerate(self.boxes):
            if b[6] == uid:
                return i
        return None

    def _push_undo(self, action):
        self._undo_stack.append(action)
        # Any genuinely new action invalidates whatever "future" redo would
        # have restored -- standard undo/redo semantics. undo()/redo()
        # themselves push their own mirror entries directly onto the other
        # stack (see below) rather than through here, precisely so that
        # replaying history doesn't wipe out the history it's replaying.
        self._redo_stack = []

    def undo(self):
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        if action["type"] == "add":
            self._undo_add(action)
        elif action["type"] == "delete":
            self._undo_delete(action)
        elif action["type"] == "reassign":
            self._undo_reassign(action)

    def redo(self):
        if not self._redo_stack:
            return
        action = self._redo_stack.pop()
        if action["type"] == "add":
            self._redo_add(action)
        elif action["type"] == "delete":
            self._redo_delete(action)
        elif action["type"] == "reassign":
            self._redo_reassign(action)

    def _undo_add(self, action):
        idx = self._index_for_uid(action["uid"])
        if idx is None:
            return
        box_copy = list(self.boxes[idx])
        img = self.boxesimgs[idx] if idx < len(self.boxesimgs) else None
        self.del_box(idx, record_undo=False)
        # Redoing an "add" is exactly "undoing a delete" of this one
        # particle -- reuse that same (index, box, image) shape so
        # _redo_add can just replay it with _undo_delete's own logic.
        self._redo_stack.append({"type": "add", "uid": action["uid"], "removed": [(idx, box_copy, img)]})

    def _redo_add(self, action):
        self._reinsert_boxes(action["removed"])
        self._undo_stack.append({"type": "add", "uid": action["uid"]})

    def _undo_delete(self, action):
        self._reinsert_boxes(action["removed"])
        # Redo re-deletes the same particles, resolved by uid at redo time
        # rather than by the original index, so it stays correct even if
        # something else happens to shift positions in between.
        self._redo_stack.append({"type": "delete", "uids": [box_copy[6] for _, box_copy, _ in action["removed"]]})

    def _redo_delete(self, action):
        idx = sorted(i for i in (self._index_for_uid(u) for u in action["uids"]) if i is not None)
        if not idx:
            return
        removed = [(i, list(self.boxes[i]), self.boxesimgs[i] if i < len(self.boxesimgs) else None) for i in idx]
        self.del_box(idx, record_undo=False)
        self._undo_stack.append({"type": "delete", "removed": removed})

    def _reinsert_boxes(self, removed):
        """Shared by _undo_delete and _redo_add: reinsert a (orig_index,
        box_copy, img) list (ascending orig_index) back into self.boxes at
        those same positions, and rebuild every shape from scratch, since
        insertion shifts every later box's index."""
        affected_classes = {box_copy[5] for _, box_copy, _ in removed}
        for orig_index, box_copy, img in removed:
            idx = min(orig_index, len(self.boxes))
            self.boxes.insert(idx, list(box_copy))
            self.boxesimgs.insert(idx, img)
        for i in range(len(self.boxes)):
            self.update_box_shape(i, self.boxes[i])
        if self.initialized:
            self.save_json(only=affected_classes)
        self.update_all(reslice=False)

    def _undo_reassign(self, action):
        uids = [self.boxes[i][6] for i, _ in action["changes"] if 0 <= i < len(self.boxes)]
        self._apply_reassign(action["changes"])
        self._redo_stack.append({"type": "reassign", "uids": uids, "new_class": action["new_class"]})

    def _redo_reassign(self, action):
        idx = [i for i in (self._index_for_uid(u) for u in action["uids"]) if i is not None]
        changes = [(i, self.boxes[i][5]) for i in idx if self.boxes[i][5] != action["new_class"]]
        if not changes:
            return
        self._apply_reassign(changes, target_id=action["new_class"])
        self._undo_stack.append({"type": "reassign", "changes": changes, "new_class": action["new_class"]})

    def _apply_reassign(self, changes, target_id=None):
        """Shared by _undo_reassign and _redo_reassign: set each (index,
        class) pair's box to target_id (undo instead restores each box to
        its own recorded prior class, when target_id is None)."""
        affected_classes = {prev for _, prev in changes}
        for i, prev_class in changes:
            if 0 <= i < len(self.boxes):
                self.boxes[i][5] = target_id if target_id is not None else prev_class
                self.update_box_shape(i, self.boxes[i])
        if target_id is not None:
            affected_classes.add(target_id)
        if self.initialized:
            self.save_json(only=affected_classes)
        self.update_all(reslice=False, only_indices=[i for i, _ in changes])

    def update_box_shape(self, n, box):
        bs2 = self.get_boxsize(box[5]) // 2
        lw = 3 if n == self.curbox else 2
        # Gallery selection (yellow, same as the gallery's own highlight)
        # takes priority over the class color -- but only for a particle
        # actually selected in the list, never just because it's the one
        # currently being picked/dragged (curbox/dragging never touch
        # _selected_uids).
        if box[6] in self._selected_uids:
            color = categories.SELECTED_COLOR.getRgbF()[:3]
        else:
            color = categories.class_color(box[5]).getRgbF()[:3]
        if self.options.mode == "3D":
            self.xyview.add_shape(n, Shape("circle", color, [box[0], box[1], bs2], lw))
            self.xzview.add_shape(n, Shape("circle", color, [box[0], box[2], bs2], lw))
            self.zyview.add_shape(n, Shape("circle", color, [box[2], box[1], bs2], lw))
        else:
            self.xyview.add_shape(n, Shape("rect", color, [box[0] - bs2, box[1] - bs2, box[0] + bs2, box[1] + bs2], 2))
            self.xzview.add_shape(n, Shape("rect", color, [box[0] - bs2, box[2] - 1, box[0] + bs2, box[2] + 1], 2))
            self.zyview.add_shape(n, Shape("rect", color, [box[2] - 1, box[1] - bs2, box[2] + 1, box[1] + bs2], 2))

    def set_selected_uids(self, uids):
        """Mirror the gallery's multi-selection as a yellow outline on the
        same particles in the tomogram views. Only the particles whose
        selection state actually changed need their shape rebuilt."""
        new_selected = set(uids or [])
        changed_uids = self._selected_uids ^ new_selected
        self._selected_uids = new_selected
        if not changed_uids:
            return
        changed_indices = [i for i, b in enumerate(self.boxes) if b[6] in changed_uids]
        for i in changed_indices:
            self.update_box_shape(i, self.boxes[i])
        self.update_all(reslice=False, only_indices=changed_indices)

    def update_box(self, n, quiet=False, axis=("x", "y", "z")):
        if n < 0 or n >= len(self.boxes):
            return
        box = self.boxes[n]
        self.update_box_shape(n, box)

        if self.initialized:
            self.update_sliceview(axis)

        if not quiet:
            proj = self.get_cube(box[0], box[1], box[2], centerslice=True, boxsz=self.get_boxsize(box[5]))
            # Keep raw intensities here (not normalize()'d) -- the gallery
            # thumbnail is rendered with the exact same contrast code and
            # levels as the tomogram views (see thumb_grid._thumbnail_pixmap
            # and ParticleGrid.set_levels), so it needs to be on the same
            # intensity scale as them to actually look like a matching crop.
            # Unrelated to save_boxes()'s own normalize() call below, which
            # is about the exported particle stack's data format, not
            # on-screen display.
            proj = proj[0]

            for _ in range(len(self.boxesimgs), n + 1):
                self.boxesimgs.append(None)
            self.boxesimgs[n] = proj

            if self.initialized:
                self.save_json()
                # The gallery thumbnail only needs refreshing when the
                # particle's own snapshot image changes (just above), which
                # only happens here in the non-quiet path -- doing this on
                # every quiet in-drag update was rebuilding the entire
                # thumbnail grid layout on every mouse-move tick for no
                # visible benefit, which is what made dragging feel slow.
                self.update_boximgs()

        self.curbox = n

    def update_boximgs(self):
        self.boxids = [i for i, m in enumerate(self.boxesimgs) if self.boxes[i][5] in self.sets_visible]
        triples = [(self.boxes[i][6], self.boxesimgs[i], self.boxes[i][5]) for i in self.boxids]
        self.boxesviewer.rebuild_tabs(self.sets)
        self.boxesviewer.sync_particles(triples)

    def img_selected(self, event, lc):
        lci = self._index_for_uid(lc[0])
        if lci is None:
            return
        # Quick gestures in the particle list: Shift+click sends this one
        # particle to the Recycle Bin, Shift+Ctrl+click sends it and
        # everything after it in the list; Space+click sends it to
        # Confirmed. (Ctrl/Alt+click without Shift/Space are used for
        # multi-select in the thumbnail grid itself, see
        # ParticleGrid.mousePressEvent.)
        if event.modifiers() & Qt.ShiftModifier:
            if event.modifiers() & Qt.ControlModifier:
                self.send_to_recycle_bin(list(range(lci, len(self.boxes))))
            else:
                self.send_to_recycle_bin(lci)
        elif space_key_state().held:
            self.send_to_confirmed(lci)
        else:
            self.curbox = lci
            box = self.boxes[lci]
            self.x_loc, self.y_loc, self.z_loc = box[0], box[1], box[2]
            self.scroll_to(self.x_loc, self.y_loc, self.z_loc)
            self.update_sliceview()

    def del_region_xy(self, x=-1, y=-1, z=-1, rad=-1):
        if rad < 0:
            rad = self.eraser_width()
        delids = []
        for i, b in enumerate(self.boxes):
            if b[5] not in self.sets_visible:
                continue
            if (x >= 0) * (b[0] - x) ** 2 + (y >= 0) * (b[1] - y) ** 2 + (z >= 0) * (b[2] - z) ** 2 < rad ** 2:
                delids.append(i)
        self.send_to_recycle_bin(delids)

    def scroll_to(self, x, y, z, axis=""):
        if axis != "z":
            self.xyview.scroll_to(x, y, True)
        if axis != "y":
            self.xzview.scroll_to(x, z, True)
        if axis != "x":
            self.zyview.scroll_to(z, y, True)

    # -- mouse click -----------------------------------------------------------
    def xy_down(self, event):
        x, y = self.xyview.scr_to_img((event.position().x(), event.position().y()))
        self.mouse_down(event, x, y, self.z_loc, "z")

    def xz_down(self, event):
        x, z = self.xzview.scr_to_img((event.position().x(), event.position().y()))
        self.mouse_down(event, x, self.y_loc, z, "y")

    def zy_down(self, event):
        z, y = self.zyview.scr_to_img((event.position().x(), event.position().y()))
        self.mouse_down(event, self.x_loc, y, z, "x")

    def mouse_down(self, event, x, y, z, axis):
        if min(x, y, z) < 0:
            return
        if self.optionviewer.erasercheckbox.isChecked():
            self.del_region_xy(x, y, z, -1)
            return

        for i in range(len(self.boxes)):
            if self.inside_box(i, x, y, z):
                if event.modifiers() & Qt.ShiftModifier:
                    self.send_to_recycle_bin(i)
                elif space_key_state().held:
                    self.send_to_confirmed(i)
                else:
                    self.dragging = i
                    self.curbox = i
                    self.scroll_to(x, y, z, axis)
                break
        else:
            if not event.modifiers() & Qt.ShiftModifier:
                self.x_loc, self.y_loc, self.z_loc = x, y, z
                self.scroll_to(x, y, z, axis)
                self.curbox = len(self.boxes)
                uid = self._new_uid()
                self.boxes.append([x, y, z, "manual", 0.0, self.currentset, uid])
                self._push_undo({"type": "add", "uid": uid})
                self.update_box(len(self.boxes) - 1)
                self.dragging = len(self.boxes) - 1

    # -- eraser preview -----------------------------------------------------------
    def xy_move(self, event):
        self.mouse_move(event, self.xyview)

    def xz_move(self, event):
        self.mouse_move(event, self.xzview)

    def zy_move(self, event):
        self.mouse_move(event, self.zyview)

    def mouse_move(self, event, view):
        if self.optionviewer.erasercheckbox.isChecked():
            self.xyview.eraser_shape = self.xzview.eraser_shape = self.zyview.eraser_shape = None
            x, y = view.scr_to_img((event.position().x(), event.position().y()))
            view.eraser_shape = Shape("circle", (1, 1, 1), [x, y, self.eraser_width()], 2)
            view.update()
        else:
            view.eraser_shape = None

    # -- dragging -----------------------------------------------------------
    def mouse_drag(self, x, y, z, skip_axis=None):
        if self.dragging < 0 or min(x, y, z) < 0:
            return
        self.x_loc, self.y_loc, self.z_loc = x, y, z
        self.boxes[self.dragging][:3] = [x, y, z]
        # Whichever view is being dragged in passes its own loc back
        # unchanged (see the three callers below), so that view's own
        # background slice can't have changed -- only its box overlay has.
        # Re-slicing and repainting it anyway on every mouse-move tick
        # while dragging is pure waste; the other two views' slice depths
        # do follow the drag live and still need a full refresh.
        axis = tuple(a for a in ("x", "y", "z") if a != skip_axis)
        self.update_box(self.dragging, True, axis=axis)

    def xy_drag(self, event):
        if self.dragging >= 0:
            x, y = self.xyview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(x, y, self.z_loc, skip_axis="z")

    def xz_drag(self, event):
        if self.dragging >= 0:
            x, z = self.xzview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(x, self.y_loc, z, skip_axis="y")

    def zy_drag(self, event):
        if self.dragging >= 0:
            z, y = self.zyview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(self.x_loc, y, z, skip_axis="x")

    def mouse_up(self, event):
        if self.dragging >= 0:
            self.update_box(self.dragging)
        self.dragging = -1

    # -- keep the same origin across the 3 views ---------------------------------
    def xy_origin(self, newor):
        xzo = self.xzview.get_origin()
        self.xzview.set_origin(newor[0], xzo[1], True)
        zyo = self.zyview.get_origin()
        self.zyview.set_origin(zyo[0], newor[1], True)

    def xz_origin(self, newor):
        xyo = self.xyview.get_origin()
        self.xyview.set_origin(newor[0], xyo[1], True)

    def zy_origin(self, newor):
        xyo = self.xyview.get_origin()
        self.xyview.set_origin(xyo[0], newor[1], True)

    # -- go up/down slices with wheel ---------------------------------------------
    def xy_wheel(self, event):
        z = int(self.z_loc + np.sign(event.angleDelta().y()))
        if 0 < z < self.data.nz:
            self.wdepth.setValue(z)

    def xz_wheel(self, event):
        y = int(self.y_loc + np.sign(event.angleDelta().y()))
        if 0 < y < self.data.ny:
            self.y_loc = y
            self.update_sliceview(("y",))

    def zy_wheel(self, event):
        x = int(self.x_loc + np.sign(event.angleDelta().y()))
        if 0 < x < self.data.nx:
            self.x_loc = x
            self.update_sliceview(("x",))

    # -- sets / classes -----------------------------------------------------------
    def set_current_set(self, name):
        self.currentset = parse_setname(name)
        self.wboxsize.setValue(self.get_boxsize())
        self.update_all()

    def hide_set(self, name):
        name = parse_setname(name)
        self.sets_visible.discard(name)
        if self.initialized:
            self.update_all()

    def show_set(self, name):
        name = parse_setname(name)
        self.sets_visible.add(name)
        if self.initialized:
            self.update_all()

    def delete_set(self, name):
        name = parse_setname(name)
        if name in categories.FIXED_CLASS_IDS:
            return  # Recycle Bin / Confirmed are fixed and can't be deleted
        delids = [i for i, b in enumerate(self.boxes) if b[5] == int(name)]
        self.del_box(delids, record_undo=False)
        self.sets_visible.discard(name)
        classname = self.sets.pop(name, None)
        self.boxsize.pop(name, None)
        if classname is not None:
            metadata.delete_class_file(self.picksdir, classname)
        self.update_all(reslice=False, only_indices=[])

    def toggle_colors(self):
        """Rotate every normal class to the next color in the remaining
        (non-reserved) palette at once, napari-style. Recycle Bin/Confirmed
        and the selection highlight are fixed and unaffected."""
        categories.cycle_colors()
        for i in range(len(self.boxes)):
            self.update_box_shape(i, self.boxes[i])
        self.setspanel.update_sets()
        self.update_all(reslice=False)

    def rename_set(self, oldname, newname):
        name = parse_setname(oldname)
        if name in categories.FIXED_CLASS_IDS:
            return  # Recycle Bin / Confirmed are fixed and can't be renamed
        if name in self.sets:
            metadata.rename_class_file(self.picksdir, self.sets[name], newname)
            self.sets[name] = newname
            if self.initialized:
                self.save_picks()

    def new_set(self, name):
        i = 0
        while i in self.sets:
            i += 1
        self.sets[i] = name
        self.sets_visible.add(i)
        # With "Apply to All Sets" on, every set is kept at the same box
        # size, so a brand-new one should join at that size too rather than
        # reverting to the generic default and breaking that invariant.
        # (hasattr guard: new_set() can be called during __init__'s initial
        # bootstrap, before self.currentset exists yet.)
        if self.wboxsize_allsets.isChecked() and hasattr(self, "currentset"):
            self.boxsize[i] = self.get_boxsize()
        else:
            self.boxsize[i] = metadata.DEFAULT_BOXSIZE
        return i

    def save_set(self):
        self.save_boxes(list(self.sets_visible))

    # -- multi-selection actions (used by the Options window's Selection panel) --
    def delete_particles_by_uid(self, uids):
        idx = [i for i, b in enumerate(self.boxes) if b[6] in uids]
        self.send_to_recycle_bin(idx)

    def confirm_particles_by_uid(self, uids):
        idx = [i for i, b in enumerate(self.boxes) if b[6] in uids]
        self.send_to_confirmed(idx)

    def move_particles_by_uid(self, uids, target_classname):
        """Reassign the given particles (by uid) to target_classname,
        creating the class if it doesn't exist yet. Undo-able."""
        target_id = None
        for k, v in self.sets.items():
            if v == target_classname:
                target_id = k
                break
        if target_id is None:
            target_id = self.new_set(target_classname)

        idx = [i for i, b in enumerate(self.boxes) if b[6] in uids]
        if not idx:
            return
        prev = [(i, self.boxes[i][5]) for i in idx]
        for i in idx:
            self.boxes[i][5] = target_id
            self.update_box_shape(i, self.boxes[i])
        self._push_undo({"type": "reassign", "changes": prev, "new_class": target_id})
        # new_set() above already makes a brand-new class visible; an
        # existing target class that the user deliberately hid should stay
        # hidden even as particles get moved into it.
        affected_classes = {target_id} | {prev for _, prev in prev}
        if self.initialized:
            self.save_picks(only=affected_classes)
        self.update_all(reslice=False, only_indices=idx)

    # -- keyboard -----------------------------------------------------------
    def key_press(self, event):
        if event.key() == 96:  # `
            self.wdepth.setValue(int(round(self.z_loc + 1)))
        elif event.key() == 49:  # 1
            self.wdepth.setValue(int(round(self.z_loc - 1)))
        else:
            self.keyPress.emit(event)

    # -- persistence -----------------------------------------------------------
    # Backward-compatible alias, kept in case anything still calls save_json().
    def save_json(self, only=None):
        self.save_picks(only=only)

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        if not self._confirm_close():
            event.ignore()
            return
        self._do_close_all()
        event.accept()

    def _confirm_close(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Close", "Are you sure you want to close?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return reply == QtWidgets.QMessageBox.Yes

    def _do_close_all(self):
        self._closing = True
        print("Exiting")
        self.save_picks()
        self.boxesviewer.close()
        self.optionviewer.close()
        try:
            self.xyview.close()
            self.xzview.close()
            self.zyview.close()
        except RuntimeError:
            pass

    def request_close_all(self):
        """Called by the particle list / options windows when the user
        tries to close them directly: confirms once, then closes every
        window in the app, including this one."""
        if self._closing:
            return
        if not self._confirm_close():
            return
        self._do_close_all()
        self.close()


class SelectionPanel(QtWidgets.QWidget):
    """Shows how many particles are currently multi-selected in the
    particle list and lets you delete them or move them to another (or a
    new) class. Inert ("No selection") when nothing is selected."""

    def __init__(self, target):
        super().__init__()
        self.target = weakref.ref(target)
        self._uids = []

        vbl = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("No selection")
        vbl.addWidget(self.status_label)

        self.delete_button = QtWidgets.QPushButton("Send to Recycle Bin")
        vbl.addWidget(self.delete_button)

        self.confirm_button = QtWidgets.QPushButton("Send to Confirmed")
        vbl.addWidget(self.confirm_button)

        moverow = QtWidgets.QHBoxLayout()
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.setEditable(True)
        self.class_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        moverow.addWidget(self.class_combo)
        self.move_button = QtWidgets.QPushButton("Move to")
        moverow.addWidget(self.move_button)
        vbl.addLayout(moverow)
        vbl.addStretch(1)

        self.delete_button.clicked.connect(self._delete)
        self.confirm_button.clicked.connect(self._confirm)
        self.move_button.clicked.connect(self._move)

        self._set_active(False)

    def refresh_classes(self):
        target = self.target()
        if target is None:
            return
        current = self.class_combo.currentText()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(sorted(target.sets.values()))
        self.class_combo.setCurrentText(current)
        self.class_combo.blockSignals(False)

    def set_selection(self, uids):
        self._uids = list(uids or [])
        active = len(self._uids) > 0
        self._set_active(active)
        if active:
            self.status_label.setText(f"{len(self._uids)} particle(s) selected")
            self.refresh_classes()
        self._update_delete_label()

    def _update_delete_label(self):
        # Sending an already-recycled particle to the Recycle Bin again
        # permanently deletes it (see TomoBoxer.send_to_recycle_bin) --
        # reflect that on the button so it isn't a surprise.
        target = self.target()
        already_binned = bool(self._uids) and target is not None and all(
            b[5] == categories.RECYCLE_BIN_ID for b in target.boxes if b[6] in self._uids)
        self.delete_button.setText("Delete Permanently" if already_binned else "Send to Recycle Bin")

    def _set_active(self, active):
        self.status_label.setText(f"{len(self._uids)} particle(s) selected" if active else "No selection")
        self.delete_button.setEnabled(active)
        self.confirm_button.setEnabled(active)
        self.move_button.setEnabled(active)
        self.class_combo.setEnabled(active)

    def _delete(self):
        target = self.target()
        if target is None or not self._uids:
            return
        target.delete_particles_by_uid(self._uids)
        self.set_selection([])

    def _confirm(self):
        target = self.target()
        if target is None or not self._uids:
            return
        target.confirm_particles_by_uid(self._uids)
        self.set_selection([])

    def _move(self):
        target = self.target()
        name = self.class_combo.currentText().strip()
        if target is None or not self._uids or not name:
            return
        target.move_particles_by_uid(self._uids, name)
        self.set_selection([])


class TomoBoxerOptions(QtWidgets.QWidget):
    def __init__(self, target):
        super().__init__()
        self.setWindowTitle("Options")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.target = weakref.ref(target)

        gbl = QtWidgets.QGridLayout(self)
        self.erasercheckbox = QtWidgets.QCheckBox("Eraser")
        gbl.addWidget(self.erasercheckbox, 0, 0)

        self.eraser_radius = LabeledSpinBox(label="Radius:", value=128, minimum=1, maximum=8192)
        gbl.addWidget(self.eraser_radius, 0, 1)

        self.tabwidget = QtWidgets.QTabWidget()
        gbl.addWidget(self.tabwidget, 1, 0, 1, 2)

    def add_panel(self, widget, name):
        self.tabwidget.addTab(widget, name)

    def closeEvent(self, event):
        target = self.target()
        if target is not None and not getattr(target, "_closing", False):
            event.ignore()
            target.request_close_all()
            return
        event.accept()


class TomoSetsPanel(QtWidgets.QWidget):
    """The set (class) display / management panel."""

    def __init__(self, target):
        super().__init__()
        self.target = weakref.ref(target)
        self.initialized = False

        hbl = QtWidgets.QHBoxLayout(self)
        self.setlist = QtWidgets.QListWidget()
        self.setlist.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        hbl.addWidget(self.setlist)

        vbl = QtWidgets.QVBoxLayout()
        self.new_set_button = QtWidgets.QPushButton("New")
        vbl.addWidget(self.new_set_button)
        self.rename_set_button = QtWidgets.QPushButton("Rename")
        vbl.addWidget(self.rename_set_button)
        self.save_set_button = QtWidgets.QPushButton("Save")
        vbl.addWidget(self.save_set_button)
        self.delete_set_button = QtWidgets.QPushButton("Delete")
        vbl.addWidget(self.delete_set_button)
        self.empty_recycle_button = QtWidgets.QPushButton("Empty Recycle Bin")
        vbl.addWidget(self.empty_recycle_button)
        self.toggle_colors_button = QtWidgets.QPushButton("Toggle Colors")
        vbl.addWidget(self.toggle_colors_button)
        hbl.addLayout(vbl)

        self.save_set_button.clicked.connect(self.save_set)
        self.new_set_button.clicked.connect(self.new_set)
        self.rename_set_button.clicked.connect(self.rename_set)
        self.delete_set_button.clicked.connect(self.delete_set)
        self.empty_recycle_button.clicked.connect(self.empty_recycle_bin)
        self.toggle_colors_button.clicked.connect(self.toggle_colors)
        self.setlist.itemChanged.connect(self.set_list_item_changed)
        self.setlist.currentRowChanged.connect(self.set_list_row_changed)

    def toggle_colors(self):
        target = self.target()
        if target is None:
            return
        target.toggle_colors()  # this also refreshes update_sets() for us

    def set_list_row_changed(self, i):
        if not self.initialized:
            return
        item = self.setlist.item(i)
        if item is None:
            return
        self.target().set_current_set(item.text())

    def set_list_item_changed(self, item):
        name = item.text()
        if item.checkState() == Qt.Checked:
            self.target().show_set(name)
        else:
            self.target().hide_set(name)

    def delete_set(self, unused=None):
        selections = self.setlist.selectedItems()
        if not selections:
            return
        names = [i.text() for i in selections]
        if parse_setname(names[0]) in categories.FIXED_CLASS_IDS:
            QtWidgets.QMessageBox.information(
                self, "Delete set", "Recycle Bin and Confirmed are fixed categories and can't be deleted.")
            return
        cancel = QtWidgets.QMessageBox.question(
            self, "Delete set",
            f"Are you sure to delete {names[0]}? This will remove all particles in that class",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if cancel == QtWidgets.QMessageBox.Yes:
            self.target().delete_set(names[0])
            self.update_sets()

    def empty_recycle_bin(self, unused=None):
        target = self.target()
        count = sum(1 for b in target.boxes if b[5] == categories.RECYCLE_BIN_ID)
        if count == 0:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Empty Recycle Bin",
            f"Permanently delete {count} particle(s) in the Recycle Bin? (Undo-able with Ctrl+Z right after.)",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            target.empty_recycle_bin()
            self.update_sets()

    def new_set(self, unused=None):
        name, ok = QtWidgets.QInputDialog.getText(self, "Set Name", "Enter a name for the new set:")
        if not ok:
            return
        name = str(name)
        if name in self.target().sets.values():
            print("Set name exists")
            return
        self.target().new_set(name)
        self.update_sets()

    def rename_set(self, unused=None):
        selections = self.setlist.selectedItems()
        if not selections:
            return
        if parse_setname(selections[0].text()) in categories.FIXED_CLASS_IDS:
            QtWidgets.QMessageBox.information(
                self, "Rename set", "Recycle Bin and Confirmed are fixed categories and can't be renamed.")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Set Name", "Enter a name for the new set:")
        if not ok:
            return
        name = str(name)
        if name in self.target().sets.values():
            print("Set name exists")
            return
        self.target().rename_set(selections[0].text(), name)
        self.update_sets()

    def save_set(self):
        self.target().save_set()
        self.update_sets()

    def update_sets(self):
        target = self.target()
        keys = categories.category_order(target.sets)
        viskeys = set(target.sets_visible)
        self.setlist.blockSignals(True)
        self.setlist.clear()

        for k in keys:
            kname = f"{int(k):02d} :: {target.sets[k]}"
            item = QtWidgets.QListWidgetItem(kname)
            item.setFlags(Qt.ItemIsEditable | Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setForeground(QtGui.QBrush(categories.class_color(k)))
            self.setlist.addItem(item)
            item.setCheckState(Qt.Checked if k in viskeys else Qt.Unchecked)
            if not self.initialized and k == target.currentset:
                self.setlist.setCurrentItem(item)
                self.initialized = True
        self.setlist.blockSignals(False)
