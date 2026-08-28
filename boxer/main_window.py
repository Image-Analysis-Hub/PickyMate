"""Standalone tomogram particle boxer main window.

This is a functional port of EMAN2's e2spt_boxer.py (EMTomoBoxer) that runs
outside the EMAN2 environment, built on plain PySide6 + numpy + mrcfile.

Feature-complete for v1: orthogonal XY/XZ/ZY slice viewing, 2D or 3D particle
picking with mouse (move/drag/shift-delete), an eraser tool, multiple
picking "sets" (classes) with visibility toggles, per-class box size, slab
thickness averaging, a lowpass filter preview, saving/loading picks to a JSON
sidecar next to the tomogram, saving picks as a plain text coordinate file,
and saving extracted particles as an MRC stack (2D) or a folder of MRC
volumes (3D).

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
from .widgets import LabeledSpinBox, LabeledSlider
from . import metadata
from .metadata import base_name
from .shapes import Shape
from .volume import Volume, normalize, ParticleStackWriter

SETCOLORS = [
    QtGui.QColor("blue"), QtGui.QColor("green"), QtGui.QColor("red"),
    QtGui.QColor("cyan"), QtGui.QColor("purple"), QtGui.QColor("orange"),
    QtGui.QColor("yellow"), QtGui.QColor("hotpink"), QtGui.QColor("gold"),
]


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

        self.wboxsize = LabeledSpinBox(label="Box Size:", value=0, minimum=1, maximum=8192)
        self.gbl2.addWidget(self.wboxsize, 1, 0)

        label0 = QtWidgets.QLabel("Thickness")
        self.gbl2.addWidget(label0, 2, 0)
        self.wnlayers = QtWidgets.QSpinBox()
        self.wnlayers.setRange(1, 256)
        self.wnlayers.setValue(1)
        self.gbl2.addWidget(self.wnlayers, 2, 1)

        self.wlocalbox = QtWidgets.QCheckBox("Limit Side Boxes")
        self.wlocalbox.setChecked(True)
        self.gbl2.addWidget(self.wlocalbox, 3, 0)

        self.wfilt = LabeledSlider(rng=(0, 150), label="Filt", value=0)
        self.gbl2.addWidget(self.wfilt, 4, 0, 1, 2)

        self.curbox = -1
        self.boxes = []       # [x, y, z, method, score, class_id, uid]
        self.boxesimgs = []
        self.dragging = -1
        self._next_uid = 0
        self._undo_stack = []

        # -- wiring -----------------------------------------------------------
        self.mfile_read_boxloc.triggered.connect(self.menu_file_read_boxloc)
        self.mfile_save_boxloc.triggered.connect(self.menu_file_save_boxloc)
        self.mfile_save_boxes_stack.triggered.connect(self.save_boxes)

        self.wdepth.valueChanged[int].connect(self.event_depth)
        self.wnlayers.valueChanged[int].connect(self.event_nlayers)
        self.wboxsize.valueChanged.connect(self.event_boxsize)
        self.wfilt.valueChanged.connect(self.event_filter)
        self.wlocalbox.stateChanged[int].connect(self.event_localbox)

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
        self.datalp = None
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

        self._arrange_windows()

        # -- metadata (picks + classes) ------------------------------------------
        # self.sets: class_id (int) -> class name (str), the name doubling
        # as the "<name>.txt" filename inside self.picksdir.
        self.sets = {}
        self.boxsize = {}

        boxsizes_by_name = metadata.load_boxsizes(self.picksdir)
        classnames = metadata.discover_classes(self.picksdir)

        self.sets_visible = set()

        for i, name in enumerate(classnames):
            self.sets[i] = name
            self.boxsize[i] = boxsizes_by_name.get(name, metadata.DEFAULT_BOXSIZE)
            for x, y, z in metadata.load_class_coords(self.picksdir, name):
                bdf = [x, y, z, "manual", 0.0, i, self._new_uid()]
                self.boxes.append(bdf)

        shape = [self.data.nx, self.data.ny, self.data.nz]
        for b in self.boxes:
            for i in range(3):
                b[i] = int(np.clip(round(b[i]), 1, shape[i] - 1))

        if len(self.sets) == 0:
            self.new_set("particles_00")

        self.currentset = sorted(self.sets.keys())[0]
        if options.label:
            sid = [i for i in self.sets if self.sets[i] == options.label]
            if sid:
                self.currentset = sid[0]
            else:
                print("cannot find specified label")

        # All classes are visible by default when the tomogram is opened.
        self.sets_visible.update(self.sets.keys())
        self.setspanel.update_sets()
        self.wboxsize.setValue(self.get_boxsize())

        if options.clean:
            self.optionviewer.erasercheckbox.setChecked(True)
            self.wlocalbox.setChecked(False)

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
        self.datalp = data
        self.apix = data.apix
        self.datasize = (data.nx, data.ny, data.nz)
        self.x_loc, self.y_loc, self.z_loc = data.nx // 2, data.ny // 2, data.nz // 2

        self.wdepth.setRange(0, data.nz - 1)
        self.wdepth.setValue(data.nz // 2)
        self.boxes = []
        self.curbox = -1
        self._next_uid = 0
        self._undo_stack = []

        if self.initialized:
            self.update_all()

    def eraser_width(self):
        return int(self.optionviewer.eraser_radius.getValue())

    def get_cube(self, x, y, z, centerslice=False, boxsz=-1):
        bs = self.get_boxsize() if boxsz < 0 else boxsz
        bz = 1 if centerslice else bs
        return self.data.get_clip_centered(x, y, z, bs, bs, bz)

    def get_slice(self, idx, thk=1, axis="z"):
        data = self.datalp if self.wfilt.getValue() != 0.0 else self.data
        return data.slab_projection(axis, idx, int(thk - 1))

    # -- box size / thickness / filter events -----------------------------------
    def event_boxsize(self):
        if self.get_boxsize() == int(self.wboxsize.getValue()):
            return
        self.boxsize[self.currentset] = int(self.wboxsize.getValue())
        self.initialized = False
        for i in range(len(self.boxes)):
            if self.boxes[i][5] == self.currentset:
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

    def event_filter(self):
        if self.wfilt.getValue() != 0.0:
            print("Filtering tomogram...")
            self.datalp = self.data.lowpass(1.0 / self.wfilt.getValue())
        else:
            self.datalp = self.data
        self.update_all()

    def event_localbox(self, tog):
        self.update_sliceview()

    def get_boxsize(self, clsid=-1):
        if clsid < 0:
            return int(self.boxsize[self.currentset])
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

    def save_picks(self):
        """Write every class's particles to its own "<name>.txt" file in
        self.picksdir, and update config.settings with any non-default box
        sizes. This is the sole persistence mechanism (no JSON)."""
        for clsid, name in self.sets.items():
            coords = [(b[0], b[1], b[2]) for b in self.boxes if b[5] == clsid]
            metadata.save_class_coords(self.picksdir, name, coords)
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
    def update_sliceview(self, axis=("x", "y", "z")):
        boxes = self.boxes
        allside = not self.wlocalbox.isChecked()

        pms = {
            "z": (2, self.xyview, self.z_loc),
            "y": (1, self.xzview, self.y_loc),
            "x": (0, self.zyview, self.x_loc),
        }

        lwi = 2 if self.boxshape == "circle" else 3  # index into geom for linewidth handled separately

        for v in (self.xyview, self.xzview, self.zyview):
            v.del_shape("xline")
            v.del_shape("yline")

        for ax in axis:
            ia, view, loc = pms[ax]
            for i, b in enumerate(boxes):
                bs = self.get_boxsize(b[5])
                dst = abs(b[ia] - loc)
                inplane = dst < bs // 2
                rad = bs // 2 - dst

                if allside:
                    inplane = True
                    rad = bs // 2

                if ax == "z" and self.options.mode == "2D":
                    inplane = dst < 1

                shp = view.get_shapes().get(i)
                if shp is None:
                    continue
                if inplane and (b[5] in self.sets_visible):
                    shp.kind = self.boxshape
                    shp.linewidth = 3 if self.curbox == i else 2
                    if self.options.mode == "3D":
                        shp.geom[2] = rad
                else:
                    shp.kind = "hidden"

            img = self.get_slice(loc, self.nlayers(), ax)
            view.set_data(img, keepcontrast=not self.wbautoc.isChecked())

        self.update_coords()

    def update_all(self):
        if self.data is None:
            return
        self.update_sliceview()
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
        if record_undo:
            removed = [(i, list(self.boxes[i]), self.boxesimgs[i] if i < len(self.boxesimgs) else None) for i in delids]
            self._push_undo({"type": "delete", "removed": removed})
        kpids = [i for i in range(len(self.boxes)) if i not in delids]
        self.boxes = [self.boxes[i] for i in kpids]
        self.boxesimgs = [self.boxesimgs[i] for i in kpids if i < len(self.boxesimgs)]
        for view in (self.xyview, self.xzview, self.zyview):
            view.shapes = {i: view.shapes[k] for i, k in enumerate(kpids) if k in view.shapes}
        self.curbox = -1
        if self.initialized:
            self.save_json()
        self.update_all()

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

    def _undo_add(self, action):
        idx = self._index_for_uid(action["uid"])
        if idx is None:
            return
        self.del_box(idx, record_undo=False)

    def _undo_delete(self, action):
        for orig_index, box_copy, img in action["removed"]:
            idx = min(orig_index, len(self.boxes))
            self.boxes.insert(idx, list(box_copy))
            self.boxesimgs.insert(idx, img)
        for i in range(len(self.boxes)):
            self.update_box_shape(i, self.boxes[i])
        if self.initialized:
            self.save_json()
        self.update_all()

    def _undo_reassign(self, action):
        for i, prev_class in action["changes"]:
            if 0 <= i < len(self.boxes):
                self.boxes[i][5] = prev_class
                self.update_box_shape(i, self.boxes[i])
        if self.initialized:
            self.save_json()
        self.update_all()

    def update_box_shape(self, n, box):
        bs2 = self.get_boxsize(box[5]) // 2
        lw = 3 if n == self.curbox else 2
        color = SETCOLORS[box[5] % len(SETCOLORS)].getRgbF()[:3]
        if self.options.mode == "3D":
            self.xyview.add_shape(n, Shape("circle", color, [box[0], box[1], bs2], lw))
            self.xzview.add_shape(n, Shape("circle", color, [box[0], box[2], bs2], lw))
            self.zyview.add_shape(n, Shape("circle", color, [box[2], box[1], bs2], lw))
        else:
            self.xyview.add_shape(n, Shape("rect", color, [box[0] - bs2, box[1] - bs2, box[0] + bs2, box[1] + bs2], 2))
            self.xzview.add_shape(n, Shape("rect", color, [box[0] - bs2, box[2] - 1, box[0] + bs2, box[2] + 1], 2))
            self.zyview.add_shape(n, Shape("rect", color, [box[2] - 1, box[1] - bs2, box[2] + 1, box[1] + bs2], 2))

    def update_box(self, n, quiet=False):
        if n < 0 or n >= len(self.boxes):
            return
        box = self.boxes[n]
        self.update_box_shape(n, box)

        if self.initialized:
            self.update_sliceview()

        if not quiet:
            proj = self.get_cube(box[0], box[1], box[2], centerslice=True, boxsz=self.get_boxsize(box[5]))
            proj = normalize(proj[0])

            for _ in range(len(self.boxesimgs), n + 1):
                self.boxesimgs.append(None)
            self.boxesimgs[n] = proj

            if self.initialized:
                self.save_json()

        if self.initialized:
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
        # Quick-delete gesture in the particle list: Shift+click deletes
        # this one particle, Shift+Ctrl+click deletes it and everything
        # after it in the list. (Ctrl/Alt+click without Shift are used for
        # multi-select in the thumbnail grid itself, see
        # ParticleGrid.mousePressEvent.)
        if event.modifiers() & Qt.ShiftModifier:
            if event.modifiers() & Qt.ControlModifier:
                self.del_box(list(range(lci, len(self.boxes))))
            else:
                self.del_box(lci)
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
        self.del_box(delids)

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
            side = self.wlocalbox.isChecked()
            xyz = {"x": x, "y": y, "z": z}
            if not side:
                xyz[axis] = -1
            self.del_region_xy(xyz["x"], xyz["y"], xyz["z"], -1)
            return

        for i in range(len(self.boxes)):
            if self.inside_box(i, x, y, z):
                if event.modifiers() & Qt.ShiftModifier:
                    self.del_box(i)
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
    def mouse_drag(self, x, y, z):
        if self.dragging < 0 or min(x, y, z) < 0:
            return
        self.x_loc, self.y_loc, self.z_loc = x, y, z
        self.boxes[self.dragging][:3] = [x, y, z]
        self.update_box(self.dragging, True)

    def xy_drag(self, event):
        if self.dragging >= 0:
            x, y = self.xyview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(x, y, self.z_loc)

    def xz_drag(self, event):
        if self.dragging >= 0:
            x, z = self.xzview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(x, self.y_loc, z)

    def zy_drag(self, event):
        if self.dragging >= 0:
            z, y = self.zyview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(self.x_loc, y, z)

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
        delids = [i for i, b in enumerate(self.boxes) if b[5] == int(name)]
        self.del_box(delids, record_undo=False)
        self.sets_visible.discard(name)
        classname = self.sets.pop(name, None)
        self.boxsize.pop(name, None)
        if classname is not None:
            metadata.delete_class_file(self.picksdir, classname)
        self.update_all()

    def rename_set(self, oldname, newname):
        name = parse_setname(oldname)
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
        self.boxsize[i] = metadata.DEFAULT_BOXSIZE
        return i

    def save_set(self):
        self.save_boxes(list(self.sets_visible))

    # -- multi-selection actions (used by the Options window's Selection panel) --
    def delete_particles_by_uid(self, uids):
        idx = [i for i, b in enumerate(self.boxes) if b[6] in uids]
        self.del_box(idx)

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
        self.sets_visible.add(target_id)
        if self.initialized:
            self.save_picks()
        self.update_all()

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
    def save_json(self):
        self.save_picks()

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

        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        vbl.addWidget(self.delete_button)

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

    def _set_active(self, active):
        self.status_label.setText(f"{len(self._uids)} particle(s) selected" if active else "No selection")
        self.delete_button.setEnabled(active)
        self.move_button.setEnabled(active)
        self.class_combo.setEnabled(active)

    def _delete(self):
        target = self.target()
        if target is None or not self._uids:
            return
        target.delete_particles_by_uid(self._uids)
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
        hbl.addLayout(vbl)

        self.save_set_button.clicked.connect(self.save_set)
        self.new_set_button.clicked.connect(self.new_set)
        self.rename_set_button.clicked.connect(self.rename_set)
        self.delete_set_button.clicked.connect(self.delete_set)
        self.setlist.itemChanged.connect(self.set_list_item_changed)
        self.setlist.currentRowChanged.connect(self.set_list_row_changed)

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
        cancel = QtWidgets.QMessageBox.question(
            self, "Delete set",
            f"Are you sure to delete {names[0]}? This will remove all particles in that class",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if cancel == QtWidgets.QMessageBox.Yes:
            self.target().delete_set(names[0])
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
        keys = sorted(target.sets.keys())
        viskeys = set(target.sets_visible)
        self.setlist.blockSignals(True)
        self.setlist.clear()

        for i, k in enumerate(keys):
            kname = f"{int(k):02d} :: {target.sets[k]}"
            item = QtWidgets.QListWidgetItem(kname)
            item.setFlags(Qt.ItemIsEditable | Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setForeground(QtGui.QBrush(SETCOLORS[i % len(SETCOLORS)]))
            self.setlist.addItem(item)
            item.setCheckState(Qt.Checked if k in viskeys else Qt.Unchecked)
            if not self.initialized and k == target.currentset:
                self.setlist.setCurrentItem(item)
                self.initialized = True
        self.setlist.blockSignals(False)
