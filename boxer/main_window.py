"""Standalone tomogram particle boxer main window.

This is a functional port of EMAN2's e2spt_boxer.py (EMTomoBoxer) that runs
outside the EMAN2 environment, built on plain PySide6 + numpy + mrcfile.

Feature-complete for v1: orthogonal XY/XZ/ZY slice viewing, 2D or 3D particle
picking with mouse (move/drag/shift-delete), an eraser tool, multiple
picking "sets" (classes) with visibility toggles, per-class box size, slab
thickness averaging, autosaving picks as one plain text coordinate file per
class next to the tomogram, and importing/exporting copies of those files
from the File menu.

Deferred to a later pass (noted in the code where relevant): PCA-based
tomogram "flatten" re-orientation, GIF export, and window-geometry
persistence.
"""
from __future__ import annotations

import os
import shutil
import weakref

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .help_dialog import ShortcutsDialog
from .image_view import ImageView2D
from .thumb_grid import ParticleListPanel
from .widgets import LabeledSpinBox, RangeSlider, TOOLBAR_BUTTON_STYLE, space_key_state
from . import categories
from . import metadata
from .shapes import Shape
from .volume import Volume

# Shared by the shape-select tool (rectangle/circle/freeform) below: index
# of each axis into a box's [x, y, z, ...] entry, and which two axes are
# "in-plane" for a given view's depth axis -- matching the exact (a1, a2)
# order update_box_shape already uses when building each view's own Shape
# geometry (xyview: x,y; xzview: x,z; zyview: z,y).
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_VIEW_PLANE_AXES = {"z": ("x", "y"), "y": ("x", "z"), "x": ("z", "y")}
_SELECT_PREVIEW_COLOR = (1.0, 1.0, 1.0)  # white -- distinct from every reserved/class color
_SELECT_CLOSE_PIXELS = 8  # screen-pixel radius for "clicked near the first freeform point"


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
        self.mfile_save_all = self.mfile.addAction("Save")
        self.mfile.addSeparator()
        self.mfile_read_boxloc = self.mfile.addAction("Read Box Coord")
        self.mfile_save_boxloc = self.mfile.addAction("Save Box Coord")

        # Populated further down, once undo()/redo() and the undo/redo
        # stacks exist -- created here so File/Edit/Help land in that order
        # on the menu bar regardless.
        self.medit = self.menuBar().addMenu("Edit")

        self._shortcuts_dialog = None
        self.mhelp = self.menuBar().addMenu("Help")
        self.mhelp_shortcuts_action = self.mhelp.addAction("Keyboard Shortcuts...")
        self.mhelp_shortcuts_action.setShortcut(QtGui.QKeySequence("F1"))
        self.mhelp_shortcuts_action.setShortcutContext(Qt.ApplicationShortcut)
        self.mhelp_shortcuts_action.triggered.connect(self.show_shortcuts_dialog)

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
        self.wbautoc.setToolTip(
            "Continuously auto-stretch contrast per slice/thumbnail. "
            "Turned off automatically when you drag the Contrast slider.")
        gbl3.addWidget(self.wbautoc)

        self.wcontrast = RangeSlider(minimum=0.0, maximum=1.0, low=0.0, high=1.0)
        self.wcontrast.setToolTip(
            "Manual black/white display levels. Dragging a handle switches "
            "off AutoContrast; the new levels apply once you release the "
            "mouse, not while dragging. Affects the tomogram views and "
            "every gallery thumbnail identically.")
        self.gbl2.addWidget(self.wcontrast, 1, 0, 1, 2)

        self.wboxsize = LabeledSpinBox(label="Box Size:", value=0, minimum=1, maximum=8192)
        self.wboxsize.setToolTip(
            "Box size in pixels. Applies once you press Enter, click "
            "elsewhere, or use the arrows -- not on every keystroke.")
        self.gbl2.addWidget(self.wboxsize, 2, 0)

        self.wboxsize_allsets = QtWidgets.QCheckBox("Apply to All Sets")
        self.wboxsize_allsets.setChecked(True)
        self.wboxsize_allsets.setToolTip(
            "Keep every set at the same box size, and size a newly created "
            "set to match. Checking this also immediately resizes every "
            "set to the number currently shown, even if you don't change it.")
        self.gbl2.addWidget(self.wboxsize_allsets, 2, 1)

        label0 = QtWidgets.QLabel("Avg Slices:")
        label0.setToolTip(
            "Averages this many slices, centered on the current depth, into "
            "each view -- reduces noise without changing overall brightness "
            "(it's a mean, not a sum).")
        self.gbl2.addWidget(label0, 3, 0)
        self.wnlayers = QtWidgets.QSpinBox()
        self.wnlayers.setRange(1, 256)
        self.wnlayers.setValue(1)
        self.wnlayers.setToolTip(label0.toolTip())
        self.gbl2.addWidget(self.wnlayers, 3, 1)

        self.curbox = -1
        self.boxes = []       # [x, y, z, method, score, class_id, uid]
        self.boxesimgs = []
        self.dragging = -1
        self._select_drawing = None  # in-progress shape-select gesture, or None
        self._next_uid = 0
        self._undo_stack = []
        self._redo_stack = []

        # -- wiring -----------------------------------------------------------
        self.mfile_save_all.triggered.connect(self.menu_file_save_all)
        self.mfile_read_boxloc.triggered.connect(self.menu_file_read_boxloc)
        self.mfile_save_boxloc.triggered.connect(self.menu_file_save_boxloc)

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
        self.xyview.mousedoubleclick.connect(self.xy_doubleclick)
        self.xyview.signal_set_scale.connect(self.event_scale)
        self.xyview.origin_update.connect(self.xy_origin)

        self.xzview.mousedown.connect(self.xz_down)
        self.xzview.mousedrag.connect(self.xz_drag)
        self.xzview.mouseup.connect(self.mouse_up)
        self.xzview.mousewheel.connect(self.xz_wheel)
        self.xzview.mousedoubleclick.connect(self.xz_doubleclick)
        self.xzview.signal_set_scale.connect(self.event_scale)
        self.xzview.origin_update.connect(self.xz_origin)
        self.xzview.mousemove.connect(self.xz_move)

        self.zyview.mousedown.connect(self.zy_down)
        self.zyview.mousedrag.connect(self.zy_drag)
        self.zyview.mouseup.connect(self.mouse_up)
        self.zyview.mousewheel.connect(self.zy_wheel)
        self.zyview.mousedoubleclick.connect(self.zy_doubleclick)
        self.zyview.signal_set_scale.connect(self.event_scale)
        self.zyview.origin_update.connect(self.zy_origin)
        self.zyview.mousemove.connect(self.zy_move)

        self.xyview.keypress.connect(self.key_press)
        # Freeform select's Enter/Escape/Backspace need to work no matter
        # which of the 3 views the polygon is being drawn in, not just the
        # main XY one.
        self.xzview.keypress.connect(self.key_press)
        self.zyview.keypress.connect(self.key_press)

        # -- tool selector: Pick / Eraser / Select ---------------------------
        # These are mutually exclusive *tools* (Photoshop/GIMP-style), not
        # independent settings -- exactly one is ever active, and each has
        # its own small set of options that only matter while it's the
        # active one. QActionGroup gives the exclusivity for free (checking
        # one unchecks the others) instead of the manual cross-wiring a set
        # of independent checkboxes would need.
        self.tool_pick_action = QtGui.QAction("Pick", self)
        self.tool_pick_action.setCheckable(True)
        self.tool_pick_action.setChecked(True)
        self.tool_pick_action.setToolTip(
            "Normal picking: click empty space to add, click + drag an "
            "existing particle to move it. Shift/Space/Ctrl+click still "
            "work for Recycle Bin/Confirmed/multi-select.")

        self.tool_eraser_action = QtGui.QAction("Eraser", self)
        self.tool_eraser_action.setCheckable(True)
        self.tool_eraser_action.setToolTip(
            "Click-drag in a tomogram view to send every particle within "
            "Radius of the cursor to the Recycle Bin.")

        self.tool_select_action = QtGui.QAction("Select", self)
        self.tool_select_action.setCheckable(True)
        self.tool_select_action.setToolTip(
            "Draw a shape in a tomogram view to select every particle "
            "inside it (replacing the current selection), instead of "
            "picking/moving particles. Use Confirm/Recycle Selected or "
            "Move Selected to act on the result.")

        self.tool_group = QtGui.QActionGroup(self)
        self.tool_group.setExclusive(True)
        for a in (self.tool_pick_action, self.tool_eraser_action, self.tool_select_action):
            self.tool_group.addAction(a)
        self.tool_group.triggered.connect(self._on_tool_changed)

        # Per-tool options: all shown at fixed positions all the time,
        # simply grayed out unless their tool is active -- rather than a
        # QStackedWidget that swaps a differently-sized page in per tool.
        # That was tried first, but QToolBar's own layout only computes
        # each added widget's position once, from its sizeHint() at the
        # time it was added; it does not reflow later items when an
        # earlier one's size changes afterwards the way an ordinary layout
        # would, and letting the toolbar's total width change every time
        # the tool changed could make the whole row overflow and rearrange
        # unpredictably. Nothing here ever changes size or position, so
        # that whole class of problem doesn't come up: switching tools
        # only ever flips enabled states in place.
        self.eraser_radius = LabeledSpinBox(label="Radius:", value=128, minimum=1, maximum=8192)
        self.eraser_radius.setToolTip(
            "Eraser radius in pixels. Applies once you press Enter, click "
            "elsewhere, or use the arrows -- not on every keystroke.")
        self.eraser_radius.setEnabled(False)

        self.select_shape_rect = QtWidgets.QRadioButton("Rectangle")
        self.select_shape_circle = QtWidgets.QRadioButton("Circle")
        self.select_shape_freeform = QtWidgets.QRadioButton("Freeform")
        self.select_shape_rect.setChecked(True)
        self.select_shape_group = QtWidgets.QButtonGroup(self)
        for rb in (self.select_shape_rect, self.select_shape_circle, self.select_shape_freeform):
            self.select_shape_group.addButton(rb)
            rb.setEnabled(False)
            # Switching shape mid-drag is ambiguous, so cancel whatever
            # shape-select was in progress rather than apply the new choice
            # retroactively to an old gesture.
            rb.toggled.connect(self._select_cancel_drawing)

        # One shared 2D/3D toggle rather than a separate one per tool: it's
        # the same concept either way (only affect what's visible on the
        # current slice, or not), and sharing it means switching tools
        # doesn't silently reset a choice you already made. Meaningless for
        # Pick, so disabled while Pick is active.
        self.tool_3d_checkbox = QtWidgets.QCheckBox("3D")
        self.tool_3d_checkbox.setToolTip(
            "2D: only affects/selects particles actually visible on the "
            "current slice. 3D: Eraser affects a full sphere of Radius "
            "regardless of depth; Select drops the depth limit entirely, "
            "extruding the shape (cuboid/cylinder/tube) through the whole "
            "volume.")
        self.tool_3d_checkbox.setEnabled(False)
        self.tool_3d_checkbox.toggled.connect(self._select_cancel_drawing)

        self.tools_toolbar = self.addToolBar("Tools")
        self.tools_toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.tools_toolbar.addAction(self.tool_pick_action)
        self.tools_toolbar.addAction(self.tool_eraser_action)
        self.tools_toolbar.addAction(self.tool_select_action)
        self.tools_toolbar.addSeparator()
        self.tools_toolbar.addWidget(self.eraser_radius)
        self.tools_toolbar.addWidget(self.select_shape_rect)
        self.tools_toolbar.addWidget(self.select_shape_circle)
        self.tools_toolbar.addWidget(self.select_shape_freeform)
        self.tools_toolbar.addWidget(self.tool_3d_checkbox)

        # QAction rather than a bare QShortcut for both: it gives Undo/Redo
        # a visible Edit-menu entry (with the shortcut shown alongside it
        # for free) and a toolbar button, in addition to still responding
        # to the keyboard shortcut -- one object serves all three.
        self.undo_action = QtGui.QAction("Undo", self)
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)
        self.undo_action.setShortcutContext(Qt.ApplicationShortcut)
        self.undo_action.setToolTip("Undo (Ctrl+Z)")
        self.undo_action.triggered.connect(self.undo)

        # QKeySequence.Redo resolves to whatever the desktop theme defines
        # as "the" redo key -- on Linux (GNOME/KDE) that's normally
        # Ctrl+Shift+Z, not Ctrl+Y (the Windows convention), so relying on
        # just QKeySequence.Redo left Ctrl+Y doing nothing. Merge Ctrl+Y
        # into the same action's key list (rather than a second action for
        # it) so it works everywhere without ever colliding with the native
        # binding as an "ambiguous shortcut" on platforms where they're
        # already the same.
        redo_keys = QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.Redo)
        ctrl_y = QtGui.QKeySequence("Ctrl+Y")
        if ctrl_y not in redo_keys:
            redo_keys.append(ctrl_y)
        self.redo_action = QtGui.QAction("Redo", self)
        self.redo_action.setShortcuts(redo_keys)
        self.redo_action.setShortcutContext(Qt.ApplicationShortcut)
        self.redo_action.setToolTip("Redo (Ctrl+Y / Ctrl+Shift+Z)")
        self.redo_action.triggered.connect(self.redo)

        self.medit.addAction(self.undo_action)
        self.medit.addAction(self.redo_action)

        self.confirm_selected_action = QtGui.QAction("Confirm Selected", self)
        self.confirm_selected_action.setToolTip("Send every currently selected particle to Confirmed.")
        self.confirm_selected_action.triggered.connect(lambda: self.confirm_particles_by_uid(self._selected_uids))

        self.recycle_selected_action = QtGui.QAction("Recycle Selected", self)
        self.recycle_selected_action.setToolTip("Send every currently selected particle to the Recycle Bin.")
        self.recycle_selected_action.triggered.connect(lambda: self.delete_particles_by_uid(self._selected_uids))

        # Editable so you can type a brand-new set name too, not just pick
        # an existing one (matches move_particles_by_uid, which creates the
        # class if it doesn't exist yet).
        self.move_to_combo = QtWidgets.QComboBox()
        self.move_to_combo.setEditable(True)
        self.move_to_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.move_to_combo.setMinimumWidth(120)
        self.move_to_combo.setToolTip("Type an existing or new set name, then Move Selected to.")
        # A placeholder rather than leaving it blank -- an empty editable
        # combo box otherwise gives no visual hint that there's an input
        # here at all, as opposed to just dead space next to the button.
        self.move_to_combo.lineEdit().setPlaceholderText("..........")

        self.move_selected_action = QtGui.QAction("Move Selected to", self)
        self.move_selected_action.setToolTip("Move every currently selected particle to the set named at right.")
        self.move_selected_action.triggered.connect(self._move_selected_via_toolbar)

        # Toggle Colors' QAction is created here (TomoBoxer owns the
        # behavior), but the button for it lives in the particle list's own
        # toolbar, not here -- see ParticleListPanel.__init__. A QAction
        # isn't tied to whichever toolbar displays it, so this is the exact
        # same object clicking that button triggers.
        self.toggle_colors_action = QtGui.QAction("Toggle Colors", self)
        self.toggle_colors_action.setToolTip(
            "Rotate every normal set to the next color at once. Recycle "
            "Bin (red), Confirmed (green), and the yellow selection "
            "highlight never change.")
        self.toggle_colors_action.triggered.connect(self.toggle_colors)

        self.edit_toolbar = self.addToolBar("Edit")
        self.edit_toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.edit_toolbar.addAction(self.undo_action)
        self.edit_toolbar.addAction(self.redo_action)

        # A separate toolbar (not just a separator inside Edit) so the
        # "acts on whatever's currently selected" actions read as one clear
        # group at a glance, with its own label, rather than blending into
        # Undo/Redo/Toggle Colors. addToolBarBreak() forces it onto its own
        # row unconditionally -- without it, Qt packs toolbars side by side
        # on the same row whenever they happen to fit, so which row this
        # landed on would depend on the window's current width.
        self.addToolBarBreak()
        self.selection_toolbar = self.addToolBar("Selection")
        self.selection_toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.selection_toolbar.addWidget(QtWidgets.QLabel(" Selection: "))
        self.selection_toolbar.addAction(self.confirm_selected_action)
        self.selection_toolbar.addAction(self.recycle_selected_action)
        self.selection_toolbar.addAction(self.move_selected_action)
        self.selection_toolbar.addWidget(self.move_to_combo)

        # Show the active channel only / Show All Channels' QActions are created here
        # (TomoBoxer owns sets_visible and the solo logic), but -- like
        # Toggle Colors above -- the buttons for them live in the particle
        # list's own toolbar, not here; see ParticleListPanel.__init__.
        self.view_only_selected_action = QtGui.QAction("Show the active channel only", self)
        self.view_only_selected_action.setCheckable(True)
        self.view_only_selected_action.setToolTip(
            "While on, only the active tab's set is shown in the tomogram "
            "views -- switching tabs (including via number keys) updates "
            "this live. Turning it off restores whatever visibility you "
            "had before.")
        self.view_only_selected_action.toggled.connect(self.toggle_view_only_selected)
        self._solo_saved_visible = None  # sets_visible snapshot from before solo turned on

        self.show_all_channels_action = QtGui.QAction("Show All Channels", self)
        self.show_all_channels_action.setToolTip("Make every set visible in the tomogram views.")
        self.show_all_channels_action.triggered.connect(self.show_all_channels)

        self.setStyleSheet(self.styleSheet() + TOOLBAR_BUTTON_STYLE)

        # A small permanent (right-aligned) status-bar label, like a file
        # manager's "N items selected" -- the only ambient feedback for the
        # current selection now that there's no dedicated Selection panel.
        self.selection_status_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.selection_status_label)

        # All start disabled (nothing to undo/redo/act on yet); kept in
        # sync by _refresh_undo_redo_actions() (end of _push_undo/undo/redo/
        # set_data) and set_selected_uids() respectively.
        self._refresh_undo_redo_actions()
        self.confirm_selected_action.setEnabled(False)
        self.recycle_selected_action.setEnabled(False)
        self.move_to_combo.setEnabled(False)
        self.move_selected_action.setEnabled(False)

        # Bare 1-9 switch the active picking class (Photoshop/Lightroom-
        # style). Application-wide like undo/redo above, so it works no
        # matter which window has focus -- but a QSpinBox/QLineEdit/editable
        # QComboBox with focus (box size, eraser radius, the "Move to" class
        # combo, ...) still claims plain digit keys for itself first via
        # Qt's own shortcut-override mechanism, so typing a number into one
        # of those fields is unaffected.
        self._class_shortcuts = []
        for n in range(1, 10):
            sc = QtGui.QShortcut(QtGui.QKeySequence(str(n)), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda n=n: self.select_set_by_number(n))
            self._class_shortcuts.append(sc)

        # Escape-to-clear-selection, also application-wide: it used to only
        # fire via ImageView2D.keypress, i.e. only while a tomogram view
        # itself had keyboard focus -- so pressing Escape right after using
        # a toolbar button or combo box (which grabs focus for itself) did
        # nothing, leaving a selection that looked "stuck" even though nothing
        # was actually wrong with the selection itself, just with reaching it.
        self._escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self)
        self._escape_shortcut.setContext(Qt.ApplicationShortcut)
        self._escape_shortcut.activated.connect(self._global_escape)

        # -- load data ---------------------------------------------------------
        self.datafilename = datafile

        data = Volume.from_file(datafile)
        if options.apix:
            data.apix = options.apix
        self.data = None
        self.set_data(data)

        # Per-class plain-text pick files live in "<basename>_picks/" next
        # to the tomogram by default, or wherever --working_dir points
        # (resolved relative to the current working directory) instead.
        self.picksdir = (
            os.path.abspath(options.working_dir) if getattr(options, "working_dir", None)
            else metadata.picks_dir(datafile))

        self.boxesviewer = ParticleListPanel(self)
        self.boxesviewer.setWindowTitle("Particle List")
        self.boxesviewer.set_mouse_mode("App")
        self.boxesviewer.show()

        self.boxesviewer.mx_image_selected.connect(self.img_selected)
        self.boxesviewer.selection_changed.connect(self.set_selected_uids)
        # A set's tab becoming the active one in the gallery also makes it
        # the active picking class -- the Sets list this used to require a
        # separate click in is gone now that sets live in the gallery's own
        # tabs. Recycle Bin/Confirmed/"All" don't emit this (see
        # ParticleListPanel.class_activated), matching number-key switching.
        self.boxesviewer.class_activated.connect(self._on_gallery_class_activated)
        # class_activated above deliberately skips "All"/Recycle Bin/
        # Confirmed (they can't become the active picking class), but View
        # Only Selected still needs to follow the gallery to *those* tabs
        # too -- active_tab_changed fires for every real tab, fixed or not.
        self.boxesviewer.active_tab_changed.connect(self._on_gallery_tab_changed)

        self._arrange_windows()

        # -- metadata (picks + classes) ------------------------------------------
        # self.sets: class_id (int) -> class name (str), the name doubling
        # as the "<name>.txt" filename inside self.picksdir.
        self.sets = {}
        self.boxsize = {}

        class_settings = metadata.load_class_settings(self.picksdir)
        classnames = metadata.discover_classes(self.picksdir)
        normal_names = [n for n in classnames if n not in categories.FIXED_CLASS_NAMES]

        self.sets_visible = set()

        for i, name in enumerate(normal_names):
            self.sets[i] = name
            s = class_settings.get(name, {})
            self.boxsize[i] = s.get("boxsize", metadata.DEFAULT_BOXSIZE)
            if s.get("color"):
                categories.set_manual_color(i, QtGui.QColor(s["color"]))
            for x, y, z in metadata.load_class_coords(self.picksdir, name):
                bdf = [x, y, z, "manual", 0.0, i, self._new_uid()]
                self.boxes.append(bdf)

        # The Recycle Bin and Confirmed categories always exist, whether or
        # not their pick files were present on disk yet.
        for name, cid in categories.FIXED_CLASS_NAMES.items():
            self.sets[cid] = name
            s = class_settings.get(name, {})
            self.boxsize[cid] = s.get("boxsize", metadata.DEFAULT_BOXSIZE)
            if s.get("color"):
                categories.set_manual_color(cid, QtGui.QColor(s["color"]))
            for x, y, z in metadata.load_class_coords(self.picksdir, name):
                bdf = [x, y, z, "manual", 0.0, cid, self._new_uid()]
                self.boxes.append(bdf)

        shape = [self.data.nx, self.data.ny, self.data.nz]
        for b in self.boxes:
            for i in range(3):
                b[i] = int(np.clip(round(b[i]), 1, shape[i] - 1))

        # --particle_lists: each path becomes its own brand-new set, copied
        # (written) into self.picksdir immediately, same as File > Read Box
        # Coord -- see import_particle_list. live=False since the boxes
        # appended here are picked up by this __init__'s own per-box
        # update_box loop just below instead of doing that twice.
        for lp in getattr(options, "particle_lists", None) or []:
            result = self.import_particle_list(lp, live=False)
            if result is None:
                print(f"warning: no coordinates found in particle list {lp!r}, skipped")
                continue
            unique_name, count = result
            print(f"Imported {count} particle(s) from {lp!r} into new set \"{unique_name}\"")

        if not any(cid not in categories.FIXED_CLASS_IDS for cid in self.sets):
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
        self.wboxsize.setValue(self.get_boxsize())

        if options.clean:
            self.tool_eraser_action.setChecked(True)
            self._on_tool_changed(self.tool_eraser_action)  # setChecked() alone doesn't emit triggered

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
        particle list."""
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
        self._refresh_undo_redo_actions()
        self.confirm_selected_action.setEnabled(False)
        self.recycle_selected_action.setEnabled(False)
        self.move_to_combo.setEnabled(False)
        self.move_selected_action.setEnabled(False)
        self.selection_status_label.setText("")

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
        return int(self.eraser_radius.getValue())

    def get_cube(self, x, y, z, centerslice=False, boxsz=-1):
        bs = self.get_boxsize() if boxsz < 0 else boxsz
        if centerslice:
            # Same slice-count formula as get_slice/slab_projection below,
            # so a gallery thumbnail is averaged over exactly the same
            # slices "Avg Slices" would put into the tomogram views at
            # this position -- not always a single slice regardless of
            # that setting.
            half_thickness = self.nlayers() - 1
            dz = 2 * half_thickness + 1
            cube = self.data.get_clip_centered(x, y, z, bs, bs, dz)
            return cube.mean(axis=0, keepdims=True)
        return self.data.get_clip_centered(x, y, z, bs, bs, bs)

    def _thumbnail_crop(self, box):
        """Raw-intensity gallery-thumbnail crop for one box (shared by
        update_box and event_nlayers). Kept as raw intensities, not
        normalize()'d -- the gallery thumbnail is rendered with the exact
        same contrast code and levels as the tomogram views (see
        thumb_grid._thumbnail_pixmap and ParticleGrid.set_levels), so it
        needs to be on the same intensity scale as them to actually look
        like a matching crop."""
        proj = self.get_cube(box[0], box[1], box[2], centerslice=True, boxsz=self.get_boxsize(box[5]))
        return proj[0]

    def _refresh_thumbnail(self, i):
        """Regenerate box i's cached gallery-thumbnail crop -- needed
        whenever its effective box size could have changed: a direct
        size edit (_resize_boxes), Avg Slices changing what a crop looks
        like (event_nlayers), or reassigning it to a class with its own
        different box size, which get_boxsize looks up by the box's
        *current* class. update_box_shape alone (used by every reassign
        path) only touches the drawn marker in the tomogram views, not
        this cached crop -- without also calling this, a particle
        confirmed/recycled/moved into a class with a different box size
        kept showing its old class's box size in the gallery thumbnail."""
        for _ in range(len(self.boxesimgs), i + 1):
            self.boxesimgs.append(None)
        self.boxesimgs[i] = self._thumbnail_crop(self.boxes[i])

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
        # Avg Slices changes what a gallery thumbnail should look like too
        # (it's averaged over the same slices as the tomogram views -- see
        # get_cube), so every existing particle's cached crop needs
        # regenerating here, not just the 3 main views re-sliced below.
        for n, box in enumerate(self.boxes):
            for _ in range(len(self.boxesimgs), n + 1):
                self.boxesimgs.append(None)
            self.boxesimgs[n] = self._thumbnail_crop(box)
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
    def menu_file_save_all(self):
        # Picks autosave continuously already; this is a manual "force it
        # now" for reassurance, plus the one thing autosave doesn't cover:
        # box sizes only get written to config.settings when the app fully
        # closes otherwise, so this flushes that immediately too.
        self.save_picks()
        QtWidgets.QMessageBox.information(
            self, "Save",
            f"Every set's picks and box sizes have been saved to:\n{self.picksdir}")

    def import_particle_list(self, path, live=True):
        """Import a plain-text "x y z" coordinate file (metadata.
        load_particle_list -- see there for the input format) as a brand
        new set, named after the file (auto-suffixed if that name's
        already taken -- deliberately: the same name can legitimately
        show up twice with different picks, e.g. --particle_lists run
        again against a working directory that already has a same-named
        set from a previous run) -- rather than dumping the imported
        particles into whichever set happens to be selected, which left
        no way to tell them apart from picks made by hand. Shared by the
        File > Read Box Coord menu action (live=True) and --particle_lists
        at startup (live=False).

        `live` covers what only makes sense once the app is already
        running: per-box marker/thumbnail generation and a view refresh.
        During __init__'s own bootstrap (live=False), the boxes appended
        here are picked up by its own trailing update_box loop instead,
        so doing that here too would just be redundant work performed
        before the window is even fully set up. Either way, this set's
        own coordinate file is always written immediately -- "imported"
        means "copied into the working directory," not "queued until
        something else happens to save."

        Returns (unique_name, count) or None if the file had no usable
        coordinates."""
        coords = metadata.load_particle_list(path)
        if not coords:
            return None

        name = metadata.base_name(path)
        unique_name = name
        n = 1
        while unique_name in self.sets.values():
            n += 1
            unique_name = f"{name}_{n}"

        clsid = self.new_set(unique_name)

        shape = [self.data.nx, self.data.ny, self.data.nz]
        start = len(self.boxes)
        for x, y, z in coords:
            x = int(np.clip(round(x), 1, shape[0] - 1))
            y = int(np.clip(round(y), 1, shape[1] - 1))
            z = int(np.clip(round(z), 1, shape[2] - 1))
            self.boxes.append([x, y, z, "manual", 0.0, clsid, self._new_uid()])

        if live:
            self.initialized = False
            for i in range(start, len(self.boxes)):
                self.update_box(i)
            self.initialized = True

        self.save_json(only={clsid})

        if live:
            self.update_all()

        return unique_name, len(coords)

    def menu_file_read_boxloc(self):
        fsp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select box coordinate text file", "", "Text files (*.txt);;All files (*)")
        if not fsp or not os.path.isfile(fsp):
            return
        name = metadata.base_name(fsp)
        result = self.import_particle_list(fsp, live=True)
        if result is None:
            QtWidgets.QMessageBox.information(self, "Read Box Coord", "No coordinates found in that file.")
            return
        unique_name, count = result
        msg = f"Imported {count} particle(s) into new set \"{unique_name}\"."
        if unique_name != name:
            msg += f' (renamed from "{name}", which was already in use.)'
        QtWidgets.QMessageBox.information(self, "Read Box Coord", msg)

    def menu_file_save_boxloc(self):
        """Picks are already autosaved continuously (see save_picks) -- this
        just copies the already-written coordinate file(s), all sets or one
        you choose, to a folder of your own for safekeeping/mental sanity,
        rather than re-deriving and writing them again from scratch."""
        order = categories.category_order(self.sets)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Save Box Coord")
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel(
            "Picks are saved automatically already. Save a copy of:"))
        combo = QtWidgets.QComboBox()
        combo.addItem("All sets", None)
        for cid in order:
            combo.addItem(self.sets[cid], cid)
        layout.addWidget(combo)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = combo.currentData()

        destdir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose destination folder")
        if not destdir:
            return

        classids = list(self.sets.keys()) if chosen is None else [chosen]
        copied = []
        for cid in classids:
            name = self.sets[cid]
            src = metadata.class_file(self.picksdir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(destdir, os.path.basename(src)))
                copied.append(name)

        if copied:
            QtWidgets.QMessageBox.information(
                self, "Save Box Coord", f"Saved a copy of {len(copied)} set(s) to {destdir}.")
        else:
            QtWidgets.QMessageBox.information(self, "Save Box Coord", "Nothing to save yet.")

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
            settings_by_name = {
                self.sets[k]: {"boxsize": self.boxsize[k], "color": categories.manual_color_hex(k)}
                for k in self.sets
            }
            metadata.save_class_settings(self.picksdir, settings_by_name)

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
            self._refresh_thumbnail(i)
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
        self._refresh_undo_redo_actions()

    def _refresh_undo_redo_actions(self):
        """Keep the Edit menu/toolbar's Undo/Redo entries enabled exactly
        when there's something on the corresponding stack -- standard
        editor convention, and the only feedback that an undo/redo actually
        did something (there's no other status message anywhere)."""
        self.undo_action.setEnabled(bool(self._undo_stack))
        self.redo_action.setEnabled(bool(self._redo_stack))

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
        self._refresh_undo_redo_actions()

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
        self._refresh_undo_redo_actions()

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
        # Highlight exactly the particles this undo/redo actually brought
        # back -- explicit, rather than the gallery guessing "a uid
        # reappeared" (it used to auto-select on that alone, which also
        # fired for unrelated reasons like a set's visibility being
        # toggled back on, since that removes and re-adds gallery items
        # the same way a real delete/restore does).
        self.boxesviewer.replace_selection([box_copy[6] for _, box_copy, _ in removed])

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
                self._refresh_thumbnail(i)
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
        has_selection = bool(self._selected_uids)
        self.confirm_selected_action.setEnabled(has_selection)
        self.recycle_selected_action.setEnabled(has_selection)
        self.move_to_combo.setEnabled(has_selection)
        self.move_selected_action.setEnabled(has_selection)
        if has_selection:
            self._refresh_move_to_combo()
        n = len(self._selected_uids)
        self.selection_status_label.setText(f"{n} particle(s) selected" if n else "")
        if not changed_uids:
            return
        changed_indices = [i for i, b in enumerate(self.boxes) if b[6] in changed_uids]
        for i in changed_indices:
            self.update_box_shape(i, self.boxes[i])
        self.update_all(reslice=False, only_indices=changed_indices)

    def _refresh_move_to_combo(self):
        current = self.move_to_combo.currentText()
        self.move_to_combo.blockSignals(True)
        self.move_to_combo.clear()
        self.move_to_combo.addItems(sorted(self.sets.values()))
        self.move_to_combo.setCurrentText(current)
        self.move_to_combo.blockSignals(False)

    def _move_selected_via_toolbar(self):
        name = self.move_to_combo.currentText().strip()
        if not name or not self._selected_uids:
            return
        self.move_particles_by_uid(self._selected_uids, name)
        self.move_to_combo.setCurrentText("")

    def update_box(self, n, quiet=False, axis=("x", "y", "z")):
        if n < 0 or n >= len(self.boxes):
            return
        box = self.boxes[n]
        self.update_box_shape(n, box)

        if self.initialized:
            self.update_sliceview(axis)

        if not quiet:
            for _ in range(len(self.boxesimgs), n + 1):
                self.boxesimgs.append(None)
            self.boxesimgs[n] = self._thumbnail_crop(box)

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

    def del_region_xy(self, x=-1, y=-1, z=-1, rad=-1, axis="z", is_3d=True):
        """3D (default, unchanged): a literal sphere of radius rad, exactly
        as before -- a bounded eraser "brush", like a sculpting tool's, not
        an unbounded volume the way Select's 3D mode is (that's fine: this
        tool destroys data immediately with no confirmation, so it should
        stay a bounded brush rather than growing to "the entire depth" the
        way a non-destructive selection reasonably can).

        2D (new): a circle of radius rad in the current view's own two
        in-plane axes only, additionally restricted to particles actually
        visible on the current slice (same in-plane test used for
        rendering/Select's own 2D mode) -- so it only erases what's drawn
        on screen, never reaching into depths you can't see."""
        if rad < 0:
            rad = self.eraser_width()
        delids = []
        if is_3d:
            for i, b in enumerate(self.boxes):
                if b[5] not in self.sets_visible:
                    continue
                if (x >= 0) * (b[0] - x) ** 2 + (y >= 0) * (b[1] - y) ** 2 + (z >= 0) * (b[2] - z) ** 2 < rad ** 2:
                    delids.append(i)
        else:
            a1, a2 = _VIEW_PLANE_AXES[axis]
            i1, i2 = _AXIS_INDEX[a1], _AXIS_INDEX[a2]
            depth_idx = _AXIS_INDEX[axis]
            click = (x, y, z)
            p1_click, p2_click, depth_click = click[i1], click[i2], click[depth_idx]
            for i, b in enumerate(self.boxes):
                if b[5] not in self.sets_visible:
                    continue
                bs = self.get_boxsize(b[5])
                if abs(b[depth_idx] - depth_click) >= bs / 2:
                    continue
                if (b[i1] - p1_click) ** 2 + (b[i2] - p2_click) ** 2 < rad ** 2:
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
        if self.tool_eraser_action.isChecked():
            self.del_region_xy(x, y, z, -1, axis=axis, is_3d=self.tool_3d_checkbox.isChecked())
            return
        if self.tool_select_action.isChecked():
            self._select_mouse_down(event, x, y, z, axis)
            return

        for i in range(len(self.boxes)):
            if self.inside_box(i, x, y, z):
                if event.modifiers() & Qt.ShiftModifier:
                    self.send_to_recycle_bin(i)
                elif space_key_state().held:
                    self.send_to_confirmed(i)
                elif event.modifiers() & Qt.ControlModifier:
                    # Toggle-select, exactly like Ctrl+click in the gallery --
                    # writes into the same selection list the gallery owns,
                    # so both windows and the Selection panel always agree.
                    self.boxesviewer.toggle_uid(self.boxes[i][6])
                else:
                    # A plain click on some other particle is "start doing
                    # something else" -- clear whatever multi-selection was
                    # active first, the same way clicking a file in a file
                    # manager deselects the others unless you hold Ctrl.
                    self.boxesviewer.clear_selection()
                    self.dragging = i
                    self.curbox = i
                    # Move the other two views onto this particle's own
                    # stored coordinate, not the raw click position -- a
                    # click only needs to land within the box's radius to
                    # select it (see inside_box), so it's almost never
                    # pixel-exact on the particle's true center. Panning
                    # alone (scroll_to) isn't enough either: each of those
                    # views' background slice depth is governed by its own
                    # x_loc/y_loc/z_loc, unrelated to this particle's
                    # position along that axis, so without also moving the
                    # depth and reslicing, the pan lands on the right pixel
                    # but the wrong slice -- the particle usually doesn't
                    # even render there. The view actually clicked in is
                    # deliberately left alone (no jump, no reslice), same
                    # as scroll_to's own axis exclusion.
                    box = self.boxes[i]
                    if axis != "x":
                        self.x_loc = box[0]
                    if axis != "y":
                        self.y_loc = box[1]
                    if axis != "z":
                        self.z_loc = box[2]
                    self.scroll_to(box[0], box[1], box[2], axis)
                    other_axes = tuple(a for a in ("x", "y", "z") if a != axis)
                    self.update_sliceview(axis=other_axes)
                break
        else:
            if not event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
                self.boxesviewer.clear_selection()
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
        if self.tool_eraser_action.isChecked():
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
        elif self._select_drawing is not None:
            x, y = self.xyview.scr_to_img((event.position().x(), event.position().y()))
            self._select_mouse_drag(self.xyview, "z", x, y, self.z_loc)

    def xz_drag(self, event):
        if self.dragging >= 0:
            x, z = self.xzview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(x, self.y_loc, z, skip_axis="y")
        elif self._select_drawing is not None:
            x, z = self.xzview.scr_to_img((event.position().x(), event.position().y()))
            self._select_mouse_drag(self.xzview, "y", x, self.y_loc, z)

    def zy_drag(self, event):
        if self.dragging >= 0:
            z, y = self.zyview.scr_to_img((event.position().x(), event.position().y()))
            self.mouse_drag(self.x_loc, y, z, skip_axis="x")
        elif self._select_drawing is not None:
            z, y = self.zyview.scr_to_img((event.position().x(), event.position().y()))
            self._select_mouse_drag(self.zyview, "x", self.x_loc, y, z)

    def mouse_up(self, event):
        if self.dragging >= 0:
            self.update_box(self.dragging)
            self.dragging = -1
            return
        if self._select_drawing is not None:
            self._select_mouse_up()

    def xy_doubleclick(self, event):
        self._select_doubleclick(self.xyview)

    def xz_doubleclick(self, event):
        self._select_doubleclick(self.xzview)

    def zy_doubleclick(self, event):
        self._select_doubleclick(self.zyview)

    def _select_doubleclick(self, view):
        drawing = self._select_drawing
        if drawing is not None and drawing["kind"] == "freeform" and drawing["view"] is view:
            self._select_finish_freeform()

    # -- tool selector (Pick/Eraser/Select) ----------------------------------
    def _on_tool_changed(self, action):
        is_eraser = action is self.tool_eraser_action
        is_select = action is self.tool_select_action
        self.eraser_radius.setEnabled(is_eraser)
        for rb in (self.select_shape_rect, self.select_shape_circle, self.select_shape_freeform):
            rb.setEnabled(is_select)
        self.tool_3d_checkbox.setEnabled(is_eraser or is_select)
        cursor = Qt.CrossCursor if is_select else None
        for view in (self.xyview, self.xzview, self.zyview):
            if cursor is not None:
                view.setCursor(cursor)
            else:
                view.unsetCursor()
        # Switching tools mid-gesture is ambiguous, so cancel whatever
        # shape-select was in progress rather than leave a stale preview.
        self._select_cancel_drawing()

    # -- shape-based multi-select tool (rectangle/circle/freeform, 2D/3D) ---
    # Non-destructive: it only replaces the gallery's selection list (same
    # one Ctrl+click above and the gallery itself write into), never
    # deletes/moves anything by itself -- act on the result via the
    # toolbar's Confirm/Recycle/Move Selected.
    def _select_shape_kind(self):
        if self.select_shape_circle.isChecked():
            return "circle"
        if self.select_shape_freeform.isChecked():
            return "freeform"
        return "rect"

    def _select_is_3d(self):
        return self.tool_3d_checkbox.isChecked()

    def _select_cancel_drawing(self, *_args):
        if self._select_drawing is None:
            return
        self._select_drawing["view"].select_preview = None
        self._select_drawing = None

    def _select_mouse_down(self, event, x, y, z, axis):
        view = {"z": self.xyview, "y": self.xzview, "x": self.zyview}[axis]
        kind = self._select_shape_kind()
        if kind == "freeform":
            self._select_freeform_click(view, axis, x, y, z)
            return
        self._select_drawing = {
            "view": view, "axis": axis, "kind": kind,
            "start": (x, y, z), "current": (x, y, z),
        }
        self._update_select_preview()

    def _select_mouse_drag(self, view, axis, x, y, z):
        drawing = self._select_drawing
        if drawing is None or drawing["view"] is not view or drawing["kind"] == "freeform":
            return
        drawing["current"] = (x, y, z)
        self._update_select_preview()

    def _select_mouse_up(self):
        drawing = self._select_drawing
        if drawing is None or drawing["kind"] == "freeform":
            return  # freeform commits by closing the loop/double-click/Enter, not on release
        uids = self._select_hit_test(drawing)
        drawing["view"].select_preview = None
        self._select_drawing = None
        self.boxesviewer.replace_selection(uids)

    def _update_select_preview(self):
        drawing = self._select_drawing
        a1, a2 = _VIEW_PLANE_AXES[drawing["axis"]]
        i1, i2 = _AXIS_INDEX[a1], _AXIS_INDEX[a2]
        p1_0, p2_0 = drawing["start"][i1], drawing["start"][i2]
        p1_1, p2_1 = drawing["current"][i1], drawing["current"][i2]
        if drawing["kind"] == "rect":
            shape = Shape("rect", _SELECT_PREVIEW_COLOR, [p1_0, p2_0, p1_1, p2_1], 2)
        else:  # circle
            r = ((p1_1 - p1_0) ** 2 + (p2_1 - p2_0) ** 2) ** 0.5
            shape = Shape("circle", _SELECT_PREVIEW_COLOR, [p1_0, p2_0, r], 2)
        drawing["view"].select_preview = shape

    def _select_freeform_click(self, view, axis, x, y, z):
        drawing = self._select_drawing
        if drawing is not None and drawing["kind"] == "freeform" and drawing["view"] is not view:
            return  # a freeform is already in progress in a different view -- finish/cancel it first
        if drawing is None or drawing["kind"] != "freeform" or drawing["view"] is not view:
            drawing = {"view": view, "axis": axis, "kind": "freeform", "points": []}
            self._select_drawing = drawing

        a1, a2 = _VIEW_PLANE_AXES[axis]
        p1, p2 = (x, y, z)[_AXIS_INDEX[a1]], (x, y, z)[_AXIS_INDEX[a2]]
        pts = drawing["points"]
        if len(pts) >= 3 and self._select_screen_close(view, pts[0], (p1, p2)):
            self._select_finish_freeform()
            return
        pts.append((p1, p2))
        self._update_select_preview_freeform()

    def _select_screen_close(self, view, plane_pt1, plane_pt2):
        s1 = view.img_to_scr(*plane_pt1)
        s2 = view.img_to_scr(*plane_pt2)
        return (s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2 <= _SELECT_CLOSE_PIXELS ** 2

    def _update_select_preview_freeform(self):
        drawing = self._select_drawing
        pts = drawing["points"]
        if len(pts) < 2:
            drawing["view"].select_preview = None
            return
        geom = [c for p in pts for c in p]
        drawing["view"].select_preview = Shape("polygon", _SELECT_PREVIEW_COLOR, geom, 2)

    def _select_finish_freeform(self):
        drawing = self._select_drawing
        if drawing is None or drawing["kind"] != "freeform":
            return
        if len(drawing["points"]) < 3:
            self._select_cancel_drawing()
            return
        uids = self._select_hit_test(drawing)
        drawing["view"].select_preview = None
        self._select_drawing = None
        self.boxesviewer.replace_selection(uids)

    def _select_hit_test(self, drawing):
        axis = drawing["axis"]
        a1, a2 = _VIEW_PLANE_AXES[axis]
        i1, i2 = _AXIS_INDEX[a1], _AXIS_INDEX[a2]
        is_3d = self._select_is_3d()

        if drawing["kind"] == "freeform":
            path = QtGui.QPainterPath()
            pts = drawing["points"]
            path.moveTo(pts[0][0], pts[0][1])
            for p1, p2 in pts[1:]:
                path.lineTo(p1, p2)
            path.closeSubpath()

            def contains(p1, p2):
                return path.contains(QtCore.QPointF(p1, p2))
        elif drawing["kind"] == "rect":
            p1_0, p2_0 = drawing["start"][i1], drawing["start"][i2]
            p1_1, p2_1 = drawing["current"][i1], drawing["current"][i2]
            lo1, hi1 = sorted((p1_0, p1_1))
            lo2, hi2 = sorted((p2_0, p2_1))

            def contains(p1, p2):
                return lo1 <= p1 <= hi1 and lo2 <= p2 <= hi2
        else:  # circle
            p1_0, p2_0 = drawing["start"][i1], drawing["start"][i2]
            p1_1, p2_1 = drawing["current"][i1], drawing["current"][i2]
            r2 = (p1_1 - p1_0) ** 2 + (p2_1 - p2_0) ** 2

            def contains(p1, p2):
                return (p1 - p1_0) ** 2 + (p2 - p2_0) ** 2 <= r2

        current_depth = {"x": self.x_loc, "y": self.y_loc, "z": self.z_loc}[axis]
        uids = []
        for b in self.boxes:
            if b[5] not in self.sets_visible:
                continue
            if not is_3d:
                bs = self.get_boxsize(b[5])
                if abs(b[_AXIS_INDEX[axis]] - current_depth) >= bs / 2:
                    continue
            if contains(b[i1], b[i2]):
                uids.append(b[6])
        return uids

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
        self._sync_solo_visibility()
        self.update_all()

    def is_solo_active(self):
        return self.view_only_selected_action.isChecked()

    def _sync_solo_visibility(self):
        """While Show the active channel only is active, force sets_visible back to
        exactly {currentset}. Called from everywhere currentset can change
        (or sets_visible can otherwise drift away from it) -- not just
        set_current_set's own callers -- since a couple of paths bypass
        set_current_set entirely: _on_gallery_class_activated skips it
        outright when the clicked tab's class already equals currentset
        (a cheap-looking optimization that predates solo mode and doesn't
        account for sets_visible having changed since, e.g. after
        visiting "All" or a fixed tab and back), and delete_set reassigns
        currentset directly when the active set itself gets deleted.
        Returns whether it actually changed anything, so callers taking a
        shortcut around update_all() know whether one is still needed."""
        if not self.view_only_selected_action.isChecked():
            return False
        if self.sets_visible == {self.currentset}:
            return False
        self.sets_visible = {self.currentset}
        self.boxesviewer.refresh_visibility_checkboxes()
        return True

    def toggle_view_only_selected(self, checked):
        if checked:
            self._solo_saved_visible = set(self.sets_visible)
            # The tab actually being viewed, not self.currentset -- which
            # never changes for "All"/Recycle Bin/Confirmed, so turning
            # solo on while looking at one of those used to solo onto
            # whatever normal class happened to be the picking class
            # instead, making the tab you were actually looking at go
            # invisible.
            cid = self.boxesviewer.active_view_classid()
            self.sets_visible = set(self.sets.keys()) if cid is None else {cid}
        elif self._solo_saved_visible is not None:
            self.sets_visible = self._solo_saved_visible
            self._solo_saved_visible = None
        self.boxesviewer.set_visibility_controls_locked(checked)
        self.boxesviewer.refresh_visibility_checkboxes()
        self.update_all()

    def show_all_channels(self):
        if self.view_only_selected_action.isChecked():
            # Turns solo off (restoring its pre-solo snapshot first) --
            # then immediately overridden below to "everything visible",
            # which is what this button actually promises.
            self.view_only_selected_action.setChecked(False)
        self.sets_visible = set(self.sets.keys())
        self.boxesviewer.refresh_visibility_checkboxes()
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
        if delids:
            # Move to the Recycle Bin rather than destroying them outright --
            # deleting a whole set shouldn't be more permanent than deleting
            # one particle at a time is. Not undoable as a single step
            # (matching this method's prior behavior): the set definition
            # itself is removed right after, so undoing just the reassign
            # would leave particles pointing at a class id that no longer
            # exists in self.sets.
            self._apply_reassign([(i, name) for i in delids], target_id=categories.RECYCLE_BIN_ID)
        self.sets_visible.discard(name)
        classname = self.sets.pop(name, None)
        self.boxsize.pop(name, None)
        if classname is not None:
            metadata.delete_class_file(self.picksdir, classname)
        if self.currentset == name:
            # The active picking class just vanished -- without this, the
            # next click-to-add still targeted this now-nonexistent class
            # id (self.currentset was never touched here), which is why
            # get_boxsize(that id) started failing ("No box size saved for
            # ..") and every new particle silently fell back to a default
            # box size instead of the set it was supposedly created in.
            remaining = sorted(cid for cid in self.sets if cid not in categories.FIXED_CLASS_IDS)
            self.currentset = remaining[0] if remaining else self.new_set("particles_00")
            self.wboxsize.setValue(self.get_boxsize())
            self._sync_solo_visibility()
        self._refresh_move_to_combo()
        self.update_all(reslice=False, only_indices=[])

    def toggle_colors(self):
        """Rotate every normal class still on its automatic color to the
        next one in the palette at once, napari-style. A class with a
        manually-set color (including Recycle Bin/Confirmed, which can
        also be recolored via the gallery's color swatch -- see
        set_class_color) is untouched: cycle_colors() only shifts the
        rotation offset the automatic formula uses, never a manual
        override, so a deliberate choice never gets silently overwritten
        by this. The selection highlight is separate and always fixed."""
        categories.cycle_colors()
        self._recolor_and_refresh()

    def set_class_color(self, classid, color):
        """Explicitly set one class's color (the gallery's color-swatch
        menu's "Choose Color..."). Works for any class, including Recycle
        Bin/Confirmed -- persisted like box sizes (see save_picks), and
        left alone by Toggle Colors once set."""
        if classid not in self.sets:
            return
        categories.set_manual_color(classid, color)
        self._recolor_and_refresh()

    def reset_class_color(self, classid):
        """Clear a class's manual color override, reverting it to the
        automatic palette (or, for Recycle Bin/Confirmed, back to their
        own fixed red/green)."""
        if classid not in self.sets:
            return
        categories.clear_manual_color(classid)
        self._recolor_and_refresh()

    def _recolor_and_refresh(self):
        for i in range(len(self.boxes)):
            self.update_box_shape(i, self.boxes[i])
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
            self._refresh_move_to_combo()

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
        # Otherwise a set created while a selection was already active (so
        # set_selected_uids' own refresh already fired earlier and won't
        # fire again) never showed up in "Move Selected to" until you
        # cleared and remade the selection.
        self._refresh_move_to_combo()
        return i

    # -- multi-selection actions (used by the Options window's Selection panel) --
    def delete_particles_by_uid(self, uids):
        idx = [i for i, b in enumerate(self.boxes) if b[6] in uids]
        self.send_to_recycle_bin(idx)
        # The selection has been acted on and is "done" -- carrying it
        # forward would just leave these particles looking selected
        # afterward for no reason, the same way a file manager doesn't
        # keep files selected in the destination once you've moved them.
        self.boxesviewer.clear_selection()

    def confirm_particles_by_uid(self, uids):
        idx = [i for i, b in enumerate(self.boxes) if b[6] in uids]
        self.send_to_confirmed(idx)
        self.boxesviewer.clear_selection()

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
            self._refresh_thumbnail(i)
        self._push_undo({"type": "reassign", "changes": prev, "new_class": target_id})
        # new_set() above already makes a brand-new class visible; an
        # existing target class that the user deliberately hid should stay
        # hidden even as particles get moved into it.
        affected_classes = {target_id} | {prev for _, prev in prev}
        if self.initialized:
            self.save_picks(only=affected_classes)
        self.boxesviewer.clear_selection()
        self.update_all(reslice=False, only_indices=idx)

    # -- help -----------------------------------------------------------
    def show_shortcuts_dialog(self):
        if self._shortcuts_dialog is None:
            self._shortcuts_dialog = ShortcutsDialog(self)
        self._shortcuts_dialog.show()
        self._shortcuts_dialog.raise_()
        self._shortcuts_dialog.activateWindow()

    # -- keyboard -----------------------------------------------------------
    def key_press(self, event):
        drawing = self._select_drawing
        if drawing is not None and drawing["kind"] == "freeform":
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._select_finish_freeform()
                return
            if event.key() == Qt.Key_Backspace:
                if drawing["points"]:
                    drawing["points"].pop()
                self._update_select_preview_freeform()
                return
            # Escape while freeform-drawing is handled by _global_escape
            # (application-wide shortcut) instead of here -- see there.
        self.keyPress.emit(event)

    def _global_escape(self):
        """Escape, application-wide (see the QShortcut set up in __init__)
        rather than only while a tomogram view happens to have keyboard
        focus: cancelling an in-progress freeform selection takes priority
        (matching key_press's old in-focus behavior), otherwise this clears
        the current multi-selection -- the standard "back out of what
        you're doing" key, and the one place that selection can actually be
        cleared from, so it's never unreachable regardless of which widget
        focus happens to be sitting on."""
        drawing = self._select_drawing
        if drawing is not None and drawing["kind"] == "freeform":
            self._select_cancel_drawing()
            return
        self.boxesviewer.clear_selection()

    def select_set_by_number(self, n):
        """Bare 1-9 keys switch the active picking class to the nth normal
        set AND switch the gallery to that set's own tab -- the same
        number shown on the tab itself (ParticleListPanel._renumber_tabs),
        which always matches whatever order the tabs are currently in
        (drag to rearrange them; the numbering follows). A detached set
        has no tab position, so it's skipped from the numbered sequence
        entirely; Recycle Bin/Confirmed are reached only via shift/
        space+click, not by number -- they aren't something you pick new
        particles into."""
        normal = self.boxesviewer.normal_class_order()
        if not (1 <= n <= len(normal)):
            return
        self.boxesviewer.activate_class_tab(normal[n - 1])

    def _on_gallery_class_activated(self, cid):
        """A set's tab becoming the active one in the gallery also makes it
        the active picking class -- see ParticleListPanel.class_activated.
        cid can already equal currentset here (e.g. clicking back onto a
        class's tab after visiting "All" or a fixed category, which don't
        change currentset) -- set_current_set itself is skippable then,
        but the solo-visibility resync isn't: sets_visible could still
        have drifted since the last time it was forced, and this tab
        becoming current is exactly the moment that should be corrected."""
        if cid == self.currentset:
            if self._sync_solo_visibility():
                self.update_all()
            return
        self.set_current_set(f"{cid}::")

    def _on_gallery_tab_changed(self, cid):
        """Show the active channel only has to follow the gallery to *every* tab,
        not just the ones that can become the active picking class --
        unlike class_activated (which only ever fires for normal sets),
        active_tab_changed also fires for Recycle Bin, Confirmed, and
        "All" (cid is None here), none of which touch currentset at all.
        Without this, switching to one of those three while solo is on
        left it not shown and whatever was visible before still visible,
        since nothing was listening for exactly that tab change. "All"
        means "show everything" -- the same thing that tab itself
        displays -- since there's no single class to isolate to there."""
        if not self.view_only_selected_action.isChecked():
            return
        target = set(self.sets.keys()) if cid is None else {cid}
        if self.sets_visible == target:
            return
        self.sets_visible = target
        self.boxesviewer.refresh_visibility_checkboxes()
        self.update_all()

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

