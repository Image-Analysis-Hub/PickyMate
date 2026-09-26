"""PickyMate's main window: the tomogram slice-view/picking widget.

Built on plain PySide6 + numpy + mrcfile.

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

from . import embeddings_io
from .embeddings_wizard import EmbeddingsImportDialog
from .help_dialog import ShortcutsDialog
from .image_view import ImageView2D
from .settings_dialog import KeyboardShortcutsDialog
from .thumb_grid import ParticleListPanel
from .widgets import LabeledSpinBox, RangeSlider, TOOLBAR_BUTTON_STYLE, space_key_state
from . import categories
from . import metadata
from . import shortcuts
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
        # Set when this window is one tomogram of a multi-tomogram session
        # (tomo_boxer.py --session): the controller owns the navigator and
        # does the switching, the session reads other tomograms' working
        # directories. Both None in ordinary single-tomogram use, and every
        # session feature below is keyed off that.
        self.session_controller = getattr(options, "session_controller", None)
        self.session = self.session_controller.session if self.session_controller else None
        self.session_name = metadata.base_name(datafile)
        self._switching = False
        # uids of the particles last selected on the UMAP map (umap_view),
        # saved to "UMAP Selection.csv" in the working directory; shown as
        # the particle list's "UMAP Selection" tab while non-empty
        self.umap_selection = set()
        self._umap_saved = None      # the uids last written to the selection file
        self.umap_window = None
        self._last_tab_was_umap = False
        self.boxshape = "circle" if options.mode == "3D" else "rect"
        self.setWindowTitle(f"PickyMate — {metadata.base_name(datafile)}")

        # Install the space-bar tracker now rather than waiting for the
        # first space+click gesture to trigger it lazily -- see
        # space_key_state()'s docstring for why that races the very first
        # keypress of the session.
        space_key_state()

        # -- menu ---------------------------------------------------------
        self.mfile = self.menuBar().addMenu("File")
        self.mfile_save_all = self.mfile.addAction("Save")
        self.mfile.addSeparator()
        self.mfile_read_boxloc = self.mfile.addAction("Import Particle List...")
        self.mfile_import_with_embeddings = self.mfile.addAction("Import Particle List with Embeddings...")
        self.mfile_attach_embeddings = self.mfile.addAction("Attach Embeddings to Current Particles...")
        self.mfile_attach_embeddings.setToolTip(
            "Add another method's embeddings to the particles already imported, so the "
            "same picks can be ranked by each method and compared.")
        self.mfile_save_boxloc = self.mfile.addAction("Export Particle Coordinates...")
        self.mfile.addSeparator()
        self.load_companion_action = QtGui.QAction("Load Companion Volume...", self)
        self.load_companion_action.setToolTip(
            "Load a second version of this same tomogram (e.g. denoised) to switch "
            "display between -- picks, sets, and box sizes are shared; only the pixel "
            "data shown changes. Must match the primary tomogram's dimensions exactly.")
        self.load_companion_action.triggered.connect(self.load_companion_volume_dialog)
        self.mfile.addAction(self.load_companion_action)

        # Each of these is populated further down, once the actions/objects
        # it needs exist -- created here (empty) so the menu bar's overall
        # order (File, Edit, Tools, View, Set, Settings, Help) is fixed
        # regardless of where in __init__ each menu's own actions happen
        # to be built.
        self.medit = self.menuBar().addMenu("Edit")
        self.mtools = self.menuBar().addMenu("Tools")
        self.mview = self.menuBar().addMenu("View")
        self.mset = self.menuBar().addMenu("Set")
        self.msettings = self.menuBar().addMenu("Settings")

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
        self.gbl.addLayout(gblh, 0, 1, 2, 2)

        self.splitter_top = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter_bottom = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter_wrapper = splitter_wrapper = QtWidgets.QSplitter(Qt.Vertical)

        self.xyview = ImageView2D(sizehint=(1024, 1024))
        self.xzview = ImageView2D(sizehint=(1024, 256))
        self.zyview = ImageView2D(sizehint=(256, 1024))

        # Three slice sliders, one per view's own depth axis -- each
        # spatially next to the view it drives, rather than all three
        # clustered arbitrarily: wdepth_x sits at the far left, parallel
        # to the top row (zyview is the leftmost view there); wdepth
        # (Z, XY view) stays at the far right, where it always was;
        # wdepth_y runs along the bottom, right below the row xzview
        # itself sits in, spanning exactly as wide as the view grid above
        # it (gblh) -- ending at the XZ view's own right edge rather than
        # the full window width. All three also respond to Shift+wheel
        # over their own view, and stay in sync however the position
        # actually changes (see update_coords/_sync_depth_sliders).
        self.wdepth_x = QtWidgets.QSlider()
        self.wdepth_x.setToolTip("X depth (ZY view) -- Shift+wheel over that view also steps it.")
        self.gbl.addWidget(self.wdepth_x, 0, 0)

        self.wdepth = QtWidgets.QSlider()
        self.wdepth.setToolTip("Z depth (XY view) -- Shift+wheel over that view also steps it.")
        self.gbl.addWidget(self.wdepth, 0, 3)

        self.wdepth_y = QtWidgets.QSlider(Qt.Horizontal)
        self.wdepth_y.setToolTip("Y depth (XZ view) -- Shift+wheel over that view also steps it.")
        self.gbl.addWidget(self.wdepth_y, 2, 1, 1, 2)

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
        self._indices_by_class = {}  # classid -> {index: None} (ordered set of
                                 # self.boxes indices in that class) -- lets
                                 # mouse_down's click hit-test skip particles in
                                 # a hidden class outright instead of visiting
                                 # and then rejecting each one; see
                                 # _visible_candidate_indices/_rebuild_indices_by_class.
        self.dragging = -1
        self._select_drawing = None  # in-progress shape-select gesture, or None
        self._next_uid = 0
        self._undo_stack = []
        self._redo_stack = []

        # -- wiring -----------------------------------------------------------
        self.mfile_save_all.triggered.connect(self.menu_file_save_all)
        self.mfile_read_boxloc.triggered.connect(self.menu_file_read_boxloc)
        self.mfile_import_with_embeddings.triggered.connect(self.menu_file_import_with_embeddings)
        self.mfile_attach_embeddings.triggered.connect(self.menu_file_attach_embeddings)
        self.mfile_save_boxloc.triggered.connect(self.menu_file_save_boxloc)

        self.wdepth.valueChanged[int].connect(self.event_depth)
        self.wdepth_y.valueChanged[int].connect(self.event_depth_y)
        self.wdepth_x.valueChanged[int].connect(self.event_depth_x)
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
        self.eraser_radius = LabeledSpinBox(label="Radius:", value=64, minimum=1, maximum=8192)
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

        self.mtools.addAction(self.tool_pick_action)
        self.mtools.addAction(self.tool_eraser_action)
        self.mtools.addAction(self.tool_select_action)

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

        self.medit.addSeparator()
        self.refresh_action = QtGui.QAction("Refresh", self)
        self.refresh_action.setShortcut(QtGui.QKeySequence(Qt.Key_F5))
        self.refresh_action.setShortcutContext(Qt.ApplicationShortcut)
        self.refresh_action.setToolTip(
            "Force everything to fully re-derive and redraw -- regenerates every "
            "gallery thumbnail from the volume and fully re-slices the tomogram "
            "views, in case rendering ever visibly gets stuck. (F5)")
        self.refresh_action.triggered.connect(self.refresh_all)
        self.medit.addAction(self.refresh_action)

        self.confirm_selected_action = QtGui.QAction("Confirm Selected", self)
        self.confirm_selected_action.setToolTip(
            "Send every currently selected particle to Confirmed. (Insert key)")
        # Same idea as Recycle Selected's Delete key below: application-wide,
        # and naturally inert (via Qt's own disabled-action-shortcut
        # behavior) whenever nothing is selected.
        self.confirm_selected_action.setShortcut(QtGui.QKeySequence(Qt.Key_Insert))
        self.confirm_selected_action.setShortcutContext(Qt.ApplicationShortcut)
        self.confirm_selected_action.triggered.connect(lambda: self.confirm_particles_by_uid(self._selected_uids))

        self.recycle_selected_action = QtGui.QAction("Recycle Selected", self)
        self.recycle_selected_action.setToolTip(
            "Send every currently selected particle to the Recycle Bin. (Delete key)")
        self.recycle_selected_action.setShortcut(QtGui.QKeySequence(Qt.Key_Delete))
        # Application-wide, like Undo/Redo/the number keys, so the Delete
        # key reaches this from anywhere rather than only while some
        # specific widget happens to have focus. A disabled QAction's
        # shortcut is inert in Qt, so this is naturally a no-op whenever
        # nothing is selected -- no separate guard needed here.
        self.recycle_selected_action.setShortcutContext(Qt.ApplicationShortcut)
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
        self.move_to_combo.lineEdit().setPlaceholderText("Type a set name…")

        self.move_selected_action = QtGui.QAction("Move Selected To", self)
        self.move_selected_action.setToolTip("Move every currently selected particle to the set named at right.")
        self.move_selected_action.triggered.connect(self._move_selected_via_toolbar)

        self.focus_selected_action = QtGui.QAction("Show Only Selected", self)
        self.focus_selected_action.setCheckable(True)
        self.focus_selected_action.setToolTip(
            "While on, a particle only shows in the tomogram views if it's both "
            "currently selected AND still in whatever set it was in when it "
            "joined the selection -- moving it to another set (Space/Shift+click, "
            "drag, ...) hides it right away without deselecting it, even though "
            "its new set is itself visible. Updates live as the selection changes.")
        self.focus_selected_action.toggled.connect(self._toggle_focus_selected)

        self.medit.addSeparator()
        self.medit.addAction(self.confirm_selected_action)
        self.medit.addAction(self.recycle_selected_action)
        self.medit.addAction(self.move_selected_action)
        self.medit.addAction(self.focus_selected_action)

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
        self.edit_toolbar.addSeparator()
        self.edit_toolbar.addAction(self.refresh_action)

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
        self.selection_toolbar.addSeparator()
        self.selection_toolbar.addAction(self.focus_selected_action)

        # Show Only Selected quietly hides every other particle, which can
        # be confusing to stumble back into later (particles look "gone"
        # for no visible reason) if you forget it's still on -- pulsing
        # its own toolbar button for as long as it's checked keeps it from
        # being a silent, easy-to-forget mode.
        self._focus_pulse_timer = QtCore.QTimer(self)
        self._focus_pulse_timer.setInterval(500)  # ms; on+off = a 1-second cycle
        self._focus_pulse_timer.timeout.connect(self._pulse_focus_button)
        self._focus_pulse_on = False
        self.focus_selected_action.toggled.connect(self._update_focus_pulse)

        # Show Active Set Only / Show All Sets' QActions are created here
        # (TomoBoxer owns sets_visible and the solo logic), but -- like
        # Toggle Colors above -- the buttons for them live in the particle
        # list's own toolbar, not here; see ParticleListPanel.__init__.
        self.view_only_selected_action = QtGui.QAction("Show Active Set Only", self)
        self.view_only_selected_action.setCheckable(True)
        self.view_only_selected_action.setToolTip(
            "While on, only the active tab's set is shown in the tomogram "
            "views -- switching tabs (including via number keys) updates "
            "this live. Turning it off restores whatever visibility you "
            "had before.")
        self.view_only_selected_action.toggled.connect(self.toggle_view_only_selected)
        self._solo_saved_visible = None  # sets_visible snapshot from before solo turned on

        self.show_all_sets_action = QtGui.QAction("Show All Sets", self)
        self.show_all_sets_action.setToolTip("Make every set visible in the tomogram views.")
        self.show_all_sets_action.triggered.connect(self.show_all_sets)

        # Disabled until a companion volume is actually loaded (see
        # load_companion_volume_dialog/_load_companion_volume below).
        self.toggle_companion_action = QtGui.QAction("Show Companion Volume", self)
        self.toggle_companion_action.setCheckable(True)
        self.toggle_companion_action.setEnabled(False)
        self.toggle_companion_action.setToolTip(
            "Switch the tomogram views and gallery thumbnails between the primary "
            "volume and its loaded companion (e.g. denoised). Contrast and Avg "
            "Slices are kept separate per volume; picks/sets are shared.")
        self.toggle_companion_action.toggled.connect(self._on_toggle_companion)

        self.mview.addAction(self.toggle_companion_action)
        self.umap_action = QtGui.QAction("UMAP Map...", self)
        self.umap_action.setToolTip(
            "Open the UMAP map of this tomogram's particles, coloured by set, and "
            "lasso-select particles on it into the \"UMAP Selection\" tab.")
        self.umap_action.triggered.connect(self.open_umap_window)
        self.mview.addAction(self.umap_action)
        self.mview.addSeparator()
        self.mview.addAction(self.toggle_colors_action)
        self.mview.addSeparator()
        self.mview.addAction(self.view_only_selected_action)
        self.mview.addAction(self.show_all_sets_action)

        self.addToolBarBreak()
        self.volume_toolbar = self.addToolBar("Volume")
        self.volume_toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.volume_toolbar.addAction(self.load_companion_action)
        self.volume_toolbar.addAction(self.toggle_companion_action)

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

        # -- customizable keyboard shortcuts (Settings > Customize Keyboard
        # Shortcuts...) -- registered here, now that every action/shortcut
        # below has its own built-in default already set, so that default
        # can just be read back rather than duplicated a second time.
        self._shortcut_registry = []
        self._register_shortcut("undo", "Undo", self.undo_action)
        self._register_shortcut("redo", "Redo", self.redo_action, multi=True)
        self._register_shortcut("refresh", "Refresh", self.refresh_action)
        self._register_shortcut("recycle_selected", "Recycle Selected (send to Recycle Bin)", self.recycle_selected_action)
        self._register_shortcut("confirm_selected", "Confirm Selected (send to Confirmed)", self.confirm_selected_action)
        self._register_shortcut("show_only_selected", "Show Only Selected", self.focus_selected_action)
        self._register_shortcut("tool_pick", "Tool: Pick", self.tool_pick_action)
        self._register_shortcut("tool_eraser", "Tool: Eraser", self.tool_eraser_action)
        self._register_shortcut("tool_select", "Tool: Select", self.tool_select_action)
        self._register_shortcut("shortcuts_reference", "Keyboard Shortcuts Reference (Help menu)", self.mhelp_shortcuts_action)
        self._register_shortcut("clear_selection", "Clear Selection", self._escape_shortcut)
        self._apply_shortcut_overrides(shortcuts.load_overrides())

        self.msettings_shortcuts_action = self.msettings.addAction("Customize Keyboard Shortcuts...")
        self.msettings_shortcuts_action.triggered.connect(self.open_shortcuts_settings)

        # -- load data ---------------------------------------------------------
        self.datafilename = datafile

        # Companion-volume support (e.g. denoised vs. non-denoised display of
        # the exact same tomogram): _volume_slots holds the two possible
        # Volume objects, _active_volume_key says which one self.data
        # currently aliases, and _volume_ui_state remembers each volume's own
        # Avg Slices / contrast settings across switches (see _switch_volume).
        # The companion is loaded lazily -- only actually read from disk the
        # first time it's switched to -- even when its path is already known
        # from --companion_tomogram, so specifying one costs nothing for
        # anyone who never uses the switch.
        self._native_memmap = bool(getattr(options, "native_memmap", False))
        self._volume_slots = {"primary": None, "companion": None}
        self._volume_ui_state = {}
        self._active_volume_key = "primary"
        self.companion_path = getattr(options, "companion_tomogram", None)
        if self.companion_path:
            self.companion_path = os.path.abspath(self.companion_path)

        data = (self.session.take_prefetched(datafile, self._native_memmap)
                if self.session is not None else None)
        if data is None:
            data = Volume.from_file(datafile, native_memmap=self._native_memmap)
        if options.apix:
            data.apix = options.apix
        self._volume_slots["primary"] = data
        self.data = None
        self.set_data(data)
        if self.companion_path:
            self.toggle_companion_action.setEnabled(True)

        # Per-class plain-text pick files live in "<basename>_picks/" next
        # to the tomogram by default, or wherever --working_dir points
        # (resolved relative to the current working directory) instead.
        self.picksdir = (
            os.path.abspath(options.working_dir) if getattr(options, "working_dir", None)
            else metadata.picks_dir(datafile))

        # Mahalanobis-distance feature: at most one embeddings bundle is
        # attached at a time (see import_particle_list_with_embeddings).
        # Auto-reload whichever one a previous session attached, so it
        # doesn't have to be re-specified on every launch just because
        # some already-imported particles still carry an embedding_row --
        # a fresh --embedded_particle_list import below (if given) simply
        # replaces this.
        # name -> {"array", "method", "kind", "n", "path", "label"}. Several
        # methods can be attached at once (CryoSIAM and TomoTwin over the
        # same picks, say) and each contributes its full embedding and, if
        # given, its UMAP as separate rankable spaces -- so the ranking
        # dropdown can offer "CryoSIAM - full (1024-d)" alongside
        # "TomoTwin - UMAP (2-d)" and they can be compared directly.
        # Insertion order is the order they were attached, which is the
        # order the dropdown shows.
        self.embedding_spaces = {}
        self.last_import_notes = []
        # A session manifest can name the current location of embedding
        # files this working directory recorded somewhere that no longer
        # exists (see SessionEntry.embedding_path_fixes); outside a session
        # there are none and a missing file is simply skipped, as before.
        path_fixes = getattr(options, "embedding_path_fixes", None) or {}
        repaired = []
        for saved in metadata.load_embeddings_settings(self.picksdir, include_missing=True):
            path = saved["path"]
            if not saved["exists"]:
                if saved["name"] not in path_fixes:
                    continue
                path = path_fixes[saved["name"]]
            try:
                if saved["kind"] == "umap":
                    array = embeddings_io.load_umap(saved["method"], path)["umap_xy"]
                else:
                    array = embeddings_io.load_embeddings(
                        saved["method"], path, n_expected=saved["n"])["embeddings"]
                self._register_embedding_space(saved["method"], saved["kind"], array, path)
                if path != saved["path"]:
                    repaired.append((saved["name"], saved["path"], path))
            except (OSError, ValueError) as e:
                print(f"warning: could not reload the {saved['name']!r} embeddings: {e}")
        if repaired:
            # Write the corrected locations back, so the working directory
            # also opens correctly on its own, outside the session. Only the
            # repaired paths change: rewriting from the spaces that loaded
            # would silently erase the record of any that could not be
            # loaded, and with it the only note of where that file was.
            fixed = {name: new_path for name, _, new_path in repaired}
            metadata.save_embeddings_settings(self.picksdir, [
                dict(sp, path=fixed.get(sp["name"], sp["path"]))
                for sp in metadata.load_embeddings_settings(self.picksdir, include_missing=True)])
            for name, old_path, new_path in repaired:
                print(f"note: {name!r} embeddings were recorded at {old_path}, which no longer "
                      f"exists; now using {new_path} from the session manifest")

        self.boxesviewer = ParticleListPanel(self)
        self.boxesviewer.setWindowTitle(f"PickyMate — Particle List — {metadata.base_name(datafile)}")
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

        # Every gallery action that only had a button/gesture until now
        # (double-click a tab to rename it, the "+" tab to create one, the
        # bottom toolbar's own actions) also gets a menu entry here, so
        # everything reachable in the app is reachable from the menu bar,
        # not just from toolbars/buttons you have to already know about.
        self.mset.addAction(self.boxesviewer.new_set_action)
        self.mset.addAction(self.boxesviewer.rename_set_action)
        self.mset.addAction(self.boxesviewer.delete_set_action)
        self.mset.addSeparator()
        self.mset.addAction(self.boxesviewer.organize_action)
        self.mset.addAction(self.boxesviewer.detach_action)
        self.mset.addSeparator()
        self.mset.addAction(self.boxesviewer.sort_by_id_action)
        self.mset.addAction(self.boxesviewer.sort_by_rank_action)
        self.mset.addSeparator()
        self.mset.addAction(self.boxesviewer.rank_action)
        self.mset.addSeparator()
        self.mset.addAction(self.boxesviewer.empty_recycle_action)

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

        # Two passes: first load every class's records and find the highest
        # explicit uid among them, so self._next_uid can start safely above
        # it -- otherwise a legacy (uid-less) entry loaded earlier in the
        # population loop below could auto-assign a uid that collides with
        # an explicit one belonging to a different class's file, loaded
        # moments later.
        normal_records = {name: metadata.load_class_records(self.picksdir, name) for name in normal_names}
        fixed_records = {name: metadata.load_class_records(self.picksdir, name) for name in categories.FIXED_CLASS_NAMES}
        max_uid = -1
        for records in list(normal_records.values()) + list(fixed_records.values()):
            for rec in records:
                if rec["uid"] is not None and rec["uid"] > max_uid:
                    max_uid = rec["uid"]
        self._next_uid = max(self._next_uid, max_uid + 1)

        for i, name in enumerate(normal_names):
            self.sets[i] = name
            s = class_settings.get(name, {})
            self.boxsize[i] = s.get("boxsize", metadata.DEFAULT_BOXSIZE)
            if s.get("color"):
                categories.set_manual_color(i, QtGui.QColor(s["color"]))
            for rec in normal_records[name]:
                uid = rec["uid"] if rec["uid"] is not None else self._new_uid()
                self.boxes.append([rec["x"], rec["y"], rec["z"], "manual", rec["score"], i, uid, rec["embedding_row"]])

        # The Recycle Bin and Confirmed categories always exist, whether or
        # not their pick files were present on disk yet.
        for name, cid in categories.FIXED_CLASS_NAMES.items():
            self.sets[cid] = name
            s = class_settings.get(name, {})
            self.boxsize[cid] = s.get("boxsize", metadata.DEFAULT_BOXSIZE)
            if s.get("color"):
                categories.set_manual_color(cid, QtGui.QColor(s["color"]))
            for rec in fixed_records[name]:
                uid = rec["uid"] if rec["uid"] is not None else self._new_uid()
                self.boxes.append([rec["x"], rec["y"], rec["z"], "manual", rec["score"], cid, uid, rec["embedding_row"]])

        # Repair duplicate uids found on disk: the prescan above only
        # protects a freshly *auto-assigned* uid from colliding with an
        # existing explicit one, but two files can each already carry the
        # same explicit uid (this happens to real data -- found in
        # practice on a working directory that predates this uid-
        # persistence scheme, where separate sessions independently
        # assigned uids without seeing each other's files). A duplicate
        # uid is silently dangerous, not just cosmetic: every gesture
        # that resolves "the particle with uid X" by scanning self.boxes
        # for the first match (_index_for_uid, used by Shift/Space-click)
        # would then act on whichever copy happens to come first, which
        # can be a completely different particle than the one actually
        # clicked -- if that first copy already belongs to the gesture's
        # target class, the reassignment silently no-ops, making the
        # particle you actually clicked look like it's being ignored.
        # Keep the first occurrence's uid as-is and give every later
        # duplicate a fresh one; the next save (already automatic)
        # persists the fix.
        seen_uids = set()
        for b in self.boxes:
            if b[6] in seen_uids:
                print(f"warning: duplicate uid {b[6]} found on disk (class "
                      f"{self.sets.get(b[5], b[5])!r}) -- reassigned to keep every "
                      f"particle uniquely identifiable")
                b[6] = self._new_uid()
            seen_uids.add(b[6])

        shape = [self.data.nx, self.data.ny, self.data.nz]
        for b in self.boxes:
            for i in range(3):
                b[i] = int(np.clip(round(b[i]), 1, shape[i] - 1))

        self._rebuild_indices_by_class()

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

        # --cryosiam_embeddings / --tomotwin_embeddings (+ their own umap
        # flags, and --embedded_particle_list): the one list that has a
        # matched embeddings bundle -- see
        # import_particle_list_with_embeddings for the validation. Any
        # failure here is a warning, not a fatal error: the launch still
        # proceeds, just without that bundle attached.
        # More than one method can be given at once: the first creates the
        # set, and each one after it attaches to those same particles, so
        # the ranking dropdown ends up offering every method's spaces side
        # by side over one set of picks.
        embedded_list = getattr(options, "embedded_particle_list", None)
        for i, (method, emb_path, umap_path) in enumerate(self._embedding_flags(options)):
            try:
                if i == 0:
                    unique_name, count = self.import_particle_list_with_embeddings(
                        embedded_list, emb_path, umap_path, live=False, method=method)
                    print(f"Imported {count} particle(s) into new set \"{unique_name}\", "
                          f"with {method} embeddings attached")
                else:
                    added = self.attach_embeddings(method, emb_path, umap_path)
                    print(f"Attached {method} embeddings to the same particles "
                          f"({', '.join(added)})")
                for note in self.last_import_notes:
                    print(f"note: {note}")
            except (ValueError, OSError) as e:
                print(f"warning: could not load {method} embeddings from {emb_path!r}: {e}")

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

        self._load_umap_selection()

        for i in range(len(self.boxes)):
            self.update_box(i)

        self.update_all()
        self.initialized = True

        # Land on the first real set's own tab, not "All" -- grouping
        # every particle across every class (ParticleGrid.
        # sync_particles_grouped) is the single most expensive gallery
        # view to build, and isn't what you want staring back at you the
        # moment a large tomogram finishes opening anyway. Whatever's
        # first in the gallery's own tab order (after Recycle Bin/
        # Confirmed), matching what the tab bar actually shows.
        first_normal = self.boxesviewer.normal_class_order()
        if first_normal:
            self.boxesviewer.activate_class_tab(first_normal[0])

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
        self.wdepth_y.setRange(0, data.ny - 1)
        self.wdepth_y.setValue(data.ny // 2)
        self.wdepth_x.setRange(0, data.nx - 1)
        self.wdepth_x.setValue(data.nx // 2)
        self.boxes = []
        self._indices_by_class = {}
        self.curbox = -1
        self._next_uid = 0
        self._undo_stack = []
        self._redo_stack = []
        self._selected_uids = set()
        self._focus_home_class = {}  # uid -> class at the moment it joined the Show-Only-Selected review; see _is_visible_particle
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
        contrast range and orders of magnitude faster.

        mean/std are computed with dtype=np.float32 explicitly rather than
        relying on the sample's own dtype: a float16-loaded volume (see
        Volume.from_file) has millions of voxels in this sample, and
        float16's accumulator overflows to inf on std() well before the
        reduction finishes -- silently breaking AutoContrast. int8/int16
        sources don't have this problem (numpy already upgrades integer
        reductions to float64), but pinning float32 here costs nothing and
        keeps this correct regardless of whatever dtype the volume ends up
        being loaded as."""
        sample = data.data[::4, ::4, ::4]
        mean, std = float(sample.mean(dtype=np.float32)), float(sample.std(dtype=np.float32))
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
            # dtype=np.float32: see Volume.slab_projection's docstring --
            # an axis-reduced mean() over a float16-loaded volume doesn't
            # auto-promote its accumulator the way a full reduction does.
            return cube.mean(axis=0, keepdims=True, dtype=np.float32)
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

    def event_depth_y(self):
        if self.y_loc != self.wdepth_y.value():
            self.y_loc = self.wdepth_y.value()
        if self.initialized:
            self.update_sliceview()

    def event_depth_x(self):
        if self.x_loc != self.wdepth_x.value():
            self.x_loc = self.wdepth_x.value()
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

    # -- companion volume (e.g. denoised alternative display) ---------------
    def load_companion_volume_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Companion Volume", os.path.dirname(self.datafilename) or ".",
            "Tomogram files (*.mrc *.rec *.mrcs *.st);;All files (*)")
        if not path:
            return
        if not self._load_companion_volume(path):
            return
        if self.toggle_companion_action.isChecked():
            # Already showing the companion slot -- replacing its file with
            # setChecked(True) below would be a silent no-op (already
            # checked), so apply the swap explicitly instead.
            self.data = self._volume_slots["companion"]
            self._init_contrast_slider(self.data)
            self._restore_volume_ui_state("companion")
            if self.initialized:
                self.refresh_all()
        else:
            # Jump straight to it -- loading one and not seeing it applied
            # would look like nothing happened.
            self.toggle_companion_action.setChecked(True)

    def _load_companion_volume(self, path):
        try:
            vol = Volume.from_file(path, native_memmap=self._native_memmap)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Companion Volume", str(e))
            return False
        primary = self._volume_slots["primary"]
        if (vol.nx, vol.ny, vol.nz) != (primary.nx, primary.ny, primary.nz):
            QtWidgets.QMessageBox.critical(
                self, "Load Companion Volume",
                f"This file is {vol.nx}x{vol.ny}x{vol.nz}, but the primary tomogram is "
                f"{primary.nx}x{primary.ny}x{primary.nz}. A companion volume must match "
                "exactly, since picks are shared between the two.")
            return False
        # Picks are defined in the primary volume's physical coordinate
        # space; matching apix keeps distances/box sizes meaning the same
        # thing regardless of which volume happens to be on screen.
        vol.apix = primary.apix
        self._volume_slots["companion"] = vol
        self.companion_path = path
        self.toggle_companion_action.setEnabled(True)
        return True

    def _on_toggle_companion(self, checked):
        self._switch_volume("companion" if checked else "primary")

    def _switch_volume(self, key):
        """Switch which Volume self.data aliases -- purely a display change.
        Picks/boxes/undo history are untouched (unlike set_data, which is
        for loading a brand new dataset); only the pixel data shown in the
        tomogram views and gallery thumbnails changes. Avg Slices and
        contrast are deliberately per-volume (a denoised volume typically
        wants neither the slab-averaging nor the intensity levels a noisy
        raw volume does), remembered across switches via
        _snapshot_volume_ui_state/_restore_volume_ui_state."""
        if key == self._active_volume_key:
            return
        if self._volume_slots[key] is None:
            if key != "companion" or not self.companion_path:
                return
            if not self._load_companion_volume(self.companion_path):
                self.toggle_companion_action.setChecked(False)
                return
        self._snapshot_volume_ui_state()
        self._active_volume_key = key
        self.data = self._volume_slots[key]
        self._init_contrast_slider(self.data)
        self._restore_volume_ui_state(key)
        if self.initialized:
            self.refresh_all()

    def _snapshot_volume_ui_state(self):
        self._volume_ui_state[self._active_volume_key] = {
            "nlayers": self.wnlayers.value(),
            "autocontrast": self.wbautoc.isChecked(),
            "manual_levels": self.wcontrast.values(),
        }

    def _restore_volume_ui_state(self, key):
        # A never-before-visited volume (state is None) starts at the same
        # defaults a freshly opened tomogram would: Avg Slices 1, AutoContrast
        # on -- not whatever the other volume happened to be using.
        state = self._volume_ui_state.get(key)
        self.wnlayers.blockSignals(True)
        self.wnlayers.setValue(state["nlayers"] if state else 1)
        self.wnlayers.blockSignals(False)
        autocontrast = state["autocontrast"] if state else True
        self.wbautoc.blockSignals(True)
        self.wbautoc.setChecked(autocontrast)
        self.wbautoc.blockSignals(False)
        if autocontrast:
            for v in (self.xyview, self.xzview, self.zyview):
                v.clear_manual_levels()
            self.boxesviewer.clear_levels()
            self.wcontrast.set_values(*self._contrast_defaults, emit=False)
        else:
            lo, hi = state["manual_levels"]
            self.wcontrast.set_values(lo, hi, emit=False)
            self._apply_manual_contrast(lo, hi)

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
        File > Import Particle List menu action (live=True) and --particle_lists
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
        return self._commit_new_set_import(metadata.base_name(path), coords, [None] * len(coords), live)

    def _commit_new_set_import(self, name, coords, embedding_rows, live):
        """Shared tail end of import_particle_list and
        import_particle_list_with_embeddings: create a uniquely-named new
        set (auto-suffixed if that name's already taken -- deliberately:
        the same name can legitimately show up twice with different
        picks, e.g. an import run again against a working directory that
        already has a same-named set from a previous run) and populate it
        with the given coordinates. embedding_rows is parallel to coords,
        entries None for a plain import with nothing to link to.

        `live` covers what only makes sense once the app is already
        running: per-box marker/thumbnail generation and a view refresh.
        During __init__'s own bootstrap (live=False), the boxes appended
        here are picked up by its own trailing update_box loop instead,
        so doing that here too would just be redundant work performed
        before the window is even fully set up. Either way, this set's
        own coordinate file is always written immediately -- "imported"
        means "copied into the working directory," not "queued until
        something else happens to save."

        Returns (unique_name, count)."""
        unique_name = name
        n = 1
        while unique_name in self.sets.values():
            n += 1
            unique_name = f"{name}_{n}"

        clsid = self.new_set(unique_name)

        shape = [self.data.nx, self.data.ny, self.data.nz]
        start = len(self.boxes)
        for (x, y, z), embedding_row in zip(coords, embedding_rows):
            x = int(np.clip(round(x), 1, shape[0] - 1))
            y = int(np.clip(round(y), 1, shape[1] - 1))
            z = int(np.clip(round(z), 1, shape[2] - 1))
            self.boxes.append([x, y, z, "manual", None, clsid, self._new_uid(), embedding_row])
        self._indices_by_class[clsid] = {i: None for i in range(start, len(self.boxes))}

        if live:
            self.initialized = False
            for i in range(start, len(self.boxes)):
                self.update_box(i)
            self.initialized = True

        self.save_json(only={clsid})

        if live:
            self.update_all()

        return unique_name, len(coords)

    @staticmethod
    def _embedding_flags(options):
        """[(method, embeddings_path, umap_path), ...] for every method
        named on the command line, in the order embeddings_io lists them.
        Several can be given at once -- attaching CryoSIAM and TomoTwin
        over the same picks is exactly how you compare them.

        Each method has its own pair of flags rather than one generic
        --embeddings, because the file formats are genuinely different
        and naming the method is clearer than having the app guess from
        an extension (see embeddings_io)."""
        found = []
        for method in embeddings_io.METHODS:
            path = getattr(options, f"{method}_embeddings", None)
            if path:
                found.append((method, path, getattr(options, f"{method}_umap", None)))
        return found

    def _register_embedding_space(self, method, kind, array, path):
        """Add one rankable space to self.embedding_spaces, keyed and
        labelled by method and kind so several methods coexist."""
        name = embeddings_io.space_key(method, kind)
        self.embedding_spaces[name] = {
            "array": array,
            "method": method,
            "kind": kind,
            "n": len(array),
            "path": path,
            "label": embeddings_io.space_label(method, kind, array.shape[1]),
        }
        return name

    def _save_embedding_spaces(self):
        metadata.save_embeddings_settings(self.picksdir, [
            {"name": name, "method": s["method"], "kind": s["kind"],
             "n": s["n"], "path": s["path"]}
            for name, s in self.embedding_spaces.items()
        ])

    def _embedded_row_coordinates(self):
        """(n, (n,3) coordinates by embedding_row) for the particles that
        already carry an embedding row, or (0, None) if none do. This is
        what a newly attached method has to agree with: rows are shared
        across every attached space, so a second method's file has to
        describe the same particles, in the same row order, as the first
        one did."""
        embedded = [(b[7], b) for b in self.boxes if b[7] is not None]
        if not embedded:
            return 0, None
        n = max(row for row, _ in embedded) + 1
        coords = np.full((n, 3), np.nan, dtype=np.float64)
        for row, box in embedded:
            coords[row] = (box[0], box[1], box[2])
        return n, coords

    def attach_embeddings(self, method, embeddings_path, umap_path=None):
        """Attach another method's embeddings to particles that are
        already imported, rather than importing a new set -- this is what
        makes "rank the same picks by CryoSIAM and by TomoTwin, and
        compare" possible.

        The new files must describe the same particles, in the same row
        order, as whatever is already attached: same count, and matching
        coordinates wherever the file records its own. A mismatch raises
        rather than silently pairing each particle with some other
        particle's embedding.

        Adds up to two spaces (the full embedding, and the UMAP if given)
        and returns their names. Notes land in self.last_import_notes."""
        self.last_import_notes = []
        n, row_coords = self._embedded_row_coordinates()
        if not n:
            raise ValueError(
                "No particles here carry an embedding row yet, so there's nothing to "
                "attach to -- import a particle list with embeddings first.")

        bundle = embeddings_io.load_embeddings(method, embeddings_path, n_expected=n)
        if len(bundle["embeddings"]) != n:
            raise ValueError(
                f"{embeddings_path!r} holds {len(bundle['embeddings'])} particle(s), but "
                f"{n} particle(s) here already carry an embedding row.")
        self._check_against_rows(bundle.get("coords_xyz"), row_coords, embeddings_path)
        self._note_volume_shape(bundle.get("volume_shape_zyx"))

        added = [self._register_embedding_space(
            method, "embeddings", bundle["embeddings"], embeddings_path)]

        if umap_path:
            umap_data = embeddings_io.load_umap(method, umap_path)
            if len(umap_data["umap_xy"]) != n:
                raise ValueError(
                    f"{umap_path!r} has {len(umap_data['umap_xy'])} row(s), but "
                    f"{n} particle(s) here carry an embedding row.")
            self._check_against_rows(umap_data.get("coords_xyz"), row_coords, umap_path)
            added.append(self._register_embedding_space(
                method, "umap", umap_data["umap_xy"], umap_path))

        self._save_embedding_spaces()
        return added

    @staticmethod
    def _check_against_rows(own_coords, row_coords, path):
        """Raise if a file's own recorded coordinates disagree with the
        particles already sitting at those embedding rows. Rows with no
        particle (nan) are skipped -- a particle can be deleted after
        import without invalidating everyone else's row."""
        if own_coords is None or row_coords is None:
            return
        known = ~np.isnan(row_coords[:, 0])
        if not known.any():
            return
        if not np.allclose(own_coords[known], row_coords[known], atol=1.0):
            raise ValueError(
                f"{path!r}'s own recorded coordinates don't match the particles "
                f"already here, row for row -- it doesn't appear to describe the "
                f"same picks in the same order.")

    def _note_volume_shape(self, shape):
        """TomoTwin records the shape of the tomogram it was computed on.
        A mismatch against the volume actually open is the signature of
        embeddings computed at a different binning, which would make
        every ranking meaningless -- but it's the user's call, not a
        reason to refuse, so it's reported rather than raised."""
        if not shape or self.data is None:
            return
        actual = (self.data.nz, self.data.ny, self.data.nx)
        if tuple(shape) != actual:
            self.last_import_notes.append(
                f"These embeddings were computed on a {shape[2]}x{shape[1]}x{shape[0]} "
                f"tomogram, but the one open is {actual[2]}x{actual[1]}x{actual[0]}. "
                f"If that's a different binning, the rankings won't mean anything.")

    def import_particle_list_with_embeddings(self, coord_path, embeddings_path, umap_path=None,
                                              live=True, method="cryosiam"):
        """Like import_particle_list, but also attaches a matched
        embeddings bundle: assigns embedding_row = 0..N-1 positionally
        (matching the row order shared by the coordinate file and the
        embeddings file) and keeps the arrays in memory
        (self.embeddings_full / self.embeddings_umap) for the
        Mahalanobis-distance feature. Attaching a new bundle replaces
        whatever was attached before -- there's only ever one active at a
        time (see metadata.save_embeddings_settings).

        `method` selects the reader (see embeddings_io). It also decides
        whether coord_path is required: CryoSIAM's .h5 carries no
        coordinates so a particle list is mandatory, while TomoTwin's
        .temb carries its own X/Y/Z and can be imported alone. Passing a
        particle list anyway is still worthwhile for TomoTwin -- the two
        are then cross-checked against each other.

        This creates the set. To add a *second* method's embeddings to
        the particles it created -- which is how one set gets ranked by
        CryoSIAM and by TomoTwin and the two compared -- use
        attach_embeddings instead.

        Everything is validated before anything is committed: particle
        counts must match exactly across every file given, and wherever a
        file records its own coordinates those must agree with the
        particle list's, particle for particle. A mismatch raises rather
        than importing a wrong association -- there'd be no way to detect
        that afterwards, and it would quietly corrupt every distance
        computed against these particles.

        Returns (unique_name, count). Non-fatal observations (e.g. the
        embeddings were computed on a differently-shaped tomogram than
        the one open) land in self.last_import_notes for the caller to
        show. Raises ValueError -- message meant to be shown to the user
        directly -- on any validation failure."""
        self.last_import_notes = []

        coords = None
        if coord_path:
            coords = metadata.load_particle_list(coord_path)
            if not coords:
                raise ValueError(f"No coordinates found in {coord_path!r}.")

        bundle = embeddings_io.load_embeddings(
            method, embeddings_path, n_expected=len(coords) if coords else None)
        full = bundle["embeddings"]
        n = len(full)

        own_coords = bundle["coords_xyz"]
        if coords is None:
            if own_coords is None:
                raise ValueError(
                    f"{embeddings_path!r} carries no coordinates, so a particle list "
                    f"is needed to import it.")
            coords = [tuple(row) for row in own_coords]
            name_source = embeddings_path
        else:
            name_source = coord_path

        if len(coords) != n:
            raise ValueError(
                f"{embeddings_path!r} holds {n} particle(s) but "
                f"{coord_path!r} has {len(coords)}.")

        listed = np.asarray(coords, dtype=np.float64)
        self._check_against_rows(own_coords if coord_path else None, listed, embeddings_path)
        self._note_volume_shape(bundle.get("volume_shape_zyx"))

        umap_xy = None
        if umap_path:
            umap_data = embeddings_io.load_umap(method, umap_path)
            if len(umap_data["umap_xy"]) != n:
                raise ValueError(
                    f"{umap_path!r} has {len(umap_data['umap_xy'])} row(s), but "
                    f"{embeddings_path!r} has {n} particle(s).")
            self._check_against_rows(umap_data.get("coords_xyz"), listed, umap_path)
            umap_xy = umap_data["umap_xy"]

        unique_name, count = self._commit_new_set_import(
            metadata.base_name(name_source), coords, list(range(n)), live)

        self._register_embedding_space(method, "embeddings", full, embeddings_path)
        if umap_xy is not None:
            self._register_embedding_space(method, "umap", umap_xy, umap_path)
        self._save_embedding_spaces()
        return unique_name, count

    # -- Mahalanobis distance -------------------------------------------------
    def ordered_space_names(self):
        """Attached embedding spaces in the order they are offered --
        TomoTwin's full embedding first (see embeddings_io.display_order)."""
        return embeddings_io.display_order(self.embedding_spaces)

    def compute_mahalanobis(self, reference_classid, target_classid, space=None,
                             leave_one_out=False, negative_classid=None,
                             reference_tomograms=None):
        """Score every embedded particle in target_classid by its
        Mahalanobis distance to reference_classid's embedding
        distribution (mean + covariance, built from whichever of the
        reference set's particles actually have an embedding), storing
        the result as that particle's score -- see
        ParticleGrid.apply_score_order for how it's displayed and
        _reassign_boxes for why moving a particle to a different class
        resets it back to None.

        negative_classid, when given, turns this into a "double distance"
        (contrastive) score instead: distance-to-reference minus
        distance-to-negative_classid, so particles simultaneously close to
        the reference and far from the negative set sort first (e.g. close
        to Confirmed, far from Recycle Bin). This is the plain difference
        of two ordinary Mahalanobis distances, not a full log-likelihood
        ratio: the missing term there (each distribution's own log-
        determinant) is a per-particle constant, so it could never change
        the resulting *order* anyway, and it's undefined here besides --
        these embeddings' covariance is severely rank-deficient (see
        _mahalanobis_distribution), so a real determinant is zero. Both
        terms use the same space and leave_one_out setting.

        space: the name of one attached embedding space (see
        self.embedding_spaces / embeddings_io.space_key) -- e.g.
        "cryosiam", "cryosiam_umap", "tomotwin", "tomotwin_umap". Several
        methods can be attached at once, so this is what picks which one
        to measure in; None means the first attached.

        leave_one_out: applied independently to whichever term(s) --
        reference, negative, or both -- actually equal target_classid
        (silently ignored for a term that doesn't, since a particle that
        isn't a member of that reference was never part of the
        distribution it's being compared to anyway). Excludes each
        particle's own contribution from that reference before scoring it
        -- the exact way, a fresh covariance per particle, which is
        genuinely expensive for anything but a small set (see the
        leave-one-out help text for real numbers).

        Particles with no embedding at all are left untouched -- no
        score, never sortable (the three-tier display); so is a particle
        leave-one-out could only score on one of the two terms (its own
        reference would have shrunk below 2 without it) -- there's
        nothing meaningful to combine it with. Returns a dict with
        "scored", "skipped_no_embedding", and "reference_size" (of the
        positive/"close to" reference) for the caller to report. Raises
        ValueError if either reference has fewer than 2 embedded
        particles, for an unrecognized `space`, or if negative_classid ==
        reference_classid (the score would be zero for every particle)."""
        if negative_classid is not None and negative_classid == reference_classid:
            raise ValueError("The \"far from\" set can't be the same as the \"close to\" set.")
        if space is None and self.embedding_spaces:
            space = self.ordered_space_names()[0]
        entry = self.embedding_spaces.get(space)
        if entry is None:
            known = ", ".join(self.embedding_spaces) or "none attached"
            raise ValueError(f"unrecognized embedding space {space!r} (have: {known})")
        array = entry["array"]

        target_rows = self._mahalanobis_rows_of(target_classid)
        target_total = sum(1 for b in self.boxes if b[5] == target_classid)
        skipped = target_total - len(target_rows)

        # Other tomograms of a session contribute the particles of their
        # own set of the same name -- their Confirmed for "close to
        # Confirmed", their Recycle Bin for "far from Recycle Bin" -- read
        # from their working directories and pooled with this tomogram's.
        extra_pos = extra_neg = None
        used, notes = [], []
        others = [t for t in (reference_tomograms or []) if t != self.session_name]
        if others and self.session is not None:
            dims = array.shape[1]
            extra_pos, used, not_used = self.session.reference_vectors(
                space, self.sets[reference_classid], others, dims)
            notes += [f"{t}: {why}" for t, why in not_used]
            if negative_classid is not None:
                extra_neg, _, not_used = self.session.reference_vectors(
                    space, self.sets[negative_classid], others, dims)
                notes += [f"{t}: {why} (far from)" for t, why in not_used]

        positive, ref_size = self._mahalanobis_distances(
            reference_classid, target_classid, target_rows, array, leave_one_out, extra=extra_pos)

        if negative_classid is not None:
            negative, _ = self._mahalanobis_distances(
                negative_classid, target_classid, target_rows, array, leave_one_out, extra=extra_neg)
            combined = {i: positive[i] - negative[i] for i in positive if i in negative}
            skipped += len(positive) - len(combined)
        else:
            combined = positive

        for i, d in combined.items():
            self.boxes[i][4] = float(d)

        self.save_json(only={target_classid})
        self.update_all(reslice=False, only_indices=[])
        return {"scored": len(combined), "skipped_no_embedding": skipped, "reference_size": ref_size,
                "space": space, "other_tomograms_used": used,
                "extra_reference_size": 0 if extra_pos is None else len(extra_pos),
                "extra_negative_size": 0 if extra_neg is None else len(extra_neg),
                "notes": notes}

    def _mahalanobis_rows_of(self, classid):
        return [(i, b[7]) for i, b in enumerate(self.boxes) if b[5] == classid and b[7] is not None]

    def _mahalanobis_distances(self, reference_classid, target_classid, target_rows, array,
                               leave_one_out, extra=None):
        """{box_index: squared Mahalanobis distance} for every (index,
        embedding_row) pair in target_rows, relative to reference_classid's
        own embedding distribution in `array`. target_rows is computed
        once by the caller (shared across both terms of a double-distance
        call) so "does this particle have an embedding at all" is only
        ever decided from the target set itself, not re-derived per
        reference. Raises ValueError if the reference set has fewer than
        2 embedded particles. Returns (distances, reference_size).

        `extra`: an (M, D) array of further reference vectors -- the same
        set's particles in other tomograms of a session -- pooled with this
        tomogram's own. They are never among the particles being scored,
        so leave-one-out only ever removes this tomogram's own."""
        ref_rows = self._mahalanobis_rows_of(reference_classid)
        n_extra = 0 if extra is None else len(extra)
        if len(ref_rows) + n_extra < 2:
            name = self.sets.get(reference_classid, reference_classid)
            raise ValueError(
                f"\"{name}\" has only {len(ref_rows) + n_extra} particle(s) with an embedding -- "
                f"need at least 2 to build a distribution.")

        def pooled(rows):
            own = array[rows] if len(rows) else np.empty((0, array.shape[1]))
            return own if extra is None else np.vstack([own, extra])

        distances = {}
        if leave_one_out and reference_classid == target_classid:
            for i, row in target_rows:
                others = [r for idx2, r in ref_rows if idx2 != i]
                if len(others) + n_extra < 2:
                    continue
                mu, covinv = self._mahalanobis_distribution(pooled(others))
                distances[i] = float(self._mahalanobis_scores(mu, covinv, array[[row]])[0])
        else:
            mu, covinv = self._mahalanobis_distribution(pooled([row for _, row in ref_rows]))
            idx_list = [i for i, _ in target_rows]
            rows_list = [row for _, row in target_rows]
            d2 = self._mahalanobis_scores(mu, covinv, array[rows_list])
            for i, d in zip(idx_list, d2):
                distances[i] = float(d)
        return distances, len(ref_rows) + n_extra

    @staticmethod
    def _mahalanobis_distribution(vectors):
        """mean + pseudo-inverse covariance of a (N, D) reference array --
        pinv rather than a true inverse since these embeddings are
        typically rank-deficient well before N reaches D (verified
        empirically against real data: covariance rank plateaus far
        below D even with every available particle as reference), so a
        true inverse would simply fail. Building the D x D covariance
        directly (rather than the N x N Gram-matrix/dual approach) is
        deliberate: at D=1024 it's sub-second regardless of N, and the
        dual approach is only cheaper when N << D, which isn't the
        regime this tool runs in."""
        vectors = np.asarray(vectors, dtype=np.float64)
        mu = vectors.mean(axis=0)
        centered = vectors - mu
        cov = (centered.T @ centered) / (len(vectors) - 1)
        covinv = np.linalg.pinv(cov, hermitian=True)
        return mu, covinv

    @staticmethod
    def _mahalanobis_scores(mu, covinv, targets):
        """Squared Mahalanobis distance of each row in targets to (mu,
        covinv). The matmul-based formulation, not einsum -- verified
        the "obvious" einsum('ij,jk,ik->i', ...) version is over 20x
        slower than this at realistic sizes, a difference real enough
        to matter at a few thousand particles."""
        centered = np.asarray(targets, dtype=np.float64) - mu
        return ((centered @ covinv) * centered).sum(axis=1)

    def menu_file_read_boxloc(self):
        fsp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select box coordinate text file", "", "Text files (*.txt);;All files (*)")
        if not fsp or not os.path.isfile(fsp):
            return
        name = metadata.base_name(fsp)
        result = self.import_particle_list(fsp, live=True)
        if result is None:
            QtWidgets.QMessageBox.information(self, "Import Particle List", "No coordinates found in that file.")
            return
        unique_name, count = result
        msg = f"Imported {count} particle(s) into new set \"{unique_name}\"."
        if unique_name != name:
            msg += f' (renamed from "{name}", which was already in use.)'
        QtWidgets.QMessageBox.information(self, "Import Particle List", msg)

    def menu_file_import_with_embeddings(self):
        dlg = EmbeddingsImportDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        method, coord_path, emb_path, umap_path = dlg.values()
        if not emb_path:
            QtWidgets.QMessageBox.information(
                self, "Import Particle List with Embeddings",
                "An embeddings file is required.")
            return
        if dlg.list_required() and not coord_path:
            QtWidgets.QMessageBox.information(
                self, "Import Particle List with Embeddings",
                "CryoSIAM embeddings carry no coordinates, so a particle list is required too.")
            return
        try:
            unique_name, count = self.import_particle_list_with_embeddings(
                coord_path, emb_path, umap_path, live=True, method=method)
        except (ValueError, OSError) as e:
            QtWidgets.QMessageBox.critical(self, "Import Particle List with Embeddings", str(e))
            return
        self.boxesviewer.rebuild_tabs(self.sets)
        msg = (f"Imported {count} particle(s) into new set \"{unique_name}\", "
               f"with {method} embeddings attached.")
        for note in self.last_import_notes:
            msg += f"\n\n{note}"
        QtWidgets.QMessageBox.information(self, "Import Particle List with Embeddings", msg)

    def menu_file_attach_embeddings(self):
        """Add another method's embeddings to the particles already
        imported -- the GUI counterpart of giving two methods' flags on
        one command line. Creates no set: the rows are already fixed by
        whatever import first attached embeddings here."""
        title = "Attach Embeddings to Current Particles"
        dlg = EmbeddingsImportDialog(self, attach=True)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        method, _coord_path, emb_path, umap_path = dlg.values()
        if not emb_path:
            QtWidgets.QMessageBox.information(self, title, "An embeddings file is required.")
            return
        try:
            added = self.attach_embeddings(method, emb_path, umap_path)
        except (ValueError, OSError) as e:
            QtWidgets.QMessageBox.critical(self, title, str(e))
            return
        self.boxesviewer.rebuild_tabs(self.sets)
        msg = ("Attached " + ", ".join(self.embedding_spaces[a]["label"] for a in added)
               + ".\n\nPick which one to rank in from the dropdown next to \"Rank\".")
        for note in self.last_import_notes:
            msg += f"\n\n{note}"
        QtWidgets.QMessageBox.information(self, title, msg)

    def run_compute_distance(self, target_classid, reference_classid, space, leave_one_out,
                             negative_classid=None, reference_tomograms=None):
        """Triggered by the particle list's own inline "Rank by Embedding Distance"
        controls (see ParticleListPanel) -- there's no dialog or menu
        entry point, this is the feature's only one, deliberately
        prominent and edited in place rather than tucked into File or
        behind a popup. The controls there are already responsible for
        disabling themselves when no embeddings are loaded (or when
        negative_classid conflicts with reference_classid), so unlike the
        old dialog-based version this doesn't re-check that itself."""
        try:
            result = self.compute_mahalanobis(
                reference_classid, target_classid, space, leave_one_out, negative_classid,
                reference_tomograms=reference_tomograms)
        except ValueError as e:
            self._show_gallery_notice(QtWidgets.QMessageBox.Critical, "Rank Particles", str(e))
            return
        # A one-shot re-layout of the target tab by the score just
        # computed, exactly like clicking "Sort by Rank" by hand --
        # not an ongoing mode, so it doesn't fight later edits (see
        # ParticleGrid.apply_score_order).
        self.boxesviewer.activate_class_tab(target_classid)
        self.boxesviewer.sort_active_tab_by_score()
        msg = (f"Scored {result['scored']} particle(s) in \"{self.sets[target_classid]}\" -- close to "
               f"{result['reference_size']} particle(s) in \"{self.sets[reference_classid]}\"")
        if negative_classid is not None:
            msg += f", far from \"{self.sets[negative_classid]}\""
        msg += "."
        if result["other_tomograms_used"]:
            msg += (f"\nReferences pooled from this tomogram and "
                    f"{len(result['other_tomograms_used'])} other(s): "
                    f"{', '.join(result['other_tomograms_used'])} "
                    f"(+{result['extra_reference_size']} \"close to\" particle(s)"
                    + (f", +{result['extra_negative_size']} \"far from\"" if negative_classid is not None else "")
                    + ").")
        if result["notes"]:
            msg += "\nNot used: " + "; ".join(result["notes"]) + "."
        if result["skipped_no_embedding"]:
            msg += f"\n{result['skipped_no_embedding']} particle(s) not scored (no embedding, or excluded by leave-one-out)."
        self._show_gallery_notice(QtWidgets.QMessageBox.Information, "Rank Particles", msg)

    def _show_gallery_notice(self, icon, title, text):
        """A result notice for something triggered from the particle list
        (not this window) -- centered over the particle list, and
        non-modal, so it doesn't block you from immediately clicking a
        particle there while it's still up (a modal QMessageBox.
        information()/.critical() parented to *this* window did both:
        appeared over the wrong window and forced you to dismiss it with
        Ok before you could touch anything else)."""
        box = QtWidgets.QMessageBox(icon, title, text, QtWidgets.QMessageBox.Ok, self.boxesviewer)
        box.setModal(False)
        box.setAttribute(Qt.WA_DeleteOnClose)
        box.show()
        geo = box.frameGeometry()
        geo.moveCenter(self.boxesviewer.geometry().center())
        box.move(geo.topLeft())

    def menu_file_save_boxloc(self):
        """Picks are already autosaved continuously (see save_picks) -- this
        just copies the already-written coordinate file(s), all sets or one
        you choose, to a folder of your own for safekeeping/mental sanity,
        rather than re-deriving and writing them again from scratch."""
        order = categories.category_order(self.sets)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export Particle Coordinates")
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
                self, "Export Particle Coordinates", f"Saved a copy of {len(copied)} set(s) to {destdir}.")
        else:
            QtWidgets.QMessageBox.information(self, "Export Particle Coordinates", "Nothing to save yet.")

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
            records = [
                {"x": b[0], "y": b[1], "z": b[2], "uid": b[6], "embedding_row": b[7], "score": b[4]}
                for b in self.boxes if b[5] == clsid
            ]
            metadata.save_class_records(self.picksdir, name, records)
        # A deleted particle must drop out of the selection file (its uid
        # stays in umap_selection, so undoing the delete brings it back).
        # Rewritten only when the particles it lists would change: moving
        # a particle between sets -- most edits -- leaves it as it is, and
        # rewriting a large selection on every click cost ~60 ms each.
        if self.umap_selection or self._umap_saved:
            present = frozenset(b[6] for b in self.boxes if b[6] in self.umap_selection)
            if present != self._umap_saved:
                self._save_umap_selection()
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

        # Focus Selected: while on, this *replaces* set-visibility with a
        # narrower rule -- a particle shows only if it's selected AND still
        # in whatever set it belonged to when it joined the current review
        # (self._focus_home_class[uid], snapshotted the moment it was
        # selected -- see _toggle_focus_selected/set_selected_uids), not
        # merely "some set that's checked visible." That's deliberate:
        # sending a selected particle to a different set (Space/Shift+click,
        # drag, Move Selected To, ...) should make it disappear from view
        # the moment it leaves its own home set -- without touching the
        # selection itself -- even though its new set (Confirmed, say) is
        # itself perfectly visible. A particle whose home set was never
        # recorded (shouldn't normally happen) is treated as still home, so
        # a bookkeeping gap fails open rather than hiding things wrongly.
        # Computed once here rather than per-box.
        focus_active = self.focus_selected_action.isChecked()
        # Show Active Set Only while the UMAP Selection tab is on screen:
        # only the selected particles, whichever sets they are in
        umap_solo = not focus_active and self._umap_solo_active()
        umap_selection = self.umap_selection

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
                if focus_active:
                    class_visible = (b[6] in self._selected_uids
                                      and self._focus_home_class.get(b[6], b[5]) == b[5])
                elif umap_solo:
                    class_visible = b[6] in umap_selection
                else:
                    class_visible = b[5] in self.sets_visible
                if inplane and class_visible:
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
        self._sync_depth_sliders()

    def _sync_depth_sliders(self):
        """Keep all three slice sliders reflecting the actual current
        position regardless of how it changed -- a slider drag, Shift+
        wheel over any view, clicking a particle, arrow-key navigation in
        the gallery, ... all eventually funnel through update_sliceview,
        which calls update_coords (and so this) at the end. Signals are
        blocked while syncing since each slider's own valueChanged
        handler (event_depth/event_depth_y/event_depth_x) is what calls
        update_sliceview in the first place -- without this, updating a
        slider here would immediately recurse back into it."""
        for slider, loc in (
            (self.wdepth, self.z_loc), (self.wdepth_y, self.y_loc), (self.wdepth_x, self.x_loc),
        ):
            value = int(loc)
            if slider.value() != value:
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)

    # -- per-class index bookkeeping (mouse_down's click hit-test) ----------
    # self._indices_by_class lets that hit-test skip every particle in a
    # currently-hidden class outright, rather than visiting each one just to
    # reject it -- at the scale this app targets, a class sitting in the
    # (hidden-by-default) Recycle Bin alone can be tens of thousands of
    # particles. Kept incrementally in O(1) per particle for add/reassign
    # (the common case), and rebuilt in one O(N) pass for del_box/
    # _reinsert_boxes, which already shift every later particle's index
    # anyway and so are no cheaper to patch up individually.
    def _rebuild_indices_by_class(self):
        d = {}
        for i, b in enumerate(self.boxes):
            d.setdefault(b[5], {})[i] = None
        self._indices_by_class = d

    def _index_class_add(self, i, classid):
        self._indices_by_class.setdefault(classid, {})[i] = None

    def _index_class_remove(self, i, classid):
        bucket = self._indices_by_class.get(classid)
        if bucket is not None:
            bucket.pop(i, None)

    def _visible_candidate_indices(self):
        """Box indices belonging to a currently-visible set -- what
        mouse_down's click hit-test actually needs to scan, instead of
        every particle regardless of which classes are hidden. Not sorted
        back into a single global index order (cheap to skip: the only
        cost would be resolving which of several *different classes'*
        particles exactly overlapping the same click point should win --
        a rare tie only the click-handler's first-match-wins loop cares
        about at all)."""
        candidates = []
        for cid in self.sets_visible:
            candidates.extend(self._indices_by_class.get(cid, ()))
        return candidates

    def _is_visible_particle(self, box):
        """Shared "is this particle currently shown in the tomogram views"
        test. Normally: its own set must be checked visible. While "Show
        only selected" is on, that's replaced (not just supplemented) by a
        narrower rule -- selected AND still in its own home set, i.e.
        whatever set it belonged to the moment it joined the current
        review (self._focus_home_class; see _toggle_focus_selected/
        set_selected_uids) -- so reassigning a selected particle to a
        different set (Space/Shift+click, drag, ...) hides it immediately
        without deselecting it, even though its new set is itself visible.
        Used by every action that scopes itself to what's on screen
        (plain-click hit-testing, the Eraser, and the Select tool's shape
        hit-test), so they can't disagree about what "on screen" means."""
        if self.focus_selected_action.isChecked():
            return (box[6] in self._selected_uids
                    and self._focus_home_class.get(box[6], box[5]) == box[5])
        if self._umap_solo_active():
            return box[6] in self.umap_selection
        return box[5] in self.sets_visible

    def inside_box(self, n, x=-1, y=-1, z=-1):
        box = self.boxes[n]
        if not self._is_visible_particle(box):
            return False
        bs = self.get_boxsize(box[5]) / 2
        if self.options.mode == "3D":
            rr = (x >= 0) * (box[0] - x) ** 2 + (y >= 0) * (box[1] - y) ** 2 + (z >= 0) * (box[2] - z) ** 2
        else:
            rr = (x >= 0) * (box[0] - x) ** 2 + (y >= 0) * (box[1] - y) ** 2 + (z >= 0) * (box[2] != z) * (1e3 * bs ** 2)
        return rr <= bs ** 2

    def _clamp_to_volume(self, x, y, z):
        # Lower bound of 1, not 0, to match every other coordinate-clamping
        # path in this file (see __init__'s load-time clip and
        # _commit_new_set_import) rather than being the one place that
        # disagrees with them at the very edge voxel.
        return (
            max(1, min(x, self.data.nx - 1)),
            max(1, min(y, self.data.ny - 1)),
            max(1, min(z, self.data.nz - 1)),
        )

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
        self._rebuild_indices_by_class()  # removal shifts every later index
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
        for i, prev_class in changed:
            self.boxes[i][5] = target_id
            self._index_class_remove(i, prev_class)
            self._index_class_add(i, target_id)
            # A Mahalanobis score is only meaningful relative to whichever
            # set it was computed against -- moving a particle to a
            # different class makes the old score stale, so it resets
            # rather than silently carrying over a number that no longer
            # means what it looks like it means. It shows up first (not
            # last) in "Sort by Rank" until recomputed -- see
            # ParticleGrid.apply_score_order.
            self.boxes[i][4] = None
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
        """Clear the Recycle Bin -- but "empty" only means "out of the
        list," not "gone with no trace": every particle in it is first
        archived to a hidden, timestamped backup file in the working
        directory (see metadata.archive_recycle_bin), which discover_classes
        never loads back as a set. Still undo-able with Ctrl+Z right
        after, same as any other delete, independent of that backup."""
        delids = [i for i, b in enumerate(self.boxes) if b[5] == categories.RECYCLE_BIN_ID]
        if not delids:
            return
        records = [
            {"x": self.boxes[i][0], "y": self.boxes[i][1], "z": self.boxes[i][2],
             "uid": self.boxes[i][6], "embedding_row": self.boxes[i][7], "score": self.boxes[i][4]}
            for i in delids
        ]
        metadata.archive_recycle_bin(self.picksdir, records)
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
        self._trim_undo_history()
        self._refresh_undo_redo_actions()

    # A "delete" entry (Eraser, Empty Recycle Bin, Delete Set, undo of a
    # bulk reassign's own reassign, ...) keeps a live reference to every
    # deleted particle's own cached thumbnail crop so undo can restore it
    # exactly -- for a large enough deletion at the particle counts this
    # app targets (tens of thousands), that can pin a lot of memory for
    # the rest of the session with no way to reclaim it short of
    # restarting. Capping the combined history size, evicting the oldest
    # entries first once it's exceeded, bounds that risk to a fixed
    # budget instead of letting it grow without limit.
    _MAX_UNDO_HISTORY_BYTES = 2 * 1024 ** 3  # ~2GB

    @staticmethod
    def _undo_entry_nbytes(action):
        """Rough memory footprint of one undo/redo stack entry. Dominated
        by any cached thumbnail arrays an "add"/"delete" entry's "removed"
        list retains -- that's what actually matters for bounding total
        history size; a flat per-item estimate covers the box record
        itself and a plain "reassign" entry (just (index, prior-class)
        pairs), neither of which is large enough to matter individually."""
        if action["type"] in ("add", "delete"):
            total = 0
            for _i, _box, img in action.get("removed", []):
                total += 200 if img is None else img.nbytes + 200
            return total
        return 100 * len(action.get("changes") or action.get("uids") or [])

    def _trim_undo_history(self):
        total = (sum(self._undo_entry_nbytes(a) for a in self._undo_stack)
                 + sum(self._undo_entry_nbytes(a) for a in self._redo_stack))
        while total > self._MAX_UNDO_HISTORY_BYTES and (self._undo_stack or self._redo_stack):
            # The undo stack's own front (index 0) is its oldest entry --
            # append()/pop() work at the end, so that's the longest-ago
            # action still on it. Once undo is empty, do the same to redo
            # (a plain new action already clears redo in _push_undo, but
            # undo()/redo() themselves keep filling both, e.g. repeatedly
            # undoing large deletes builds up redo instead).
            oldest = self._undo_stack.pop(0) if self._undo_stack else self._redo_stack.pop(0)
            total -= self._undo_entry_nbytes(oldest)

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
        self._trim_undo_history()
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
        self._trim_undo_history()
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
        self._rebuild_indices_by_class()  # insertion shifts every later index
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
                # The class to remove the index from is whatever it
                # currently, actually is right now -- not necessarily
                # prev_class: for an undo, prev_class is what the particle
                # should revert *to*, while its current class (what this
                # is about to overwrite) is the *other* end of the
                # original reassignment.
                old_class = self.boxes[i][5]
                new_class = target_id if target_id is not None else prev_class
                self.boxes[i][5] = new_class
                self._index_class_remove(i, old_class)
                self._index_class_add(i, new_class)
                self.boxes[i][4] = None  # stale relative to the new class -- see _reassign_boxes
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
        added_uids = new_selected - self._selected_uids
        changed_uids = self._selected_uids ^ new_selected
        self._selected_uids = new_selected
        # Show Only Selected's home-class bookkeeping (see
        # _is_visible_particle): a particle newly joining the selection
        # while it's already on records its *current* class as home right
        # now, same as if Focus had just been turned on with it already
        # selected -- so it shows immediately rather than needing a
        # reassignment-free round trip first.
        if self.focus_selected_action.isChecked() and added_uids:
            self._focus_home_class.update(
                (b[6], b[5]) for b in self.boxes if b[6] in added_uids)
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

    def _toggle_focus_selected(self, checked):
        """Turning Focus Selected on/off doesn't change sets_visible at
        all (unlike Show Active Set Only) -- while on, it replaces the
        set-visibility check entirely (see _is_visible_particle/
        update_sliceview) with a per-particle "still in its home set" test,
        so turning it on snapshots every currently selected particle's
        current class as its home (discarding any stale snapshot from a
        previous time it was on -- re-selecting after turning it off starts
        fresh). Particles selected *while* it's already on get their home
        class recorded by set_selected_uids instead, the moment they join.
        Every particle's visibility outcome can change the moment this
        flips, so this needs the full (only_indices=None) recompute;
        ongoing selection changes while it's already on are handled by
        set_selected_uids's own scoped update instead, since only the
        particles whose selection actually changed can have a different
        outcome."""
        if checked:
            self._focus_home_class = {b[6]: b[5] for b in self.boxes if b[6] in self._selected_uids}
        self.update_all(reslice=False, only_indices=None)

    def _update_focus_pulse(self, checked):
        if checked:
            self._focus_pulse_on = True
            self._set_focus_button_pulsing(True)
            self._focus_pulse_timer.start()
        else:
            self._focus_pulse_timer.stop()
            self._set_focus_button_pulsing(False)

    def _pulse_focus_button(self):
        self._focus_pulse_on = not self._focus_pulse_on
        self._set_focus_button_pulsing(self._focus_pulse_on)

    def _set_focus_button_pulsing(self, on):
        widget = self.selection_toolbar.widgetForAction(self.focus_selected_action)
        if widget is None:
            return
        widget.setObjectName("pulsingAttention" if on else "")
        widget.style().unpolish(widget)
        widget.style().polish(widget)

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
                # Scoped to this one box's own class -- a single add/drag
                # only changes that class's file, so rewriting every other
                # class's pick file too (a full self.boxes scan per class)
                # was pure waste, at scale enough to make each pick or
                # drag-release synchronously stall on disk I/O.
                self.save_json(only={box[5]})
                # The gallery thumbnail only needs refreshing when the
                # particle's own snapshot image changes (just above), which
                # only happens here in the non-quiet path -- doing this on
                # every quiet in-drag update was rebuilding the entire
                # thumbnail grid layout on every mouse-move tick for no
                # visible benefit, which is what made dragging feel slow.
                self.update_boximgs()

        self.curbox = n

    def update_boximgs(self):
        # The gallery is a complete inventory of every picked particle,
        # regardless of self.sets_visible -- that's a *tomogram-view-only*
        # concept (see update_sliceview/inside_box, and the visibility
        # checkbox's own tooltip), unrelated to what the gallery shows.
        # Filtering by it here used to also empty a set's own gallery tab
        # the moment it was hidden from the tomogram views, then rebuild
        # every one of its thumbnails from scratch when shown again.
        self.boxids = list(range(len(self.boxesimgs)))
        triples = [(self.boxes[i][6], self.boxesimgs[i], self.boxes[i][5]) for i in self.boxids]
        # (score, has_embedding) per uid -- feeds the gallery's "Sort by
        # Score" display (see ParticleGrid.apply_score_order).
        scores = {self.boxes[i][6]: (self.boxes[i][4], self.boxes[i][7] is not None) for i in self.boxids}
        self.boxesviewer.rebuild_tabs(self.sets)
        self.boxesviewer.sync_particles(triples, scores)
        if self.umap_window is not None and self.umap_window.isVisible():
            self.umap_window.schedule_refresh()

    def refresh_all(self):
        """Force everything to fully re-derive and redraw -- unlike the
        normal update_all/sync_particles path (deliberately incremental,
        so a single click or edit stays fast at tens of thousands of
        particles), this regenerates every particle's cached gallery-
        thumbnail crop straight from the volume, fully re-slices every
        tomogram view and recomputes every particle's shape, and fully
        relayouts whatever's currently on screen in the gallery. A
        deliberate, explicit "make everything consistent again" escape
        hatch for if rendering ever visibly gets stuck -- Edit > Refresh,
        or F5 -- not something that needs to run on its own."""
        for i, box in enumerate(self.boxes):
            self.boxesimgs[i] = self._thumbnail_crop(box)
        self.update_all(reslice=True, only_indices=None)
        self.boxesviewer.force_refresh()

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
                # "The rest of the list" means the rest of what's actually
                # on screen right now -- self.boxes' own storage order
                # (effectively original ID) is wrong the moment a set has
                # been Sort by Rank'd, and would also reach across class
                # boundaries the gallery itself never shows mixed
                # together. active_uids_from asks the grid actually being
                # looked at for its own current visual order instead.
                rest_uids = set(self.boxesviewer.active_uids_from(lc[0]))
                idx = [i for i, b in enumerate(self.boxes) if b[6] in rest_uids]
                self.send_to_recycle_bin(idx)
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
                if not self._is_visible_particle(b):
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
                if not self._is_visible_particle(b):
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
        if self.tool_select_action.isChecked():
            # Clamped rather than rejected: dragging a selection shape
            # (once started) was already unbounded -- see xy_drag et al,
            # which call _select_mouse_drag directly with no bounds check
            # at all -- so starting one from just past the edge, snapped
            # to the boundary, makes starting a selection right at the
            # edge of the volume easier instead of needing a pixel-exact
            # first click inside it.
            x, y, z = self._clamp_to_volume(x, y, z)
            self._select_mouse_down(event, x, y, z, axis)
            return
        if min(x, y, z) < 0:
            return
        if self.tool_eraser_action.isChecked():
            self.del_region_xy(x, y, z, -1, axis=axis, is_3d=self.tool_3d_checkbox.isChecked())
            return

        for i in self._visible_candidate_indices():
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
                    # But if this particle is itself already selected (e.g.
                    # it's the sole selected particle and the user is about
                    # to drag it), leave the selection alone -- dragging a
                    # selected particle shouldn't unselect it.
                    if self.boxes[i][6] not in self._selected_uids:
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
                self.boxes.append([x, y, z, "manual", None, self.currentset, uid, None])
                self._index_class_add(len(self.boxes) - 1, self.currentset)
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
            if not self._is_visible_particle(b):
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
        if 0 <= z < self.data.nz:
            self.wdepth.setValue(z)

    def xz_wheel(self, event):
        y = int(self.y_loc + np.sign(event.angleDelta().y()))
        if 0 <= y < self.data.ny:
            self.y_loc = y
            self.update_sliceview(("y",))

    def zy_wheel(self, event):
        x = int(self.x_loc + np.sign(event.angleDelta().y()))
        if 0 <= x < self.data.nx:
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
        """While Show Active Set Only is active, force sets_visible back to
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

    def show_all_sets(self):
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
        self._indices_by_class.pop(name, None)  # already emptied by the reassign above
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
        for i, prev_class in prev:
            self.boxes[i][5] = target_id
            self._index_class_remove(i, prev_class)
            self._index_class_add(i, target_id)
            self.boxes[i][4] = None  # stale relative to the new class -- see _reassign_boxes
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

    # -- customizable keyboard shortcuts ---------------------------------
    def _register_shortcut(self, id_, label, target, multi=False):
        """target: a QAction (its own default already set via setShortcut/
        setShortcuts) or a QShortcut (setKey). multi=True is for actions
        like Redo that can carry more than one QKeySequence at once (see
        the platform-native-Redo-key handling above) -- customizing it
        here always narrows it down to the single sequence the user
        picks, replacing whatever list it started with."""
        if isinstance(target, QtGui.QShortcut):
            get, set_ = target.key, target.setKey
        elif multi:
            def get(t=target):
                seqs = t.shortcuts()
                return seqs[0] if seqs else QtGui.QKeySequence()

            def set_(seq, t=target):
                t.setShortcuts([] if seq.isEmpty() else [seq])
        else:
            get, set_ = target.shortcut, target.setShortcut
        self._shortcut_registry.append({
            "id": id_, "label": label, "get": get, "set": set_,
            "default": QtGui.QKeySequence(get()),
        })

    def _apply_shortcut_overrides(self, overrides):
        for entry in self._shortcut_registry:
            text = overrides.get(entry["id"])
            if text:
                entry["set"](QtGui.QKeySequence(text))

    def open_shortcuts_settings(self):
        dlg = KeyboardShortcutsDialog(self._shortcut_registry, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        new_values = dlg.values()
        overrides = {}
        for entry in self._shortcut_registry:
            seq = new_values[entry["id"]]
            entry["set"](seq)
            if seq.toString() != entry["default"].toString():
                overrides[entry["id"]] = seq.toString()
        shortcuts.save_overrides(overrides)

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
        win = self.umap_window
        if win is not None and win.isVisible() and QtWidgets.QApplication.activeWindow() is win:
            # the map's own selection, not the particle list's: cancel a
            # lasso in progress, otherwise clear the UMAP selection
            if not win.canvas.cancel_lasso():
                win.clear_selection()     # this tomogram's, or every tomogram's on a session map
            return
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
        """Show Active Set Only has to follow the gallery to *every* tab,
        not just the ones that can become the active picking class --
        unlike class_activated (which only ever fires for normal sets),
        active_tab_changed also fires for Recycle Bin, Confirmed, and
        "All" (cid is None here), none of which touch currentset at all.
        Without this, switching to one of those three while solo is on
        left it not shown and whatever was visible before still visible,
        since nothing was listening for exactly that tab change. "All"
        means "show everything" -- the same thing that tab itself
        displays -- since there's no single class to isolate to there."""
        umap_now = self.boxesviewer.is_umap_selection_active()
        umap_changed = umap_now != self._last_tab_was_umap
        self._last_tab_was_umap = umap_now
        if not self.view_only_selected_action.isChecked():
            return
        target = set(self.sets.keys()) if cid is None else {cid}
        if self.sets_visible == target and not umap_changed:
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
        if not self._switching:
            print("Exiting")
        self.save_picks()
        if self.umap_window is not None:
            self.umap_window.close()
        self.boxesviewer.close()
        try:
            self.xyview.close()
            self.xzview.close()
            self.zyview.close()
        except RuntimeError:
            pass
        # Closing a tomogram's windows by hand ends the whole session, the
        # same as it ends the program in single-tomogram mode; a switch
        # closes them too but must leave the session running.
        if self.session_controller is not None and not self._switching:
            self.session_controller.tomogram_closed_by_user(self)

    # -- UMAP map and its selection ----------------------------------------------
    def open_umap_window(self):
        from .umap_view import UmapWindow
        if self.umap_window is None:
            self.umap_window = UmapWindow(self)
        self.umap_window.show()
        self.umap_window.raise_()
        self.umap_window.activateWindow()

    def set_umap_selection(self, uids, mode="replace", from_map=False):
        """mode: "replace" (the lasso's particles become the selection),
        "add" (Ctrl+lasso), or "toggle" (Ctrl+click adds or removes).
        Saved straight away, like every other edit. A selection made on the
        map also brings its tab to the front, since seeing those particles
        is why you made it."""
        uids = {int(u) for u in uids}
        if mode == "add":
            new = self.umap_selection | uids
        elif mode == "toggle":
            new = self.umap_selection ^ uids
        else:
            new = uids
        if new == self.umap_selection:
            return
        self.umap_selection = new
        self._save_umap_selection()
        if self._umap_solo_active():
            self.update_all()
        else:
            self.update_boximgs()
        if from_map and new:
            self.boxesviewer.activate_umap_tab()

    def clear_umap_selection(self):
        self.set_umap_selection([], "replace")

    def _umap_solo_active(self):
        # the slice views can be drawn during start-up, before the particle
        # list (and so any tab) exists
        viewer = getattr(self, "boxesviewer", None)
        return (viewer is not None and self.view_only_selected_action.isChecked()
                and viewer.is_umap_selection_active())

    def _save_umap_selection(self):
        records = [{"x": b[0], "y": b[1], "z": b[2], "uid": b[6], "embedding_row": b[7], "score": b[4]}
                   for b in self.boxes if b[6] in self.umap_selection]
        metadata.save_umap_selection(self.picksdir, records)
        self._umap_saved = frozenset(r["uid"] for r in records)

    def _load_umap_selection(self):
        """Match the saved selection back to particles by uid, which is
        stable across saves and reassignments; fall back to the embedding
        row for a record whose uid no longer exists (uids are reassigned
        on load only for duplicates, which should not happen)."""
        records = metadata.load_umap_selection(self.picksdir)
        if not records:
            return
        uids = {b[6] for b in self.boxes}
        by_row = {b[7]: b[6] for b in self.boxes if b[7] is not None}
        selection = set()
        for r in records:
            if r["uid"] is not None and r["uid"] in uids:
                selection.add(r["uid"])
            elif r["embedding_row"] is not None and r["embedding_row"] in by_row:
                selection.add(by_row[r["embedding_row"]])
        self.umap_selection = selection
        self._umap_saved = frozenset(selection)

    def session_windows(self):
        """This tomogram's top-level windows, for the session controller to
        carry their positions over to the next tomogram it opens. Only real
        windows belong here: the three slice views are panels inside this
        window's splitters, and restoring a saved window geometry onto a
        panel moves it behind the splitter's back (offset by a title bar's
        height), leaving it misplaced until something forces a relayout --
        their proportions are carried by session_layout instead."""
        return {"main": self, "gallery": self.boxesviewer}

    def session_layout(self):
        """The proportions of the slice-view panels, as splitter state."""
        return {"top": self.splitter_top.saveState(),
                "bottom": self.splitter_bottom.saveState(),
                "wrapper": self.splitter_wrapper.saveState()}

    def restore_session_layout(self, state):
        for key, splitter in (("top", self.splitter_top), ("bottom", self.splitter_bottom),
                              ("wrapper", self.splitter_wrapper)):
            if key in state:
                splitter.restoreState(state[key])

    def close_for_switch(self):
        """Save and close every window without asking -- the session is
        about to open another tomogram in its place -- then let go of
        everything this tomogram holds in memory.

        Closing alone frees nothing. The particle list is its own top-level
        window, not a child of this one, so deleting this window does not
        delete it; and its signals hold references back to this window's
        methods, a cycle that runs through Qt where Python's garbage
        collector cannot see it. Left like that, every switch kept the
        previous tomogram's volume, embeddings and thumbnails alive -- about
        a gigabyte per switch -- so the heavy data is dropped explicitly
        and the particle list is deleted along with this window."""
        self._switching = True
        self._do_close_all()
        self.close()
        self._focus_pulse_timer.stop()
        if self.umap_window is not None:
            self.umap_window.dispose()
            self.umap_window = None
        self.boxesviewer.dispose()
        self.data = None
        self._volume_slots = {"primary": None, "companion": None}
        self.embedding_spaces = {}
        self.boxesimgs = []
        self.boxes = []

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

