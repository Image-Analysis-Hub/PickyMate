"""The "Help > Keyboard Shortcuts" reference dialog.

Most of PickyMate's gestures (shift+click, space+click, right-drag-to-pan,
ctrl/alt+click multi-select, ...) are raw modifier checks inside mouse
handlers, not QAction/QShortcut objects -- there's no on-screen widget to
hang a tooltip off of, and no live object to introspect a list of them from.
This dialog is the one place they're all written down for in-app lookup,
reachable from the Help menu or F1 from anywhere in the app.

Keep _SHORTCUTS/_NOTES here in sync by hand with README.md's own "Shortcut
reference" table and prose -- they're two hand-maintained copies of the same
information, not generated from one shared source.
"""
from __future__ import annotations

from PySide6 import QtWidgets

_SHORTCUTS = [
    ("Add particle", "Click empty space (tomogram view)"),
    ("Move particle", "Click + drag (tomogram view)"),
    ("Switch active picking class", "1-9"),
    ("Pan", "Right-click + drag (tomogram view or gallery)"),
    ("Zoom", "Mouse wheel"),
    ("Step depth/slice", "Shift + wheel (tomogram view)"),
    ("Send to Recycle Bin / delete from bin", "Shift + click"),
    ("Send this + rest of list to Recycle Bin", "Shift + Ctrl + click (gallery)"),
    ("Send to Confirmed", "Space + click"),
    ("Multi-select toggle", "Ctrl + click (gallery or tomogram view)"),
    ("Multi-select range", "Alt + click (gallery)"),
    ("Shape-select (rectangle/circle)", "Click + drag, Select tool active"),
    ("Shape-select (freeform)", "Click to place points, Select tool active"),
    ("Finish freeform selection", "Double-click, Enter, or click near the start point"),
    ("Undo last freeform point", "Backspace (while placing points)"),
    ("Cancel freeform selection", "Escape"),
    ("New set", "Click the gallery's \"+\" tab"),
    ("Rename set", "Double-click its tab"),
    ("Undo", "Ctrl+Z"),
    ("Redo", "Ctrl+Y, Ctrl+Shift+Z, or your platform's native Redo key"),
    ("This reference", "F1"),
]

_NOTES = [
    "Box size and eraser radius apply once you press Enter, click elsewhere, "
    "or use the arrows -- not on every keystroke.",
    "“Apply to All Sets” (Box Size) also applies immediately when "
    "you check it, even without changing the number.",
    "Dragging the Contrast slider switches off AutoContrast and applies "
    "once you release the mouse, not while dragging.",
    "The Recycle Bin starts hidden -- toggle it from its tab's checkbox in "
    "the gallery.",
    "Shift+click a particle already in the Recycle Bin to delete it "
    "permanently (undoable).",
    "A particle selected in the gallery is outlined yellow in the tomogram "
    "views too -- only for selection, never while picking or dragging.",
    "The Edit toolbar's Confirm/Recycle/Move Selected act on whatever's "
    "currently selected, from either window; a label at the bottom-right of "
    "the tomogram window shows how many particles are selected.",
    "The shared 2D/3D checkbox (Tools toolbar) applies to whichever of "
    "Eraser/Select is active: 2D only affects particles visible on the "
    "current slice; 3D drops that limit -- Eraser becomes a full sphere of "
    "Radius, Select extrudes the shape (cuboid/cylinder/tube) through the "
    "entire volume.",
    "Pick, Eraser, and Select are mutually exclusive tools (Tools toolbar) "
    "-- only one is ever active.",
    "Sets are managed entirely through the gallery's tabs now -- the \"+\" "
    "tab creates one, double-click renames it, and Delete Set/Empty Recycle "
    "Bin live in its button row.",
    "File > Save forces an immediate write of every set's picks and box "
    "sizes (happens automatically anyway) and tells you which folder.",
]


class ShortcutsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(560, 480)

        layout = QtWidgets.QVBoxLayout(self)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._html())
        layout.addWidget(browser)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _html():
        rows = "".join(
            f"<tr><td>{action}</td>"
            f"<td style='padding-left:20px'><b>{keys}</b></td></tr>"
            for action, keys in _SHORTCUTS
        )
        notes = "".join(f"<li style='margin-bottom:6px'>{n}</li>" for n in _NOTES)
        return (
            "<h3>Shortcuts</h3>"
            f"<table cellspacing='4'>{rows}</table>"
            "<h3>Notes</h3>"
            f"<ul>{notes}</ul>"
        )
