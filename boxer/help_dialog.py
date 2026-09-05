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
    ("Switch active picking class + gallery tab", "1-9"),
    ("Reorder a set's tab", "Drag it in the gallery"),
    ("Pan", "Right-click + drag (tomogram view or gallery)"),
    ("Zoom", "Mouse wheel"),
    ("Step depth/slice", "Shift + wheel (tomogram view)"),
    ("Send to Recycle Bin / delete from bin", "Shift + click"),
    ("Send this + rest of list to Recycle Bin", "Shift + Ctrl + click (gallery)"),
    ("Send to Confirmed", "Space + click"),
    ("Multi-select toggle", "Ctrl + click (gallery or tomogram view)"),
    ("Multi-select range", "Alt + click (gallery)"),
    ("Send every selected particle to Recycle Bin", "Delete"),
    ("Shape-select (rectangle/circle)", "Click + drag, Select tool active"),
    ("Shape-select (freeform)", "Click to place points, Select tool active"),
    ("Finish freeform selection", "Double-click, Enter, or click near the start point"),
    ("Undo last freeform point", "Backspace (while placing points)"),
    ("Cancel freeform selection", "Escape"),
    ("Clear the current selection (anywhere in the app)", "Escape"),
    ("New set", "Click the gallery's \"+\" tab"),
    ("Rename set", "Double-click its tab"),
    ("Undo", "Ctrl+Z"),
    ("Redo", "Ctrl+Y, Ctrl+Shift+Z, or your platform's native Redo key"),
    ("Force everything to re-derive and redraw", "F5 (Edit > Refresh)"),
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
    "The Selection toolbar's Confirm/Recycle/Move Selected act on "
    "whatever's currently selected, from either window, and clear the "
    "selection afterward; a label at the bottom-right of the tomogram "
    "window shows how many particles are selected.",
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
    "Each tab has a color swatch -- click it to choose any color, or "
    "revert it (Reset to Automatic, or Revert to Red/Green for Recycle "
    "Bin/Confirmed, which can be recolored too). A manual color is saved "
    "and is never changed by Toggle Colors (particle list toolbar).",
    "Each normal set's tab shows its current number-key position (1-9) "
    "and can be dragged to reorder it -- the numbering always follows "
    "whatever's on screen. \"All\", Recycle Bin, Confirmed, and \"+\" stay "
    "fixed in place and can't be dragged; a detached set has no tab "
    "position and is skipped from numbering until reattached.",
    "\"Show the active channel only\" and \"Show All Channels\" (particle "
    "list toolbar, alongside Toggle Colors) show only the active tab's "
    "set and follow you as you switch tabs (including by number key), "
    "Recycle Bin/Confirmed included; following \"All\" shows every set "
    "instead, matching what All itself displays. Turning \"Show the "
    "active channel only\" off restores exactly the visibility you had "
    "before; Show All Channels makes every set visible in one click and "
    "turns it off if it was on.",
    "\"Show only selected\" (Selection toolbar) shows only the currently "
    "selected particle(s) in the tomogram views, regardless of any set's "
    "own visibility -- if nothing is selected, nothing shows. Updates "
    "live as the selection changes; unlike \"Show the active channel "
    "only\", there's nothing to restore when you turn it back off.",
    "\"Refresh\" (Edit menu, toolbar, or F5) forces everything to fully "
    "re-derive and redraw -- regenerating every gallery thumbnail from "
    "the volume and fully re-slicing the tomogram views -- in case "
    "rendering ever visibly gets stuck. Not needed for normal use.",
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
