"""The "Help > Keyboard Shortcuts" reference dialog.

Most of PickyMate's gestures (shift+click, space+click, right-drag-to-pan,
ctrl/alt+click multi-select, ...) are raw modifier checks inside mouse
handlers, not QAction/QShortcut objects -- there's no on-screen widget to
hang a tooltip off of, and no live object to introspect a list of them from.
This dialog is the one place they're all written down for in-app lookup,
reachable from the Help menu or F1 from anywhere in the app.

The table itself is shortcuts.REFERENCE_SHORTCUTS -- one shared source,
also used to generate the documentation's table -- so this dialog can never
drift from the docs. The notes below are in-app-only prose.
"""
from __future__ import annotations

from PySide6 import QtWidgets

from .shortcuts import REFERENCE_SHORTCUTS

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
    "\"Show Active Set Only\" and \"Show All Sets\" (particle "
    "list toolbar, alongside Toggle Colors) show only the active tab's "
    "set and follow you as you switch tabs (including by number key), "
    "Recycle Bin/Confirmed included; following \"All\" shows every set "
    "instead, matching what All itself displays. Turning \"Show Active "
    "Set Only\" off restores exactly the visibility you had "
    "before; Show All Sets makes every set visible in one click and "
    "turns it off if it was on.",
    "\"Show Only Selected\" (Selection toolbar) shows a particle in the "
    "tomogram views only if it's both currently selected AND still in "
    "whatever set it was in when it joined the selection -- moving it to "
    "another set (Space/Shift+click, drag, ...) hides it right away "
    "without deselecting it, even though its new set is itself visible. "
    "Updates live as the selection changes; unlike \"Show Active Set "
    "Only\", there's nothing to restore when you turn it back off.",
    "\"Refresh\" (Edit menu, toolbar, or F5) forces everything to fully "
    "re-derive and redraw -- regenerating every gallery thumbnail from "
    "the volume and fully re-slicing the tomogram views -- in case "
    "rendering ever visibly gets stuck. Not needed for normal use.",
    "Most of the shortcuts above can be rebound from Settings > Customize "
    "Keyboard Shortcuts... -- optional, and stored in "
    "~/.config/pickymate/shortcuts.json (only ever created once you "
    "actually change one).",
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
            for action, keys in REFERENCE_SHORTCUTS
        )
        notes = "".join(f"<li style='margin-bottom:6px'>{n}</li>" for n in _NOTES)
        return (
            "<h3>Shortcuts</h3>"
            f"<table cellspacing='4'>{rows}</table>"
            "<h3>Notes</h3>"
            f"<ul>{notes}</ul>"
        )
