"""Fixed particle categories: "Recycle Bin" and "Confirmed" are always
present alongside the user's own picking classes, so a cleanup workflow
(shift+click to reject, space+click to confirm) always has somewhere to
put a particle.

Negative ids keep them from colliding with the auto-incrementing user
class ids (0, 1, 2, ...) or with the -1 sentinel `parse_setname` (in
main_window.py) returns when it can't parse a class id at all.
"""
from __future__ import annotations

from PySide6 import QtGui

RECYCLE_BIN_ID = -3
CONFIRMED_ID = -2
RECYCLE_BIN_NAME = "Recycle Bin"
CONFIRMED_NAME = "Confirmed"
FIXED_CLASS_IDS = (RECYCLE_BIN_ID, CONFIRMED_ID)
FIXED_CLASS_NAMES = {RECYCLE_BIN_NAME: RECYCLE_BIN_ID, CONFIRMED_NAME: CONFIRMED_ID}

SETCOLORS = [
    QtGui.QColor("blue"), QtGui.QColor("green"), QtGui.QColor("red"),
    QtGui.QColor("cyan"), QtGui.QColor("purple"), QtGui.QColor("orange"),
    QtGui.QColor("yellow"), QtGui.QColor("hotpink"), QtGui.QColor("gold"),
]
_FIXED_COLORS = {RECYCLE_BIN_ID: QtGui.QColor("crimson"), CONFIRMED_ID: QtGui.QColor("limegreen")}


def class_color(classid: int) -> QtGui.QColor:
    if classid in _FIXED_COLORS:
        return _FIXED_COLORS[classid]
    return SETCOLORS[classid % len(SETCOLORS)]


def category_order(sets: dict) -> list:
    """Fixed categories first (Recycle Bin, then Confirmed), then every
    other class alphabetically by name -- the display order used for both
    the particle-list tabs and the grouped "All" gallery."""
    fixed = [cid for cid in FIXED_CLASS_IDS if cid in sets]
    normal = sorted((cid for cid in sets if cid not in FIXED_CLASS_IDS), key=lambda cid: sets[cid])
    return fixed + normal
