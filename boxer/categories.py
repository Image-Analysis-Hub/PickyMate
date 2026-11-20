"""Fixed particle categories: "Recycle Bin" and "Confirmed" are always
present alongside the user's own picking classes, so a cleanup workflow
(shift+click to reject, space+click to confirm) always has somewhere to
put a particle.

Negative ids keep them from colliding with the auto-incrementing user
class ids (0, 1, 2, ...) or with the -1 sentinel `parse_setname` returns
when it can't parse a class id at all.
"""
from __future__ import annotations

from PySide6 import QtGui

RECYCLE_BIN_ID = -3
CONFIRMED_ID = -2
RECYCLE_BIN_NAME = "Recycle Bin"
CONFIRMED_NAME = "Confirmed"
FIXED_CLASS_IDS = (RECYCLE_BIN_ID, CONFIRMED_ID)
FIXED_CLASS_NAMES = {RECYCLE_BIN_NAME: RECYCLE_BIN_ID, CONFIRMED_NAME: CONFIRMED_ID}


def parse_setname(name):
    """The class id in a "<id>::<name>" label (the form the particle list's
    tabs hand back for a set), or -1 if it has none."""
    p0 = name.find("::")
    if p0 > 0:
        try:
            return int(name[:p0])
        except ValueError:
            pass
    return -1

# Red/green/yellow are reserved for Recycle Bin/Confirmed/selection (see
# below) and deliberately excluded here so a normal class's color can never
# be confused with one of those three meanings.
SETCOLORS = [
    QtGui.QColor("blue"), QtGui.QColor("cyan"), QtGui.QColor("purple"),
    QtGui.QColor("orange"), QtGui.QColor("hotpink"), QtGui.QColor("gold"),
    QtGui.QColor("turquoise"), QtGui.QColor("chocolate"), QtGui.QColor("slateblue"),
]
_FIXED_COLORS = {RECYCLE_BIN_ID: QtGui.QColor("crimson"), CONFIRMED_ID: QtGui.QColor("limegreen")}
SELECTED_COLOR = QtGui.QColor("yellow")

# A global rotation offset into SETCOLORS, advanced by cycle_colors() (the
# "Toggle Colors" button) so every normal class shifts to the next color at
# once, napari-style -- in-memory only, resets each session.
_color_offset = 0

# Explicit per-class overrides (persisted -- see metadata.load/save_class_
# settings) that take priority over everything else: the rotation formula
# for a normal class, or Recycle Bin/Confirmed's own fixed red/green. Set
# via the gallery's per-tab color swatch. Not touched by cycle_colors(), so
# a manual pick is never silently overwritten by clicking Toggle Colors --
# only the classes still on "automatic" shift when that's clicked.
_manual_colors = {}  # classid -> QColor


def class_color(classid: int) -> QtGui.QColor:
    if classid in _manual_colors:
        return _manual_colors[classid]
    if classid in _FIXED_COLORS:
        return _FIXED_COLORS[classid]
    return SETCOLORS[(classid + _color_offset) % len(SETCOLORS)]


def cycle_colors() -> None:
    global _color_offset
    _color_offset = (_color_offset + 1) % len(SETCOLORS)


def set_manual_color(classid: int, color: QtGui.QColor) -> None:
    _manual_colors[classid] = QtGui.QColor(color)


def clear_manual_color(classid: int) -> None:
    _manual_colors.pop(classid, None)


def clear_all_manual_colors() -> None:
    """Forget every per-class override. Class ids are only unique within
    one tomogram, and each tomogram's overrides are loaded from its own
    working directory, so a multi-tomogram session calls this before
    opening the next tomogram -- otherwise one tomogram's colour for its
    set 0 would silently paint the next tomogram's set 0. The Toggle
    Colors rotation is deliberately kept: it is a display preference for
    the whole run, not a property of any one tomogram."""
    _manual_colors.clear()


def has_manual_color(classid: int) -> bool:
    return classid in _manual_colors


def manual_color_hex(classid: int):
    """"#rrggbb" for a class with a manual override, else None -- the form
    saved to config.settings."""
    color = _manual_colors.get(classid)
    return color.name() if color is not None else None


def category_order(sets: dict) -> list:
    """Fixed categories first (Recycle Bin, then Confirmed), then every
    other class alphabetically by name -- the display order used for both
    the particle-list tabs and the grouped "All" gallery."""
    fixed = [cid for cid in FIXED_CLASS_IDS if cid in sets]
    normal = sorted((cid for cid in sets if cid not in FIXED_CLASS_IDS), key=lambda cid: sets[cid])
    return fixed + normal
