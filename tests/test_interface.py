"""The other windows use only what TomoBoxer's docstring says they may."""
import os
import re

from PySide6 import QtWidgets

from boxer.main_window import TomoBoxer

BOXER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "boxer")
# module -> the names its code calls the main window by
HOLDERS = {"thumb_grid": ["target"], "ranking_panel": ["target"], "umap_view": ["target"],
           "session_browser": ["target"], "session_window": ["win", "old", r"self\.current"]}


def documented():
    block = TomoBoxer.__doc__.split("checks that:")[1]
    return set(re.findall(r"\b[a-z_]+\b", block)) - {"state", "verbs", "actions"}


def test_other_windows_use_only_the_documented_interface():
    allowed = documented()
    assert len(allowed) > 20, allowed
    for module, holders in HOLDERS.items():
        with open(os.path.join(BOXER, f"{module}.py")) as f:
            src = f.read()
        used = set()
        for holder in holders:
            used |= set(re.findall(rf"\b{holder}\.([A-Za-z_]+)", src))
            used |= set(re.findall(rf"getattr\({holder}, \"([A-Za-z_]+)\"", src))
        used -= {n for n in used if hasattr(QtWidgets.QWidget, n)}  # show, close, deleteLater, ...
        assert used <= allowed, (module, sorted(used - allowed))
