"""Optional, user-customizable keyboard shortcuts.

If ~/.config/pickymate/shortcuts.json (or $XDG_CONFIG_HOME/pickymate/...)
exists, it overrides the built-in default for whichever shortcuts it
names by id; anything not listed keeps its default. The file is written
by the "Settings > Customize Keyboard Shortcuts..." dialog (see
settings_dialog.py) and is never required to exist -- a fresh install
with no such file just uses every built-in default as-is.
"""
from __future__ import annotations

import json
import os

CONFIG_DIRNAME = "pickymate"
SETTINGS_FILENAME = "shortcuts.json"


def config_dir() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, CONFIG_DIRNAME)


def settings_path() -> str:
    return os.path.join(config_dir(), SETTINGS_FILENAME)


def load_overrides() -> dict:
    """{shortcut_id: key sequence string}, empty if the file doesn't
    exist or can't be parsed -- a missing or broken settings file should
    never stop the app from starting, just mean "use the defaults"."""
    path = settings_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}
    except (OSError, ValueError) as e:
        print(f"warning: could not read {path!r}, using default shortcuts: {e}")
        return {}


def save_overrides(overrides: dict) -> None:
    """Write the given {id: key sequence string} overrides -- only ones
    that actually differ from their default need to be included (see
    KeyboardShortcutsDialog.accept). An empty dict removes the file
    entirely, so "Restore All Defaults" cleanly leaves no trace rather
    than an empty-but-present settings file."""
    path = settings_path()
    if not overrides:
        if os.path.isfile(path):
            os.remove(path)
        return
    os.makedirs(config_dir(), exist_ok=True)
    with open(path, "w") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)
