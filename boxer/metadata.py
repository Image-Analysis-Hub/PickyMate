"""JSON sidecar metadata helpers, replacing EMAN2's js_open_dict/info_name/
base_name. Picks and per-class box sizes are stored in an
"info/<basename>_info.json" file next to the tomogram, using the same
"boxes_3d"/"class_list" keys EMAN2 uses so the file stays human-readable."""
from __future__ import annotations

import json
import os


def base_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def info_name(datafile: str) -> str:
    dirname = os.path.dirname(os.path.abspath(datafile))
    infodir = os.path.join(dirname, "info")
    return os.path.join(infodir, base_name(datafile) + "_info.json")


class InfoDict(dict):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        if os.path.isfile(path):
            with open(path) as f:
                try:
                    self.update(json.load(f))
                except (json.JSONDecodeError, OSError):
                    pass

    def close(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self, f, indent=2)


def js_open_dict(path: str) -> InfoDict:
    return InfoDict(path)
