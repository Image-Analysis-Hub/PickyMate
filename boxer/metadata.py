"""Plain-text pick storage.

Picks are stored as one plain-text coordinate file per particle class,
inside a "<basename>_picks" folder next to the tomogram:

    TS_0005.rec
    TS_0005_picks/
        Ribosomes.txt      # "x y z" per line
        FAS.txt
        config.settings    # optional, only present if a box size != default
                           # or a color was manually set

Every "*.txt" file found in the picks folder (other than config.settings)
is treated as a class, named after the file (without extension). Drop any
whitespace-separated x/y/z coordinate text file into the folder and it
will show up as a class next time the tomogram is opened.

Box size is a purely cosmetic (visualization/extraction box) setting and
defaults to DEFAULT_BOXSIZE for every class. Color is likewise cosmetic
and defaults to whatever the automatic palette (or, for Recycle Bin/
Confirmed, their own fixed red/green) would assign. Both live together in
config.settings, one "name boxsize [color]" line per class that has
either set -- a class needing neither gets no line at all, and the file
itself isn't required to exist.
"""
from __future__ import annotations

import os
import re

DEFAULT_BOXSIZE = 32
_COLOR_RE = re.compile(r"#[0-9a-fA-F]{6}")
CONFIG_FILENAME = "config.settings"


def base_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def picks_dir(datafile: str) -> str:
    dirname = os.path.dirname(os.path.abspath(datafile))
    return os.path.join(dirname, base_name(datafile) + "_picks")


def config_path(pdir: str) -> str:
    return os.path.join(pdir, CONFIG_FILENAME)


def class_file(pdir: str, classname: str) -> str:
    return os.path.join(pdir, classname + ".txt")


def discover_classes(pdir: str) -> list:
    """Sorted (alphabetical) class names found as *.txt files in the picks
    folder, excluding the config file. Returns [] if the folder doesn't
    exist yet."""
    if not os.path.isdir(pdir):
        return []
    names = []
    for fn in os.listdir(pdir):
        if not fn.lower().endswith(".txt"):
            continue
        if fn == CONFIG_FILENAME:
            continue
        names.append(os.path.splitext(fn)[0])
    return sorted(names)


def load_class_coords(pdir: str, classname: str) -> list:
    path = class_file(pdir, classname)
    coords = []
    if not os.path.isfile(path):
        return coords
    with open(path) as f:
        for line in f:
            parts = line.split()[:3]
            if len(parts) < 3:
                continue
            try:
                coords.append(tuple(float(p) for p in parts))
            except ValueError:
                continue
    return coords


def save_class_coords(pdir: str, classname: str, coords) -> None:
    os.makedirs(pdir, exist_ok=True)
    path = class_file(pdir, classname)
    with open(path, "w") as out:
        for c in coords:
            out.write(f"{c[0]} {c[1]} {c[2]}\n")


def delete_class_file(pdir: str, classname: str) -> None:
    path = class_file(pdir, classname)
    if os.path.isfile(path):
        os.remove(path)


def rename_class_file(pdir: str, oldname: str, newname: str) -> None:
    oldpath = class_file(pdir, oldname)
    newpath = class_file(pdir, newname)
    if os.path.isfile(oldpath):
        os.makedirs(pdir, exist_ok=True)
        os.replace(oldpath, newpath)


def load_class_settings(pdir: str) -> dict:
    """Read config.settings: whitespace-separated "name boxsize [color]"
    per line, color an optional "#rrggbb" hex string. Returns
    {name: {"boxsize": int, "color": str or None}}. Missing file simply
    means every class uses the default box size and automatic color.

    Parsed from the *right*, not split on the first token: a class name
    can itself contain spaces ("Recycle Bin" always has one), so the
    boundary between "the name" and "the boxsize [color] that follow it"
    has to be found from the trailing, unambiguously-shaped fields
    (boxsize is always an int, color always "#rrggbb") inward, not by
    assuming the name is a single token."""
    path = config_path(pdir)
    settings = {}
    if not os.path.isfile(path):
        return settings
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            color = None
            if _COLOR_RE.fullmatch(parts[-1]):
                color = parts[-1]
                parts = parts[:-1]
            if len(parts) < 2:
                continue
            try:
                boxsize = int(float(parts[-1]))
            except ValueError:
                continue
            name = " ".join(parts[:-1])
            if not name:
                continue
            settings[name] = {"boxsize": boxsize, "color": color}
    return settings


def save_class_settings(pdir: str, settings: dict) -> None:
    """settings: {name: {"boxsize": int, "color": str or None}}. Only a
    class with a non-default box size or an explicit color gets a line
    written; if none do, the file is removed rather than left empty."""
    to_write = {
        name: s for name, s in settings.items()
        if int(s.get("boxsize", DEFAULT_BOXSIZE)) != DEFAULT_BOXSIZE or s.get("color")
    }
    path = config_path(pdir)
    if not to_write:
        if os.path.isfile(path):
            os.remove(path)
        return
    os.makedirs(pdir, exist_ok=True)
    with open(path, "w") as out:
        for name in sorted(to_write):
            s = to_write[name]
            line = f"{name} {int(s.get('boxsize', DEFAULT_BOXSIZE))}"
            if s.get("color"):
                line += f" {s['color']}"
            out.write(line + "\n")
