"""Plain-text pick storage.

Picks are stored as one plain-text coordinate file per particle class,
inside a "<basename>_picks" folder next to the tomogram:

    TS_0005.rec
    TS_0005_picks/
        Ribosomes.txt      # "x y z" per line
        FAS.txt
        config.settings    # optional, only present if a box size != default

Every "*.txt" file found in the picks folder (other than config.settings)
is treated as a class, named after the file (without extension). Drop any
whitespace-separated x/y/z coordinate text file into the folder and it
will show up as a class next time the tomogram is opened.

Box size is a purely cosmetic (visualization/extraction box) setting and
defaults to DEFAULT_BOXSIZE for every class. It is only ever recorded in
config.settings when a class's box size differs from the default, and the
file is not required to exist at all.
"""
from __future__ import annotations

import os

DEFAULT_BOXSIZE = 32
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


def load_boxsizes(pdir: str) -> dict:
    """Read config.settings (whitespace-separated "name boxsize" per
    line). Missing file simply means every class uses DEFAULT_BOXSIZE."""
    path = config_path(pdir)
    sizes = {}
    if not os.path.isfile(path):
        return sizes
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                sizes[name] = int(float(parts[1]))
            except ValueError:
                continue
    return sizes


def save_boxsizes(pdir: str, boxsizes: dict) -> None:
    """Write config.settings with only the non-default entries. If there
    are none, remove the file (or don't create it) rather than leaving an
    empty file around."""
    non_default = {name: bs for name, bs in boxsizes.items() if int(bs) != DEFAULT_BOXSIZE}
    path = config_path(pdir)
    if not non_default:
        if os.path.isfile(path):
            os.remove(path)
        return
    os.makedirs(pdir, exist_ok=True)
    with open(path, "w") as out:
        for name in sorted(non_default):
            out.write(f"{name} {non_default[name]}\n")
