"""Plain-text pick storage.

Picks are stored as one text file per particle class, inside a
"<basename>_picks" folder next to the tomogram:

    TS_0005.rec
    TS_0005_picks/
        Ribosomes.txt      # "x y z" per line -- see below for both formats
        FAS.txt
        config.settings    # optional, only present if a box size != default
                           # or a color was manually set

Every "*.txt" file found in the picks folder (other than config.settings)
is treated as a class, named after the file (without extension).

Two per-class file formats are supported, auto-detected from the first
line -- see _parse_particle_file:

- Legacy: bare whitespace-separated "x y z" per line, no header. This is
  the original format and is still fully supported for reading; drop any
  such file into the folder and it shows up as a class next time the
  tomogram is opened, exactly as before.
- Rich: a comma-separated file with a header row naming columns, used to
  carry a particle's stable identity (uid) and, when it has one, a link
  to an external embedding (embedding_row) and the last computed distance
  score (score) -- all three optional per particle (a None/absent value
  is written as an empty cell, never a placeholder). Column order doesn't
  matter, and any column this module doesn't recognize is read and then
  ignored rather than causing an error -- tolerant of foreign per-particle
  metadata some other pipeline stage might have added, at the cost of not
  preserving it: an unrecognized column that round-trips through a save
  here will not be written back out.

save_class_records always writes the rich format -- this is what silently
upgrades an old plain "x y z" file the first time PickyMate saves it, no
migration step needed. A particle's uid is what makes it possible to
match a particle back to a specific row of an external embeddings file
(TomoBoxer.import_particle_list's embedding_row) reliably across saves,
reassignments between sets, and reloads -- something bare coordinates
alone can never do, since a particle's line position isn't stable once it
can move between files.

Box size is a purely cosmetic (visualization/extraction box) setting and
defaults to DEFAULT_BOXSIZE for every class. Color is likewise cosmetic
and defaults to whatever the automatic palette (or, for Recycle Bin/
Confirmed, their own fixed red/green) would assign. Both live together in
config.settings, one "name boxsize [color]" line per class that has
either set -- a class needing neither gets no line at all, and the file
itself isn't required to exist.
"""
from __future__ import annotations

import csv
import os
import re
import time

import numpy as np

DEFAULT_BOXSIZE = 32
_COLOR_RE = re.compile(r"#[0-9a-fA-F]{6}")
CONFIG_FILENAME = "config.settings"
EMBEDDINGS_SETTINGS_FILENAME = "embeddings.settings"


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


_RECORD_FIELDS = {"uid": int, "embedding_row": int, "score": float}


def _parse_particle_file(path: str) -> list:
    """Read a particle file, returning a list of dicts each with 'x','y',
    'z' (always present, float) plus 'uid', 'embedding_row', 'score'
    (always present as keys, but individually None when this particular
    file/row doesn't carry that field). [] if the file doesn't exist.

    Format is auto-detected from the first line, comma-split: if "x",
    "y", and "z" all appear (case-insensitively) among its fields, this
    is the rich header'd format, and every field is looked up by column
    name (any order; unrecognized column names are read and dropped,
    never causing an error). Otherwise it's the legacy bare
    whitespace-separated "x y z" per line (extra fields after the first
    three ignored), and the first line is real data, not a header -- a
    coordinate line can never look like a header since it has no commas.

    This is the one place either format's shape lives -- both a class's
    own on-disk pick file (load_class_records, below) and an ad-hoc
    imported particle list (TomoBoxer.import_particle_list, for File >
    Read Box Coord, --particle_lists, and the embeddings import wizard)
    go through here."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        first_line = f.readline()
        rest = f.readlines()

    header = [h.strip().lower() for h in first_line.split(",")]
    is_rich = len(header) > 1 and {"x", "y", "z"} <= set(header)

    records = []
    if is_rich:
        col = {name: i for i, name in enumerate(header)}

        def cell(parts, name, cast):
            i = col.get(name)
            if i is None or i >= len(parts):
                return None
            v = parts[i].strip()
            if not v:
                return None
            try:
                return cast(v)
            except ValueError:
                return None

        for line in rest:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            x, y, z = cell(parts, "x", float), cell(parts, "y", float), cell(parts, "z", float)
            if x is None or y is None or z is None:
                continue
            rec = {"x": x, "y": y, "z": z}
            for name, cast in _RECORD_FIELDS.items():
                rec[name] = cell(parts, name, cast)
            records.append(rec)
    else:
        for line in [first_line, *rest]:
            parts = line.split()[:3]
            if len(parts) < 3:
                continue
            try:
                x, y, z = (float(p) for p in parts)
            except ValueError:
                continue
            records.append({"x": x, "y": y, "z": z, "uid": None, "embedding_row": None, "score": None})
    return records


def load_particle_list(path: str) -> list:
    """[(x, y, z), ...] from either file format -- see _parse_particle_file.
    Used where only bare coordinates matter (a plain coordinate import),
    not a class's own persisted file, which needs the full record; see
    load_class_records for that."""
    return [(r["x"], r["y"], r["z"]) for r in _parse_particle_file(path)]


def load_class_records(pdir: str, classname: str) -> list:
    """Like load_particle_list, but returns the full per-particle dict
    (uid/embedding_row/score, None for any the file doesn't carry) --
    used when loading the app's own persisted per-class files, which need
    to recover a particle's stable identity and any embedding link."""
    return _parse_particle_file(class_file(pdir, classname))


def _write_records_csv(path: str, records) -> None:
    with open(path, "w") as out:
        out.write("x,y,z,uid,embedding_row,score\n")
        for r in records:
            fields = [r["x"], r["y"], r["z"]] + [r.get(name) for name in _RECORD_FIELDS]
            out.write(",".join("" if v is None else str(v) for v in fields) + "\n")


def save_class_records(pdir: str, classname: str, records) -> None:
    """records: iterable of dicts with 'x','y','z' and optionally 'uid',
    'embedding_row','score' (missing or None written as an empty cell,
    never a fabricated placeholder like 0). Always writes the rich
    header'd format -- this is what upgrades an old plain "x y z" file
    the first time it's saved through here."""
    os.makedirs(pdir, exist_ok=True)
    _write_records_csv(class_file(pdir, classname), records)


def archive_recycle_bin(pdir: str, records) -> str:
    """Write a timestamped, hidden backup of the given records -- used
    when "emptying" the Recycle Bin, so that only ever means "out of the
    way," never "gone with no trace." Deliberately named ".recyclebin_
    <timestamp>", not "<name>.txt": discover_classes only ever looks for
    "*.txt" files, so this is never picked back up as a set on the next
    launch, with no separate exclusion list to maintain. Returns the
    path written to."""
    os.makedirs(pdir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(pdir, f".recyclebin_{timestamp}")
    n = 1
    while os.path.isfile(path):
        # Two empties within the same second (rare, but possible via undo
        # + immediately emptying again) -- disambiguate rather than
        # silently overwriting the previous archive.
        n += 1
        path = os.path.join(pdir, f".recyclebin_{timestamp}_{n}")
    _write_records_csv(path, records)
    return path


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


# -- embeddings (Mahalanobis-distance feature) -------------------------------
def load_embeddings_h5(path: str) -> np.ndarray:
    """Load a full-dimensionality embeddings array from an HDF5 file.
    Expects exactly one dataset in the file (which one to use would
    otherwise be ambiguous) and returns it as-is, 2D, whatever its own
    (particle, feature) vs. (feature, particle) orientation is --
    TomoBoxer.import_particle_list_with_embeddings is what normalizes
    that against the actual particle count, since this function alone
    has no way to know which axis is which."""
    import h5py

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) != 1:
            raise ValueError(f"expected exactly one dataset in {path!r}, found {keys}")
        arr = np.asarray(f[keys[0]][:])
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D embeddings array in {path!r}, got shape {arr.shape}")
    return arr


def load_embeddings_umap(path: str) -> dict:
    """Read a UMAP-reduced-embedding CSV: a header row naming its columns
    (needs "x","y" -- the 2D reduced coordinates -- and "centroid-0",
    "centroid-1","centroid-2" -- the original z,y,x the embedding was
    computed at), one row per particle in the same order as the
    corresponding coordinate file. Returns {"xy": (N,2) array of the
    UMAP coordinates, "centroid_xyz": (N,3) array of the *original*
    coordinates, already reordered here from the file's own z,y,x into
    x,y,z so nothing downstream has to think about that swap again}."""
    xy, centroid_xyz = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xy.append((float(row["x"]), float(row["y"])))
            centroid_xyz.append((
                float(row["centroid-2"]), float(row["centroid-1"]), float(row["centroid-0"]),
            ))
    return {
        "xy": np.array(xy, dtype=np.float64).reshape(-1, 2),
        "centroid_xyz": np.array(centroid_xyz, dtype=np.float64).reshape(-1, 3),
    }


def embeddings_settings_path(pdir: str) -> str:
    return os.path.join(pdir, EMBEDDINGS_SETTINGS_FILENAME)


def save_embeddings_settings(pdir: str, h5_path: str, n: int, umap_path: str = None) -> None:
    """Record which embeddings bundle is attached to this working
    directory (absolute paths, so they still resolve regardless of the
    current directory at reload time) -- one bundle per directory,
    matching the "only one embedding [source]" design -- so a later
    launch can auto-reload it without repeating the CLI flags or
    re-running the import wizard. n is the particle count the bundle was
    attached with, needed on reload to normalize the .h5 array's
    (particle, feature) vs. (feature, particle) orientation the same way
    the original import did -- there's no coordinate file being imported
    at reload time to re-derive it from."""
    os.makedirs(pdir, exist_ok=True)
    with open(embeddings_settings_path(pdir), "w") as out:
        out.write(f"h5 {os.path.abspath(h5_path)}\n")
        out.write(f"n {n}\n")
        if umap_path:
            out.write(f"umap {os.path.abspath(umap_path)}\n")


def load_embeddings_settings(pdir: str):
    """{"h5": path, "umap": path or None, "n": int}, or None if nothing's
    attached -- including when the recorded file(s) have since been moved
    or deleted, or the record predates this field being tracked, which
    the caller should treat as "nothing to auto-load" rather than an
    error at startup."""
    path = embeddings_settings_path(pdir)
    if not os.path.isfile(path):
        return None
    result = {"h5": None, "umap": None, "n": None}
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0] in result:
                result[parts[0]] = parts[1].strip()
    if result["n"] is not None:
        try:
            result["n"] = int(result["n"])
        except ValueError:
            result["n"] = None
    if not result["h5"] or not os.path.isfile(result["h5"]) or result["n"] is None:
        return None
    if result["umap"] and not os.path.isfile(result["umap"]):
        result["umap"] = None
    return result
