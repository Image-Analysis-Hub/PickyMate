"""A picking session: several tomograms worked on together.

A session is described by a manifest -- a CSV with one row per tomogram,
whose columns mirror the single-tomogram command-line flags:

    tomogram,working_dir,embedded_particle_list,cryosiam_embeddings,cryosiam_umap,tomotwin_embeddings,tomotwin_umap,companion_tomogram
    ../Tomograms/TS_026.rec,TS_026_picks,../TemplateMatching_Ribosomes_PicksOnly/TS_026.txt,...

Relative paths are resolved against the manifest's own folder, so a
manifest can sit next to the working directories and be moved with them.
Only `tomogram` is required; `working_dir` defaults to "<tomogram>_picks"
exactly as in single-tomogram mode, and every other column may be blank.

Only one tomogram is ever open -- its volume is the only one in memory
(plus its companion, if switched to) -- and switching closes it and opens
the next. What the session adds is everything that can be done without
opening a tomogram at all, straight from the working directories on disk:
per-tomogram progress counts for the navigator, and the Confirmed/Recycle
Bin particles of other tomograms as extra references when ranking. That
works because every edit is saved to disk as it happens, so a tomogram's
working directory is always its current state whenever it is not open.

The import columns (embedded_particle_list and the embeddings files) are
only passed on a tomogram's *first* open -- the one that creates its
working directory's embeddings.settings. Passing them again would import
the same particle list a second time as a duplicate set, which is what the
single-tomogram flags do if repeated; after the first open the working
directory already records everything, and it is simply resumed.
"""
from __future__ import annotations

import argparse
import csv
import os
import threading

import numpy as np

from . import categories, embeddings_io, metadata
from .volume import Volume

IMPORT_COLUMNS = ("embedded_particle_list", "cryosiam_embeddings", "cryosiam_umap",
                  "tomotwin_embeddings", "tomotwin_umap")
PATH_COLUMNS = ("tomogram", "working_dir", "companion_tomogram") + IMPORT_COLUMNS
# which manifest column supplies each embedding space's file
SPACE_COLUMN = {"cryosiam": "cryosiam_embeddings", "cryosiam_umap": "cryosiam_umap",
                "tomotwin": "tomotwin_embeddings", "tomotwin_umap": "tomotwin_umap"}


class SessionEntry:
    def __init__(self, row: dict, base_dir: str):
        def resolve(key):
            value = (row.get(key) or "").strip()
            if not value:
                return None
            return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))

        self.tomogram = resolve("tomogram")
        if not self.tomogram:
            raise ValueError("every manifest row needs a 'tomogram'")
        self.name = metadata.base_name(self.tomogram)
        self.working_dir = resolve("working_dir") or metadata.picks_dir(self.tomogram)
        self.companion = resolve("companion_tomogram")
        self.import_flags = {k: resolve(k) for k in IMPORT_COLUMNS}

    @property
    def initialized(self) -> bool:
        """Has this tomogram been opened (and its embeddings imported)
        before? Decides whether the import columns are passed."""
        return os.path.isfile(metadata.embeddings_settings_path(self.working_dir))

    def embedding_path_fixes(self) -> dict:
        """{space: path} for every embedding file this working directory
        records at a location that no longer exists, where the manifest
        names the file's current location.

        This is what a working directory looks like after the data moved
        to another machine or disk: embeddings.settings still holds the old
        absolute paths, so the spaces silently fail to reload and the
        tomogram opens with nothing to rank in. The manifest's path is only
        accepted when it exists and has the same file name as the recorded
        one -- the same file in a new place, not a different file that
        happens to fill the same column, since a mismatch there would pair
        particles with the wrong embeddings without any error."""
        fixes = {}
        for saved in metadata.load_embeddings_settings(self.working_dir, include_missing=True):
            if saved["exists"]:
                continue
            candidate = self.import_flags.get(SPACE_COLUMN.get(saved["name"], ""))
            if (candidate and os.path.isfile(candidate)
                    and os.path.basename(candidate) == os.path.basename(saved["path"])):
                fixes[saved["name"]] = candidate
        return fixes

    def counts(self) -> dict:
        """Particles per category, read from disk: Confirmed, Recycle Bin,
        and everything still sitting in an ordinary set ("unlabelled")."""
        out = {"confirmed": 0, "recycle": 0, "unlabelled": 0}
        if not os.path.isdir(self.working_dir):
            return out
        for name in metadata.discover_classes(self.working_dir):
            n = _count_records(metadata.class_file(self.working_dir, name))
            if name == categories.CONFIRMED_NAME:
                out["confirmed"] += n
            elif name == categories.RECYCLE_BIN_NAME:
                out["recycle"] += n
            else:
                out["unlabelled"] += n
        return out


def _count_records(path: str) -> int:
    """Lines holding a particle, without parsing them -- the navigator
    refreshes every tomogram's counts after each switch."""
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        first = f.readline()
        n = sum(1 for line in f if line.strip())
    is_header = "," in first and {"x", "y", "z"} <= {h.strip().lower() for h in first.split(",")}
    return n + (0 if is_header or not first.strip() else 1)


class Session:
    def __init__(self, entries, manifest_path=None):
        if not entries:
            raise ValueError("the session manifest lists no tomograms")
        names = [e.name for e in entries]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"tomogram names must be unique within a session: {sorted(dupes)}")
        self.entries = entries
        self.manifest_path = manifest_path
        self._ref_cache = {}
        self._table_cache = {}
        self._selection_cache = {}
        self._prefetched = {}
        self._prefetch_lock = threading.Lock()

    @classmethod
    def from_manifest(cls, path: str) -> "Session":
        path = os.path.abspath(path)
        base = os.path.dirname(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "tomogram" not in [c.strip() for c in reader.fieldnames]:
                raise ValueError(f"{path}: needs a header row with at least a 'tomogram' column")
            rows = [{(k or "").strip(): v for k, v in r.items()} for r in reader]
        return cls([SessionEntry(r, base) for r in rows if (r.get("tomogram") or "").strip()],
                   manifest_path=path)

    # -- lookup ------------------------------------------------------------
    @property
    def names(self):
        return [e.name for e in self.entries]

    def entry(self, name) -> SessionEntry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(name)

    def index_of(self, name) -> int:
        return self.names.index(name)

    # -- opening a tomogram ---------------------------------------------------
    def options_for(self, name, base_options) -> argparse.Namespace:
        """The single-tomogram options this entry would be launched with.
        Import flags only on first open -- see the module docstring."""
        e = self.entry(name)
        o = argparse.Namespace(**vars(base_options))
        o.tomogram = e.tomogram
        o.working_dir = e.working_dir
        o.companion_tomogram = e.companion
        o.particle_lists = None
        first_open = not e.initialized
        for key in IMPORT_COLUMNS:
            setattr(o, key, e.import_flags[key] if first_open else None)
        o.embedding_path_fixes = {} if first_open else e.embedding_path_fixes()
        if first_open and not any(e.import_flags.values()):
            o.embedded_particle_list = None
        return o

    def prefetch(self, name, native_memmap=False):
        """Read a tomogram's volume on a background thread, so the next
        switch to it doesn't wait on the disk -- reading the volume is
        about half the time a switch takes. Only one prefetched volume is
        kept; asking for another drops the previous one."""
        path = self.entry(name).tomogram
        key = (path, bool(native_memmap))
        with self._prefetch_lock:
            if key in self._prefetched:
                return
            self._prefetched = {key: None}

        def work():
            try:
                vol = Volume.from_file(path, native_memmap=native_memmap)
            except Exception as e:  # noqa: BLE001 -- a failed prefetch just means no head start
                print(f"note: could not prefetch {path}: {e}")
                vol = None
            with self._prefetch_lock:
                if key in self._prefetched:
                    self._prefetched[key] = vol

        threading.Thread(target=work, daemon=True, name=f"prefetch-{name}").start()

    def take_prefetched(self, path, native_memmap=False):
        """The volume prefetched for `path`, if it has finished loading;
        None otherwise (the caller then reads it itself)."""
        key = (os.path.abspath(path), bool(native_memmap))
        with self._prefetch_lock:
            vol = self._prefetched.get(key)
            if vol is not None:
                self._prefetched.pop(key)
            return vol

    # -- references from other tomograms ---------------------------------------
    def reference_vectors(self, space, set_name, names, dims):
        """Embedding vectors of the particles in set `set_name` of each
        tomogram in `names`, stacked, read from their working directories.

        Returns (vectors or None, used, skipped) where skipped is a list
        of (tomogram, reason). A tomogram is skipped rather than failing
        the whole rank when it lacks that set, has none of its particles
        embedded, never had this embedding space attached, or has it at a
        different dimensionality -- vectors from different spaces cannot
        be mixed into one distribution.

        Gathered vectors are cached against the set file's size and
        modification time, so repeating a rank costs nothing until that
        tomogram is edited again; the full embedding arrays themselves are
        read transiently and not kept, since ten CryoSIAM arrays would
        cost the better part of a gigabyte."""
        blocks, used, skipped = [], [], []
        for name in names:
            e = self.entry(name)
            path = metadata.class_file(e.working_dir, set_name)
            if not os.path.isfile(path):
                skipped.append((name, f"no \"{set_name}\" set"))
                continue
            st = os.stat(path)
            key = (name, space, set_name)
            stamp = (st.st_mtime_ns, st.st_size)
            cached = self._ref_cache.get(key)
            if cached is not None and cached[0] == stamp:
                vec = cached[1]
            else:
                rows = [r["embedding_row"] for r in metadata.load_class_records(e.working_dir, set_name)
                        if r["embedding_row"] is not None]
                if not rows:
                    vec = None
                else:
                    array, why = self._space_array(e, space)
                    if array is None:
                        skipped.append((name, why))
                        continue
                    rows = np.asarray(rows, dtype=np.int64)
                    rows = rows[(rows >= 0) & (rows < len(array))]
                    vec = np.asarray(array[rows], dtype=np.float64)
                self._ref_cache[key] = (stamp, vec)
            if vec is None or len(vec) == 0:
                skipped.append((name, f"no embedded particles in \"{set_name}\""))
                continue
            if vec.shape[1] != dims:
                skipped.append((name, f"{space} has {vec.shape[1]} dimensions there, {dims} here"))
                continue
            blocks.append(vec)
            used.append(name)
        return (np.vstack(blocks) if blocks else None), used, skipped

    @staticmethod
    def space_file(entry, space):
        """(record, None) for the file holding `space` in this tomogram's
        working directory -- {"method", "kind", "n", "path"}, with the
        manifest's location standing in for a recorded one that no longer
        exists -- or (None, reason)."""
        fixes = entry.embedding_path_fixes()
        for saved in metadata.load_embeddings_settings(entry.working_dir, include_missing=True):
            if saved["name"] != space:
                continue
            path = saved["path"] if saved["exists"] else fixes.get(space)
            if path is None:
                return None, f"its {space} file is missing ({saved['path']})"
            return dict(saved, path=path), None
        return None, f"no {space} embeddings attached"

    @classmethod
    def _space_array(cls, entry, space):
        """(array, None), or (None, reason) when it can't be read."""
        rec, why = cls.space_file(entry, space)
        if rec is None:
            return None, why
        if rec["kind"] == "umap":
            return embeddings_io.load_umap(rec["method"], rec["path"])["umap_xy"], None
        return embeddings_io.load_embeddings(
            rec["method"], rec["path"], n_expected=rec["n"])["embeddings"], None

    # -- particles and UMAP selections of tomograms that are not open --------------
    def particle_table(self, name):
        """Every embedded particle of a tomogram, read from its working
        directory: {"rows", "uids", "cats", "records"} where rows are
        embedding rows, cats the category of each (CAT_* below) and records
        the full per-particle dicts keyed by uid. What the session map needs
        to place, colour and select particles of a tomogram that is not
        open. Cached until any of its set files changes on disk."""
        e = self.entry(name)
        classes = metadata.discover_classes(e.working_dir) if os.path.isdir(e.working_dir) else []
        stamp = tuple(sorted((c, *_stat(metadata.class_file(e.working_dir, c))) for c in classes))
        cached = self._table_cache.get(name)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        rows, uids, cats, records = [], [], [], {}
        for c in classes:
            cat = (CAT_CONFIRMED if c == categories.CONFIRMED_NAME else
                   CAT_RECYCLE if c == categories.RECYCLE_BIN_NAME else CAT_UNLABELLED)
            for r in metadata.load_class_records(e.working_dir, c):
                if r["embedding_row"] is None or r["uid"] is None:
                    continue
                rows.append(r["embedding_row"])
                uids.append(r["uid"])
                cats.append(cat)
                records[r["uid"]] = dict(r, set=c)
        table = {"rows": np.asarray(rows, dtype=np.int64), "uids": np.asarray(uids, dtype=np.int64),
                 "cats": np.asarray(cats, dtype=np.int8), "records": records}
        self._table_cache[name] = (stamp, table)
        return table

    def selection_uids(self, name):
        """uids in a tomogram's saved UMAP selection (its "UMAP Selection.csv")."""
        path = metadata.umap_selection_path(self.entry(name).working_dir)
        stamp = _stat(path)
        cached = self._selection_cache.get(name)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        uids = {r["uid"] for r in metadata.load_umap_selection(self.entry(name).working_dir)
                if r["uid"] is not None}
        self._selection_cache[name] = (stamp, uids)
        return uids

    def write_selection(self, name, uids):
        """Replace a (not open) tomogram's saved UMAP selection -- it is
        loaded as that tomogram's "UMAP Selection" tab when it is opened.
        Only particles that exist in its sets are written."""
        table = self.particle_table(name)
        records = [table["records"][u] for u in sorted(uids) if u in table["records"]]
        metadata.save_umap_selection(self.entry(name).working_dir, records)


CAT_UNLABELLED, CAT_RECYCLE, CAT_CONFIRMED = 0, 1, 2


def _stat(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, -1)
