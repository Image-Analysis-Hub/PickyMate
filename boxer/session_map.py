"""The session map: one UMAP fitted on every tomogram of a session at once.

Each tomogram's own UMAP (Task 2's map) is fitted on that tomogram alone,
so its coordinates mean nothing next to another tomogram's. This fits a
single UMAP on the full embeddings of all of them stacked together --
TomoTwin's 32 dimensions as they are, CryoSIAM's 1024 reduced to 50 by PCA
first (on EMPIAR-10988 those 50 keep all of its variance: its embeddings
have numerical rank ~150 and 10 components already hold 99.99%) -- so a
position on the map means the same thing whichever tomogram a particle
came from, and one lasso can select across all of them.

The fit runs in the background: embeddings are read on a thread, then
umap_worker.py runs as a separate process of the same Python PickyMate runs
in (umap-learn is one of PickyMate's requirements). A process rather than a
thread because a UMAP fit cannot be interrupted from inside -- killing a
process is what makes Cancel work -- and because it keeps the interface
responsive throughout. Its output streams into progress and status
signals.

Results are cached per embedding in a hidden ".pickymate_session" folder
next to the manifest, as coordinates per (tomogram, embedding row), so the
map is computed once and reused on every launch. A map records, per
tomogram, how many rows it was fitted on and the embedding file's size; if
either changes it is reported as out of date, and if the row counts no
longer match it cannot be used at all until recomputed.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
from PySide6 import QtCore

from .umap_worker import pca

MAPPABLE = {"tomotwin": "TomoTwin full", "cryosiam": "CryoSIAM full"}
CACHE_DIRNAME = ".pickymate_session"
PCA_DIMS = 50
UMAP_PARAMS = {"neighbors": 15, "min_dist": 0.1}
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "umap_worker.py")
MISSING_UMAP = ("umap-learn is not installed in this environment, and the session map needs "
                "it. Install PickyMate's requirements (pip install -r requirements.txt), or run "
                "PickyMate from its conda environment (conda env create -f environment.yml).")


class SessionMap:
    """A computed map: coordinates for each tomogram's embedding rows."""

    def __init__(self, coords, layout, fingerprint, params):
        self.coords = coords
        self.layout = layout            # [(name, n_rows), ...] in stacking order
        self.fingerprint = fingerprint
        self.params = params
        self._offsets = {}
        start = 0
        for name, n in layout:
            self._offsets[name] = (start, n)
            start += n

    def coords_for(self, name):
        """(n_rows, 2) coordinates of one tomogram's embedding rows, or None."""
        if name not in self._offsets:
            return None
        start, n = self._offsets[name]
        return self.coords[start:start + n]


class SessionMapStore:
    def __init__(self, session):
        self.session = session
        base = (os.path.dirname(session.manifest_path) if session.manifest_path
                else os.path.dirname(session.entries[0].working_dir))
        self.dir = os.path.join(base, CACHE_DIRNAME)
        self._loaded = {}

    def path(self, space):
        return os.path.join(self.dir, f"umap_{space}.npz")

    def fingerprint(self, space):
        """[(name, n_rows, file size)] for every tomogram with this space --
        not modification times, which change when a file is merely copied
        or touched, and not paths, which change when data moves disk."""
        out = []
        for e in self.session.entries:
            rec, _ = self.session.space_file(e, space)
            if rec is None:
                continue
            try:
                size = os.path.getsize(rec["path"])
            except OSError:
                continue
            out.append([e.name, int(rec["n"]), int(size)])
        return out

    def load(self, space):
        """(SessionMap or None, status) -- status is "missing", "ok",
        "stale" (computed from files that have since changed, still usable
        because every tomogram's row count matches) or "incompatible"."""
        path = self.path(space)
        if not os.path.isfile(path):
            return None, "missing"
        stamp = os.stat(path).st_mtime_ns
        cached = self._loaded.get(space)
        if cached is None or cached[0] != stamp:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                m = SessionMap(z["coords"], [tuple(x) for x in meta["layout"]],
                               meta["fingerprint"], meta["params"])
            self._loaded[space] = cached = (stamp, m)
        m = cached[1]
        now = {name: (n, size) for name, n, size in self.fingerprint(space)}
        then = {name: (n, size) for name, n, size in m.fingerprint}
        # row counts first: if any tomogram's no longer matches, its rows
        # cannot be mapped to coordinates at all, whatever else agrees
        if not all(name in now and now[name][0] == n for name, n in m.layout):
            return m, "incompatible"
        return m, ("ok" if now == then else "stale")

    def save(self, space, coords, layout, fingerprint, params):
        os.makedirs(self.dir, exist_ok=True)
        meta = {"layout": layout, "fingerprint": fingerprint, "params": params,
                "space": space, "computed": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = self.path(space) + ".tmp.npz"
        np.savez(tmp, coords=np.asarray(coords, dtype=np.float32), meta=np.array(json.dumps(meta)))
        os.replace(tmp, self.path(space))
        self._loaded.pop(space, None)


def worker_python():
    """The worker runs under the very Python PickyMate is running in."""
    return sys.executable


class SessionMapJob(QtCore.QObject):
    """Compute one session map in the background. Signals: status(text),
    progress(done, total), finished(ok, space, message)."""

    status = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(bool, str, str)
    _gathered = QtCore.Signal(object)

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.space = None
        self._process = None
        self._cancelled = False
        self._error = ""
        self._payload = None
        self._gathered.connect(self._launch)

    def running(self):
        return self.space is not None

    def start(self, space):
        if self.running():
            return
        self.space = space
        self._cancelled = False
        self._error = ""
        self.status.emit("Reading the embeddings...")
        self.progress.emit(0, 0)
        threading.Thread(target=self._gather, daemon=True, name="session-map-gather").start()

    def cancel(self):
        self._cancelled = True
        if self._process is not None:
            self._process.kill()

    # -- step 1, on a thread: read every tomogram's embeddings -----------------
    def _gather(self):
        space, session = self.space, self.store.session
        try:
            blocks, layout = [], []
            for i, e in enumerate(session.entries):
                if self._cancelled:
                    self._gathered.emit(None)
                    return
                self.status.emit(f"Reading {MAPPABLE.get(space, space)} embeddings of {e.name} "
                                 f"({i + 1}/{len(session.entries)})...")
                array, _why = session._space_array(e, space)
                if array is None:
                    continue
                blocks.append(np.asarray(array, dtype=np.float32))
                layout.append([e.name, len(array)])
            if not blocks:
                self._gathered.emit(("error", f"No tomogram in the session has {space} embeddings."))
                return
            X = np.vstack(blocks)
            params = dict(UMAP_PARAMS, input_dims=int(X.shape[1]))
            if X.shape[1] > PCA_DIMS:
                self.status.emit(f"Reducing {X.shape[1]} dimensions to {PCA_DIMS} by PCA...")
                X, kept = pca(X, PCA_DIMS)
                params.update(pca_dims=PCA_DIMS, pca_variance_kept=kept)
            if self._cancelled:
                self._gathered.emit(None)
                return
            os.makedirs(self.store.dir, exist_ok=True)
            inp = os.path.join(self.store.dir, f"umap_{space}_input.npy")
            np.save(inp, X)
            self._gathered.emit({"input": inp, "layout": layout, "params": params,
                                 "fingerprint": self.store.fingerprint(space)})
        except Exception as e:  # noqa: BLE001 -- surfaced to the user
            self._gathered.emit(("error", f"{type(e).__name__}: {e}"))

    # -- step 2, in the GUI thread: run the worker -----------------------------
    def _launch(self, payload):
        if payload is None:
            return self._done(False, "Cancelled.")
        if isinstance(payload, tuple):
            return self._done(False, payload[1])
        self._payload = payload
        out = os.path.join(self.store.dir, f"umap_{self.space}_output.npy")
        payload["output"] = out
        python = worker_python()
        self._process = p = QtCore.QProcess(self)
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._exited)
        p.errorOccurred.connect(self._failed)
        self.status.emit(f"Starting the UMAP worker ({python})...")
        p.start(python, ["-u", WORKER, payload["input"], out, "--pca", str(PCA_DIMS),
                         "--neighbors", str(UMAP_PARAMS["neighbors"]),
                         "--min_dist", str(UMAP_PARAMS["min_dist"])])

    def _read(self):
        data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        for line in data.splitlines():
            kind, _, text = line.strip().partition(" ")
            if kind == "STAGE":
                self.status.emit(text + "...")
            elif kind == "PROGRESS":
                try:
                    n, total = (int(v) for v in text.split())
                    self.progress.emit(n, total)
                except ValueError:
                    pass
            elif kind == "ERROR":
                self._error = text
            elif kind == "DONE":
                self.status.emit(f"Finished: {text}.")

    def _failed(self, error):
        # a process that never starts never reports finishing either
        if error == QtCore.QProcess.FailedToStart and self.space is not None:
            payload, self._payload = self._payload, None
            self._process = None
            try:
                os.remove(payload["input"])
            except (OSError, KeyError, TypeError):
                pass
            self._done(False, f"Could not start {worker_python()} to run the UMAP worker.")

    def _exited(self, code, _status):
        payload, self._payload = self._payload, None
        self._process = None
        try:
            if self._cancelled:
                return self._done(False, "Cancelled.")
            if code == 2:
                return self._done(False, MISSING_UMAP)
            if code != 0 or not os.path.isfile(payload["output"]):
                return self._done(False, self._error or f"The UMAP worker failed (exit code {code}).")
            coords = np.load(payload["output"])
            self.store.save(self.space, coords, payload["layout"], payload["fingerprint"],
                            payload["params"])
            self._done(True, f"Computed the map of {len(coords)} particles in "
                             f"{len(payload['layout'])} tomograms.")
        finally:
            for key in ("input", "output"):
                try:
                    os.remove(payload[key])
                except (OSError, KeyError, TypeError):
                    pass

    def _done(self, ok, message):
        space, self.space = self.space, None
        self.finished.emit(ok, space or "", message)
