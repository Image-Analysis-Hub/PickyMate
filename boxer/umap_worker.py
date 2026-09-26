#!/usr/bin/env python3
"""Compute a 2-D UMAP of a stacked embedding array, as a separate process.

    python umap_worker.py input.npy output.npy [--pca 50] [--neighbors 15]
                          [--min_dist 0.1] [--seed N]

PickyMate runs this for the session map (see boxer/session_map.py), as a
separate process of its own Python: a UMAP fit cannot be interrupted from
inside, so running it as a process is what lets Cancel stop it, and it
keeps the interface responsive during the fit. It imports nothing from
PickyMate, only numpy and umap-learn, so it can also be run by hand.

It reports on stdout, one line per event, for the caller to parse:

    STAGE <text>          what it is doing now
    PROGRESS <n> <total>  layout optimisation epochs done
    DONE <text>           finished; output written
    ERROR <text>          failed (exit code 2: umap-learn missing, 1: other)
"""
from __future__ import annotations

import argparse
import re
import sys
import time


def say(kind, text=""):
    print(f"{kind} {text}".rstrip(), flush=True)


class _EpochProgress:
    """A file-like object handed to umap-learn's tqdm bar in place of a
    terminal. tqdm redraws its bar with carriage returns; this turns each
    redraw into one clean PROGRESS line, and announces the layout stage the
    first time a count arrives (umap-learn reports nothing for the
    nearest-neighbour search that precedes it)."""

    def __init__(self):
        self._last = None

    def write(self, text):
        m = re.search(r"(\d+)/(\d+)", text)
        if not m:
            return
        n, total = int(m.group(1)), int(m.group(2))
        if self._last is None:
            say("STAGE", "Optimising the layout")
        if (n, total) != self._last:
            self._last = (n, total)
            say("PROGRESS", f"{n} {total}")

    def flush(self):
        pass


def pca(X, dims):
    """Project onto the `dims` directions of largest variance -- plain
    principal component analysis, by eigendecomposition of the covariance,
    so it needs nothing beyond numpy. Returns (projected, fraction of the
    variance kept)."""
    import numpy as np
    Xc = X.astype(np.float64) - X.mean(axis=0)
    w, v = np.linalg.eigh(Xc.T @ Xc / max(len(X) - 1, 1))
    top = v[:, ::-1][:, :dims]
    kept = float(w[::-1][:dims].sum() / w.sum()) if w.sum() > 0 else 1.0
    return (Xc @ top).astype(np.float32), kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pca", type=int, default=50,
                    help="reduce to this many dimensions first when there are more")
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the random state for a reproducible map; umap-learn then "
                         "runs single-threaded, which is considerably slower")
    a = ap.parse_args()
    t0 = time.time()
    try:
        import numpy as np
        say("STAGE", "Loading the embeddings")
        X = np.load(a.input).astype(np.float32)
        say("STAGE", f"{X.shape[0]} particles x {X.shape[1]} dimensions")
        if X.shape[1] > a.pca:
            say("STAGE", f"Reducing {X.shape[1]} dimensions to {a.pca} by PCA")
            X, kept = pca(X, a.pca)
            say("STAGE", f"PCA kept {kept * 100:.4f}% of the variance")
        try:
            import umap
        except ImportError as e:
            say("ERROR", f"umap-learn is not installed in {sys.executable} ({e}). "
                         f"Install it there with: pip install umap-learn")
            return 2
        say("STAGE", "Finding nearest neighbours (no progress is reported for this step)")
        reducer = umap.UMAP(
            n_neighbors=a.neighbors, min_dist=a.min_dist, random_state=a.seed,
            low_memory=True, verbose=False,
            tqdm_kwds={"file": _EpochProgress(), "bar_format": "{n}/{total}",
                       "mininterval": 0.25, "disable": False})
        xy = reducer.fit_transform(X)
        say("STAGE", "Writing the map")
        with open(a.output, "wb") as f:
            np.save(f, np.asarray(xy, dtype=np.float32))
        say("DONE", f"{X.shape[0]} particles in {time.time() - t0:.0f} s")
        return 0
    except Exception as e:  # noqa: BLE001 -- report anything, the caller shows it
        say("ERROR", f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
