# PickyMate

A standalone tomogram particle-picking tool, built on plain PySide6 + numpy +
mrcfile (design loosely inspired by EMAN2's `e2spt_boxer.py`, though the two
have since diverged considerably). Pick particles from an MRC/REC tomogram in
three orthogonal views, organize them into sets, clean up with a Recycle-Bin
/ Confirmed workflow, rank them by embedding similarity, and export
coordinates.

## Quickstart

```
pip install -r requirements.txt
python tomo_boxer.py tomogram.mrc
```

## Documentation

Full documentation, including screenshots and a walkthrough of every
feature, lives in [docs/](docs/index.md):

- [Getting Started](docs/getting-started.md) — installing, running, and
  all command-line options.
- [Picking and Tools](docs/picking-and-tools.md) — Pick/Eraser/Select, the
  Recycle Bin/Confirmed workflow, box size, and contrast.
- [The Particle List](docs/particle-list.md) — the thumbnail gallery and
  managing sets through its tabs.
- [Selection, Undo, and Saving](docs/selection-and-saving.md)
- [Ranking by Embedding Distance](docs/ranking.md) — Mahalanobis-distance
  ranking, including the "close to / far from" double-distance mode.
- [Keyboard Shortcuts](docs/keyboard-shortcuts.md) — the full shortcut
  reference and how to customize it.
