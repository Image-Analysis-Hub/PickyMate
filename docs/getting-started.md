# Getting Started

## Installing

```
pip install -r requirements.txt
```

## Running

```
python tomo_boxer.py tomogram.mrc [--box2d | --box3d] [--label NAME] [--clean] [--apix APIX]
    [--working_dir DIR] [--particle_lists PATH [PATH ...]]
    [--embedded_particle_list PATH --embeddings_h5 PATH [--embeddings_umap PATH]]
```

- `--box2d` / `--box3d` — pick 2D particles from slices, or 3D sub-volumes
  (default).
- `--label NAME` — open with the named set already selected.
- `--clean` — start with the Eraser tool already active.
- `--apix` — override the pixel size read from the file header.
- `--working_dir DIR` — store/load picks from `DIR` instead of the default
  `<tomogram>_picks` next to the tomogram file. Resolved relative to the
  current directory.
- `--particle_lists PATH [PATH ...]` — one or more plain-text `x y z`
  coordinate files, space-separated; each becomes its own brand new set
  (named after the file) copied into the working directory immediately,
  exactly like File > Import Particle List. A name already in use (including
  from a previous run against the same working directory) gets a numeric
  suffix rather than being skipped or overwritten, since the same name can
  legitimately carry different picks run to run.
- `--embedded_particle_list PATH` — a single `x y z` coordinate file to
  import with a matched embeddings bundle attached (requires
  `--embeddings_h5`). Separate from `--particle_lists` since only one
  particle list can have embeddings attached at a time. See
  [Ranking by Embedding Distance](ranking.md) for what this unlocks.
- `--embeddings_h5 PATH` — an HDF5 file with the full-dimensionality
  embedding for each particle in `--embedded_particle_list`, in the same
  order.
- `--embeddings_umap PATH` — an optional companion CSV with a 2D
  UMAP-reduced embedding per particle, usable as an alternate ranking space.

## The two windows

![Main tomogram window, annotated](images/main-window-overview.png)

1. **Tools toolbar** — Pick, Eraser, and Select; exactly one is ever active.
   See [Picking and Tools](picking-and-tools.md).
2. **XY view** — the main slice view, at the current Z depth.
3. **ZY view** — the far-left vertical slider steps through X; this view
   updates as it moves (Shift+wheel over the view itself also steps it).
4. **XZ view** — the bottom slider (not pictured above the view, but sized
   to match it) steps through Y the same way.
5. **Selection toolbar** — acts on whatever particles are currently
   multi-selected, from either window. See
   [Selection, Undo, and Saving](selection-and-saving.md).

The second window, the **Particle List**, is the thumbnail gallery and also
where sets are created, renamed, colored, and deleted — there's no separate
"Sets" window. See [The Particle List](particle-list.md).
