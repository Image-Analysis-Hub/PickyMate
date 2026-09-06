# PickyMate Documentation

PickyMate is a standalone tomogram particle-picking tool: pick particles from
an MRC/REC tomogram in three orthogonal views, organize them into sets, clean
up with a Recycle Bin / Confirmed workflow, rank them by similarity to a
reference set, and export coordinates.

Two windows make up the app: the **tomogram window** (orthogonal views,
toolbars, controls) and the **Particle List** (thumbnail gallery, where sets
are managed).

## Contents

- [Getting Started](getting-started.md) — installing, running, command-line
  options, and a tour of the two windows.
- [Picking and Tools](picking-and-tools.md) — the Pick/Eraser/Select tools,
  the Recycle Bin/Confirmed workflow, box size, and contrast.
- [The Particle List](particle-list.md) — the thumbnail gallery, managing
  sets through its tabs, and its toolbar.
- [Selection, Undo, and Saving](selection-and-saving.md) — multi-selecting
  particles from the tomogram window, undo/redo, and how/where picks are
  saved.
- [Ranking by Embedding Distance](ranking.md) — importing an embeddings
  bundle and using Mahalanobis-distance ranking (including the "close
  to / far from" double-distance mode) to sort a set by how well each
  particle matches a reference.
- [Keyboard Shortcuts](keyboard-shortcuts.md) — the full shortcut reference,
  and how to customize them from Settings.

Screenshots throughout use a small synthetic example dataset purely to
illustrate the UI; annotated numbers in each screenshot are explained in the
text right below it.
