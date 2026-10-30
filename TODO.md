# TODO

Agreed backlog, in rough priority order. (Done items are removed, not
struck through.)

## Documentation / onboarding

- **Video tutorials**: Mohamad records himself teaching PickyMate; turn the
  recordings into the tutorial layer (hosting, per-page links, chapter
  timestamps). Keep videos workflow-level rather than button-level so UI
  changes don't invalidate them; the written docs stay the canonical
  reference.
- **Input-preparation guides** (placeholders already in docs/index.md): how
  the template-matching picks, TomoTwin embeddings, and CryoSIAM embeddings
  are produced — likely with the scripts below.
- **Searchable docs site** (MkDocs Material on GitHub Pages) — low priority;
  worth it only once the tutorial layer exists.

## Features

- **STAR import/export** (RELION/Warp interoperability), probably alongside
  scripts that generate the TomoTwin/CryoSIAM inputs from the same STAR
  files, so the whole pipeline speaks one format. Design not started —
  revisit deliberately.

## Known performance debts (measured earlier, unfixed)

- The gallery builds ~3 QGraphicsItems per particle up front (~25 s at
  100k); real fix is virtualization.
