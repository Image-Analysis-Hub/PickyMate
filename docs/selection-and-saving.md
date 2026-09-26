# Selection, Undo, and Saving

## Selecting from the tomogram window

- **Ctrl+click** a particle in any view toggles it in the multi-selection —
  the same selection the gallery and the UMAP map use, interchangeably. A
  count shows at the bottom-right of the tomogram window.
- The **Selection** toolbar acts on it: **Confirm Selected**, **Recycle
  Selected**, **Move Selected To** (type a set name — existing or new). Each
  clears the selection afterwards.
- **Delete** = Recycle Selected; **Insert** = Confirm Selected.
- **Escape** clears the selection, from anywhere in the app.
- **Show Only Selected** shows a particle only while it is selected *and*
  still in the set it was in when selected — so moving it (Space/Shift+click,
  drag, Move Selected To) hides it immediately, without deselecting it.
  That makes it a review mode: what remains visible is what you have not
  acted on yet.

## Undo / redo

Every add, delete, and reassignment is undoable and redoable, including
permanent deletes from the Recycle Bin. **Refresh** (F5) forces a full
redraw if rendering ever gets stuck; not needed in normal use.

## Saving

Picks save automatically, continuously: one plain-text file per set inside
the working directory (default `<tomogram>_picks/`), plus `config.settings`
for box sizes and colors. No save step. Every file is documented in
[Files on Disk](files-on-disk.md).

The **File** menu adds:

- **Save** — force an immediate write and confirm where it went.
- **Import Particle List...** — an `x y z` text file becomes a new set.
- **Export Particle Coordinates...** — copy the saved coordinate files (all
  sets or one) to a folder of your choice.
