# Selection, Undo, and Saving

## Selecting from the tomogram window

- **Ctrl+click** a particle in any of the three tomogram views toggles it
  in/out of the multi-selection — the exact same selection the gallery
  uses, so it works interchangeably with Ctrl/Alt+click in the gallery.
- A small label at the bottom-right of the tomogram window shows "N
  particle(s) selected", the way a file manager shows a selection count.
- The **Selection** toolbar acts on that selection directly: **Confirm
  Selected**, **Recycle Selected**, and **Move Selected To** (type a set
  name — existing or new — into the box beside it, then Move Selected To).
  Each one clears the selection afterward, the same way a file manager
  doesn't keep files selected once you've moved them.
- **Escape** clears the current selection from anywhere in the app, not
  just while a tomogram view has keyboard focus.
- **Show Only Selected** (also in the Selection toolbar) shows only the
  currently selected particle(s) in the tomogram views, regardless of any
  set's own visibility — if nothing is selected, nothing shows. Updates
  live as the selection changes.
- **Delete** sends every currently selected particle to the Recycle Bin,
  the same as clicking Recycle Selected; **Insert** sends them to
  Confirmed instead, the same as Confirm Selected.

## Undo / redo

Every add, delete, and reassignment (recycle/confirm/move) is undoable and
redoable, including permanent deletes from the Recycle Bin. The **Edit**
menu and toolbar mirror Undo/Redo with visible buttons alongside the
keyboard shortcuts. **Refresh** (Edit menu, toolbar, or F5) forces
everything to fully re-derive and redraw — regenerating every gallery
thumbnail from the volume and fully re-slicing the tomogram views — in
case rendering ever visibly gets stuck; not needed for normal use.

## Saving

Picks persist automatically as one plain-text `x y z` file per set (plus
Recycle Bin/Confirmed) inside `<tomogram>_picks/`, and a `config.settings`
file for any non-default box sizes — no separate save step needed.

The **File** menu offers:

- **Save** — forces an immediate write of every set's picks and box sizes
  (happens automatically anyway) and confirms with the folder it wrote to.
- **Import Particle List...** — import an `x y z` coordinate text file as a
  brand new set, named after the file (auto-renamed with a numeric suffix if
  that name's already taken).
- **Export Particle Coordinates...** — copies the already-saved coordinate
  file(s) — all sets, or one you pick — to a folder of your choice, for your
  own records.
