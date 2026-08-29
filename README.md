# PickyMate

A standalone tomogram particle boxer (a PySide6 + numpy + mrcfile port of
EMAN2's `e2spt_boxer.py` that runs without EMAN2). Pick particles from an
MRC/REC tomogram in three orthogonal views, organize them into sets, clean
up with a Recycle-Bin / Confirmed workflow, and export coordinates or
particle stacks.

## Running it

```
python tomo_boxer.py tomogram.mrc [--box2d | --box3d] [--label NAME] [--clean] [--apix APIX]
```

- `--box2d` / `--box3d` — pick 2D particles from slices, or 3D sub-volumes (default).
- `--label NAME` — open with the named set already selected.
- `--clean` — start with the eraser tool already enabled.
- `--apix` — override the pixel size read from the file header.

Three windows open: the **tomogram window** (orthogonal XY/XZ/ZY views +
controls), the **Particle List** (thumbnail gallery), and **Options**
(Sets / Selection tabs, always on top).

## Picking and moving particles

- **Click empty space** — add a new particle to the current set.
- **Click an existing particle, then drag** — move it.
- **Right-click + drag**, in any view — pan.
- **Mouse wheel** — zoom, centered on the cursor.
- **Shift + wheel** — step through depth/slices instead of zooming.
- **Eraser** (Options → checkbox + Radius) — click-drag to delete every
  visible particle within the radius of the cursor, sent to the Recycle
  Bin like any other delete.

## Recycle Bin, Confirmed, and cleanup

Two fixed categories always exist alongside your own sets:

- **Shift+click** a particle — send it to the **Recycle Bin**. Shift+click
  a particle *already* in the Recycle Bin and it's deleted permanently
  (undoable). Shift+Ctrl+click in the particle list sends that particle and
  everything after it in the list.
- **Space+click** a particle — send it to **Confirmed**.
- Recycle Bin starts hidden by default; toggle visibility for any set from
  its checkbox in the **Sets** panel.
- **Empty Recycle Bin** (Sets panel) permanently deletes everything in it
  at once.
- Sending particles into a class never overrides a visibility you've
  already set — a hidden set stays hidden even as more particles land in it.

Sets panel buttons: **New**, **Rename**, **Save**, **Delete** (all disabled
for Recycle Bin/Confirmed, which are fixed), **Empty Recycle Bin**, and
**Toggle Colors** (rotates every normal set to the next color in the
palette at once — Recycle Bin's red, Confirmed's green, and the yellow
selection highlight are reserved and never change).

## Box size and contrast

- **Box Size** + **Apply to All Sets** (checked by default): editing the
  box size updates every set at once, and a newly created set joins at
  that same size. Uncheck it to size each set independently. Checking the
  box also immediately syncs everything to whatever size is currently
  shown, even if you didn't type a new number.
- Box size and eraser radius only apply once you press **Enter**, click
  elsewhere, or use the spinbox's arrows — not on every keystroke.
- **AutoContrast** (default on) continuously auto-stretches each slice/
  thumbnail. Dragging the **Contrast** slider switches to manual mode and
  applies the exact same black/white levels to all three tomogram views
  and every gallery thumbnail; the slider only applies its change when you
  release the mouse, not while dragging. Re-check AutoContrast to go back.
- **Thickness** averages that many slices into each slab view.

## Particle list (gallery)

- The **All** tab groups particles by category (Recycle Bin, Confirmed,
  then your sets alphabetically) instead of mixing them, with a colored
  border per category. Each set also gets its own tab.
- Each thumbnail shows its particle's numeric ID.
- **Click** — select and jump to that particle in the tomogram views.
  **Ctrl+click** toggles one item in/out of a multi-selection; **Alt+click**
  range-selects from the last clicked item.
- A particle selected in the gallery is outlined **yellow** in the
  tomogram views too — only for selection, never while you're just picking
  or dragging.
- **Right-click + drag** — pan the gallery (does not select).
- **Organize Grid** compacts gaps left by deleted particles; **Detach Tab**
  pops a set's grid into its own window (close it, or its Reattach button,
  to dock it back).
- The **Selection** tab (Options window) shows the current multi-selection
  and lets you **Send to Recycle Bin** (relabels to **Delete Permanently**
  once everything selected is already in the bin), **Send to Confirmed**,
  or **Move to** another (or a new) set.

## Undo / redo

Every add, delete, and reassignment (recycle/confirm/move) is undoable and
redoable, including permanent deletes from the Recycle Bin.

## Saving

Picks persist automatically as one plain-text `x y z` file per set (plus
Recycle Bin/Confirmed) inside `<tomogram>_picks/`, and a `config.settings`
file for any non-default box sizes — no separate save step needed. The
**File** menu also offers **Read/Save Box Coord** (plain coordinate text
files) and **Save Boxes as Stack** (extracted particles as an MRC stack,
or a folder of MRC volumes in 3D mode).

## Shortcut reference

| Action | Shortcut |
|---|---|
| Add particle | Click empty space (tomogram view) |
| Move particle | Click + drag (tomogram view) |
| Pan | Right-click + drag (tomogram view or gallery) |
| Zoom | Mouse wheel |
| Step depth/slice | Shift + wheel (tomogram view) |
| Send to Recycle Bin / delete from bin | Shift + click |
| Send this + rest of list to Recycle Bin | Shift + Ctrl + click (gallery) |
| Send to Confirmed | Space + click |
| Multi-select toggle | Ctrl + click (gallery) |
| Multi-select range | Alt + click (gallery) |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y, Ctrl+Shift+Z, or your platform's native Redo key |
