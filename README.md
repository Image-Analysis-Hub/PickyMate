# PickyMate

A standalone tomogram particle boxer (a PySide6 + numpy + mrcfile port of
EMAN2's `e2spt_boxer.py` that runs without EMAN2). Pick particles from an
MRC/REC tomogram in three orthogonal views, organize them into sets, clean
up with a Recycle-Bin / Confirmed workflow, and export coordinates.

## Running it

```
python tomo_boxer.py tomogram.mrc [--box2d | --box3d] [--label NAME] [--clean] [--apix APIX]
```

- `--box2d` / `--box3d` — pick 2D particles from slices, or 3D sub-volumes (default).
- `--label NAME` — open with the named set already selected.
- `--clean` — start with the Eraser tool already active.
- `--apix` — override the pixel size read from the file header.

Two windows open: the **tomogram window** (orthogonal XY/XZ/ZY views,
toolbars, and controls) and the **Particle List** (thumbnail gallery, also
where sets are managed).

## Tools: Pick / Eraser / Select

The **Tools** toolbar holds three mutually exclusive tools — exactly one is
ever active. Every tool's options sit at the same fixed spot next to the
tool buttons at all times; only the active tool's options are enabled:

- **Pick** (default) — click empty space to add a particle, click + drag an
  existing one to move it. Shift/Space/Ctrl+click still work for Recycle
  Bin/Confirmed/multi-select (see below).
- **Eraser** — click-drag in a tomogram view to send every particle within
  **Radius** of the cursor to the Recycle Bin.
- **Select** — click-drag (or click-to-place points for **Freeform**) to
  select every particle inside a **Rectangle**, **Circle**, or **Freeform**
  shape, replacing the current selection. Non-destructive — use the
  **Edit** toolbar or gallery Ctrl+click to act on the result.

Both Eraser and Select share one **2D/3D** checkbox (disabled for Pick,
which has no use for it):
- **2D** — only affects particles actually visible on the current slice.
- **3D** — Eraser affects a full sphere of Radius regardless of depth;
  Select drops the depth limit entirely, extruding the shape
  (cuboid/cylinder/tube) through the whole volume.

For Freeform: click to place each corner point (a live preview closes back
to the start as you go); finish with a double-click, Enter, or by clicking
back near the first point. Backspace removes the last point, Escape cancels.

## Recycle Bin, Confirmed, and cleanup

Two fixed categories always exist alongside your own sets:

- **Shift+click** a particle — send it to the **Recycle Bin**. Shift+click
  a particle *already* in the Recycle Bin and it's deleted permanently
  (undoable). Shift+Ctrl+click in the particle list sends that particle and
  everything after it in the list.
- **Space+click** a particle — send it to **Confirmed**.
- Recycle Bin starts hidden by default; toggle visibility from its tab's
  checkbox in the gallery (see below).
- Sending particles into a class never overrides a visibility you've
  already set — a hidden set stays hidden even as more particles land in it.
- **Number keys 1-9** switch the active picking class to the 1st-9th normal
  set (in the same order the gallery's tabs list them); Recycle Bin/Confirmed
  aren't reachable this way since you never pick new particles into them
  directly.

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
- **Avg Slices** averages that many slices, centered on the current depth,
  into each view — a mean, not a sum, so it reduces noise without changing
  overall brightness.

## Particle list (gallery) and sets management

Sets are managed directly through the gallery's tabs — there's no separate
Sets window:

- The **+** tab (always last) prompts for a name and creates a new set.
- **Double-click** a set's tab to rename it.
- The **Delete Set** button deletes the current tab's set, sending every
  particle in it to the Recycle Bin (disabled on "All", "+", Recycle Bin,
  and Confirmed).
- Each set's tab has its own **visibility checkbox**, independent of which
  tab you're currently looking at — any number of sets can be visible in
  the tomogram views at once.
- **Empty Recycle Bin** only appears while the Recycle Bin tab is active.
- Switching to a set's tab also makes it the active picking class — the
  same as clicking a Sets-panel row used to.

Beyond sets management:

- The **All** tab groups particles by category (Recycle Bin, Confirmed,
  then your sets alphabetically) instead of mixing them, with a colored
  border per category.
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

## Selecting from the tomogram window

- **Ctrl+click** a particle in any of the three tomogram views toggles it
  in/out of the multi-selection — the exact same selection the gallery
  uses, so it works interchangeably with Ctrl/Alt+click in the gallery.
- A small label at the bottom-right of the tomogram window shows "N
  particle(s) selected", the way a file manager shows a selection count.
- The **Edit** toolbar acts on that selection directly: **Confirm
  Selected**, **Recycle Selected**, and **Move Selected** (type a set name
  — existing or new — into the box beside it, then Move Selected).
- **Toggle Colors** (Edit toolbar) rotates every normal set to the next
  color in the palette at once — Recycle Bin's red, Confirmed's green, and
  the yellow selection highlight are reserved and never change.

## Undo / redo

Every add, delete, and reassignment (recycle/confirm/move) is undoable and
redoable, including permanent deletes from the Recycle Bin. The **Edit**
menu and toolbar mirror Undo/Redo with visible buttons alongside the
keyboard shortcuts.

## Saving

Picks persist automatically as one plain-text `x y z` file per set (plus
Recycle Bin/Confirmed) inside `<tomogram>_picks/`, and a `config.settings`
file for any non-default box sizes — no separate save step needed.

The **File** menu offers:
- **Save** — forces an immediate write of every set's picks and box sizes
  (happens automatically anyway) and confirms with the folder it wrote to.
- **Read Box Coord** — import an `x y z` coordinate text file as a brand
  new set, named after the file (auto-renamed with a numeric suffix if
  that name's already taken).
- **Save Box Coord** — copies the already-saved coordinate file(s) — all
  sets, or one you pick — to a folder of your choice, for your own records.

## Shortcut reference

| Action | Shortcut |
|---|---|
| Add particle | Click empty space (tomogram view, Pick tool) |
| Move particle | Click + drag (tomogram view, Pick tool) |
| Switch active picking class | 1-9 |
| Pan | Right-click + drag (tomogram view or gallery) |
| Zoom | Mouse wheel |
| Step depth/slice | Shift + wheel (tomogram view) |
| Send to Recycle Bin / delete from bin | Shift + click |
| Send this + rest of list to Recycle Bin | Shift + Ctrl + click (gallery) |
| Send to Confirmed | Space + click |
| Multi-select toggle | Ctrl + click (gallery or tomogram view) |
| Multi-select range | Alt + click (gallery) |
| Shape-select (rectangle/circle) | Click + drag, Select tool active |
| Shape-select (freeform) | Click to place points, Select tool active |
| Finish freeform selection | Double-click, Enter, or click near the start point |
| Undo last freeform point | Backspace (while placing points) |
| Cancel freeform selection | Escape |
| New set | Click the gallery's "+" tab |
| Rename set | Double-click its tab |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y, Ctrl+Shift+Z, or your platform's native Redo key |
| Open this reference | F1 |
