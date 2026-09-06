# Picking and Tools

## Pick / Eraser / Select

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
  checkbox in the gallery.
- Sending particles into a class never overrides a visibility you've
  already set — a hidden set stays hidden even as more particles land in it.
- **Number keys 1-9** switch the active picking class to the 1st-9th normal
  set and switch the gallery to that set's own tab — the same number shown
  right on the tab, which always matches whatever order the tabs are
  currently in (see "Reordering tabs" in [The Particle List](particle-list.md)),
  so the mapping is never a guess. A detached set has no tab to number, so
  it's skipped; Recycle Bin/Confirmed aren't reachable this way either,
  since you never pick new particles into them directly.
- **Empty Recycle Bin** doesn't discard anything outright — see
  [The Particle List](particle-list.md#sets-management).

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
