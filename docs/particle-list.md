# The Particle List

The Particle List window is the thumbnail gallery and where sets are managed,
directly through its tabs.

![Particle List window on the All tab, annotated](images/particle-list-overview.png)

1. **Tabs** — one per set, plus **All**, **Recycle Bin**, **Confirmed**, and
   **+**. A normal set's tab shows its number-key position, a visibility
   checkbox, and a color swatch.
2. **Thumbnail grid** — here the **All** tab: particles grouped by set, each
   bordered in its set's color (red Recycle Bin, green Confirmed). A set's
   own tab shows just that set, in pick order.
3. **Ranking row** — see [Ranking by Embedding Distance](ranking.md). Active
   on a set's tab (grayed on All, which is not one set to rank).
4. **Gallery toolbar** — Organize Grid, Detach Tab, sorting, Delete Set /
   Empty Recycle Bin.

## Sets

- **+** creates a set; **double-click** a tab renames it; **Delete Set**
  sends its particles to the Recycle Bin.
- Any number of sets can be visible at once (the checkboxes); switching to a
  set's tab makes it the active picking set.
- The **color swatch** picks any color, or reverts to the automatic one. A
  manual color is remembered and never touched by Toggle Colors.
- **Drag** a set's tab to reorder your sets; number keys follow the tab
  order. All / Recycle Bin / Confirmed / + stay pinned.
- **Empty Recycle Bin** (on its tab) archives everything to a hidden
  timestamped file, and is still undoable right after.

## Gallery toolbar

- **Toggle Colors** rotates every automatic set color at once.
- **Show Active Set Only** shows only the tab you are looking at, following
  you from tab to tab; turning it off restores the previous visibility.
- **Show All Sets** makes everything visible again in one click.
- **Organize Grid** compacts the gaps deleted particles leave; **Detach
  Tab** pops a set into its own window.

## Browsing and selecting

- **Click** a thumbnail to jump to that particle in the tomogram views.
  **Ctrl+click** toggles it in a multi-selection, **Alt+click**
  range-selects, **arrow keys** step to neighbors.
- Selected particles are outlined **yellow** everywhere.
- **Right-drag** pans; the **wheel** zooms, centered on the selection.
