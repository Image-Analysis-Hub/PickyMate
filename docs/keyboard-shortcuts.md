# Keyboard Shortcuts

<!-- shortcuts:begin -- generated from boxer/shortcuts.py by docs/make_figures.py; do not edit the table by hand -->
| Action | Shortcut |
|---|---|
| Add particle | Click empty space (tomogram view, Pick tool) |
| Move particle | Click + drag (tomogram view, Pick tool) |
| Switch active picking set + gallery tab | 1-9 |
| Next / previous tomogram in a session | Ctrl+PgDown / Ctrl+PgUp |
| Reorder a set's tab | Drag it in the gallery |
| Pan | Right-click + drag (tomogram view, gallery, or UMAP map) |
| Zoom | Mouse wheel |
| Step depth/slice | Shift + wheel (tomogram view) |
| Send to Recycle Bin / delete from bin | Shift + click |
| Send this + rest of list to Recycle Bin | Shift + Ctrl + click (gallery) |
| Send to Confirmed | Space + click |
| Multi-select toggle | Ctrl + click (gallery, tomogram view, or UMAP map) |
| Multi-select range | Alt + click (gallery) |
| Send every selected particle to Recycle Bin | Delete |
| Send every selected particle to Confirmed | Insert |
| Navigate to the neighboring particle | Arrow keys (gallery) |
| Shape-select (rectangle/circle) | Click + drag, Select tool active |
| Shape-select (freeform) | Click to place points, Select tool active |
| Finish freeform selection | Double-click, Enter, or click near the start point |
| Undo last freeform point | Backspace (while placing points) |
| Cancel freeform selection | Escape |
| Lasso-select on the UMAP map | Drag (Ctrl+drag adds) |
| Clear the current selection (anywhere in the app) | Escape |
| New set | Click the gallery's "+" tab, or Set > New Set... |
| Rename set | Double-click its tab, or Set > Rename Set... |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y, Ctrl+Shift+Z, or your platform's native Redo key |
| Force everything to re-derive and redraw | F5 (Edit > Refresh) |
| Open this reference | F1 (Help > Keyboard Shortcuts) |
<!-- shortcuts:end -->

This reference is also in-app: **Help > Keyboard Shortcuts** or **F1**. The
table above is generated from the same source the app shows, so the two
cannot drift apart.

## Customizing shortcuts

**Settings > Customize Keyboard Shortcuts...** rebinds most of the above:

![Customize Keyboard Shortcuts dialog, annotated](images/shortcuts-settings.png)

1. Click a field and press the new combination. **Reset** restores one row,
   **Restore All Defaults** everything; clashes are flagged in red.

Saved to `~/.config/pickymate/shortcuts.json` (or under
`$XDG_CONFIG_HOME`), separate from any tomogram's picks.
