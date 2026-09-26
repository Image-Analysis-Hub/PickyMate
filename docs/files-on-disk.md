# Files on Disk

Everything PickyMate stores is plain text next to your data — nothing hides
in a database, and every file can be read, diffed, or produced by your own
scripts. A working directory after some curation looks like this (real
example, EMPIAR-10988 TS_026):

```
TS_026_picks/                    ← default: <tomogram>_picks, or --working_dir
├── TS_026.txt                   ← one file per set, named after the set
├── Confirmed.txt                ← the two fixed categories
├── Recycle Bin.txt
├── UMAP Selection.csv           ← the current UMAP-map selection, if any
├── config.settings              ← box sizes and manual colors
├── embeddings.settings          ← which embedding files are attached
└── .recyclebin_20260924_181432  ← hidden archive from "Empty Recycle Bin"
```

## Set files (`<set name>.txt`)

Each set is one comma-separated file with a header:

```
x,y,z,uid,embedding_row,score
563,368,225,13,13,
756,653,215,0,0,32.1609099408694
```

- `x,y,z` — voxel coordinates.
- `uid` — the particle's stable identity: it survives saves, moves between
  sets, and restarts (the number shown on its gallery thumbnail).
- `embedding_row` — which row of the attached embedding files describes this
  particle; empty for hand-picked particles.
- `score` — the last ranking score, if any (lower = closer to "close to").

A plain, headerless `x y z` file (one particle per line, space-separated) is
also accepted anywhere a set file is read — drop one into the working
directory and it loads as a set; PickyMate upgrades it to the rich format on
the first save. That is also the import/export interchange format.

## `config.settings`

Only exists once some set has a non-default box size or a manually chosen
color — one line per such set, `name boxsize [color]`:

```
Confirmed 30
TS_026 30 #1f77b4
```

## `embeddings.settings`

One line per attached embedding space, written when embeddings are imported
so later launches reload them without any flags:

```
space tomotwin tomotwin embeddings 3081 /data/TomoTwin/TS_026.temb
space tomotwin_umap tomotwin umap 3081 /data/TomoTwin/TS_026.tumap
space cryosiam cryosiam embeddings 3081 /data/CryoSiam/TS_026_embeds.h5
```

Fields: space name, method, kind (`embeddings`/`umap`), particle count, and
the file's path (last, so paths containing spaces survive). If a recorded
file no longer exists — data moved to another disk — a
[session](sessions-and-umap.md) manifest naming the file's new location
repairs the path automatically on the next open.

## `UMAP Selection.csv`

The particles currently selected on the [UMAP map](sessions-and-umap.md), in
the same format as a set file. It is `.csv` rather than `.txt` deliberately:
every `.txt` in the folder is loaded as a set, and the selection is a view,
not a set. Deleted when the selection is cleared.

## `.recyclebin_<date>_<time>`

"Empty Recycle Bin" never discards particles — it moves them to one of these
hidden, timestamped archives (same format as a set file). They are never
loaded back automatically; rename one to `something.txt` to recover it as a
set.

## Session files

```
my_dataset/
├── session.csv                  ← the manifest: one row per tomogram
├── .pickymate_session/          ← cached session UMAP maps
│   └── umap_tomotwin.npz
├── TS_026_picks/
├── TS_027_picks/
└── ...
```

The manifest's columns mirror the command-line options (see
[Sessions and UMAP Maps](sessions-and-umap.md)); relative paths resolve
against the manifest's own folder, so the whole tree can be moved or shared
as one piece. The `.npz` map caches are safe to delete — they are recomputed
on demand.

## Outside the working directory

`~/.config/pickymate/shortcuts.json` (or under `$XDG_CONFIG_HOME`) — your
keyboard-shortcut overrides. Created only if you customize something.
