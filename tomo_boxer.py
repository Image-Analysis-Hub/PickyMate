#!/usr/bin/env python3
"""PickyMate: a standalone tomogram particle-picking tool.

Usage:
    python tomo_boxer.py tomogram.mrc [--box2d | --box3d] [--label NAME] [--clean] [--apix APIX]
        [--working_dir DIR] [--particle_lists PATH [PATH ...]]
        [--embedded_particle_list PATH]
        [--cryosiam_embeddings PATH [--cryosiam_umap PATH]]
        [--tomotwin_embeddings PATH [--tomotwin_umap PATH]]
        [--companion_tomogram PATH] [--native_memmap]

Each embedding method has its own pair of flags -- their file formats differ
enough that naming the method is clearer than having the app guess from an
extension (see boxer/embeddings_io.py). Give more than one method at once to
rank the same picks by each and compare: the first creates the set, the rest
attach to those same particles, and every method's full embedding and UMAP
show up as separate choices in the ranking dropdown. For example:

    python tomo_boxer.py TS_026.rec \\
        --embedded_particle_list Picks_3DTM/TS_026.txt \\
        --cryosiam_embeddings CryoSIAM/TS_026_embeds.h5 \\
        --cryosiam_umap       CryoSIAM/TS_026_embeds_umap_data.csv \\
        --tomotwin_embeddings TomoTwin/TS_026.temb \\
        --tomotwin_umap       TomoTwin/TS_026.tumap
"""
from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from boxer.main_window import TomoBoxer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tomogram", help="Tomogram (MRC/REC) to pick particles from")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--box2d", action="store_true", help="Pick 2D particles from tomogram slices")
    mode.add_argument("--box3d", action="store_true", help="Pick 3D sub-volumes (default)")
    parser.add_argument("--label", type=str, default=None, help="Start viewing the specified particle set/label")
    parser.add_argument("--clean", action="store_true", help="Start in eraser (cleaning) mode")
    parser.add_argument("--apix", type=float, default=None, help="Override pixel size (Angstrom/pixel)")
    parser.add_argument(
        "--working_dir", type=str, default=None,
        help="Directory to store/load picks from (default: <tomogram>_picks next to the tomogram), "
             "resolved relative to the current directory")
    parser.add_argument(
        "--particle_lists", type=str, nargs="+", default=None, metavar="PATH",
        help="One or more 'x y z' coordinate text files, each imported as its own new set "
             "(named after the file, auto-suffixed on a name collision) and copied into "
             "the working directory")
    parser.add_argument(
        "--embedded_particle_list", type=str, default=None, metavar="PATH",
        help="A single 'x y z' coordinate text file to import with a matched embeddings "
             "bundle attached. Required with --cryosiam_embeddings (whose .h5 carries no "
             "coordinates); optional with --tomotwin_embeddings (whose .temb carries its "
             "own, and is then cross-checked against this list). Separate from "
             "--particle_lists since only one particle list can have embeddings attached "
             "at a time.")
    parser.add_argument(
        "--cryosiam_embeddings", type=str, default=None, metavar="PATH",
        help="CryoSIAM embeddings: an HDF5 file with one dataset holding the "
             "full-dimensionality embedding per particle, in the same order as "
             "--embedded_particle_list (either (particle, feature) or (feature, particle) "
             "orientation is fine)")
    parser.add_argument(
        "--cryosiam_umap", type=str, default=None, metavar="PATH",
        help="Optional CryoSIAM UMAP CSV (needs 'x','y' and 'centroid-0/1/2' columns), "
             "used both as an alternate ranking space and to cross-check the particle "
             "list's own coordinates")
    parser.add_argument(
        "--tomotwin_embeddings", type=str, default=None, metavar="PATH",
        help="TomoTwin embeddings: a .temb file (a pickled pandas DataFrame with "
             "'coord_index', 'Z', 'Y', 'X' and one column per embedding dimension). "
             "Carries its own coordinates, so --embedded_particle_list is optional here.")
    parser.add_argument(
        "--tomotwin_umap", type=str, default=None, metavar="PATH",
        help="Optional TomoTwin UMAP: a .tumap file (a pickled pandas DataFrame with "
             "'X','Y','Z' and 'umap_0','umap_1'), used both as an alternate ranking "
             "space and to cross-check coordinates")
    parser.add_argument(
        "--companion_tomogram", type=str, default=None, metavar="PATH",
        help="A second version of the same tomogram (e.g. denoised), of identical "
             "dimensions, to switch display between via View > Show Companion Volume. "
             "Picks/sets are shared -- only the pixel data shown changes. Not actually "
             "read from disk until first switched to.")
    parser.add_argument(
        "--native_memmap", action="store_true",
        help="Load tomogram(s) at their exact on-disk dtype via a memory-mapped file "
             "instead of the RAM-saving default (native integer dtype kept as-is, "
             "float32/float64 downcast to float16). Trades some access latency for "
             "full original precision and a much lower RAM ceiling.")
    options = parser.parse_args()

    if options.cryosiam_umap and not options.cryosiam_embeddings:
        parser.error("--cryosiam_umap requires --cryosiam_embeddings")
    if options.tomotwin_umap and not options.tomotwin_embeddings:
        parser.error("--tomotwin_umap requires --tomotwin_embeddings")
    # CryoSIAM's .h5 holds no coordinates, so it can't be imported alone;
    # TomoTwin's .temb carries its own, so a list is optional there.
    if options.cryosiam_embeddings and not options.embedded_particle_list:
        parser.error("--cryosiam_embeddings requires --embedded_particle_list "
                     "(the .h5 carries no coordinates of its own)")
    if (options.embedded_particle_list
            and not (options.cryosiam_embeddings or options.tomotwin_embeddings)):
        parser.error("--embedded_particle_list is for importing a list *with* embeddings; "
                     "use --particle_lists for a plain coordinate file")

    options.mode = "2D" if options.box2d and not options.box3d else "3D"

    app = QApplication(sys.argv)
    boxer = TomoBoxer(options, datafile=options.tomogram)
    boxer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
