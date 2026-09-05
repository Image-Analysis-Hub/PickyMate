#!/usr/bin/env python3
"""Standalone tomogram particle boxer (functional port of EMAN2's
e2spt_boxer.py that runs outside the EMAN2 environment).

Usage:
    python tomo_boxer.py tomogram.mrc [--box2d | --box3d] [--label NAME] [--clean] [--apix APIX]
        [--working_dir DIR] [--particle_lists PATH [PATH ...]]
        [--embedded_particle_list PATH --embeddings_h5 PATH [--embeddings_umap PATH]]
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
             "bundle attached (requires --embeddings_h5). Separate from --particle_lists "
             "since only one particle list can have embeddings attached at a time.")
    parser.add_argument(
        "--embeddings_h5", type=str, default=None, metavar="PATH",
        help="HDF5 file with one dataset holding the full-dimensionality embedding for "
             "each particle in --embedded_particle_list, in the same order (either "
             "(particle, feature) or (feature, particle) orientation is fine)")
    parser.add_argument(
        "--embeddings_umap", type=str, default=None, metavar="PATH",
        help="Optional companion CSV with a 2D UMAP-reduced embedding per particle (needs "
             "'x','y' and 'centroid-0/1/2' columns), used both as an alternate distance "
             "space and to cross-check --embedded_particle_list's own coordinates")
    options = parser.parse_args()

    if (options.embeddings_h5 or options.embeddings_umap) and not options.embedded_particle_list:
        parser.error("--embeddings_h5/--embeddings_umap require --embedded_particle_list")
    if options.embedded_particle_list and not options.embeddings_h5:
        parser.error("--embedded_particle_list requires --embeddings_h5")

    options.mode = "2D" if options.box2d and not options.box3d else "3D"

    app = QApplication(sys.argv)
    boxer = TomoBoxer(options, datafile=options.tomogram)
    boxer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
