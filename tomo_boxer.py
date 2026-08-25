#!/usr/bin/env python3
"""Standalone tomogram particle boxer (functional port of EMAN2's
e2spt_boxer.py that runs outside the EMAN2 environment).

Usage:
    python tomo_boxer.py tomogram.mrc [--box2d | --box3d] [--label NAME] [--clean] [--apix APIX]
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
    options = parser.parse_args()

    options.mode = "2D" if options.box2d and not options.box3d else "3D"

    app = QApplication(sys.argv)
    boxer = TomoBoxer(options, datafile=options.tomogram)
    boxer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
