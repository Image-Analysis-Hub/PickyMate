"""Shared fixtures: a QApplication on the offscreen platform, and a synthetic
tomogram with TomoTwin-style embeddings that opens in well under a second.

Everything in test_synthetic.py and test_session_synthetic.py runs on that
fixture, on any machine. The three original scripts (test_session.py,
test_session_map.py, test_umap_selection.py) drive the real EMPIAR-10988
data and are run by test_scripts.py:

    pytest                      # everything; the scripts skip without the dataset
    pytest -m "not dataset"     # the fast loop: synthetic data only
"""
import argparse
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mrcfile
import numpy as np
import pandas as pd
import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets

from boxer import metadata

# the original scripts run their whole test at import -- never collect them
collect_ignore = ["test_session.py", "test_session_map.py", "test_umap_selection.py"]

SHAPE = 64      # voxels per side of the synthetic tomogram
DIMS = 8        # embedding dimensions
# 16 particles on a 4x4 grid at z=32 and 4 more at z=8: every pair is further
# apart than half the default box size (16), so a click or an eraser stroke
# on one particle can never also hit a neighbour, and nothing is within 16 of
# (32, 32, 60), so a click there is a click on empty space. Embedding rows
# 0-9 form one cluster and 10-19 another, so a ranking from a few Confirmed
# picks has a known right answer.
GRID = (5, 23, 41, 59)
COORDS = ([(x, y, 32) for y in GRID for x in GRID]
          + [(14, 14, 8), (50, 14, 8), (14, 50, 8), (50, 50, 8)])
CLUSTER_A = list(range(10))
CLUSTER_B = list(range(10, 20))
EMPTY_SPOT = (32, 32, 60)
IMPORT_FLAGS = ("embedded_particle_list", "tomotwin_embeddings", "tomotwin_umap")


def write_tomogram(folder, name, seed=0):
    """A synthetic tomogram plus the files `--embedded_particle_list`,
    `--tomotwin_embeddings` and `--tomotwin_umap` would point at, for
    COORDS; returns {flag: path}, with the tomogram under "tomogram"."""
    rng = np.random.default_rng(seed)
    folder = str(folder)
    tomogram = os.path.join(folder, f"{name}.mrc")
    with mrcfile.new(tomogram, overwrite=True) as mrc:
        mrc.set_data(rng.standard_normal((SHAPE, SHAPE, SHAPE)).astype(np.float32))
        mrc.voxel_size = 10.0
    particle_list = os.path.join(folder, f"{name}.txt")
    with open(particle_list, "w") as f:
        f.writelines(f"{x} {y} {z}\n" for x, y, z in COORDS)
    emb = rng.standard_normal((len(COORDS), DIMS)) * 0.1
    emb[CLUSTER_A] += 1.0
    emb[CLUSTER_B] -= 1.0
    xyz = np.array(COORDS, dtype=float)
    # the .temb layout: coord_index, Z, Y, X, then one float16 column per dimension
    temb = pd.DataFrame({"coord_index": range(len(COORDS)),
                         "Z": xyz[:, 2], "Y": xyz[:, 1], "X": xyz[:, 0]})
    for d in range(DIMS):
        temb[str(d)] = emb[:, d].astype(np.float16)
    temb.attrs["tomogram_input_shape"] = (SHAPE, SHAPE, SHAPE)
    temb_path = os.path.join(folder, f"{name}.temb")
    temb.to_pickle(temb_path)
    tumap = pd.DataFrame({"X": xyz[:, 0], "Y": xyz[:, 1], "Z": xyz[:, 2],
                          "umap_0": emb[:, 0], "umap_1": emb[:, 1]})
    tumap_path = os.path.join(folder, f"{name}.tumap")
    tumap.to_pickle(tumap_path)
    return {"tomogram": tomogram, "embedded_particle_list": particle_list,
            "tomotwin_embeddings": temb_path, "tomotwin_umap": tumap_path}


def options(working_dir=None, **flags):
    """The argparse.Namespace tomo_boxer.py builds, every flag off unless given."""
    o = argparse.Namespace(mode="3D", label=None, clean=False, apix=None, working_dir=working_dir,
                           particle_lists=None, native_memmap=False, companion_tomogram=None,
                           embedded_particle_list=None, cryosiam_embeddings=None, cryosiam_umap=None,
                           tomotwin_embeddings=None, tomotwin_umap=None)
    vars(o).update(flags)
    return o


def close_window(win):
    """Close a tomogram the way a session switch does -- save, close every
    window, drop the volume, no confirmation -- then delete the windows so
    they don't pile up, hidden, across tests."""
    app = QtWidgets.QApplication.instance()
    if shiboken6.isValid(win) and not win.closing:
        win.close_for_switch()
    for w in (win, getattr(win, "boxesviewer", None)):
        if w is not None and shiboken6.isValid(w):
            w.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture(scope="session")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tomo(tmp_path):
    paths = write_tomogram(tmp_path, "TS_A")
    paths["picks_dir"] = metadata.picks_dir(paths["tomogram"])
    return paths


@pytest.fixture
def open_window(qapp):
    """Open a synthetic tomogram as `pickymate TS_A.mrc --embedded_particle_list
    ... --tomotwin_embeddings ... --tomotwin_umap ...` would (first_open=True),
    or as a plain `pickymate TS_A.mrc` on its existing working directory.
    Whatever a test leaves open is closed at the end."""
    from boxer.main_window import TomoBoxer
    opened = []

    def _open(paths, first_open=True):
        flags = {k: paths[k] for k in IMPORT_FLAGS} if first_open else {}
        win = TomoBoxer(options(**flags), datafile=paths["tomogram"])
        win.show()
        qapp.processEvents()
        opened.append(win)
        return win

    yield _open
    for win in opened:
        close_window(win)
