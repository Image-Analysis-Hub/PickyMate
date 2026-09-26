#!/usr/bin/env python3
"""Embed picking coordinates with a stock TomoTwin installation.

Extracts a subvolume around every coordinate of a picks file directly from
the tomogram and embeds it with a TomoTwin model, preserving the row order
of the picks file: row i of the output .temb always describes pick i. Run
it with the Python of an environment where TomoTwin is installed:

    conda activate tomotwin
    python tomotwin_embed_picks.py -m tomotwin_latest.pth \\
        -v Tomograms/TS_026.rec -c TemplateMatching/TS_026.txt -o out/

The picks file is whitespace-separated X Y Z in voxels, no header (a header
line is skipped if present). The output .temb has the same shape as the one
`tomotwin_embed.py tomogram` writes (Z, Y, X plus embedding columns
"0".."N", float16), so the stock `tomotwin_tools.py umap` runs on it
unchanged, and PickyMate reads it directly.

How this differs from stock sliding-window embedding, and why:

- Border picks are kept: the tomogram is reflect-padded by half a box
  (default; --no-padding disables it), so a pick near the border still gets
  a full, centered box instead of being dropped.
- Picks up to 2 px outside the volume - a common rounding artifact of
  picking tools - are clamped to the nearest valid voxel.
- Anything farther outside aborts with the offending row numbers listed.
  The row-for-row match between the picks file and the .temb is the
  contract everything downstream relies on, so nothing is ever skipped.
- TomoTwin's normalization is patched at runtime to divide by
  (std + 1e-12): a completely flat window (masked or padded background)
  has std 0 and would otherwise embed as NaN.

Written against TomoTwin 0.9.1. It reaches into three TomoTwin internals
(embedor, volumedata, preprocess) and checks at startup that they still
look the way it expects.
"""
import argparse
import hashlib
import inspect
import os
import sys

import numpy as np
import pandas as pd

try:
    import tomotwin
    import tomotwin.modules.common.preprocess as pp
    import tomotwin.modules.inference.embedor as embedor_module
    from tomotwin.modules.common.io.mrc_format import MrcFormat
    from tomotwin.modules.inference.embedor import TorchEmbedor
    from tomotwin.modules.inference.volumedata import VolumeDataset
except ImportError as e:
    sys.exit(f"TomoTwin is not importable from this Python ({e}).\n"
             f"Run this script with the Python of the environment where "
             f"TomoTwin is installed, e.g. 'conda activate tomotwin' first.")

# Picks this many voxels outside the tomogram are snapped back to the edge:
# picking and template-matching tools routinely report x == nx for the last
# voxel of the axis, which is rounding, not a bad pick.
BOUNDARY_CLAMP_TOLERANCE = 2


def check_internals() -> None:
    """Fail early, and clearly, if the TomoTwin internals this script relies
    on have changed shape in the installed version."""
    problems = []
    if not callable(getattr(pp, "norm", None)):
        problems.append("tomotwin.modules.common.preprocess.norm is missing")
    dataset = getattr(embedor_module, "TorchVolumeDataset", None)
    if dataset is None:
        problems.append("tomotwin...embedor.TorchVolumeDataset is missing")
    elif "pp.norm" not in inspect.getsource(dataset.__getitem__):
        problems.append("TorchVolumeDataset no longer normalizes via pp.norm, "
                        "so the flat-window NaN patch would not take effect")
    if problems:
        sys.exit("This TomoTwin version ({}) is not the one this script was "
                 "written against:\n- {}".format(
                     tomotwin.__version__, "\n- ".join(problems)))


def patch_norm() -> None:
    """Replace TomoTwin's normalization with an epsilon-guarded one.

    TorchVolumeDataset looks norm up as an attribute of the preprocess
    module on every call, so patching the module is enough - including in
    DataLoader worker processes, which fork after this runs."""
    def norm(vol: np.array) -> np.array:
        return (vol - np.mean(vol)) / (np.std(vol) + 1e-12)
    pp.norm = norm


class PicksVolumeDataset(VolumeDataset):
    """Subvolumes of a single in-memory volume at explicit center
    coordinates. Centers are in the volume's own array index order (Z, Y, X)
    and must already account for any padding applied to the volume. Order is
    preserved: item i always corresponds to centers[i]."""

    def __init__(self, volume: np.array, centers: np.array, box_size: int):
        self.volume = volume
        self.centers = centers
        self.box_size = box_size
        odd_factor = box_size % 2
        self._half_lo = int((box_size - odd_factor) // 2)
        self._half_hi = self._half_lo + odd_factor

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, itemindex) -> np.array:
        c = self.centers[itemindex]
        lo = c - self._half_lo
        hi = c + self._half_hi
        return self.volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    def get_localization(self, itemindex):
        return self.centers[itemindex]


def read_coords(path: str) -> pd.DataFrame:
    """Whitespace-separated X Y Z, one pick per row; a single header line is
    tolerated. The DataFrame index is the row number in the file."""
    try:
        coords = pd.read_csv(path, sep=r"\s+", header=None, index_col=False,
                             dtype=float)
    except (ValueError, pd.errors.ParserError):
        print("Could not parse the first row as numbers; skipping it as a header.")
        coords = pd.read_csv(path, sep=r"\s+", header=None, index_col=False,
                             dtype=float, skiprows=1)
    if coords.shape[1] != 3:
        sys.exit(f"{path}: expected 3 columns (X Y Z), found {coords.shape[1]}.")
    coords.columns = ["X", "Y", "Z"]
    return coords


def window_size_of(embedor: TorchEmbedor) -> int:
    config = embedor.tomotwin_config or {}
    if "window_size" in config:
        return int(config["window_size"][0])
    print("Can't find window size in model. Use window size of 37.")
    return 37


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def embed_picks(tomo: np.array, coords: pd.DataFrame, embedor: TorchEmbedor,
                box_size: int, padding: bool) -> pd.DataFrame:
    orig_shape = tomo.shape
    odd_factor = box_size % 2
    half_lo = int((box_size - odd_factor) // 2)
    half_hi = half_lo + odd_factor

    if padding:
        tomo = np.pad(tomo, half_lo, mode="reflect")
        print(f"Padded the tomogram by {half_lo} (reflect), new shape {tomo.shape}")

    # coords file is X Y Z; volume array indices are Z Y X
    zyx = coords[["X", "Y", "Z"]].to_numpy()[:, [2, 1, 0]]

    max_idx = np.array(orig_shape) - 1
    below = np.clip(-zyx, 0, None)
    above = np.clip(zyx - max_idx, 0, None)
    out_of_range = np.maximum(below, above)
    outside = np.any(out_of_range > 0, axis=1)
    clampable = outside & np.all(out_of_range <= BOUNDARY_CLAMP_TOLERANCE, axis=1)

    # clamp only where the box is extracted from; the output keeps the
    # original coordinates, so they stay identical to the picks file
    zyx_extract = zyx
    if clampable.any():
        print(f"NOTE: {int(clampable.sum())} pick(s) within "
              f"{BOUNDARY_CLAMP_TOLERANCE}px of the tomogram boundary (likely "
              f"a rounding artifact) were clamped to the nearest valid voxel "
              f"(picks file rows: {coords.index[clampable].tolist()}).")
        zyx_extract = zyx.copy()
        zyx_extract[clampable] = np.clip(zyx_extract[clampable], 0, max_idx)

    centers = (zyx_extract + (half_lo if padding else 0)).astype(int)
    lo_ok = (centers - half_lo >= 0).all(axis=1)
    hi_ok = (centers + half_hi <= np.array(tomo.shape)).all(axis=1)
    bad = ~(lo_ok & hi_ok)
    if bad.any():
        sys.exit(f"{int(bad.sum())} pick(s) cannot be extracted (picks file "
                 f"rows: {coords.index[bad].tolist()}).\nEvery output row must "
                 f"correspond to the same row of the picks file, so nothing is "
                 f"ever skipped. Fix or remove these coordinates"
                 + ("" if padding else ", or drop --no-padding") +
                 "; with padding this can only happen when a pick is genuinely "
                 "far outside the tomogram, or the box is larger than the "
                 "tomogram itself.")

    dataset = PicksVolumeDataset(volume=tomo, centers=centers, box_size=box_size)
    embeddings = embedor.embed(volume_data=dataset)

    n_nan = int(np.isnan(embeddings).any(axis=1).sum())
    if n_nan:
        sys.exit(f"{n_nan} embedding(s) contain NaN - this should be impossible "
                 f"with the normalization patch; not writing an output.")

    df = pd.DataFrame(
        np.hstack([zyx, embeddings]),
        columns=["Z", "Y", "X"] + [str(i) for i in range(embeddings.shape[1])])
    df.index.name = "index"
    return df.astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--modelpth", required=True,
                        help="Path to the TomoTwin model (.pth)")
    parser.add_argument("-v", "--volumes", required=True,
                        help="Path to a single tomogram file (.mrc/.rec)")
    parser.add_argument("-c", "--coords", required=True,
                        help="Whitespace-separated picks file, columns X Y Z, no header")
    parser.add_argument("-o", "--outpath", required=True,
                        help="Output directory; the .temb is named after the picks file")
    parser.add_argument("-b", "--batchsize", type=int, default=128,
                        help="Batch size for the embedding (default: 128)")
    parser.add_argument("--size", type=int, default=None,
                        help="Box size in voxels around each pick "
                             "(default: the window size stored in the model)")
    parser.add_argument("--no-padding", dest="padding", action="store_false",
                        default=True,
                        help="Do not reflect-pad the tomogram; border picks "
                             "then abort instead of getting a full box")
    args = parser.parse_args()

    check_internals()
    patch_norm()

    embedor = TorchEmbedor(weightspth=args.modelpth,
                           batchsize=args.batchsize, workers=4)
    box_size = args.size or window_size_of(embedor)
    tomo = -1 * MrcFormat.read(args.volumes)  # -1 inverts the contrast, as TomoTwin expects
    coords = read_coords(args.coords)
    print(f"Embedding {len(coords)} picks from {args.volumes} "
          f"(box {box_size}, padding {'on' if args.padding else 'off'})")

    df = embed_picks(tomo, coords, embedor, box_size, args.padding)
    if df is None:
        sys.exit("No embeddings produced.")

    df.attrs["tt_version_embed"] = tomotwin.__version__
    df.attrs["filepath"] = args.volumes
    df.attrs["coords_path"] = args.coords
    df.attrs["modelpth"] = args.modelpth
    df.attrs["modelmd5"] = file_md5(args.modelpth)
    df.attrs["window_size"] = box_size
    df.attrs["tomogram_input_shape"] = tomo.shape
    df.attrs["tomotwin_config"] = embedor.tomotwin_config
    df.attrs["padding"] = args.padding
    df.attrs["row_order"] = "row i is row i of the picks file"

    os.makedirs(args.outpath, exist_ok=True)
    out = os.path.join(
        args.outpath,
        os.path.splitext(os.path.basename(args.coords))[0] + ".temb")
    df.to_pickle(out)
    print(f"Wrote {len(df)} embeddings to {out}")


if __name__ == "__main__":
    main()
