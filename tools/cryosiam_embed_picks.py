#!/usr/bin/env python3
"""Embed picking coordinates with an existing CryoSIAM installation.

Extracts a patch around every coordinate of a picks file from a tomogram
and embeds it with a trained CryoSIAM SimSiam model, preserving the row
order of the picks file: column i of the output embeddings always
describes pick i. Run it with the Python of an environment where CryoSIAM
is installed:

    conda activate cryosiam
    python cryosiam_embed_picks.py -m simsiam_embeds_denoised_no_masking.ckpt \\
        -v denoised/TS_026.rec -c TemplateMatching/TS_026.txt -o out/

The picks file is whitespace-separated X Y Z in voxels, no header (a header
line is skipped if present). The tomogram must be the kind of volume the
model was trained on - for the usual pipeline that is the CryoSIAM-denoised
tomogram, since the normalization stored in the checkpoint clips intensities
to its training range before anything is extracted.

Outputs, exactly the shape CryoSIAM's own tools and PickyMate read:

    <picks>_embeds.h5            'embeddings' (dim, n_picks) float32
    <picks>_instance_labels.h5   'labels' (n_picks,) int64, 1..n in pick order
    <picks>_instance_regions.csv label + centroid-0/1/2 (Z, Y, X) per pick
    <picks>_instance_regions.star  the same, in RELION column names

Why this script exists instead of CryoSIAM's own
extract_patches_embeddings_from_centers app:

- The released app computes every patch's slice start as min(center - box,
  0), which is never positive - every patch is anchored at index 0 and the
  center crop lands far from the pick. This script uses the fixed slicing
  (clamped to the volume on both sides).
- The app materializes every patch in memory up front and holds the full
  embedding matrix besides; here patches are extracted lazily, one batch at
  a time, and embeddings stream straight into the HDF5, so memory stays
  flat however many picks there are.
- The app is driven by a config file spanning whole folders of tomograms;
  this is one tomogram + one picks file + one model, and everything
  model-side (network architecture, normalization) is read from the
  checkpoint itself.

Border picks are never dropped: the raw crop is clamped to the volume and
the patch transforms pad and center-crop it to the network size (which
shifts the effective center for picks closer to the border than the crop
half-size - the same behaviour as the app). A pick so far outside the
volume that its crop would be empty aborts with the offending row numbers
listed, because the row-for-row match with the picks file is the contract
everything downstream relies on.

Written against CryoSIAM as of 2026-01 (SimSiam constructor and MrcReader);
it checks the checkpoint's layout at startup and says what is missing.
"""
import argparse
import os
import sys
import collections

import numpy as np
import pandas as pd

try:
    import h5py
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from monai.transforms import (
        Compose,
        EnsureType,
        LoadImage,
        SpatialPad,
        CenterSpatialCrop,
        EnsureChannelFirst,
        NormalizeIntensity,
        ScaleIntensityRange,
    )
    from cryosiam.data import MrcReader
    from cryosiam.networks.nets import SimSiam
except ImportError as e:
    sys.exit(f"CryoSIAM (or one of its dependencies) is not importable from "
             f"this Python ({e}).\nRun this script with the Python of the "
             f"environment where CryoSIAM is installed, e.g. "
             f"'conda activate cryosiam' first.")


def load_checkpoint_config(checkpoint: dict, contrastive: bool) -> dict:
    """The training configuration stored inside the checkpoint: network
    constructor arguments and the intensity normalization of the training
    data. Fails with a clear message if the checkpoint is not a CryoSIAM
    SimSiam checkpoint of the expected layout."""
    key = "backbone_config" if contrastive else "config"
    try:
        config = checkpoint["hyper_parameters"][key]
        config["parameters"]["network"]
        config["parameters"]["data"]
    except (KeyError, TypeError):
        sys.exit(f"The checkpoint does not carry hyper_parameters['{key}'] "
                 f"with parameters.network / parameters.data - is it a "
                 f"CryoSIAM SimSiam checkpoint"
                 + ("" if contrastive else
                    ", or does it need --contrastive dropped/added") + "?")
    return config


def load_prediction_model(checkpoint: dict, config: dict, device) -> SimSiam:
    net = config["parameters"]["network"]
    model = SimSiam(block_type=net["block_type"],
                    n_input_channels=net["in_channels"],
                    spatial_dims=net["spatial_dims"],
                    num_layers=net["num_layers"],
                    num_filters=net["num_filters"],
                    no_max_pool=net["no_max_pool"],
                    dim=net["dim"],
                    pred_dim=net["pred_dim"])
    state = collections.OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        state[k.replace("_model.", "")] = v
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


class CentersPatchDataset(TorchDataset):
    """Lazily extracts a patch around each center from an already loaded
    tomogram: only one batch of patches exists in memory at any time. The
    raw crop is clamped to the volume; the transforms pad and center-crop
    it to the network patch size. Order is preserved: item i always
    corresponds to centers_zyx[i]."""

    def __init__(self, image, centers_zyx, box_size, transform):
        self.image = image
        self.centers = centers_zyx
        self.box_size = box_size
        self.transform = transform
        self.max_z, self.max_y, self.max_x = image.shape

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        c = self.centers[idx]
        slices = (slice(max(int(c[0]) - self.box_size, 0),
                        min(int(c[0]) + self.box_size, self.max_z)),
                  slice(max(int(c[1]) - self.box_size, 0),
                        min(int(c[1]) + self.box_size, self.max_y)),
                  slice(max(int(c[2]) - self.box_size, 0),
                        min(int(c[2]) + self.box_size, self.max_x)))
        return self.transform(self.image[slices]), idx


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


def load_volume(path: str, data_cfg: dict) -> np.array:
    """The tomogram, normalized exactly the way CryoSIAM's prediction apps
    normalize it: intensities clipped and scaled from the checkpoint's
    [min, max] to [0, 1], then (x - mean) / std."""
    transforms = Compose([
        LoadImage(reader=MrcReader(read_in_mem=True), image_only=True),
        ScaleIntensityRange(a_min=data_cfg["min"], a_max=data_cfg["max"],
                            b_min=0, b_max=1, clip=True),
        NormalizeIntensity(subtrahend=data_cfg["mean"], divisor=data_cfg["std"]),
        EnsureType(data_type="numpy"),
    ])
    return np.asarray(transforms(path), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--modelpth", required=True,
                        help="Path to the trained CryoSIAM checkpoint (.ckpt)")
    parser.add_argument("-v", "--volumes", required=True,
                        help="Path to a single tomogram (.mrc/.rec) - the kind "
                             "the model expects, usually the denoised one")
    parser.add_argument("-c", "--coords", required=True,
                        help="Whitespace-separated picks file, columns X Y Z, no header")
    parser.add_argument("-o", "--outpath", required=True,
                        help="Output directory; files are named after the picks file")
    parser.add_argument("-b", "--batchsize", type=int, default=10,
                        help="Batch size for the embedding (default: 10)")
    parser.add_argument("--centers-box", type=int, default=32,
                        help="Half-size in voxels of the raw crop around each "
                             "pick, before the center crop (default: 32, the "
                             "value the EMPIAR-10988 reference embeddings used)")
    parser.add_argument("--patch-size", type=int, default=32,
                        help="Network input size in voxels: the raw crop is "
                             "padded/center-cropped to this cube (default: 32, "
                             "as the EMPIAR-10988 reference embeddings; the "
                             "checkpoint's own training value may differ)")
    parser.add_argument("--no-contrastive", dest="contrastive",
                        action="store_false", default=True,
                        help="For models trained without the contrastive head: "
                             "read hyper_parameters['config'] and embed with "
                             "the plain encoder instead of forward_one")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device (default: cuda:0)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or "cuda" not in args.device else "cpu")
    if str(device) != args.device:
        print(f"CUDA not available, using {device}")

    checkpoint = torch.load(args.modelpth, map_location="cpu", weights_only=False)
    config = load_checkpoint_config(checkpoint, args.contrastive)
    net = load_prediction_model(checkpoint, config, device)
    dim = config["parameters"]["network"]["dim"]

    image = load_volume(args.volumes, config["parameters"]["data"])
    coords = read_coords(args.coords)
    # picks file is X Y Z; the volume's array indices are Z Y X
    centers_zyx = coords[["Z", "Y", "X"]].to_numpy()

    # a pick whose whole crop lies outside the volume would yield an empty
    # patch; nothing may ever be skipped, so refuse and name the rows
    lo = np.maximum(centers_zyx.astype(int) - args.centers_box, 0)
    hi = np.minimum(centers_zyx.astype(int) + args.centers_box, image.shape)
    bad = (hi <= lo).any(axis=1)
    if bad.any():
        sys.exit(f"{int(bad.sum())} pick(s) lie entirely outside the tomogram "
                 f"{image.shape} (picks file rows: {coords.index[bad].tolist()}).\n"
                 f"Every output column must correspond to the same row of the "
                 f"picks file, so nothing is ever skipped - fix or remove "
                 f"these coordinates.")

    patch = [args.patch_size] * 3
    patch_transforms = Compose([EnsureChannelFirst(channel_dim="no_channel"),
                                SpatialPad(patch),
                                CenterSpatialCrop(roi_size=patch),
                                EnsureType(data_type="tensor")])
    post_pred = Compose([EnsureType("numpy", device=torch.device("cpu"))])
    dataset = CentersPatchDataset(image, centers_zyx, args.centers_box,
                                  patch_transforms)
    loader = DataLoader(dataset, batch_size=args.batchsize, num_workers=2)

    os.makedirs(args.outpath, exist_ok=True)
    stem = os.path.join(
        args.outpath, os.path.splitext(os.path.basename(args.coords))[0])
    n = len(dataset)
    print(f"Embedding {n} picks from {args.volumes} "
          f"(crop half-size {args.centers_box}, network input {args.patch_size}, "
          f"dim {dim}, {'contrastive' if args.contrastive else 'encoder'})")

    n_nan = 0
    with h5py.File(stem + "_embeds.h5", "w") as f_embeds, \
            h5py.File(stem + "_instance_labels.h5", "w") as f_labels, \
            torch.no_grad():
        embeds = f_embeds.create_dataset("embeddings", shape=(dim, n),
                                         dtype=np.float32)
        labels = f_labels.create_dataset("labels", shape=(n,), dtype=np.int64)
        done = 0
        for img, idx in loader:
            if args.contrastive:
                _, out = net.forward_one(img.to(device))
            else:
                out = net.encoder(img.to(device))
            embed = post_pred(out)
            n_nan += int(np.isnan(embed).any(axis=1).sum())
            embeds[:, done:done + len(idx)] = embed.T
            labels[done:done + len(idx)] = idx.numpy() + 1  # 1-based, in pick order
            done += len(idx)
            print(f"\rEmbedded {done}/{n} picks", end="", flush=True)
    print()
    if n_nan:
        for suffix in ("_embeds.h5", "_instance_labels.h5"):
            os.remove(stem + suffix)
        sys.exit(f"{n_nan} embedding(s) contain NaN; outputs removed.")

    regions = pd.DataFrame({
        "label": np.arange(1, n + 1),
        "centroid-0": centers_zyx[:, 0],
        "centroid-1": centers_zyx[:, 1],
        "centroid-2": centers_zyx[:, 2],
        # the true crop volume; CryoSIAM's app writes box ^ 3 here, which is
        # a XOR, but the column is only ever plot-hover metadata
        "area": (2 * args.centers_box) ** 3,
    })
    regions.to_csv(stem + "_instance_regions.csv", index=False)
    try:
        import starfile
        starfile.write(
            regions.rename(columns={"centroid-0": "rlnCoordinateZ",
                                    "centroid-1": "rlnCoordinateY",
                                    "centroid-2": "rlnCoordinateX",
                                    "label": "rlnLabel", "area": "rlnArea"}),
            stem + "_instance_regions.star", overwrite=True)
    except ImportError:
        print("starfile is not installed; skipped the .star copy of the regions.")
    print(f"Wrote {n} embeddings to {stem}_embeds.h5 "
          f"(+ instance_labels.h5, instance_regions.csv/star)")


if __name__ == "__main__":
    main()
