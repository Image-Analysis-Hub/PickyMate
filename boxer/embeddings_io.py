"""Per-method readers for particle embeddings and their UMAP reductions.

Every embedding method ships its own file format, and they differ in ways
that can't be sniffed reliably -- so each gets an explicit loader here and
the caller says which one it's using, rather than this module guessing.
What the two currently supported methods actually look like, as measured
against real files:

CryoSIAM
    <name>_embeds.h5            one HDF5 dataset, shape (feature, particle)
                                -- i.e. transposed relative to everything
                                else here -- float32, 1024 features. Carries
                                no coordinates, so a separate particle list
                                is required to know which particle each
                                column belongs to.
    <name>_umap_data.csv        header'd CSV; "centroid-0/1/2" are the
                                particle's z,y,x and "x","y" are the 2-D
                                UMAP coordinates.

TomoTwin
    <name>.temb                 a pickled pandas DataFrame, shape
                                (particle, 4 + D): "coord_index", "Z", "Y",
                                "X", then one column per embedding
                                dimension named "0".."D-1". float16, and
                                L2-normalised (the model is trained with a
                                cosine metric, so every row is a unit
                                vector). Carries its own coordinates, so a
                                separate particle list is optional.
    <name>.tumap                a pickled pandas DataFrame with "X","Y","Z"
                                and "umap_0","umap_1".

Note the coordinate columns are ordered Z,Y,X in .temb but X,Y,Z in .tumap
-- which is exactly why nothing here reads columns positionally. Column
names are matched case-insensitively and by name only, so a future version
of either tool can add, drop or reorder columns without silently shifting
which numbers get read as an embedding.

Every loader returns a plain dict rather than a bare array, because the
useful thing to hand back varies: whether coordinates came with the file,
and (for TomoTwin) the tomogram shape recorded in the DataFrame's own
metadata, which is worth cross-checking against the volume actually open.
"""
from __future__ import annotations

import numpy as np

METHODS = ("cryosiam", "tomotwin")

# How each method is spelled in the UI. Every method contributes up to two
# rankable "spaces" -- its full embedding and, when a reduction file was
# given too, its UMAP -- and each is named and labelled independently so
# several methods can be attached at once and compared against each other
# from the same dropdown.
DISPLAY_NAMES = {"cryosiam": "CryoSIAM", "tomotwin": "TomoTwin"}


def space_key(method: str, kind: str) -> str:
    """Stable identifier for one space, used as the dropdown's data and as
    the name persisted in embeddings.settings."""
    return method if kind == "embeddings" else f"{method}_umap"


# The order spaces are offered in, wherever a choice of space is shown.
# TomoTwin's full embedding comes first because it was the most reliable
# ranking space in every benchmark on EMPIAR-10988 -- it never fell below
# the template-matching order in any tomogram, and it transfers between
# tomograms -- so it is also the default when nothing else is chosen.
SPACE_DISPLAY_ORDER = ("tomotwin", "tomotwin_umap", "cryosiam", "cryosiam_umap")


def display_order(names):
    """`names` (space keys) sorted into SPACE_DISPLAY_ORDER; anything not
    listed there keeps its original relative order, after the known ones."""
    names = list(names)
    rank = {n: i for i, n in enumerate(SPACE_DISPLAY_ORDER)}
    return sorted(names, key=lambda n: (rank.get(n, len(rank)), names.index(n)))


def space_label(method: str, kind: str, dims: int) -> str:
    """What the ranking dropdown shows, e.g. "CryoSIAM - full (1024-d)"."""
    pretty = DISPLAY_NAMES.get(method, method)
    what = "full" if kind == "embeddings" else "UMAP"
    return f"{pretty} - {what} ({dims}-d)"

# Columns in a TomoTwin .temb/.tumap that are never part of the embedding.
# Matched case-insensitively; anything else numeric is a dimension.
_TOMOTWIN_NON_EMBEDDING = {"coord_index", "index", "x", "y", "z", "filepath",
                            "umap_0", "umap_1"}


def _read_pickled_frame(path):
    """TomoTwin writes pandas pickles. pandas is imported here rather than
    at module import so the app still starts without it installed for
    anyone who never touches TomoTwin files."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ValueError(
            f"Reading TomoTwin files needs pandas, which isn't installed "
            f"({e}). Install it with: pip install pandas") from e
    try:
        frame = pd.read_pickle(path)
    except Exception as e:
        raise ValueError(f"Could not read {path!r} as a pandas pickle: {e}") from e
    if not hasattr(frame, "columns"):
        raise ValueError(
            f"{path!r} unpickled to {type(frame).__name__}, not a DataFrame -- "
            f"this doesn't look like a TomoTwin embeddings file.")
    return frame


def _coordinate_columns(frame, path):
    """The frame's X/Y/Z columns, found by name (case-insensitively) and
    returned in x,y,z order regardless of the order they appear in --
    .temb stores them Z,Y,X and .tumap stores them X,Y,Z."""
    by_lower = {str(c).lower(): c for c in frame.columns}
    missing = [axis for axis in ("x", "y", "z") if axis not in by_lower]
    if missing:
        raise ValueError(
            f"{path!r} has no {'/'.join(m.upper() for m in missing)} column "
            f"-- found {list(frame.columns)[:8]}...")
    return [by_lower["x"], by_lower["y"], by_lower["z"]]


def _tomotwin_embedding_columns(frame, path):
    """Which columns hold the embedding itself.

    TomoTwin names them "0".."D-1", so those are preferred and sorted
    numerically -- sorting them as strings would order 0, 1, 10, 11, 2 and
    silently permute the dimensions. (Mahalanobis distance happens to be
    invariant to a consistent permutation, but a wrong order would still
    make these embeddings disagree with the same vectors read anywhere
    else, so it's not worth relying on.) If a future version stops using
    digit names, fall back to every numeric column that isn't known
    metadata."""
    digit_cols = [c for c in frame.columns if str(c).isdigit()]
    if digit_cols:
        return sorted(digit_cols, key=lambda c: int(str(c)))
    numeric = [
        c for c in frame.columns
        if str(c).lower() not in _TOMOTWIN_NON_EMBEDDING
        and np.issubdtype(frame[c].dtype, np.number)
    ]
    if not numeric:
        raise ValueError(
            f"{path!r} has no columns that look like embedding dimensions "
            f"-- found {list(frame.columns)[:12]}...")
    return numeric


def load_tomotwin_embeddings(path: str) -> dict:
    """{"embeddings": (N, D) float32, "coords_xyz": (N, 3) float64,
    "volume_shape_zyx": (nz, ny, nx) or None, "columns": [...],
    "description": str}.

    Coordinates come from the file itself, so importing TomoTwin
    embeddings doesn't require a separate particle list (though one can
    still be given, and is then cross-checked against these)."""
    frame = _read_pickled_frame(path)
    emb_cols = _tomotwin_embedding_columns(frame, path)
    coord_cols = _coordinate_columns(frame, path)

    # float16 on disk (that's how TomoTwin writes it); widened to float32
    # here so every downstream reduction has a sane accumulator -- see
    # Volume.slab_projection for the same issue on volume data.
    embeddings = np.ascontiguousarray(frame[emb_cols].to_numpy(dtype=np.float32))
    coords = frame[coord_cols].to_numpy(dtype=np.float64)

    attrs = dict(getattr(frame, "attrs", {}) or {})
    shape = attrs.get("tomogram_input_shape")
    volume_shape_zyx = tuple(int(v) for v in shape) if shape and len(shape) == 3 else None

    return {
        "embeddings": embeddings,
        "coords_xyz": coords,
        "volume_shape_zyx": volume_shape_zyx,
        "columns": [str(c) for c in emb_cols],
        "description": (
            f"TomoTwin: {len(embeddings)} particles x {embeddings.shape[1]} dimensions "
            f"(columns {emb_cols[0]}..{emb_cols[-1]}), coordinates read from "
            f"{'/'.join(str(c) for c in coord_cols)}"),
    }


def load_tomotwin_umap(path: str) -> dict:
    """{"umap_xy": (N, 2) float64, "coords_xyz": (N, 3) float64}. The 2-D
    columns are "umap_0"/"umap_1"; coordinates ride along and are used to
    confirm this file describes the same particles as the .temb."""
    frame = _read_pickled_frame(path)
    by_lower = {str(c).lower(): c for c in frame.columns}
    umap_cols = [by_lower[name] for name in ("umap_0", "umap_1") if name in by_lower]
    if len(umap_cols) != 2:
        # Not the documented names -- fall back to any two numeric columns
        # that aren't coordinates, rather than failing outright.
        umap_cols = [
            c for c in frame.columns
            if str(c).lower() not in ("x", "y", "z", "coord_index", "index")
            and np.issubdtype(frame[c].dtype, np.number)
        ][:2]
    if len(umap_cols) != 2:
        raise ValueError(
            f"{path!r} has no umap_0/umap_1 columns -- found {list(frame.columns)}")
    coord_cols = _coordinate_columns(frame, path)
    return {
        "umap_xy": frame[umap_cols].to_numpy(dtype=np.float64),
        "coords_xyz": frame[coord_cols].to_numpy(dtype=np.float64),
        "description": f"TomoTwin UMAP: {len(frame)} particles, columns "
                        f"{umap_cols[0]}/{umap_cols[1]}",
    }


def load_cryosiam_embeddings(path: str, n_expected: int) -> dict:
    """{"embeddings": (N, D) float32, "coords_xyz": None, ...}.

    The HDF5 holds a single dataset written (feature, particle), so it
    needs transposing -- but which axis is which can only be decided
    against the particle count from the coordinate file, since a square
    array would otherwise be ambiguous. That's why n_expected is required
    here and not for TomoTwin."""
    from . import metadata

    array = metadata.load_embeddings_h5(path)
    if array.shape[0] == n_expected:
        oriented = array
    elif array.shape[1] == n_expected:
        oriented = array.T
    else:
        raise ValueError(
            f"Embeddings in {path!r} have shape {array.shape}, which doesn't "
            f"match the {n_expected} particle(s) in the coordinate file on "
            f"either axis.")
    oriented = np.ascontiguousarray(oriented, dtype=np.float32)
    return {
        "embeddings": oriented,
        "coords_xyz": None,
        "volume_shape_zyx": None,
        "columns": None,
        "description": (f"CryoSIAM: {oriented.shape[0]} particles x "
                        f"{oriented.shape[1]} dimensions"),
    }


def load_cryosiam_umap(path: str) -> dict:
    """{"umap_xy": (N, 2), "coords_xyz": (N, 3)} -- the existing
    centroid-0/1/2 (z,y,x) plus x,y CSV, read via metadata."""
    from . import metadata

    data = metadata.load_embeddings_umap(path)
    return {
        "umap_xy": data["xy"],
        "coords_xyz": data["centroid_xyz"],
        "description": f"CryoSIAM UMAP: {len(data['xy'])} particles",
    }


def load_embeddings(method: str, path: str, n_expected: int = None) -> dict:
    if method == "tomotwin":
        return load_tomotwin_embeddings(path)
    if method == "cryosiam":
        if n_expected is None:
            raise ValueError(
                "CryoSIAM embeddings carry no coordinates, so a particle list "
                "is required to import them.")
        return load_cryosiam_embeddings(path, n_expected)
    raise ValueError(f"unknown embedding method {method!r} (expected one of {METHODS})")


def load_umap(method: str, path: str) -> dict:
    if method == "tomotwin":
        return load_tomotwin_umap(path)
    if method == "cryosiam":
        return load_cryosiam_umap(path)
    raise ValueError(f"unknown embedding method {method!r} (expected one of {METHODS})")
