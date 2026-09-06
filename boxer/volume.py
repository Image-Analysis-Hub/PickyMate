"""Lightweight tomogram-volume wrapper: MRC I/O, centered sub-volume
extraction (with zero-padding), and thin-slab axis-aligned projections.

Internally the volume is stored as a numpy array shaped (nz, ny, nx), matching
the on-disk convention used by mrcfile. ``nx``/``ny``/``nz`` properties are
provided for readability wherever header-style axis access reads better than
indexing the array shape directly.
"""
from __future__ import annotations

import numpy as np
import mrcfile


class Volume:
    def __init__(self, data: np.ndarray, apix: float = 1.0):
        if data.ndim != 3:
            raise ValueError("Volume data must be a 3D array (nz, ny, nx)")
        self.data = data
        self.apix = float(apix) if apix else 1.0
        self._mmap_owner = None  # keeps the backing mrcfile alive for native_memmap loads

    @property
    def nx(self) -> int:
        return self.data.shape[2]

    @property
    def ny(self) -> int:
        return self.data.shape[1]

    @property
    def nz(self) -> int:
        return self.data.shape[0]

    @classmethod
    def from_file(cls, path: str, native_memmap: bool = False) -> "Volume":
        """Load a tomogram. By default this picks whichever in-memory dtype
        wastes the least RAM without throwing away information the file
        didn't already have: an integer volume (int8/int16/... -- the common
        case for cryo-ET tomograms) is kept at its native on-disk dtype
        instead of upcast to float32, and a floating-point volume (float32/
        float64) is downcast to float16. Display/contrast code already works
        in plain Python floats applied via numpy broadcasting, so it's
        dtype-agnostic and needs no changes for either case.

        native_memmap=True instead memory-maps the file at its exact
        on-disk dtype via mrcfile.mmap(), for anyone who wants the original
        full-precision data and is willing to trade a bit of access latency
        (the OS pages in slices/crops from disk as they're actually touched)
        for a RAM footprint far below even a single fully-materialized copy.
        """
        if native_memmap:
            mrc = mrcfile.mmap(path, mode="r", permissive=True)
            data = mrc.data
            apix = float(mrc.voxel_size.x) if mrc.voxel_size.x else 1.0
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            vol = cls(data, apix)
            vol._mmap_owner = mrc
            return vol

        with mrcfile.open(path, permissive=True) as mrc:
            raw = mrc.data
            dtype = np.float16 if np.issubdtype(raw.dtype, np.floating) else raw.dtype
            data = np.array(raw, dtype=dtype)
            apix = float(mrc.voxel_size.x) if mrc.voxel_size.x else 1.0
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        return cls(data, apix)

    def get_clip_centered(self, cx: float, cy: float, cz: float,
                           dx: int, dy: int, dz: int) -> np.ndarray:
        """Return a (dz, dy, dx) sub-volume centered at image coordinate
        (cx, cy, cz), zero-padded where it extends past the volume edges."""
        x0 = int(round(cx - dx // 2))
        y0 = int(round(cy - dy // 2))
        z0 = int(round(cz - dz // 2))
        out = np.zeros((dz, dy, dx), dtype=self.data.dtype)

        sx0, sx1 = max(x0, 0), min(x0 + dx, self.nx)
        sy0, sy1 = max(y0, 0), min(y0 + dy, self.ny)
        sz0, sz1 = max(z0, 0), min(z0 + dz, self.nz)
        if sx0 >= sx1 or sy0 >= sy1 or sz0 >= sz1:
            return out

        ox, oy, oz = sx0 - x0, sy0 - y0, sz0 - z0
        out[oz:oz + (sz1 - sz0), oy:oy + (sy1 - sy0), ox:ox + (sx1 - sx0)] = \
            self.data[sz0:sz1, sy0:sy1, sx0:sx1]
        return out

    def slab_projection(self, axis: str, center: float, half_thickness: int) -> np.ndarray:
        """Mean projection over a thin slab centered on ``center`` along
        ``axis`` ('x', 'y' or 'z'). Returned as a 2D array using
        (row, col) = (vertical, horizontal) display convention:

          axis='z' -> XY image,  row=y, col=x
          axis='y' -> XZ image,  row=z, col=x
          axis='x' -> ZY image,  row=y, col=z

        dtype=np.float32 on every mean() below is deliberate, not
        cosmetic: unlike a full unaxed reduction, numpy's axis-reduced
        mean() does *not* auto-promote a float16 input's accumulator, so a
        float16-loaded volume (see from_file) would silently produce a
        float16 slice here -- and a float16 image, once averaged over
        anywhere near 1000 slices/pixels, overflows to inf the moment any
        downstream .std() (autocontrast) touches it. Forcing float32 here
        keeps every consumer downstream exactly as dtype-safe as it was
        when volumes were always loaded as float32.
        """
        lo = int(round(center)) - int(half_thickness)
        hi = int(round(center)) + int(half_thickness) + 1

        if axis == "z":
            lo_c, hi_c = max(lo, 0), min(hi, self.nz)
            if hi_c <= lo_c:
                return np.zeros((self.ny, self.nx), dtype=np.float32)
            return self.data[lo_c:hi_c].mean(axis=0, dtype=np.float32)
        elif axis == "y":
            lo_c, hi_c = max(lo, 0), min(hi, self.ny)
            if hi_c <= lo_c:
                return np.zeros((self.nz, self.nx), dtype=np.float32)
            return self.data[:, lo_c:hi_c, :].mean(axis=1, dtype=np.float32)
        elif axis == "x":
            lo_c, hi_c = max(lo, 0), min(hi, self.nx)
            if hi_c <= lo_c:
                return np.zeros((self.ny, self.nz), dtype=np.float32)
            return self.data[:, :, lo_c:hi_c].mean(axis=2, dtype=np.float32).T
        else:
            raise ValueError(f"unknown axis {axis!r}")


