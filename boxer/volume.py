"""Lightweight, dependency-free replacement for EMAN2's EMData for tomogram
volumes, providing only the subset of functionality that the tomogram boxer
needs: MRC I/O, centered sub-volume extraction (with zero-padding), thin-slab
axis-aligned projections, and normalization.

Internally the volume is stored as a numpy array shaped (nz, ny, nx), matching
the on-disk convention used by mrcfile. ``nx``/``ny``/``nz`` properties are
provided for readability when porting code that used EMAN2's EMData["nx"]
style header access.
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
    def from_file(cls, path: str) -> "Volume":
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data, dtype=np.float32)
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
        """
        lo = int(round(center)) - int(half_thickness)
        hi = int(round(center)) + int(half_thickness) + 1

        if axis == "z":
            lo_c, hi_c = max(lo, 0), min(hi, self.nz)
            if hi_c <= lo_c:
                return np.zeros((self.ny, self.nx), dtype=np.float32)
            return self.data[lo_c:hi_c].mean(axis=0)
        elif axis == "y":
            lo_c, hi_c = max(lo, 0), min(hi, self.ny)
            if hi_c <= lo_c:
                return np.zeros((self.nz, self.nx), dtype=np.float32)
            return self.data[:, lo_c:hi_c, :].mean(axis=1)
        elif axis == "x":
            lo_c, hi_c = max(lo, 0), min(hi, self.nx)
            if hi_c <= lo_c:
                return np.zeros((self.ny, self.nz), dtype=np.float32)
            return self.data[:, :, lo_c:hi_c].mean(axis=2).T
        else:
            raise ValueError(f"unknown axis {axis!r}")


def normalize(img: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization (EMData "normalize" processor)."""
    std = float(img.std())
    if std == 0:
        return img - img.mean()
    return (img - img.mean()) / std


class ParticleStackWriter:
    """Accumulates 2D or 3D particle images and writes them to a single MRC
    stack (2D particles -> .mrcs with particle index along the first axis)
    or a directory of individual .mrc files (3D particles)."""

    def __init__(self, path: str, is3d: bool, apix: float = 1.0):
        self.path = path
        self.is3d = is3d
        self.apix = apix
        self._frames = []

    def add(self, img: np.ndarray, extra_header=None):
        self._frames.append(np.asarray(img, dtype=np.float32))

    def close(self):
        import os
        if not self._frames:
            return
        if self.is3d:
            os.makedirs(self.path, exist_ok=True)
            for i, frame in enumerate(self._frames):
                fname = os.path.join(self.path, f"ptcl_{i:05d}.mrc")
                with mrcfile.new(fname, overwrite=True) as mrc:
                    mrc.set_data(frame)
                    mrc.voxel_size = self.apix
        else:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            stack = np.stack(self._frames, axis=0)
            with mrcfile.new(self.path, overwrite=True) as mrc:
                mrc.set_data(stack)
                mrc.voxel_size = self.apix
