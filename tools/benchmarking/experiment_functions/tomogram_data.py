"""Reading one tomogram's candidates, embeddings, ground truth and mask.

A benchmark tomogram is a candidate list produced by template matching, one
embedding file per model, the deposited particle coordinates, and the mask that
marks the region of interest. The files are those PickyMate itself reads, through
the same readers, so a benchmark cannot drift away from what the application
loads.

Two properties of this layout the experiments rely on, and which this module
checks rather than assumes:

* the candidate list is in descending template-matching score order, so the
  order it arrives in is a meaningful baseline to beat;
* every embedding file is in the row order of the candidate file, so a candidate
  is addressed by its line number in every space.
"""
from __future__ import annotations

import os

import numpy as np

from boxer import embeddings_io, metadata

#: Where each particle type keeps its files, relative to the dataset root.
LAYOUT = {
    'ribosome': {
        'candidates': 'TemplateMatching_Ribosomes_PicksOnly/{t}.txt',
        'particles': 'Ribosomes_GroundTruth/{t}.txt',
        'cryosiam': 'CryoSiam/Ribosomes_embeds/{t}_embeds.h5',
        'cryosiam_umap': 'CryoSiam/Ribosomes_embeds/vis/{t}_embeds_umap_data.csv',
        'tomotwin': 'TomoTwin/Ribosomes_embeddings/{t}.temb',
        'tomotwin_umap': 'TomoTwin/Ribosomes_embeddings/{t}.tumap',
    },
    'fas': {
        'candidates': 'TemplateMatching_FAS_PicksOnly/{t}.txt',
        'particles': 'FAS_GroundTruth/{t}.txt',
        'cryosiam': 'CryoSiam/FAS_embeds/{t}_embeds.h5',
        'cryosiam_umap': 'CryoSiam/FAS_embeds/vis/{t}_embeds_umap_data.csv',
        'tomotwin': 'TomoTwin/FAS_embeddings/{t}.temb',
        'tomotwin_umap': 'TomoTwin/FAS_embeddings/{t}.tumap',
    },
}
MASK = 'labels/{t}_cytosol.mrc'
TOMOGRAMS = ['TS_026', 'TS_027', 'TS_028', 'TS_029', 'TS_030',
             'TS_034', 'TS_037', 'TS_041', 'TS_043', 'TS_045']
SPACES = ('tomotwin', 'tomotwin_umap', 'cryosiam', 'cryosiam_umap')


def paths_for(root, particle, tomogram):
    """Every file of one tomogram, as absolute paths."""
    layout = LAYOUT[particle]
    out = {key: os.path.join(root, template.format(t=tomogram))
           for key, template in layout.items()}
    out['mask'] = os.path.join(root, MASK.format(t=tomogram))
    return out


def match_candidates_to_particles(candidates, particles, radius):
    """Which deposited particle each candidate sits on, or -1 for none.

    A candidate counts as a true pick when a deposited particle lies within
    `radius`; the nearest particle wins. The assignment is returned rather than
    a boolean, so that recall can be counted against particles rather than
    against candidates when several candidates share one particle.
    """
    owner = np.full(len(candidates), -1, dtype=np.int64)
    if len(particles) == 0:
        return owner
    cell = max(radius, 1e-6)
    grid = {}
    for index, point in enumerate(particles):
        key = tuple((point // cell).astype(np.int64))
        grid.setdefault(key, []).append(index)
    for i, point in enumerate(candidates):
        key = (point // cell).astype(np.int64)
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near.extend(grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
        if near:
            near = np.asarray(near)
            distance = np.linalg.norm(particles[near] - point, axis=1)
            if distance.min() <= radius:
                owner[i] = near[int(np.argmin(distance))]
    return owner


def _inside_mask(path, points):
    """Whether each point falls inside the mask, sampled point by point.

    A mask of a full tomogram is large, so it is memory mapped and read only at
    the coordinates asked for.
    """
    import mrcfile
    with mrcfile.mmap(path, mode='r', permissive=True) as handle:
        shape = handle.data.shape
        z = np.clip(np.rint(points[:, 2]).astype(int), 0, shape[0] - 1)
        y = np.clip(np.rint(points[:, 1]).astype(int), 0, shape[1] - 1)
        x = np.clip(np.rint(points[:, 0]).astype(int), 0, shape[2] - 1)
        return np.asarray(handle.data[z, y, x]) > 0.5


class Tomogram:
    """One tomogram: its candidates, their labels, the mask and the spaces.

    Attributes
    ----------
    candidates : (n, 3) array of coordinates, in template-matching score order
    is_true_pick : (n,) boolean, a candidate that matches a deposited particle
    owner : (n,) index of the matched particle, or -1
    inside_region : (n,) boolean, a candidate inside the mask
    spaces : name -> (label, (n, d) array)
    """

    def __init__(self, root, particle, name, radius=10.0, spaces=SPACES):
        self.name, self.particle, self.radius = name, particle, radius
        self.paths = paths_for(root, particle, name)
        self.candidates = np.array(metadata.load_particle_list(self.paths['candidates']),
                                   dtype=np.float64)
        self.particles = np.array(metadata.load_particle_list(self.paths['particles']),
                                  dtype=np.float64)
        self.n = len(self.candidates)
        self.owner = match_candidates_to_particles(self.candidates, self.particles, radius)
        self.is_true_pick = self.owner >= 0
        self.inside_region = _inside_mask(self.paths['mask'], self.candidates)
        self.particles_inside_region = _inside_mask(self.paths['mask'], self.particles)
        self.spaces = self._load_spaces(spaces)

    def _load_spaces(self, wanted):
        out = {}
        for key in wanted:
            path = self.paths.get(key)
            if not path or not os.path.isfile(path):
                continue
            if key == 'cryosiam':
                array = embeddings_io.load_cryosiam_embeddings(path, self.n)['embeddings']
            elif key == 'cryosiam_umap':
                array = embeddings_io.load_cryosiam_umap(path)['umap_xy']
            elif key == 'tomotwin':
                array = embeddings_io.load_tomotwin_embeddings(path)['embeddings']
            else:
                array = embeddings_io.load_tomotwin_umap(path)['umap_xy']
            array = np.asarray(array, dtype=np.float64)
            if len(array) != self.n:
                raise SystemExit(f'{path}: {len(array)} rows for {self.n} candidates')
            method = key.split('_')[0]
            kind = 'umap' if key.endswith('umap') else 'embeddings'
            out[key] = (embeddings_io.space_label(method, kind, array.shape[1]), array)
        return out

    def kept(self, use_region):
        """Which candidates enter the analysis.

        With the region in use, the candidates outside the mask are removed
        before anything is scored, which is the bulk rejection a curator would
        otherwise perform by hand.
        """
        return self.inside_region if use_region else np.ones(self.n, dtype=bool)

    def particle_count(self, use_region):
        """The denominator of recall.

        With the region in use, particles outside it are not counted: no
        surviving candidate can reach them, so counting them would penalise the
        mask for doing its work.
        """
        return (int(self.particles_inside_region.sum()) if use_region
                else len(self.particles))

    def describe(self, use_region=False):
        kept = self.kept(use_region)
        return (f'{self.name:>7} {self.particle:>9} candidates {self.n:>6} '
                f'kept {int(kept.sum()):>6} particles {self.particle_count(use_region):>5} '
                f'true picks {int(self.is_true_pick[kept].sum()):>5}')


def load_tomograms(root, particle, names=None, radius=10.0, spaces=SPACES):
    return [Tomogram(root, particle, name, radius, spaces) for name in (names or TOMOGRAMS)]
