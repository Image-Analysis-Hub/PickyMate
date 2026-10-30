"""The particles of one tomogram: each pick as a Particle, and the Picks
collection that holds them together with their sets and the bookkeeping
that keeps them findable. The class ids and names live in categories.py;
how a particle is written to disk in metadata.py."""
from __future__ import annotations


class Particle:
    """A pick: its position, the set (class id) it is in, a uid that stays
    the same for the life of the working directory, and -- for a pick that
    came from an embedded particle list -- the row of its embedding in every
    attached space (None for a hand pick, which can't take part in ranking)
    and the score of the last ranking it took part in (None until ranked,
    and reset to None when it changes set; see Picks.set_class)."""
    __slots__ = ("x", "y", "z", "classid", "uid", "embedding_row", "score")

    def __init__(self, x, y, z, classid, uid, embedding_row=None, score=None):
        self.x = x
        self.y = y
        self.z = z
        self.classid = classid
        self.uid = uid
        self.embedding_row = embedding_row
        self.score = score

    def copy(self):
        return Particle(self.x, self.y, self.z, self.classid, self.uid, self.embedding_row, self.score)

    def record(self):
        """The dict metadata.save_class_records writes."""
        return {"x": self.x, "y": self.y, "z": self.z, "uid": self.uid,
                "embedding_row": self.embedding_row, "score": self.score}


class Picks:
    """Every particle of one tomogram, the sets they are in, and the two
    indices that keep them findable: which list positions belong to each
    class, and which position holds each uid. Adding, removing and moving
    a particle goes through here so those can't drift from the list;
    drawing, saving and undo stay with the window."""

    def __init__(self):
        self.boxes = []        # Particle, in the order they are drawn and saved
        self.sets = {}         # class id -> name, which is also its "<name>.txt" file
        self.boxsize = {}      # class id -> box size in voxels
        self._next_uid = 0
        # class id -> {index: None}, an ordered set of the positions in
        # self.boxes that belong to that class -- lets a click hit-test skip
        # every particle of a hidden class outright, rather than visiting
        # each one just to reject it (a hidden-by-default Recycle Bin alone
        # can hold tens of thousands). Kept in O(1) per particle by add and
        # set_class, and rebuilt in one O(N) pass by remove and after
        # insert, which shift every later position anyway.
        self._indices_by_class = {}
        self._index_of_uid = {}

    # -- uids ---------------------------------------------------------------
    def new_uid(self):
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def reserve_uids(self, max_uid):
        """Never hand out a uid up to max_uid -- called with the highest uid
        found on disk before the files are loaded, so a legacy particle
        without one can't be given a uid some other file already uses."""
        self._next_uid = max(self._next_uid, max_uid + 1)

    # -- lookup ---------------------------------------------------------------
    def index_for_uid(self, uid):
        return self._index_of_uid.get(uid)

    def indices_of(self, classid):
        return self._indices_by_class.get(classid, ())

    def embedded_rows(self, classid):
        """[(index, embedding row)] of the particles of a set that carry an
        embedding -- the ones a ranking can score, or use as a reference."""
        return [(i, b.embedding_row) for i, b in enumerate(self.boxes)
                if b.classid == classid and b.embedding_row is not None]

    def rebuild_indices(self):
        by_class, of_uid = {}, {}
        for i, b in enumerate(self.boxes):
            by_class.setdefault(b.classid, {})[i] = None
            of_uid[b.uid] = i
        self._indices_by_class = by_class
        self._index_of_uid = of_uid

    def _index_class_add(self, i, classid):
        self._indices_by_class.setdefault(classid, {})[i] = None

    def _index_class_remove(self, i, classid):
        bucket = self._indices_by_class.get(classid)
        if bucket is not None:
            bucket.pop(i, None)

    # -- particles --------------------------------------------------------------
    def add(self, particle):
        """Append; returns its index."""
        i = len(self.boxes)
        self.boxes.append(particle)
        self._index_class_add(i, particle.classid)
        self._index_of_uid[particle.uid] = i
        return i

    def insert(self, index, particle):
        """Put a particle back at a position (undo of a delete). Both
        indices are stale until rebuild_indices(), so a run of inserts can
        rebuild once."""
        self.boxes.insert(index, particle)

    def remove(self, indices):
        """Drop the particles at these positions. Returns (removed, kept):
        removed as [(old index, particle)] ascending, kept as the old
        indices of the survivors in their new order, for whoever holds a
        list parallel to boxes (thumbnails, the views' shapes)."""
        gone = set(indices)
        kept = [i for i in range(len(self.boxes)) if i not in gone]
        removed = [(i, self.boxes[i]) for i in sorted(gone)]
        self.boxes = [self.boxes[i] for i in kept]
        self.rebuild_indices()
        return removed, kept

    def set_class(self, index, classid):
        """Move one particle to another set; returns the set it was in. A
        score is only meaningful relative to the set it was computed
        against, so it resets rather than silently carrying over a number
        that no longer means what it looks like it means -- the particle
        shows up first in "Sort by Rank" until recomputed (see
        ParticleGrid.apply_score_order)."""
        b = self.boxes[index]
        prev = b.classid
        b.classid = classid
        self._index_class_remove(index, prev)
        self._index_class_add(index, classid)
        b.score = None
        return prev

    def reassign(self, indices, classid):
        """set_class for every particle not already in that set; returns
        [(index, previous class id)] for those actually moved."""
        return [(i, self.set_class(i, classid)) for i in indices if self.boxes[i].classid != classid]

    def clear(self):
        """Let go of every particle (the sets stay)."""
        self.boxes = []
        self._indices_by_class = {}
        self._index_of_uid = {}
        self._next_uid = 0

    # -- sets -------------------------------------------------------------------
    def new_set(self, name, boxsize):
        """A new set at the lowest free class id; returns the id."""
        i = 0
        while i in self.sets:
            i += 1
        self.sets[i] = name
        self.boxsize[i] = boxsize
        return i

    def delete_set(self, classid):
        """Forget a set that holds no particles any more; returns its name,
        or None if there was no such set."""
        self.boxsize.pop(classid, None)
        self._indices_by_class.pop(classid, None)
        return self.sets.pop(classid, None)
