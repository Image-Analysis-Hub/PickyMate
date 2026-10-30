"""One picked particle, as the window, the gallery, the map and the tests
hold it. The class ids and names it refers to live in categories.py; how it
is written to disk in metadata.py."""
from __future__ import annotations


class Particle:
    """A pick: its position, the set (class id) it is in, a uid that stays
    the same for the life of the working directory, and -- for a pick that
    came from an embedded particle list -- the row of its embedding in every
    attached space (None for a hand pick, which can't take part in ranking)
    and the score of the last ranking it took part in (None until ranked,
    and reset to None when it changes set; see TomoBoxer._reassign_boxes)."""
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
