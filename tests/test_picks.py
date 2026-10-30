"""The Picks model on its own: no window, no Qt, no files."""
import time

from boxer.picks import Particle, Picks


def consistent(p):
    """Both indices agree with the list itself."""
    assert {i for idx in p._indices_by_class.values() for i in idx} == set(range(len(p.boxes)))
    for cid, idx in p._indices_by_class.items():
        assert all(p.boxes[i].classid == cid for i in idx), cid
    assert p._index_of_uid == {b.uid: i for i, b in enumerate(p.boxes)}


def filled(n, classid=0):
    p = Picks()
    assert p.new_set("a", 32) == classid
    for k in range(n):
        p.add(Particle(k, k, k, classid, p.new_uid()))
    return p


def test_add_reassign_remove_insert_keep_the_indices_right():
    p = filled(10)
    b = p.new_set("b", 32)
    consistent(p)
    assert p.indices_of(b) == ()

    assert p.reassign([1, 3, 5], b) == [(1, 0), (3, 0), (5, 0)]
    assert p.reassign([1], b) == [], "already there: nothing moved"
    assert p.boxes[1].score is None and list(p.indices_of(b)) == [1, 3, 5]
    consistent(p)

    removed, kept = p.remove([0, 5, 9])
    assert [i for i, _ in removed] == [0, 5, 9] and kept == [1, 2, 3, 4, 6, 7, 8]
    assert len(p.boxes) == 7 and p.index_for_uid(removed[1][1].uid) is None
    consistent(p)

    p.insert(0, removed[0][1])
    p.rebuild_indices()
    assert p.index_for_uid(removed[0][1].uid) == 0
    consistent(p)

    p.reassign(list(p.indices_of(b)), 0)
    assert p.delete_set(b) == "b" and b not in p.sets and b not in p.boxsize
    assert p.delete_set(b) is None
    consistent(p)


def test_set_class_resets_the_score():
    p = filled(2)
    p.boxes[0].score = 1.5
    assert p.set_class(0, 0) == 0, "same set still returns the previous one"
    assert p.boxes[0].score is None


def test_uids_never_repeat():
    p = Picks()
    p.reserve_uids(41)
    assert p.new_uid() == 42
    p.reserve_uids(10)
    assert p.new_uid() == 43


def test_new_set_takes_the_lowest_free_id():
    p = Picks()
    assert p.new_set("a", 32) == 0 and p.new_set("b", 32) == 1
    p.delete_set(0)
    assert p.new_set("c", 32) == 0


def test_remove_is_linear():
    # emptying a 20k Recycle Bin out of 40k picks took 7 s with list membership
    p = filled(40_000)
    t = time.perf_counter()
    p.remove(range(0, 40_000, 2))
    assert time.perf_counter() - t < 1.0
    assert len(p.boxes) == 20_000
    consistent(p)
