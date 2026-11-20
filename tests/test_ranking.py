"""The ranking maths on its own: no window, no Qt."""
import numpy as np
import pytest

from boxer import ranking


def test_distances_match_the_formula():
    rng = np.random.default_rng(0)
    ref, cand = rng.standard_normal((30, 8)), rng.standard_normal((5, 8))
    mu = ref.mean(0)
    cen = ref - mu
    ci = np.linalg.pinv(cen.T @ cen / 29, hermitian=True)
    d = cand - mu
    assert np.allclose(ranking.distances(ref, cand), ((d @ ci) * d).sum(1))


def test_scores_by_index_and_leave_one_out():
    rng = np.random.default_rng(1)
    array = rng.standard_normal((6, 4))
    rows = [(i, i) for i in range(6)]       # box index == embedding row
    plain, size = ranking.scores(rows[:3], rows[3:], array)
    assert size == 3
    assert plain == pytest.approx(dict(zip(range(3), ranking.distances(array[3:], array[:3]))))

    loo, size = ranking.scores(rows, rows, array, leave_one_out=True)
    assert size == 6
    for i in range(6):
        others = [j for j in range(6) if j != i]
        assert loo[i] == pytest.approx(ranking.distances(array[others], array[[i]])[0])

    two, _ = ranking.scores(rows[:2], rows[:2], array, leave_one_out=True)
    assert two == {}, "a reference that shrinks below 2 without the particle leaves it unscored"
    with pytest.raises(ValueError, match='"Confirmed" has only 1'):
        ranking.scores(rows, rows[:1], array, reference_name="Confirmed")


def test_extra_vectors_pool_with_the_reference():
    rng = np.random.default_rng(2)
    array = rng.standard_normal((5, 4))
    extra = rng.standard_normal((7, 4))
    rows = [(i, i) for i in range(5)]
    pooled, size = ranking.scores(rows, rows[:2], array, extra=extra)
    assert size == 9
    assert [pooled[i] for i in range(5)] == pytest.approx(
        list(ranking.distances(np.vstack([array[:2], extra]), array)))
    one, size = ranking.scores(rows, rows[:1], array, extra=extra)
    assert size == 8 and len(one) == 5, "other tomograms' vectors make a lone reference rankable"
