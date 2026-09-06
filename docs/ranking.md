# Ranking by Embedding Distance

If one of your sets came from an external classifier or feature extractor
that produced a per-particle embedding vector, PickyMate can rank that
set's particles by Mahalanobis distance to a reference set — useful for
pushing likely junk (or likely gems) to one end of the gallery instead of
reviewing everything in pick order.

## Bringing embeddings in

Embeddings attach to one particle list at import time, either from the
command line (`--embedded_particle_list`, `--embeddings_h5`, and optionally
`--embeddings_umap` — see [Getting Started](getting-started.md)) or from
**File > Import Particle List with Embeddings...**. Only one particle list
can carry embeddings at a time. Once loaded, the ranking row appears at the
bottom of the Particle List window, above its toolbar:

![Ranking panel, annotated](images/ranking-panel.png)

1. **"close to"** — the reference set to measure distance *toward*. A
   particle scores low (ranks first after sorting) the more it resembles
   this set in embedding space.
2. **"far from"** — optional second reference set to measure distance
   *away from* at the same time (defaults to **(none)**, which is the
   plain single-distance behavior described below).
3. **Leave-One-Out** — only selectable when the current tab's own set is
   used as a reference (either "close to" or "far from"). Without it, a
   particle scored against its own set would trivially look close to
   itself; Leave-One-Out excludes each particle from the reference
   statistics while scoring it against its own set.
4. **Rank by Embedding Distance** — runs the computation and offers to sort
   the current tab by the resulting score immediately after.

The space dropdown next to "far from" picks which embedding to measure in:
the full-dimensionality vector from `--embeddings_h5`, or (if
`--embeddings_umap` was also supplied) the 2D UMAP projection instead.

## Single distance ("close to" only)

With "far from" left at **(none)**, each particle in the current tab is
scored by its Mahalanobis distance to the "close to" set: how many standard
deviations away it sits from that set's mean, accounting for how the
reference set's own points are correlated/spread out in embedding space —
not just a plain Euclidean distance. Lower means "more like the reference
set."

## Double distance ("close to" + "far from")

Setting both fields scores each particle by the difference of two
Mahalanobis distances:

```
score = distance_to("close to" set) − distance_to("far from" set)
```

This is useful when you have two positive/negative example sets already —
say, particles you've moved to **Confirmed** and particles you've sent to
the **Recycle Bin** — and want to rank the rest by how much they look like
the good examples *and* how little they look like the bad ones, in one
pass. A lower (more negative) score means "closer to the confirmed
examples, farther from the discarded ones."

The two fields can't be set to the same set — doing so disables only the
**Rank by Embedding Distance** button itself (with a tooltip explaining
why), never the dropdowns, so you can always change either one back to fix
it.

## After ranking

The result reports how many particles were actually scored (a particle
with no embedding row, or excluded by Leave-One-Out, is skipped) and offers
to sort the current tab by score. You can re-sort by score later at any
time with the gallery toolbar's **Sort by Rank** button — it's the same
ranking values computed here, not a separate concept.
