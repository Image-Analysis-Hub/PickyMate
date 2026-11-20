# PickyMate benchmark

Code accompanying Supplementary Note 5, *Benchmark of the embedding-guided ranking in
PickyMate*. The three experiments reported there are produced by the three scripts in this
directory. They call the ranking functions of the interactive application on the files the
application reads, so that a benchmark result corresponds to the behaviour of the released
program rather than to a reimplementation of it.

## Contents

| Path | Purpose |
|:---|:---|
| `experiment_1_confirmed_picks.py` | Average precision of a single re-ranking against the number of confirmed picks (Supplementary Table 8, Supplementary Figures 19 and 20) |
| `experiment_2_reranking_schedule.py` | Decisions spent under each re-ranking schedule (Supplementary Table 9, Supplementary Figures 21 and 22) |
| `experiment_3_reference_reuse.py` | Average precision and decisions when a tomogram is ranked from the picks of tomograms already curated (Supplementary Tables 10 and 11, Supplementary Figures 23 and 24) |
| `experiment_functions/` | The functions the three experiments share, one module per concern |
| `reproduce_published_results.py` | Runs the experiments and compares every value with the published results |

## Requirements

Python 3.10 or later, NumPy, h5py and mrcfile, and an importable PickyMate installation,
from which the benchmark uses the particle-list and embedding readers
(`boxer.metadata`, `boxer.embeddings_io`). No other dependency is required; the
experiments produce comma-separated tables and leave plotting to the user.

```bash
pip install -e ../..        # PickyMate itself, from the clone this folder sits in
```

## Data

The experiments were run on the defocus dataset of EMPIAR-10988 (de Teresa-Trueba et al.,
2023), with candidate lists produced by template matching as described in Supplementary
Note 1, embeddings from TomoTwin and CryoSiam as described in Supplementary Notes 2 and 3,
and the cytosol masks deposited with the dataset. The file layout expected under the
dataset root is declared in `experiment_functions/tomogram_data.py` and can be edited there
for a different dataset. Experiment 3 additionally requires the curated working directories
produced by PickyMate, one per tomogram, from which the confirmed and rejected picks are
read.

## Reproducing the published results

The following three commands produce the tables of Supplementary Note 5 for the ribosome
target. `DATA` is the dataset root and `CURATED` the folder of curated working
directories.

```bash
python3 experiment_1_confirmed_picks.py   --data DATA --out results/experiment_1
python3 experiment_2_reranking_schedule.py --data DATA --out results/experiment_2 \
                                           --clear-region-first
python3 experiment_3_reference_reuse.py   --data DATA --curated CURATED \
                                           --out results/experiment_3 --orders 5
```

Supplementary Table 8 and Supplementary Figure 20 compare the full embeddings of each model
with the two-dimensional UMAP supplied with it. Both are produced by the first script with
the four spaces requested:

```bash
python3 experiment_1_confirmed_picks.py --data DATA --out results/experiment_1 \
        --spaces tomotwin tomotwin_umap cryosiam cryosiam_umap
```

`--tomograms`, `--particle`, `--confirmed-picks`, `--intervals`, `--recall-levels` and
`--radius` select subsets and parameters; `--help` on each script lists them with their
defaults.

Expected runtime on one core, for ten tomograms: about seven minutes for Experiment 1 in
the TomoTwin embeddings and forty for the two embedding models together; about two hours
for Experiment 2 with both models; and about ninety minutes for Experiment 3 with five
accumulation orders in both models. The UMAP spaces add a few minutes, being
two-dimensional.

## Output

Each script writes one table and one `run_report.json` recording the arguments, the checks
that passed, the runtime, and the Python and NumPy versions. The principal columns are:

| Column | Meaning |
|:---|:---|
| `tomogram` / `target` | the tomogram scored |
| `condition` / `region` | which of the three uses of the cytosol mask applies |
| `space` | the embedding space in which the ranking was computed |
| `distance` | `single`, the distance to the confirmed picks; `relative`, that distance minus the distance to the rejected picks |
| `confirmed_picks`, `examined_candidates` | the curation that produced the reference set, and its cost |
| `average_precision`, `improvement_percent` | the ranked list, and its change against the template-matching order of the same list |
| `decisions`, `decisions_to_target`, `saving_percent` | the effort of the walk, and its reduction against never re-ranking |
| `curated_tomograms`, `accumulation_order`, `reference_picks` | the pooled reference of Experiment 3 |

## Definitions

**True pick.** A candidate lying within `--radius` of a deposited particle, the nearest
particle taking the candidate.

**Average precision.** The area under the precision–recall curve of the list read from the
top, computed without interpolation. Recall is counted against the true picks present in
the list being ranked, so the measure describes the ordering of that list. On the scale of
Supplementary Note 1, where recall is counted against all deposited particles, the values
reported here are multiplied by the maximum achievable recall of the list.

**Decisions.** Walking a ranked list, a maximal run of consecutive true picks is accepted
in one action and a maximal run of other candidates is rejected in one action; each
re-ranking is one further decision, as is restoring the template-matching order after a
re-ranking that places no acceptable run at the front.

**The three uses of the cytosol mask.** The mask is not used; the candidates outside it are
removed before anything is scored; or the complete list is kept and the candidates outside
the mask are rejected in a single action, which supplies rejected picks before any have
been rejected individually.

**Splitting.** In Experiments 1 and 2 the candidate list is dealt alternately into a
labelling half, in which picks are confirmed and rejected, and an evaluation half, which is
re-ranked and scored; dealing alternately gives both halves the same range of
template-matching scores. In Experiment 3 no reference comes from the target, so its
complete list is ranked and scored.

## Validation

Each run verifies its own invariants and terminates without writing a table if one fails:
the two halves are disjoint and cover the list; a reference set holds exactly the requested
number of confirmed picks and shares no candidate with the evaluation half; a target never
contributes to its own reference set; confirmed and rejected sets are disjoint; reaching a
larger share of the true picks never costs fewer decisions; and every score is finite. A
condition that cannot be formed — the relative distance requires at least two confirmed and
two rejected picks — is recorded as absent rather than replaced by a baseline value.

`reproduce_published_results.py` runs the three experiments on a subset of tomograms and
compares every value with the published results, exiting with a non-zero status if any
differs:

```bash
python3 reproduce_published_results.py --data DATA --curated CURATED --published DIR
```

On the system used to prepare Supplementary Note 5 it reports 79 of 79 average precisions
for Experiment 1, 28 of 28 decision counts for Experiment 2 and 12 of 12 average precisions
for Experiment 3 identical to the published values.

## Citation

Please cite the PickyMate paper and, for the dataset, de Teresa-Trueba et al.,
*Nature Methods* 20, 284–294 (2023), EMPIAR-10988.
