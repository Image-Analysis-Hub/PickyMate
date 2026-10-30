"""Functions shared by the three benchmark experiments of PickyMate.

    tomogram_data        reading candidates, embeddings, particles and the mask
    evaluation_split     dealing a list into a labelling and an evaluation half
    curation_simulation  what a curator does: examining, and clearing a region
    mahalanobis_ranking  the ranking the application performs
    ranking_metrics      average precision, and decisions
    reranking_schedules  walking a list while re-ranking it
    reference_pooling    pooling the decisions of finished tomograms
    run_report           writing results, and the checks a run makes on itself
"""
import sys
from pathlib import Path

# The PickyMate checkout this folder sits in, ahead of any installed copy:
# the benchmark must rank (boxer.ranking) and read working directories
# (boxer.metadata) with the code it is committed next to.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import boxer  # noqa: E402
if Path(boxer.__file__).resolve().parents[1] != Path(__file__).resolve().parents[2]:
    raise ImportError(f"boxer was already imported from {boxer.__file__}; import "
                      f"experiment_functions first so the benchmark uses the checkout it sits in")

from . import (curation_simulation, evaluation_split, mahalanobis_ranking,  # noqa: E402
               ranking_metrics, reference_pooling, reranking_schedules,
               run_report, tomogram_data)

__all__ = ['curation_simulation', 'evaluation_split', 'mahalanobis_ranking',
           'ranking_metrics', 'reference_pooling', 'reranking_schedules',
           'run_report', 'tomogram_data']
