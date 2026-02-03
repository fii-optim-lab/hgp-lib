from typing import Callable, NamedTuple
from numpy import ndarray

from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import MutationExecutor
from hgp_lib.populations import PopulationGenerator
from hgp_lib.rules import Rule
from hgp_lib.selections import BaseSelection


class BenchmarkConfig(NamedTuple):
    data: ndarray
    labels: ndarray
    test_size: float
    n_folds: int
    num_epochs: int
    score_fn: Callable[[ndarray, ndarray], float]
    val_score_fn: Callable[[ndarray, ndarray], float]
    optimize_scorer: bool
    check_valid: Callable[[Rule], bool] | None
    population_generator: PopulationGenerator | None
    mutation_executor: MutationExecutor | None
    crossover_executor: CrossoverExecutor | None
    selection: BaseSelection | None
    regeneration: bool
    regeneration_patience: int
    val_every: int
    show_fold_progress: bool
    show_epoch_progress: bool
    progress_bar: bool
