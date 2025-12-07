from typing import Callable, TypedDict

from numpy import ndarray
import numpy as np

from hgp_lib.mutations import MutationExecutor
from hgp_lib.populations import PopulationGenerator
from hgp_lib.rules import Rule
from hgp_lib.utils.validation import check_isinstance, validate_callable

from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.selections import StubSelection


class StepMetrics(TypedDict):
    best: float
    best_rule: Rule
    current_best: float
    population_scores: ndarray
    epoch: int


class ValidateBestMetrics(TypedDict):
    best: float
    best_rule: Rule


class ValidatePopulationMetrics(TypedDict):
    best: float
    best_rule: Rule
    population_scores: ndarray


class BooleanGP:
    def __init__(
        self,
        score_fn: Callable[[ndarray, ndarray], float],
        population_generator: PopulationGenerator,
        mutation_executor: MutationExecutor,
        crossover_executor: CrossoverExecutor | None = None,
        selection: StubSelection | None = None,
        regeneration: bool = False,
        regeneration_patience: int = 100,
    ):
        validate_callable(score_fn)
        check_isinstance(population_generator, PopulationGenerator)
        check_isinstance(mutation_executor, MutationExecutor)
        check_isinstance(regeneration, bool)
        check_isinstance(regeneration_patience, int)

        if crossover_executor is not None:
            check_isinstance(crossover_executor, CrossoverExecutor)
        else:
            crossover_executor = CrossoverExecutor()
        if selection is not None:
            check_isinstance(selection, StubSelection)
        else:
            # TODO: Implement a real selection strategy
            selection = StubSelection()
        if regeneration and regeneration_patience < 1:
            raise ValueError("regeneration_patience must be a positive integer")

        self.score_fn = score_fn
        self.population_generator = population_generator
        self.mutation_executor = mutation_executor
        self.crossover_executor = crossover_executor
        self.selection = selection

        self.population = self.population_generator.generate()
        self.population_size = len(self.population)

        self.best_score = -float("inf")
        self.best_rule = None
        self._epoch = 0

    def step(self, train_data: ndarray, train_labels: ndarray) -> StepMetrics:
        self.population += self.crossover_executor.apply(self.population)
        self.mutation_executor.apply(self.population)
        scores = self._evaluate_population(train_data, train_labels)
        metrics = self._handle_metrics(scores)
        # We must calculate the metrics before the selection, otherwise population is changed and metrics are not correct
        self.population = self.selection.select(
            self.population, scores, self.population_size
        )
        self._epoch += 1
        return metrics

    def _handle_metrics(self, scores: ndarray) -> StepMetrics:
        best_idx = np.argmax(scores)
        current_best = scores[best_idx]
        if current_best > self.best_score:
            self.best_score = current_best
            self.best_rule = self.population[best_idx].copy()

        return {
            "best": self.best_score,
            "best_rule": self.best_rule,
            "current_best": current_best,
            "population_scores": scores,
            "epoch": self._epoch,
        }

    def _evaluate_population(self, data: ndarray, labels: ndarray) -> ndarray:
        # TODO: we should also support batched evaluation or free-threaded evaluation
        n = len(self.population)
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = self.score_fn(self.population[i].evaluate(data), labels)
        return scores

    def validate_best(self, data: ndarray, labels: ndarray) -> ValidateBestMetrics:
        if self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        return {
            "best": self.score_fn(self.best_rule.evaluate(data), labels),
            "best_rule": self.best_rule,
        }

    def validate_population(
        self, data: ndarray, labels: ndarray
    ) -> ValidatePopulationMetrics:
        if self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        return {
            "best": self.score_fn(self.best_rule.evaluate(data), labels),
            "best_rule": self.best_rule,
            "population_scores": self._evaluate_population(data, labels),
        }
