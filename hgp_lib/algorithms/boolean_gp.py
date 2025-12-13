from typing import Callable, TypedDict

from numpy import ndarray
import numpy as np

from hgp_lib.mutations import MutationExecutor
from hgp_lib.populations import PopulationGenerator
from hgp_lib.rules import Rule
from hgp_lib.utils.validation import check_isinstance, validate_callable

from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.selections import BaseSelection, RouletteSelection


class StepMetrics(TypedDict):
    best: float
    best_rule: Rule
    real_best: float
    real_best_rule: Rule
    current_best: float
    population_scores: ndarray
    epoch: int
    best_not_improved_epochs: int
    regenerated: bool


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
        selection: BaseSelection | None = None,
        regeneration: bool = False,
        regeneration_patience: int = 100,
    ):
        # TODO: We should reconsider the ordering of the arguments for score fn. Pred, GT or GT, Pred?
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
            check_isinstance(selection, BaseSelection)
        else:
            selection = RouletteSelection()
        if regeneration and regeneration_patience < 1:
            raise ValueError("regeneration_patience must be a positive integer")

        self.score_fn = score_fn
        self.population_generator = population_generator
        self.mutation_executor = mutation_executor
        self.crossover_executor = crossover_executor
        self.selection = selection
        self.regeneration = regeneration
        self.regeneration_patience = regeneration_patience

        self.population = self.population_generator.generate()
        self.population_size = len(self.population)

        self.best_score = -float("inf")
        self.best_rule: Rule | None = None
        self.best_not_improved_epochs = 0
        self.real_best_score = -float("inf")
        self.real_best_rule: Rule | None = None
        self._epoch = -1

    def step(self, train_data: ndarray, train_labels: ndarray) -> StepMetrics:
        self._epoch += 1
        self.population += self.crossover_executor.apply(self.population)
        self.mutation_executor.apply(self.population)
        scores = self._evaluate_population(train_data, train_labels, self.score_fn)
        return self._new_generation(scores)

    def _new_generation(self, scores: ndarray) -> StepMetrics:
        best_idx = np.argmax(scores)
        current_best = scores[best_idx]

        self._update_best(current_best, self.population[best_idx])

        regenerated = False

        if (
            self.regeneration
            and self.best_not_improved_epochs >= self.regeneration_patience
        ):
            regenerated = True

        metrics = StepMetrics(
            best=self.best_score,
            best_rule=self.best_rule,
            real_best=self.real_best_score,
            real_best_rule=self.real_best_rule,
            current_best=current_best,
            population_scores=scores,
            epoch=self._epoch,
            best_not_improved_epochs=self.best_not_improved_epochs,
            regenerated=regenerated,
        )

        if regenerated:
            self.population = self.population_generator.generate()
            self.best_score = -float("inf")
            self.best_not_improved_epochs = 0
        else:
            self.population = self.selection.select(
                self.population, scores, self.population_size
            )
        return metrics

    def _evaluate_population(
        self,
        data: ndarray,
        labels: ndarray,
        score_fn: Callable[[ndarray, ndarray], float],
    ) -> ndarray:
        # TODO: we should also support batched evaluation or free-threaded evaluation
        n = len(self.population)
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = score_fn(self.population[i].evaluate(data), labels)
        return scores

    def _update_best(self, current_best: float, current_best_rule: Rule):
        if current_best >= self.best_score:
            self.best_not_improved_epochs = 0
            self.best_score = current_best
            self.best_rule = current_best_rule.copy()
            if self.best_score > self.real_best_score:
                self.real_best_score = self.best_score
                self.real_best_rule = self.best_rule
        else:
            self.best_not_improved_epochs += 1

    def validate_best(
        self,
        data: ndarray,
        labels: ndarray,
        score_fn: Callable[[ndarray, ndarray], float] | None = None,
        all_time_best: bool = False,
    ) -> ValidateBestMetrics:
        if self.real_best_rule is None or self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        best_rule = self.real_best_rule if all_time_best else self.best_rule
        score_fn = self.score_fn if score_fn is None else score_fn
        return ValidateBestMetrics(
            best=score_fn(best_rule.evaluate(data), labels), best_rule=best_rule
        )

    def validate_population(
        self,
        data: ndarray,
        labels: ndarray,
        score_fn: Callable[[ndarray, ndarray], float] | None = None,
        all_time_best: bool = False,
    ) -> ValidatePopulationMetrics:
        if self.real_best_rule is None or self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")
        best_rule = self.real_best_rule if all_time_best else self.best_rule
        score_fn = self.score_fn if score_fn is None else score_fn
        return ValidatePopulationMetrics(
            best=score_fn(best_rule.evaluate(data), labels),
            best_rule=best_rule,
            population_scores=self._evaluate_population(data, labels, score_fn),
        )
