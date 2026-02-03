from typing import Callable

import numpy as np
from numpy import ndarray

from ..configs import BooleanGPConfig, validate_gp_config
from ..crossover import CrossoverExecutor
from ..metrics import StepMetrics, ValidateBestMetrics, ValidatePopulationMetrics
from ..mutations import (
    MutationExecutor,
    create_standard_literal_mutations,
    create_standard_operator_mutations,
)
from ..populations import PopulationGenerator, RandomStrategy
from ..rules import Rule
from ..selections import RouletteSelection
from ..utils.metrics import optimize_scorer_for_data


class BooleanGP:
    """
    Boolean Genetic Programming algorithm for evolving rule-based classifiers.

    This class implements a genetic programming algorithm that evolves a population of
    boolean rules to optimize a fitness function. Each generation applies crossover and
    mutation operations, evaluates the population, and selects the best individuals.

    The algorithm tracks both the current best rule (best in the current run) and the
    real best rule (all-time best across regenerations). When enabled, regeneration
    resets the population if no improvement is observed for a specified number of epochs.

    Training data and labels are provided via BooleanGPConfig. The number of features
    (num_features) is derived from the data shape, simplifying PopulationGenerator and
    MutationExecutor setup when defaults are used.

    Args:
        config (BooleanGPConfig): Configuration containing train_data, train_labels,
            score_fn, and optional components. If population_generator or
            mutation_executor are None, they are created using num_features derived
            from config.train_data.shape[1].

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig
        >>> from hgp_lib.algorithms import BooleanGP
        >>>
        >>> def accuracy(predictions, labels):
        ...     return np.mean(predictions == labels)
        >>>
        >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
        >>> train_labels = np.array([1, 0])
        >>> config = BooleanGPConfig(
        ...     score_fn=accuracy,
        ...     train_data=train_data,
        ...     train_labels=train_labels,
        ... )
        >>> gp = BooleanGP(config)
        >>> metrics = gp.step()
        >>> metrics.epoch
        0
    """

    def __init__(self, config: BooleanGPConfig):
        validate_gp_config(config)

        train_data = config.train_data
        train_labels = config.train_labels
        score_fn = config.score_fn

        if config.optimize_scorer:
            score_fn, train_data, train_labels = optimize_scorer_for_data(
                config.score_fn, config.train_data, config.train_labels
            )

        self.score_fn = score_fn
        self._train_data = train_data
        self._train_labels = train_labels
        num_features = train_data.shape[1]

        population_generator = config.population_generator
        if population_generator is None:
            random_strategy = RandomStrategy(num_literals=num_features)
            population_generator = PopulationGenerator(
                strategies=[random_strategy],
                population_size=100,
            )

        mutation_executor = config.mutation_executor
        if mutation_executor is None:
            literal_mutations = create_standard_literal_mutations(num_features)
            operator_mutations = create_standard_operator_mutations(num_features)
            mutation_executor = MutationExecutor(
                literal_mutations=literal_mutations,
                operator_mutations=operator_mutations,
                check_valid=config.check_valid,
            )

        crossover_executor = config.crossover_executor
        if crossover_executor is None:
            crossover_executor = CrossoverExecutor(check_valid=config.check_valid)

        selection = config.selection
        if selection is None:
            selection = RouletteSelection()

        self.population_generator = population_generator
        self.mutation_executor = mutation_executor
        self.crossover_executor = crossover_executor
        self.selection = selection
        self.regeneration = config.regeneration
        self.regeneration_patience = config.regeneration_patience

        self.population = self.population_generator.generate()
        self.population_size = len(self.population)

        self.best_score = -float("inf")
        self.best_rule: Rule | None = None
        self.best_not_improved_epochs = 0
        self.real_best_score = -float("inf")
        self.real_best_rule: Rule | None = None
        self._epoch = -1

    def step(self) -> StepMetrics:
        """
        Performs one training step (generation) of the genetic programming algorithm.

        Uses the training data and labels from the config. Each step applies crossover
        to create offspring, mutates the population, evaluates all rules, updates the
        best rule, and selects individuals for the next generation. If regeneration
        is enabled and no improvement has been observed for `regeneration_patience`
        epochs, the population is regenerated.

        Returns:
            StepMetrics: Dataclass containing best, best_rule, real_best, real_best_rule,
                current_best, mean_score, std_score, population_scores, epoch,
                best_not_improved_epochs, and regenerated.
        """
        self._epoch += 1
        self.population += self.crossover_executor.apply(self.population)
        self.mutation_executor.apply(self.population)
        scores = self._evaluate_population(
            self._train_data, self._train_labels, self.score_fn
        )
        return self._new_generation(scores)

    def _new_generation(self, scores: ndarray) -> StepMetrics:
        """
        Creates a new generation by selecting individuals and optionally regenerating the population.

        Updates the best rule tracking, checks if regeneration is needed, and selects the next
        generation using the configured selection strategy. If regeneration is triggered,
        the population is completely regenerated and best tracking is reset.

        Args:
            scores (ndarray):
                Fitness scores for all rules in the current population. Must have the same
                length as `self.population`.

        Returns:
            StepMetrics: Metrics about the generation step, including mean_score and std_score.
        """
        best_idx = int(np.argmax(scores))
        current_best = float(scores[best_idx])
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

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
            mean_score=mean_score,
            std_score=std_score,
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
        """
        Evaluates all rules in the population against the given data.

        Args:
            data (ndarray): Data to evaluate rules on (2D boolean array).
            labels (ndarray): True labels (1D integer array).
            score_fn (Callable): Function to compute fitness scores.

        Returns:
            ndarray: Array of fitness scores, one for each rule in the population.
        """
        n = len(self.population)
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = score_fn(self.population[i].evaluate(data), labels)
        return scores

    def _update_best(self, current_best: float, current_best_rule: Rule) -> None:
        """
        Updates the best rule tracking based on the current generation's best.
        """
        if current_best >= self.best_score:
            self.best_not_improved_epochs = 0
            self.best_score = current_best
            self.best_rule = current_best_rule.copy()
            if self.best_score > self.real_best_score:
                self.real_best_score = self.best_score
                self.real_best_rule = self.best_rule.copy()
        else:
            self.best_not_improved_epochs += 1

    def validate_best(
        self,
        data: ndarray,
        labels: ndarray,
        score_fn: Callable[[ndarray, ndarray], float] | None = None,
        all_time_best: bool = False,
    ) -> ValidateBestMetrics:
        """
        Evaluates the best rule on validation or test data.

        Args:
            data (ndarray): Validation/test data (2D boolean array).
            labels (ndarray): Validation/test labels (1D integer array).
            score_fn (Callable | None): Optional; uses instance score_fn if None.
            all_time_best (bool): If True, evaluate all-time best rule; else current run's best.

        Returns:
            ValidateBestMetrics: best (float) and best_rule (Rule).

        Raises:
            RuntimeError: If no best rule is available (run at least one step first).
        """
        if self.real_best_rule is None or self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        best_rule = self.real_best_rule if all_time_best else self.best_rule
        fn = self.score_fn if score_fn is None else score_fn
        best_score = float(fn(best_rule.evaluate(data), labels))
        return ValidateBestMetrics(best=best_score, best_rule=best_rule)

    def validate_population(
        self,
        data: ndarray,
        labels: ndarray,
        score_fn: Callable[[ndarray, ndarray], float] | None = None,
        all_time_best: bool = False,
    ) -> ValidatePopulationMetrics:
        """
        Evaluates the best rule and entire population on validation or test data.

        Args:
            data (ndarray): Validation/test data (2D boolean array).
            labels (ndarray): Validation/test labels (1D integer array).
            score_fn (Callable | None): Optional; uses instance score_fn if None.
            all_time_best (bool): If True, evaluate all-time best rule; else current run's best.

        Returns:
            ValidatePopulationMetrics: best, best_rule, and population_scores.

        Raises:
            RuntimeError: If no best rule is available (run at least one step first).
        """
        if self.real_best_rule is None or self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        best_rule = self.real_best_rule if all_time_best else self.best_rule
        fn = self.score_fn if score_fn is None else score_fn
        best_score = float(fn(best_rule.evaluate(data), labels))
        population_scores = self._evaluate_population(data, labels, fn)
        return ValidatePopulationMetrics(
            best=best_score,
            best_rule=best_rule,
            population_scores=population_scores,
        )
