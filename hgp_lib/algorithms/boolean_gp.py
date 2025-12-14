from typing import Callable

from numpy import ndarray
import numpy as np

from hgp_lib.mutations import MutationExecutor
from hgp_lib.populations import PopulationGenerator
from hgp_lib.rules import Rule
from hgp_lib.utils.validation import check_isinstance, validate_callable

from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.selections import BaseSelection, RouletteSelection
from .metrics import StepMetrics, ValidateBestMetrics, ValidatePopulationMetrics


class BooleanGP:
    """
    Boolean Genetic Programming algorithm for evolving rule-based classifiers.

    This class implements a genetic programming algorithm that evolves a population of
    boolean rules to optimize a fitness function. Each generation applies crossover and
    mutation operations, evaluates the population, and selects the best individuals.

    The algorithm tracks both the current best rule (best in the current run) and the
    real best rule (all-time best across regenerations). When enabled, regeneration
    resets the population if no improvement is observed for a specified number of epochs.

    Args:
        score_fn (Callable[[ndarray, ndarray], float]):
            Function that computes fitness scores. Signature: `score_fn(predictions, labels) -> float`.
            Higher scores indicate better fitness. Must accept boolean predictions and integer labels.
        population_generator (PopulationGenerator):
            Generator that creates the initial population of rules.
        mutation_executor (MutationExecutor):
            Executor that applies mutations to the population.
        crossover_executor (CrossoverExecutor | None, optional):
            Executor that applies crossover operations. If `None`, a default `CrossoverExecutor`
            is created. Default: `None`.
        selection (BaseSelection | None, optional):
            Selection strategy for choosing individuals for the next generation. If `None`,
            a default `RouletteSelection` is created. Default: `None`.
        regeneration (bool, optional):
            Whether to regenerate the population when no improvement is observed.
            Default: `False`.
        regeneration_patience (int, optional):
            Number of epochs without improvement before regenerating the population.
            Must be positive when `regeneration=True`. Default: `100`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.algorithms import BooleanGP
        >>> from hgp_lib.mutations import (
        ...     MutationExecutor, create_standard_literal_mutations, create_standard_operator_mutations
        ... )
        >>> from hgp_lib.populations import PopulationGenerator, RandomStrategy
        >>> from hgp_lib.rules import Literal
        >>>
        >>> def accuracy(predictions, labels):
        ...     return np.mean(predictions == labels)
        >>>
        >>> generator = PopulationGenerator(
        ...     strategies=[RandomStrategy(num_literals=4)],
        ...     population_size=10
        ... )
        >>> mutation_executor = MutationExecutor(
        ...     literal_mutations=create_standard_literal_mutations(4),
        ...     operator_mutations=create_standard_operator_mutations(4)
        ... )
        >>>
        >>> gp = BooleanGP(
        ...     score_fn=accuracy,
        ...     population_generator=generator,
        ...     mutation_executor=mutation_executor
        ... )
        >>>
        >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
        >>> train_labels = np.array([1, 0])
        >>> metrics = gp.step(train_data, train_labels)
        >>> metrics['epoch']
        0
    """

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
        """
        Performs one training step (generation) of the genetic programming algorithm.

        Each step applies crossover to create offspring, mutates the population,
        evaluates all rules, updates the best rule, and selects individuals for
        the next generation. If regeneration is enabled and no improvement has
        been observed for `regeneration_patience` epochs, the population is
        regenerated.

        Args:
            train_data (ndarray):
                Training data as a 2D boolean array with instances on rows and features on columns.
            train_labels (ndarray):
                Training labels as a 1D integer array (0 or 1 for binary classification).

        Returns:
            StepMetrics: Dictionary containing:
                - `best` (float): Best fitness score in current run.
                - `best_rule` (Rule): Best rule in current run.
                - `real_best` (float): All-time best fitness score.
                - `real_best_rule` (Rule): All-time best rule.
                - `current_best` (float): Best score in current generation.
                - `population_scores` (ndarray): Fitness scores for all rules.
                - `epoch` (int): Current epoch number (0-indexed).
                - `best_not_improved_epochs` (int): Epochs since last improvement.
                - `regenerated` (bool): Whether population was regenerated this step.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.algorithms import BooleanGP
            >>> from hgp_lib.mutations import (
            ...     MutationExecutor, create_standard_literal_mutations, create_standard_operator_mutations
            ... )
            >>> from hgp_lib.populations import PopulationGenerator, RandomStrategy
            >>>
            >>> def accuracy(predictions, labels):
            ...     return np.mean(predictions == labels)
            >>>
            >>> generator = PopulationGenerator(
            ...     strategies=[RandomStrategy(num_literals=4)],
            ...     population_size=5
            ... )
            >>> mutation_executor = MutationExecutor(
            ...     literal_mutations=create_standard_literal_mutations(4),
            ...     operator_mutations=create_standard_operator_mutations(4)
            ... )
            >>>
            >>> gp = BooleanGP(
            ...     score_fn=accuracy,
            ...     population_generator=generator,
            ...     mutation_executor=mutation_executor
            ... )
            >>>
            >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
            >>> train_labels = np.array([1, 0])
            >>> metrics = gp.step(train_data, train_labels)
            >>> 'best' in metrics
            True
            >>> metrics['epoch']
            0
        """
        self._epoch += 1
        self.population += self.crossover_executor.apply(self.population)
        self.mutation_executor.apply(self.population)
        scores = self._evaluate_population(train_data, train_labels, self.score_fn)
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
            StepMetrics: Dictionary containing metrics about the generation step, including
                whether regeneration occurred and the current best scores.
        """
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
        """
        Evaluates all rules in the population against the given data.

        Computes fitness scores for each rule by evaluating it on the data and applying
        the scoring function. This is done sequentially for each rule in the population.

        Args:
            data (ndarray):
                Data to evaluate rules on. A 2D boolean array with instances on rows
                and features on columns.
            labels (ndarray):
                True labels for the data. A 1D integer array (0 or 1 for binary classification).
            score_fn (Callable[[ndarray, ndarray], float]):
                Function to compute fitness scores. Signature: `score_fn(predictions, labels) -> float`.

        Returns:
            ndarray: Array of fitness scores, one for each rule in the population.

        Note:
            TODO: we should also support batched evaluation or free-threaded evaluation
        """
        n = len(self.population)
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = score_fn(self.population[i].evaluate(data), labels)
        return scores

    def _update_best(self, current_best: float, current_best_rule: Rule):
        """
        Updates the best rule tracking based on the current generation's best.

        If the current best score is greater than or equal to the stored best score,
        updates both the current run's best and the all-time best (if it's a new record).
        Otherwise, increments the counter for epochs without improvement.

        Args:
            current_best (float):
                The best fitness score from the current generation.
            current_best_rule (Rule):
                The rule that achieved the best score in the current generation.
                This rule will be copied when stored.
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

        Computes the fitness score of either the current best rule or the all-time
        best rule on the provided data. This is useful for validation during training
        or final evaluation on test sets.

        Args:
            data (ndarray):
                Validation/test data as a 2D boolean array with instances on rows
                and features on columns.
            labels (ndarray):
                Validation/test labels as a 1D integer array (0 or 1 for binary classification).
            score_fn (Callable[[ndarray, ndarray], float] | None, optional):
                Optional custom scoring function. If `None`, uses the instance's `score_fn`.
                Default: `None`.
            all_time_best (bool, optional):
                If `True`, evaluates the all-time best rule. If `False`, evaluates
                the current run's best rule. Default: `False`.

        Returns:
            ValidateBestMetrics: Dictionary containing:
                - `best` (float): Fitness score of the best rule on the data.
                - `best_rule` (Rule): The evaluated rule.

        Raises:
            RuntimeError:
                If no best rule is available (algorithm hasn't run any steps yet).

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.algorithms import BooleanGP
            >>> from hgp_lib.mutations import (
            ...     MutationExecutor, create_standard_literal_mutations, create_standard_operator_mutations
            ... )
            >>> from hgp_lib.populations import PopulationGenerator, RandomStrategy
            >>>
            >>> def accuracy(predictions, labels):
            ...     return np.mean(predictions == labels)
            >>>
            >>> generator = PopulationGenerator(
            ...     strategies=[RandomStrategy(num_literals=4)],
            ...     population_size=5
            ... )
            >>> mutation_executor = MutationExecutor(
            ...     literal_mutations=create_standard_literal_mutations(4),
            ...     operator_mutations=create_standard_operator_mutations(4)
            ... )
            >>>
            >>> gp = BooleanGP(
            ...     score_fn=accuracy,
            ...     population_generator=generator,
            ...     mutation_executor=mutation_executor
            ... )
            >>>
            >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
            >>> train_labels = np.array([1, 0])
            >>> _ = gp.step(train_data, train_labels)
            >>> val_data = np.array([[True, True, False, False]])
            >>> val_labels = np.array([1])
            >>> metrics = gp.validate_best(val_data, val_labels)
            >>> 'best' in metrics
            True
            >>> 'best_rule' in metrics
            True
        """
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
        """
        Evaluates the best rule and entire population on validation or test data.

        Computes fitness scores for both the best rule and all rules in the current
        population. This provides insight into both the best individual's performance
        and the overall population quality.

        Args:
            data (ndarray):
                Validation/test data as a 2D boolean array with instances on rows
                and features on columns.
            labels (ndarray):
                Validation/test labels as a 1D integer array (0 or 1 for binary classification).
            score_fn (Callable[[ndarray, ndarray], float] | None, optional):
                Optional custom scoring function. If `None`, uses the instance's `score_fn`.
                Default: `None`.
            all_time_best (bool, optional):
                If `True`, evaluates the all-time best rule. If `False`, evaluates
                the current run's best rule. Default: `False`.

        Returns:
            ValidatePopulationMetrics: Dictionary containing:
                - `best` (float): Fitness score of the best rule on the data.
                - `best_rule` (Rule): The evaluated best rule.
                - `population_scores` (ndarray): Fitness scores for all rules in the population.

        Raises:
            RuntimeError:
                If no best rule is available (algorithm hasn't run any steps yet).

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.algorithms import BooleanGP
            >>> from hgp_lib.mutations import (
            ...     MutationExecutor, create_standard_literal_mutations, create_standard_operator_mutations
            ... )
            >>> from hgp_lib.populations import PopulationGenerator, RandomStrategy
            >>>
            >>> def accuracy(predictions, labels):
            ...     return np.mean(predictions == labels)
            >>>
            >>> generator = PopulationGenerator(
            ...     strategies=[RandomStrategy(num_literals=4)],
            ...     population_size=5
            ... )
            >>> mutation_executor = MutationExecutor(
            ...     literal_mutations=create_standard_literal_mutations(4),
            ...     operator_mutations=create_standard_operator_mutations(4)
            ... )
            >>>
            >>> gp = BooleanGP(
            ...     score_fn=accuracy,
            ...     population_generator=generator,
            ...     mutation_executor=mutation_executor
            ... )
            >>>
            >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
            >>> train_labels = np.array([1, 0])
            >>> _ = gp.step(train_data, train_labels)
            >>>
            >>> val_data = np.array([[True, True, False, False]])
            >>> val_labels = np.array([1])
            >>> metrics = gp.validate_population(val_data, val_labels)
            >>> 'best' in metrics
            True
            >>> 'population_scores' in metrics
            True
            >>> len(metrics['population_scores']) == len(gp.population)
            True
        """
        if self.real_best_rule is None or self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")
        best_rule = self.real_best_rule if all_time_best else self.best_rule
        score_fn = self.score_fn if score_fn is None else score_fn
        return ValidatePopulationMetrics(
            best=score_fn(best_rule.evaluate(data), labels),
            best_rule=best_rule,
            population_scores=self._evaluate_population(data, labels, score_fn),
        )
