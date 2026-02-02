from typing import Callable, List

from numpy import ndarray
import numpy as np

from hgp_lib.mutations import MutationExecutor
from hgp_lib.populations import PopulationGenerator
from hgp_lib.rules import Rule
from hgp_lib.utils.metrics import normalize
from hgp_lib.utils.validation import check_isinstance, validate_callable

from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.selections import BaseSelection, RouletteSelection
from hgp_lib.metrics import StepMetrics, ValidateBestMetrics, ValidatePopulationMetrics


class BooleanGP:
    """
    Boolean Genetic Programming algorithm for evolving rule-based classifiers.

    This class implements a genetic programming algorithm that evolves a population of
    boolean rules to optimize a fitness function. Each generation applies crossover and
    mutation operations, evaluates the population, and selects the best individuals.

    The algorithm supports hierarchical GP, where child populations can contribute rules
    to the parent's crossover pool. Child populations may operate on feature subsets
    (feature bagging), with feature mappings translating indices during crossover.
    Fitness signals propagate bidirectionally: children contribute rules during the
    forward pass, and receive score feedback during the backward pass.

    The algorithm tracks both the current best rule (best in the current run) and the
    real best rule (all-time best across regenerations). When enabled, regeneration
    resets the population if no improvement is observed for a specified number of epochs.

    Args:
        score_fn (Callable[[ndarray, ndarray], float]):
            Function that computes fitness scores. Signature: `score_fn(predictions, labels) -> float`.
            Higher scores indicate better fitness. Must accept boolean predictions and integer labels.
        train_data (ndarray):
            Training data as a 2D boolean array with instances on rows and features on columns.
        train_labels (ndarray):
            Training labels as a 1D integer array (0 or 1 for binary classification).
            Must have the same length as the number of rows in `train_data`.
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

    Attributes:
        child_populations (List[BooleanGP]):
            List of child BooleanGP instances for hierarchical GP. Empty by default.
        feature_mapping (dict[int, int] | None):
            Mapping from this population's feature indices to the parent's indices.
            Used when this population operates on a feature subset. None for root populations.

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
        >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
        >>> train_labels = np.array([1, 0])
        >>> gp = BooleanGP(
        ...     score_fn=accuracy,
        ...     train_data=train_data,
        ...     train_labels=train_labels,
        ...     population_generator=generator,
        ...     mutation_executor=mutation_executor
        ... )
        >>>
        >>> metrics = gp.step()
        >>> metrics['epoch']
        0
    """

    def __init__(
        self,
        score_fn: Callable[[ndarray, ndarray], float],
        train_data: ndarray,
        train_labels: ndarray,
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

        check_isinstance(train_data, ndarray)
        check_isinstance(train_labels, ndarray)
        # TODO: Add checks for dim

        if len(train_labels) != train_data.shape[0]:
            raise ValueError(
                f"train_labels length ({len(train_labels)}) must match "
                f"train_data rows ({train_data.shape[0]})"
            )

        self.train_data = train_data
        self.train_labels = train_labels

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

        self.child_populations: List["BooleanGP"] = []
        self.feature_mapping: dict[int, int] | None = None
        self.child_population_sizes: List[int] = []
        self.parent_rule_indices: List[int] = []

    def step(self) -> StepMetrics:
        """
        Performs one training step (generation) of the genetic programming algorithm.

        Each step consists of a forward pass (crossover and mutation) followed by a
        backward pass (evaluation and selection). In hierarchical GP, child populations
        also perform their forward/backward passes, with scores propagating between levels.

        The forward pass applies crossover to create offspring (including rules from
        child populations when present), then mutates the population. The backward pass
        evaluates all rules, updates the best rule tracking, and selects individuals
        for the next generation.

        If regeneration is enabled and no improvement has been observed for
        `regeneration_patience` epochs, the population is regenerated.

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
            >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
            >>> train_labels = np.array([1, 0])
            >>> gp = BooleanGP(
            ...     score_fn=accuracy,
            ...     train_data=train_data,
            ...     train_labels=train_labels,
            ...     population_generator=generator,
            ...     mutation_executor=mutation_executor
            ... )
            >>>
            >>> metrics = gp.step()
            >>> 'best' in metrics
            True
            >>> metrics['epoch']
            0
        """
        self._epoch += 1
        self.forward()
        return self.backward()

    def forward(self) -> None:
        """
        Performs the forward pass of the genetic programming algorithm.

        The forward pass consists of:
        1. Recursively calling forward() on all child populations
        2. Collecting rules from child populations with their feature mappings
        3. Applying crossover to create offspring (combining rules from current
           and child populations)
        4. Applying mutations to the expanded population

        In hierarchical GP, child populations may operate on different feature
        subsets. Their rules are translated to the parent's feature space via
        feature mappings before crossover.

        This method updates:
            - `self.child_population_sizes`: Sizes of each child population
            - `self.parent_rule_indices`: Indices tracking which parents contributed to children
            - `self.population`: Extended with new offspring from crossover
        """
        child_populations = []
        self.child_population_sizes = []
        feature_mappings: List[dict[int, int] | None] = [None] * self.population_size

        for child in self.child_populations:
            child.forward()
            child_populations += child.population
            child_population_size = len(child.population)
            feature_mappings += [child.feature_mapping] * child_population_size
            self.child_population_sizes.append(child_population_size)

        next_generation, self.parent_rule_indices = self.crossover_executor.apply(
            self.population + child_populations, feature_mappings
        )
        self.population += next_generation

        self.mutation_executor.apply(self.population)

    def get_index_from_helper(self, index: int) -> tuple[int, int]:
        """
        Maps a flattened child population index to (child_index, local_index).

        When rules from multiple child populations are concatenated, this method
        determines which child population a given index belongs to and the
        corresponding local index within that child's population.

        Args:
            index (int): The flattened index into the concatenated child populations
                (already offset by `self.population_size`).

        Returns:
            tuple[int, int]: A tuple of (child_population_index, local_index) where:
                - child_population_index: Which child population (0-indexed)
                - local_index: Index within that child's population

        Raises:
            RuntimeError: If the index is out of bounds for all child populations.
        """
        for child_population_index, child_population_size in enumerate(
            self.child_population_sizes
        ):
            if index >= child_population_size:
                index -= child_population_size
            else:
                return child_population_index, index
        raise RuntimeError("Unreachable code")

    def process_helper_scores(self, scores: ndarray) -> List[ndarray]:
        """
        Distributes normalized scores to child populations based on parent contributions.

        When child population rules contribute to offspring via crossover, this method
        propagates fitness signals back to those child populations. Scores are normalized
        and accumulated for each rule in each child population based on how often that
        rule was selected as a parent.

        Args:
            scores (ndarray): Fitness scores for the current population's rules.

        Returns:
            List[ndarray]: A list of score arrays, one per child population. Each array
                has the same length as the corresponding child population and contains
                accumulated normalized scores for rules that contributed to offspring.
        """
        normalized_scores = normalize(scores)
        child_population_scores = [
            np.zeros(child_population_size)
            for child_population_size in self.child_population_sizes
        ]

        for index in self.parent_rule_indices:
            if (
                index >= self.population_size
            ):  # index is not from the current population
                child_population_index, remainder = self.get_index_from_helper(index - self.population_size)
                child_population_scores[child_population_index][remainder] += (
                    normalized_scores[index]
                )

        return child_population_scores

    def backward(self, parent_scores: ndarray | None = None) -> StepMetrics:
        """
        Performs the backward pass of the genetic programming algorithm.

        The backward pass consists of:
        1. Evaluating all rules in the population against training data
        2. Adding any scores propagated from parent populations (in hierarchical GP)
        3. Propagating scores to child populations based on their contributions
        4. Recursively calling backward() on child populations
        5. Creating the next generation via selection

        In hierarchical GP, scores flow bidirectionally: parent populations receive
        rules from children during crossover (forward), and children receive fitness
        signals based on how well their contributed rules performed (backward).

        Args:
            parent_scores (ndarray | None): Optional scores propagated from a parent
                population. These are added to the local evaluation scores. Used in
                hierarchical GP to reward child rules that contributed to successful
                offspring in the parent. Default: None.

        Returns:
            StepMetrics: Dictionary containing metrics about the generation step.
        """
        scores = self._evaluate_population(
            self.train_data, self.train_labels, self.score_fn
        )
        if parent_scores is not None:
            scores += parent_scores

        if len(self.child_populations):
            child_population_scores = self.process_helper_scores(scores)
            for child, child_scores in zip(
                self.child_populations, child_population_scores
            ):
                child.backward(child_scores)

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
