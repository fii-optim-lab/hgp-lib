from dataclasses import replace
from typing import Callable, List, Tuple

import numpy as np
from numpy import ndarray

from ..configs import BooleanGPConfig, validate_gp_config
from ..crossover import CrossoverExecutor
from ..metrics import StepMetrics, ValidateBestMetrics, ValidatePopulationMetrics
from ..mutations import (
    create_mutation_executor,
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
        ...     optimize_scorer=False,
        ... )
        >>> gp = BooleanGP(config)
        >>> metrics = gp.step()
        >>> metrics.epoch
        0
    """

    def __init__(self, config: BooleanGPConfig, current_depth: int = 0):
        validate_gp_config(config)

        train_data = config.train_data
        train_labels = config.train_labels
        score_fn = config.score_fn

        self._original_score_fn = config.score_fn
        self.current_depth = current_depth

        if config.optimize_scorer:
            score_fn, train_data, train_labels = optimize_scorer_for_data(
                config.score_fn, config.train_data, config.train_labels
            )

        self.score_fn = score_fn
        self.train_data = train_data
        self.train_labels = train_labels
        num_features = train_data.shape[1]

        population_generator = config.population_generator
        if population_generator is None:
            # TODO: We should rethink a system that is easy to initialize for both the user and child populations:
            # Maybe a common configuration for all strategies? This should be analyzed further
            random_strategy = RandomStrategy(num_literals=num_features)
            population_generator = PopulationGenerator(strategies=[random_strategy])

        mutation_executor = config.mutation_executor
        if mutation_executor is None:
            mutation_executor = create_mutation_executor(
                num_literals=num_features, check_valid=config.check_valid
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

        if config.top_k_transfer > self.population_size:
            raise ValueError(
                f"top_k_transfer ({config.top_k_transfer}) must be less than"
                f" or equal to population_size ({self.population_size})"
            )
        self._top_k = config.top_k_transfer

        self.best_score = -float("inf")
        self.best_rule: Rule | None = None
        self.best_not_improved_epochs = 0
        self.real_best_score = -float("inf")
        self.real_best_rule: Rule | None = None
        self._epoch = -1

        self.config = config
        self.child_populations: List["BooleanGP"] = []
        self.feature_mapping: dict[int, int] | None = None
        self._transfer_size: int = 0
        self.parent_rule_indices: List[int] = []
        self.last_scores: ndarray | None = None

        if config.max_depth > current_depth and config.num_child_populations > 0:
            self._create_child_populations()

    def _create_child_populations(self) -> None:
        """Create child populations using the sampling strategy."""
        if self.config.sampling_strategy is None:
            raise RuntimeError(
                "Cannot create child populations without a sampling strategy"
            )
        num_features = self.train_data.shape[1]
        for _ in range(self.config.num_child_populations):
            result = self.config.sampling_strategy.sample(
                self.train_data,
                self.train_labels,
                num_features,
                self.config.num_child_populations,
            )
            num_sampled_features = len(result.feature_indices)
            child_generator = PopulationGenerator(
                strategies=[RandomStrategy(num_literals=num_sampled_features)],
                population_size=self.population_generator.population_size,
            )
            child_mutation_executor = create_mutation_executor(
                num_literals=num_sampled_features,
                check_valid=self.config.check_valid,
                mutation_p=self.mutation_executor.mutation_p,
                num_tries=self.mutation_executor.num_tries,
            )
            child_config = replace(
                self.config,
                train_data=result.data,
                train_labels=result.labels,
                population_generator=child_generator,
                mutation_executor=child_mutation_executor,
            )
            child = BooleanGP(child_config, current_depth=self.current_depth + 1)

            # Feature mapping is only needed when features are subsampled (feature bagging).
            # For instance-only sampling, all features are preserved, so no mapping is needed.
            needs_feature_mapping = (
                num_sampled_features != num_features
                or not np.array_equal(result.feature_indices, np.arange(num_features))
            )
            if needs_feature_mapping:
                child.feature_mapping = {
                    i: int(result.feature_indices[i])
                    for i in range(num_sampled_features)
                }

            self.child_populations.append(child)

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
        # TODO: Add doctests here.
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
            - self._transfer_size: Total number of rules transferred from children
            - self.parent_rule_indices: Indices tracking which parents contributed to children
            - self.population: Extended with new offspring from crossover
        """
        for child in self.child_populations:
            child.forward()

        crossover_pool = []
        feature_mappings = []
        self._transfer_size = 0

        for child in self.child_populations:
            crossover_pool.extend(child._get_top_k_rules())
            self._transfer_size += child._top_k
            feature_mappings.extend([child.feature_mapping] * child._top_k)

        crossover_pool.extend(self.population)
        feature_mappings.extend([None] * len(self.population))
        offspring, self.parent_rule_indices = self.crossover_executor.apply(
            crossover_pool, feature_mappings
        )
        self.population += offspring
        self.mutation_executor.apply(self.population)

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
            self._apply_feedback(scores, parent_scores)

        self.last_scores = scores[: self.population_size].copy()

        children_metrics = []

        if self.child_populations:
            child_feedbacks = self._generate_child_feedback(scores)
            for child, feedback in zip(self.child_populations, child_feedbacks):
                children_metrics.append(child.backward(feedback))

        return self._new_generation(scores, children_metrics)

    def _get_top_k_rules(self) -> List[Rule]:
        """Select top-K rules for transfer to parent (indices 0.._top_k-1).
        The rules are already sorted by score in descending order.
        During the first epoch, the rules are not sorted by score.
        TODO: Add better documentation and doctests.
        """
        return self.population[: self._top_k]

    def _apply_feedback(self, scores: ndarray, parent_scores: ndarray) -> ndarray:
        """Apply incoming feedback from parent to the first self._top_k scores."""
        parent_scores *= self.config.feedback_strength
        if self.config.feedback_type == "multiplicative":
            scores[: self._top_k] *= 1 + parent_scores
        else:
            scores[: self._top_k] += parent_scores
        return scores

    def _get_child_and_local_index(self, flat_idx: int) -> Tuple[int, int]:
        """Map flattened index (into transferred rules) to (child_idx, local_idx)."""
        child_idx = flat_idx // self._top_k
        local_idx = flat_idx % self._top_k
        return child_idx, local_idx

    def _generate_child_feedback(self, scores: ndarray) -> List[ndarray]:
        """Generate feedback signals for each child from offspring performance.

        Uses mean/max/min over the entire population (current + offspring) so the
        signal is relative to the full evaluated set. Crossover pool order is
        [child0 rules, child1 rules, ..., current population]; parent indices
        below num_transferred refer to child rules.
        """
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        child_feedbacks = [np.zeros(child._top_k) for child in self.child_populations]
        child_counts = [np.zeros(child._top_k) for child in self.child_populations]

        num_offspring = len(scores) - self.population_size
        for offspring_idx in range(num_offspring):
            offspring_score = float(scores[self.population_size + offspring_idx])
            parent1 = self.parent_rule_indices[2 * offspring_idx]
            parent2 = self.parent_rule_indices[2 * offspring_idx + 1]
            for parent_idx in (parent1, parent2):
                if parent_idx < self._transfer_size:
                    child_idx, local_idx = self._get_child_and_local_index(parent_idx)
                    if offspring_score == mean_score:
                        signal = 0.0
                    elif offspring_score > mean_score:
                        denom = (
                            max_score - mean_score if max_score > mean_score else 1.0
                        )
                        signal = min(
                            1.0,
                            max(0.0, (offspring_score - mean_score) / denom),
                        )
                    else:
                        denom = (
                            mean_score - min_score if mean_score > min_score else 1.0
                        )
                        signal = max(
                            -1.0,
                            min(0.0, (offspring_score - mean_score) / denom),
                        )
                    child_feedbacks[child_idx][local_idx] += signal
                    child_counts[child_idx][local_idx] += 1

        for child in self.child_populations:
            mask = child_counts[child] > 0
            child_feedbacks[child][mask] /= child_counts[child][mask]
        return child_feedbacks

    def _new_generation(
        self, scores: ndarray, children_metrics: List[StepMetrics]
    ) -> StepMetrics:
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
            children_metrics=children_metrics,
        )

        if regenerated:
            self.population = self.population_generator.generate()
            self.best_score = -float("inf")
            self.best_not_improved_epochs = 0
        else:
            self.population, selected_scores = self.selection.select(
                self.population, scores, self.population_size
            )
            # Non-root populations need reordering so top-K rules are at the front
            # for transfer to parent population during the next forward pass.
            if self.current_depth > 0:
                sorted_indices = np.argsort(-selected_scores, self._top_k)
                self.population = [self.population[i] for i in sorted_indices]
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

        Note:
            TODO: we should also support batched evaluation or free-threaded evaluation.
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

        Args:
            data (ndarray): Validation/test data (2D boolean array).
            labels (ndarray): Validation/test labels (1D integer array).
            score_fn (Callable | None): Optional; uses original score_fn if None.
                Note: Uses the original (non-optimized) scorer by default since
                the optimized scorer has sample_weight bound to training data.
                Default: `None`.
            all_time_best (bool): If True, evaluate all-time best rule; else current run's best.
                Default: `False`.

        Returns:
            ValidateBestMetrics: best (float) and best_rule (Rule).

        Raises:
            RuntimeError: If no best rule is available (run at least one step first).
        """
        # TODO: Add doctests here.
        if self.real_best_rule is None or self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        best_rule = self.real_best_rule if all_time_best else self.best_rule
        fn = self._original_score_fn if score_fn is None else score_fn
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
            score_fn (Callable | None): Optional; uses original score_fn if None.
                Note: Uses the original (non-optimized) scorer by default since
                the optimized scorer has sample_weight bound to training data.
                Default: `None`.
            all_time_best (bool): If True, evaluate all-time best rule; else current run's best.
                Default: `False`.

        Returns:
            ValidatePopulationMetrics: best, best_rule, and population_scores.

        Raises:
            RuntimeError: If no best rule is available (run at least one step first).
        """
        # TODO: Add doctests here.
        if self.real_best_rule is None or self.best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        best_rule = self.real_best_rule if all_time_best else self.best_rule
        fn = self._original_score_fn if score_fn is None else score_fn
        best_score = float(fn(best_rule.evaluate(data), labels))
        population_scores = self._evaluate_population(data, labels, fn)
        return ValidatePopulationMetrics(
            best=best_score,
            best_rule=best_rule,
            population_scores=population_scores,
        )
