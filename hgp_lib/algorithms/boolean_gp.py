from dataclasses import replace
from typing import Callable, List

import numpy as np
from numpy import ndarray

from ..configs import BooleanGPConfig, validate_gp_config
from ..crossover import CrossoverExecutor
from ..metrics import GenerationMetrics
from ..rules import Rule
from ..selections import TournamentSelection
from ..utils.metrics import optimize_scorer_for_data


class BooleanGP:
    """
    Boolean Genetic Programming algorithm for evolving rule-based classifiers.

    This class implements a genetic programming algorithm that evolves a population of
    boolean rules to optimize a fitness function. Each generation applies crossover and
    mutation operations, evaluates the population, and selects the best individuals.

    The algorithm tracks the current best rule (best in the current run). When enabled,
    regeneration resets the population if no improvement is observed for a specified
    number of epochs.

    Training data and labels are provided via `BooleanGPConfig`. The number of features
    (`num_features`) is derived from the data shape and passed to the configured
    `population_factory` and `mutation_factory` for runtime construction of the
    `PopulationGenerator` and `MutationExecutor`.

    Args:
        config (BooleanGPConfig): Configuration containing `train_data`,
            `train_labels`, `score_fn`, `population_factory`,
            `mutation_factory`, and optional components.

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
        >>> gen_metrics = gp.step()
    """

    def __init__(self, config: BooleanGPConfig, current_depth: int = 0):
        validate_gp_config(config)

        train_data = config.train_data
        train_labels = config.train_labels
        # TODO: We should add in documentation that our score_fn follows the sklearn
        #  standard of (predictions, labels) and sample_weight support is recommended for optimization.
        # Careful! the sklearn pattern is labels, predictions!

        score_fn = config.score_fn
        self._original_score_fn = score_fn

        if config.optimize_scorer:
            score_fn, train_data, train_labels = optimize_scorer_for_data(
                config.score_fn, config.train_data, config.train_labels
            )

        self.score_fn = score_fn
        self.complexity_penalty = config.complexity_penalty
        self.train_data = train_data
        self.train_labels = train_labels

        self.current_depth = current_depth
        num_features = train_data.shape[1]

        population_generator = config.population_factory.create(
            num_features, score_fn, train_data, train_labels
        )
        mutation_executor = config.mutation_factory.create(
            num_features, config.check_valid
        )

        crossover_executor = config.crossover_executor
        if crossover_executor is None:
            crossover_executor = CrossoverExecutor(check_valid=config.check_valid)

        selection = config.selection
        if selection is None:
            selection = TournamentSelection()

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
        self.global_best_score = -float("inf")
        self.best_rule: Rule | None = None
        self.global_best_rule: Rule | None = None
        self.best_not_improved_epochs = 0
        self._epoch = -1

        self.config = config
        self.child_populations: List["BooleanGP"] = []
        self.feature_mapping: dict[int, int] | None = None
        self._transfer_size: int = 0
        self.parent_rule_indices: List[int] = []

        # TODO: We should always have num_child_populations > 0 if max_depth si greater than 0.
        # Check if this is true and add the check.
        if config.max_depth > current_depth and config.num_child_populations > 0:
            self._create_child_populations()

    def _create_child_populations(self) -> None:
        """Create child populations using the sampling strategy.

        Calls sample() once for all children to ensure correct overlap/partitioning
        behavior controlled by the replace parameter. Each SamplingResult in the
        returned list is used to configure one child population.
        """
        if self.config.sampling_strategy is None:
            # TODO: We should have checked this in the validate function for the config.
            raise RuntimeError(
                "Cannot create child populations without a sampling strategy"
            )

        results = self.config.sampling_strategy.sample(
            self.train_data,
            self.train_labels,
            self.config.num_child_populations,
        )

        for result in results:
            child_config = replace(
                self.config,
                train_data=result.data,
                train_labels=result.labels,
            )
            child = BooleanGP(child_config, current_depth=self.current_depth + 1)
            child.feature_mapping = result.feature_mapping

            self.child_populations.append(child)

    def step(self) -> GenerationMetrics:
        """
        Performs one training step (generation) of the genetic programming algorithm.

        Uses the training data and labels from the config. Each step applies crossover
        to create offspring, mutates the population, evaluates all rules, updates the
        best rule, and selects individuals for the next generation. If regeneration
        is enabled and no improvement has been observed for `regeneration_patience`
        epochs, the population is regenerated.

        Returns:
            GenerationMetrics: Metrics for this generation including rules, scores, etc.
        """
        # TODO: Add doctests here.
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

    def backward(self, parent_scores: ndarray | None = None) -> GenerationMetrics:
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
            GenerationMetrics: Metrics for this generation.
        """
        scores = self.evaluate_population(
            self.train_data, self.train_labels, self.score_fn
        )
        if parent_scores is not None:
            self._apply_feedback(scores, parent_scores)

        children_metrics = []

        if self.child_populations:
            child_feedbacks = self._generate_child_feedback(scores)
            for child, feedback in zip(self.child_populations, child_feedbacks):
                children_metrics.append(child.backward(feedback))

        return self._new_generation(scores, children_metrics)

    def _get_top_k_rules(self) -> List[Rule]:
        """Retrieve the top-K rules for transfer to the parent population.

        Returns the first `_top_k` rules from the population, which are expected
        to be the highest-scoring rules. After each generation (except the first),
        non-root populations sort their rules by score in descending order during
        `_new_generation()`, ensuring that indices 0 through `_top_k - 1` contain
        the best-performing rules.

        Note:
            During the first epoch (before any `backward()` call completes), the
            population has not yet been sorted by score. In this case, the returned
            rules are simply the first `_top_k` rules from the initial population,
            which are effectively random. This is acceptable since all populations
            start with randomly generated rules.

        Returns:
            List[Rule]: The top-K rules from this population, to be used in the
                parent's crossover pool during hierarchical GP.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.algorithms import BooleanGP
            >>> from hgp_lib.configs import BooleanGPConfig
            >>> from hgp_lib.populations import FeatureSamplingStrategy
            >>> from sklearn.metrics import accuracy_score
            >>> data = np.random.rand(50, 10) > 0.5
            >>> labels = np.random.randint(0, 2, 50)
            >>> config = BooleanGPConfig(
            ...     score_fn=accuracy_score,
            ...     train_data=data,
            ...     train_labels=labels,
            ...     max_depth=1,
            ...     num_child_populations=2,
            ...     sampling_strategy=FeatureSamplingStrategy(),
            ...     top_k_transfer=5,
            ... )
            >>> gp = BooleanGP(config)
            >>> child = gp.child_populations[0]
            >>> top_rules = child._get_top_k_rules()
            >>> len(top_rules)
            5
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

    def _generate_child_feedback(self, scores: ndarray) -> ndarray:
        """Generate feedback signals for each child from offspring performance.

        Each parent that came from a child population receives a signal based on
        how well the offspring it helped create performed. The signal is normalized
        relative to the population's score distribution.

        Args:
            scores (ndarray): Fitness scores for all rules in the current population,
                including both the original population and offspring from crossover.

        Returns:
            ndarray: A 2D array of shape (num_children, top_k) containing feedback
                signals for each child population's transferred rules.
        """
        num_children = len(self.child_populations)

        mean_s = float(np.mean(scores))
        min_s = float(np.min(scores))
        max_s = float(np.max(scores))

        if max_s == min_s:
            return np.zeros((num_children, self._top_k))

        # Offspring are situated after the original parent population in the scores array
        offspring_scores = scores[self.population_size :]
        above = offspring_scores >= mean_s
        signals = np.where(
            above,
            (offspring_scores - mean_s) / (max_s - mean_s),
            (offspring_scores - mean_s) / (mean_s - min_s),
        )
        child_feedbacks = np.zeros((num_children, self._top_k))
        child_counts = np.zeros((num_children, self._top_k))

        # parent_rule_indices stores TWO indices per offspring (one for each parent).
        # For offspring i, parent_rule_indices[2*i] and parent_rule_indices[2*i + 1]
        # are the indices of its two parents in the crossover pool.
        # Therefore, j // 2 maps from the parent_rule_indices position to the
        # corresponding offspring index in offspring_scores.
        for j, parent_idx in enumerate(self.parent_rule_indices):
            if parent_idx < self._transfer_size:
                # parent_idx < _transfer_size means this parent came from a child population
                # Determine which child and which local rule index within that child
                child_idx, local_idx = divmod(parent_idx, self._top_k)
                offspring_idx = j // 2  # Two parent indices per offspring
                child_feedbacks[child_idx, local_idx] += signals[offspring_idx]
                child_counts[child_idx, local_idx] += 1

        mask = child_counts > 0
        child_feedbacks[mask] /= child_counts[mask]

        return child_feedbacks

    def _compute_regularized_scores(
        self, scores: ndarray, complexities: List[int]
    ) -> ndarray:
        """Compute regularized scores with complexity penalty.

        regularized_score = score - complexity_penalty * ln(complexity)

        Args:
            scores (ndarray): Scores for the population.

        Returns:
            ndarray: Regularized scores for selection.
        """
        if self.complexity_penalty == 0:
            return scores
        return scores - self.complexity_penalty * np.log(complexities)

    def _new_generation(
        self, scores: ndarray, children_metrics: List[GenerationMetrics]
    ) -> GenerationMetrics:
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
            GenerationMetrics: Metrics about the generation step.
        """
        # TODO: Check this
        best_idx = int(np.argmax(scores))
        current_best = float(scores[best_idx])
        current_best_rule = self.population[best_idx].copy()

        self._update_best(current_best, current_best_rule)

        regenerated = False
        if (
            self.regeneration
            and self.best_not_improved_epochs >= self.regeneration_patience
        ):
            regenerated = True

        self._epoch += 1

        # Create GenerationMetrics
        complexities = [len(rule) for rule in self.population]

        metrics = GenerationMetrics.from_population(
            best_idx=best_idx,
            best_rule=current_best_rule,
            train_scores=scores.tolist(),
            complexities=complexities,
            child_population_generation_metrics=children_metrics,
        )

        if regenerated:
            self.population = self.population_generator.generate()
            self.best_score = -float("inf")
            self.best_not_improved_epochs = 0
        else:
            regularized_scores = self._compute_regularized_scores(scores, complexities)
            self.population, selected_scores = self.selection.select(
                self.population, regularized_scores, self.population_size
            )
            # Non-root populations need reordering so top-K rules are at the front
            # for transfer to parent population during the next forward pass.
            if self.current_depth > 0:  # top_k must be positive if current_depth > 0
                # TODO: We had a ValueError: kth(=50) out of bounds (50)
                # ValueError: kth(=100) out of bounds (100)
                sorted_indices = np.argpartition(-selected_scores, self._top_k)
                self.population = [self.population[i] for i in sorted_indices]

        return metrics

    def evaluate_population(
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
            TODO: we should also support batched evaluation or free-threaded evaluation if needed.
        """
        return np.array(
            [score_fn(rule.evaluate(data), labels) for rule in self.population]
        )

    def _update_best(self, current_best: float, current_best_rule: Rule):
        """
        Updates the best rule tracking based on the current generation's best.

        If the current best score is greater than or equal to the stored best score,
        updates the current run's best. Otherwise, increments the counter for epochs
        without improvement.

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
            self.best_rule = current_best_rule
            if current_best >= self.global_best_score:
                self.global_best_score = current_best
                self.global_best_rule = current_best_rule
        else:
            self.best_not_improved_epochs += 1

    def evaluate_best(
        self,
        data: ndarray,
        labels: ndarray,
        score_fn: Callable[[ndarray, ndarray], float] | None = None,
    ) -> float:
        """
        Evaluates the best rule on validation or test data.

        Args:
            data (ndarray): Validation/test data (2D boolean array).
            labels (ndarray): Validation/test labels (1D integer array).
            score_fn (Callable | None): Optional; uses original score_fn if None.
                Note: Uses the original (non-optimized) scorer by default since
                the optimized scorer has sample_weight bound to training data.
                Default: `None`.

        Returns:
            float: best_score

        Raises:
            RuntimeError: If no best rule is available (run at least one step first).
        """
        # TODO: Add doctests here.
        if self.global_best_rule is None:
            raise RuntimeError("No best rule available. Run at least one step first.")

        fn = self._original_score_fn if score_fn is None else score_fn
        return float(fn(self.global_best_rule.evaluate(data), labels))
