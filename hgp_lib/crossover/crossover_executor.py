import random
from typing import Callable, List, Sequence, Tuple

import numpy as np

from ..rules import Rule, Literal
from ..rules.utils import deep_swap, apply_feature_mapping
from ..utils.validation import validate_callable, check_isinstance


class CrossoverExecutor:
    """
    Coordinates subtree crossover operations across a collection of `Rule` trees.

    The executor randomly selects pairs of rules based on `crossover_p`, then exchanges
    subtrees between the paired parents to produce offspring. An optional validator can
    reject invalid children, with configurable retry attempts.

    Args:
        crossover_p (float, optional):
            Probability of selecting each rule for crossover. Default: `0.7`.
        crossover_strategy (str, optional):
            Strategy for pairing rules. Must be `"best"` or `"random"`. Default: `"random"`.
        check_valid (Callable[[Rule], bool] | None, optional):
            Optional validator executed after crossover. When supplied, each child must
            pass validation or the crossover is retried up to `num_tries` times. All children that
            pass validation are kept until two children pass validation. Default: `None`.
            Note: The validator is called once during initialization to verify it returns a bool.
            Stateful validators should account for this extra call.
        num_tries (int, optional):
            Maximum number of crossover attempts per pair when validation fails.
            Must be `1` when no validator is provided. Default: `1`.

    Examples:
        >>> import random
        >>> import numpy as np
        >>> from hgp_lib.crossover import CrossoverExecutor
        >>> from hgp_lib.rules import And, Or, Literal
        >>> random.seed(42); np.random.seed(42)
        >>> executor = CrossoverExecutor(crossover_p=1.0)
        >>> rules = [
        ...     And([Literal(value=0), Literal(value=1)]),
        ...     Or([Literal(value=2), Literal(value=3)])
        ... ]
        >>> children, parent_indices = executor.apply(rules, [None, None])
        >>> len(children)
        2
    """

    def __init__(
        self,
        crossover_p: float = 0.7,
        crossover_strategy: str = "random",
        check_valid: Callable[[Rule], bool] | None = None,
        num_tries: int = 1,
    ):
        self._validate_params(crossover_p, crossover_strategy, check_valid, num_tries)
        self.crossover_p: float = crossover_p
        self.crossover_strategy: str = crossover_strategy
        self.check_valid: Callable[[Rule], bool] | None = check_valid
        self.num_tries: int = num_tries

    @staticmethod
    def _validate_params(
        crossover_p: float,
        crossover_strategy: str,
        check_valid: Callable[[Rule], bool] | None,
        num_tries: int,
    ):
        check_isinstance(crossover_p, float)
        check_isinstance(crossover_strategy, str)
        check_isinstance(num_tries, int)

        if crossover_p < 0.0 or crossover_p > 1.0:
            raise ValueError(
                f"crossover_p must be a float between 0.0 and 1.0, is '{crossover_p}'"
            )

        accepted_strategies = ("best", "random")
        if crossover_strategy not in accepted_strategies:
            raise ValueError(
                f"crossover_strategy must be one of {accepted_strategies}, is '{crossover_strategy}'"
            )

        if check_valid is not None:
            error_msg = f"check_valid must be a callable that accepts a Rule and returns bool, is {type(check_valid)}"
            validate_callable(check_valid, error_msg)
            try:
                boolean = check_valid(Literal(value=0))
                if not isinstance(boolean, bool):
                    raise TypeError(error_msg)
            except Exception as e:
                raise TypeError(error_msg) from e

        if num_tries < 1:
            raise ValueError(f"num_tries must be greater than 0, is '{num_tries}'")
        if num_tries > 1 and check_valid is None:
            raise ValueError("num_tries must be 1 if check_valid is None")

    def apply(
        self, rules: List[Rule], feature_mappings: List[dict | None]
    ) -> Tuple[List[Rule], List[int]]:
        """
        Applies crossover to the provided list of rules and returns children with parent tracking.

        Rules are randomly selected for crossover based on `crossover_p`, paired
        consecutively, and their subtrees are exchanged. Before crossover, feature mappings
        are applied to translate rules from child populations (which may use different
        feature indices) into the parent's feature space.

        This method supports hierarchical GP by tracking which parent rules contributed
        to each child, enabling score propagation back to child populations.

        Args:
            rules (List[Rule]):
                The collection of parent rules that will undergo crossover. May include
                rules from both the current population and child populations.
            feature_mappings (List[dict | None]):
                A list of feature mapping dictionaries, one per rule. Each mapping translates
                feature indices from a child population's space to the parent's space.
                Use `None` for rules that don't need remapping (i.e., from the current population).

        Returns:
            Tuple[List[Rule], List[int]]: A tuple containing:
                - List[Rule]: The children produced by crossover operations.
                - List[int]: The indices of parent rules that contributed to each child.
                  For each child, both parent indices are recorded (so the list length
                  is 2x the number of children).

        Examples:
            >>> import random
            >>> import numpy as np
            >>> from hgp_lib.crossover import CrossoverExecutor
            >>> from hgp_lib.rules import And, Or, Literal
            >>> random.seed(42); np.random.seed(42)
            >>> executor = CrossoverExecutor(crossover_p=1.0)
            >>> rules = [
            ...     And([Literal(value=0), Literal(value=1)]),
            ...     Or([Literal(value=2), Literal(value=3)])
            ... ]
            >>> children, parent_indices = executor.apply(rules, [None, None])
            >>> len(children)
            2
        """
        n = len(rules)
        if n == 0:
            return [], []

        if self.crossover_strategy == "random":
            probabilities = np.random.rand(n)
        else:
            # self.crossover_strategy == "best"
            raise NotImplementedError()

        idx_sorted = np.argsort(probabilities)
        partition_point = np.argmax(probabilities[idx_sorted] > self.crossover_p)
        if partition_point == 0 and probabilities[0] < self.crossover_p:
            # partition_point is 0 if all probabilities are below the crossover probability
            partition_point = n

        if partition_point % 2 == 1:
            partition_point = (
                partition_point + 1 if partition_point + 1 < n else partition_point - 1
            )

        children = []
        parent_indices = []
        for i in range(0, partition_point, 2):
            i1 = idx_sorted[i]
            i2 = idx_sorted[i + 1]
            parent_a = apply_feature_mapping(rules[i1], feature_mappings[i1])
            parent_b = apply_feature_mapping(rules[i2], feature_mappings[i2])
            for child in self.crossover(parent_a, parent_b):
                parent_indices.append(i1)
                parent_indices.append(i2)
                children.append(child)
        return children, parent_indices

    def crossover(self, parent_a: Rule, parent_b: Rule) -> Sequence[Rule]:
        """
        Performs subtree crossover between two parent rules.

        A random node is selected from each parent, and the subtrees rooted at those
        nodes are exchanged using `deep_swap`. When a validator is provided, each child
        is validated individually and accepted children are collected until at least two pass
        validation or `num_tries` attempts are exhausted.

        Args:
            parent_a (Rule): First parent rule.
            parent_b (Rule): Second parent rule.

        Returns:
            Sequence[Rule]: Children with exchanged subtrees. Returns two children when
                no validator is provided, or up to two valid children when a validator
                is used.

        Examples:
            >>> import random
            >>> from hgp_lib.crossover import CrossoverExecutor
            >>> from hgp_lib.rules import And, Or, Literal
            >>> random.seed(0)
            >>> executor = CrossoverExecutor()
            >>> parent_a = And([Literal(value=0), Literal(value=1)])
            >>> parent_b = Or([Literal(value=2), Literal(value=3)])
            >>> child_a, child_b = executor.crossover(parent_a, parent_b)
            >>> parent_a is child_a
            False
        """
        accepted = []
        for _ in range(self.num_tries):
            child_a, child_b = parent_a.copy(), parent_b.copy()
            flat_a, flat_b = child_a.flatten(), child_b.flatten()
            node_a = random.choice(flat_a)
            node_b = random.choice(flat_b)

            deep_swap(node_a, node_b)

            if self.check_valid is None:
                return child_a, child_b

            if self.check_valid(child_a):
                accepted.append(child_a)
            if self.check_valid(child_b):
                accepted.append(child_b)
            if len(accepted) >= 2:
                break
        return accepted
