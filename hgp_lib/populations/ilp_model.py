"""ILP model for optimal literal selection in boolean rules.

Formulates and solves a Mixed-Integer Program that selects the best subset
of literals to combine with AND or OR to maximise accuracy on a given
boolean dataset.  Solved with HiGHS via Pyomo.

This module is data-in / data-out and has no dependency on the Rule class
hierarchy — the translation from ILP solution to Rule objects lives in the
strategy layer (``strategies.ILPStrategy``).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyomo.environ as pyo


def _get_solver():
    """Return a HiGHS solver, trying ``appsi_highs`` first then ``highs``."""
    for name in ("appsi_highs", "highs"):
        solver = pyo.SolverFactory(name)
        if solver.available():
            return solver
    raise RuntimeError(
        "No HiGHS solver found. Install highspy: pip install highspy"
    )


@dataclass
class ILPSolution:
    """Result returned by :func:`solve_best_rule_ilp`.

    Attributes:
        selected_features: Indices into the *subsampled* feature space of the
            literals that were selected.
        negations: Boolean array, same length as ``selected_features``.
            ``True`` means the literal is negated.
        use_and: Whether the solution is an AND rule (``True``) or OR rule.
        feasible: Whether the solver found a feasible solution.
    """

    selected_features: np.ndarray
    negations: np.ndarray
    use_and: bool
    feasible: bool


def solve_best_rule_ilp(
    data: np.ndarray,
    labels: np.ndarray,
    use_and: bool,
    min_literals: int = 2,
    max_literals: int = 5,
    time_limit: float = 5.0,
) -> ILPSolution:
    """Build and solve the ILP for the best AND or OR rule.

    Decision variables
    ------------------
    For each feature *j* in the data (``F`` features total) we create two
    binary selection variables:

    * ``x[2j]``   — include literal ``j`` (positive)
    * ``x[2j+1]`` — include literal ``~j`` (negated)

    At most one of the two can be active per feature.

    Prediction variables
    --------------------
    * ``y[i]`` ∈ {0, 1} — predicted label for instance *i*.

    **OR rule** (``ŷ_i = 1`` iff *any* selected literal is true for *i*):

        For every candidate literal *c* that is true for instance *i*:
            ``y[i] >= x[c]``

        Upper bound:
            ``y[i] <= Σ_c  x[c] · d[i,c]``

    **AND rule** (``ŷ_i = 1`` iff *all* selected literals are true for *i*):

        For every candidate literal *c* that is false for instance *i*:
            ``y[i] <= 1 - x[c]``

        Lower bound:
            ``y[i] >= 1 - Σ_c  x[c] · (1 - d[i,c])``

    Objective
    ---------
    Maximise accuracy: ``Σ_i  match[i]`` where ``match[i] = 1`` iff
    ``y[i] == label[i]``.  Linearised via ``m[i] = y[i]`` when ``label=1``
    and ``m[i] = 1 - y[i]`` when ``label=0``.

    Parameters
    ----------
    data : np.ndarray
        Boolean data, shape ``(N, F)``.
    labels : np.ndarray
        Binary labels, shape ``(N,)``.
    use_and : bool
        ``True`` → AND rule, ``False`` → OR rule.
    min_literals : int
        Minimum number of literals in the rule.
    max_literals : int
        Maximum number of literals in the rule.
    time_limit : float
        Solver wall-clock limit in seconds.

    Returns
    -------
    ILPSolution
        The solution containing selected feature indices, negation flags,
        operator type, and feasibility status.
    """
    n_instances, n_features = data.shape
    n_candidates = 2 * n_features

    # Truth table for all 2·F candidate literals.
    # Candidate 2j   = feature j positive
    # Candidate 2j+1 = feature j negated
    D = np.empty((n_instances, n_candidates), dtype=bool)
    D[:, 0::2] = data
    D[:, 1::2] = ~data

    model = pyo.ConcreteModel()

    model.I = pyo.RangeSet(0, n_instances - 1)
    model.C = pyo.RangeSet(0, n_candidates - 1)

    # --- decision variables ---
    model.x = pyo.Var(model.C, domain=pyo.Binary)   # select candidate c
    model.y = pyo.Var(model.I, domain=pyo.Binary)   # prediction for instance i
    model.m = pyo.Var(model.I, domain=pyo.Binary)   # match indicator

    # --- at most one polarity per feature ---
    model.one_polarity = pyo.ConstraintList()
    for j in range(n_features):
        model.one_polarity.add(model.x[2 * j] + model.x[2 * j + 1] <= 1)

    # --- cardinality bounds ---
    total_selected = sum(model.x[c] for c in model.C)
    model.min_card = pyo.Constraint(expr=total_selected >= min_literals)
    model.max_card = pyo.Constraint(expr=total_selected <= max_literals)

    # --- link y to x via operator semantics ---
    model.link = pyo.ConstraintList()

    if not use_and:
        # OR rule
        for i in range(n_instances):
            model.link.add(
                model.y[i]
                <= sum(int(D[i, c]) * model.x[c] for c in range(n_candidates))
            )
            for c in range(n_candidates):
                if D[i, c]:
                    model.link.add(model.y[i] >= model.x[c])
    else:
        # AND rule
        for i in range(n_instances):
            model.link.add(
                model.y[i]
                >= 1
                - sum(int(not D[i, c]) * model.x[c] for c in range(n_candidates))
            )
            for c in range(n_candidates):
                if not D[i, c]:
                    model.link.add(model.y[i] <= 1 - model.x[c])

    # --- match indicator: m[i] = 1 iff y[i] == label[i] ---
    model.match_link = pyo.ConstraintList()
    for i in range(n_instances):
        li = int(labels[i])
        if li == 1:
            model.match_link.add(model.m[i] <= model.y[i])
            model.match_link.add(model.m[i] >= model.y[i])
        else:
            model.match_link.add(model.m[i] <= 1 - model.y[i])
            model.match_link.add(model.m[i] >= 1 - model.y[i])

    # --- objective: maximise accuracy ---
    model.obj = pyo.Objective(
        expr=sum(model.m[i] for i in model.I), sense=pyo.maximize
    )

    # --- solve ---
    solver = _get_solver()
    solver.options["time_limit"] = time_limit
    solver.options["output_flag"] = False
    result = solver.solve(model, tee=False, load_solutions=False)

    # --- extract solution ---
    feasible = result.solver.termination_condition in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )

    if not feasible:
        return ILPSolution(
            selected_features=np.array([], dtype=int),
            negations=np.array([], dtype=bool),
            use_and=use_and,
            feasible=False,
        )

    # Load the solution into the model variables now that we know it's feasible
    model.solutions.load_from(result)

    selected_features = []
    negations = []
    for j in range(n_features):
        if pyo.value(model.x[2 * j]) > 0.5:
            selected_features.append(j)
            negations.append(False)
        elif pyo.value(model.x[2 * j + 1]) > 0.5:
            selected_features.append(j)
            negations.append(True)

    return ILPSolution(
        selected_features=np.array(selected_features, dtype=int),
        negations=np.array(negations, dtype=bool),
        use_and=use_and,
        feasible=len(selected_features) >= min_literals,
    )
