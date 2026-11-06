"""
Express the models of DT, RIPPER, and BoolXAI as hgp-lib rules, so that their size and
number of literals are counted exactly as hgp-lib counts its own rules.

Importing this module only defines the conversions. Running it checks them:

    python scripts/rule_conversion.py                     # hgp-lib, DT, and RIPPER tests
    .venv/boolxai/bin/python scripts/rule_conversion.py   # also the BoolXAI tests

The DT and RIPPER tests need scikit-learn and wittgenstein; the BoolXAI tests need
boolxai and are skipped when it is not installed.
"""

import sys
import traceback
import warnings

import numpy as np

from hgp_lib.rules import And, Literal, Or, Rule


def rule_stats(rule: Rule | None) -> tuple[int, int]:
    """
    Size and number of literals of a rule, counted as in hgp-lib.

    The size is ``len(rule)``, the number of nodes (operators and literals). The
    literals are the leaves, each testing one binary feature. ``None`` stands for a
    constant prediction (an empty RIPPER rule set, a single-leaf tree, or a trivial
    BoolXAI rule) and counts as 0 for both.
    """
    if rule is None:
        return 0, 0
    return len(rule), sum(isinstance(node, Literal) for node in rule.flatten())


def _combine(operator: type[Rule], subrules: list[Rule]) -> Rule | None:
    """Join subrules under an operator; a single subrule is returned unchanged."""
    if not subrules:
        return None
    if len(subrules) == 1:
        return subrules[0]
    return operator(subrules=subrules, copy_subrules=False)


def dt_literals(model) -> list[tuple[int, float]]:
    """
    The test behind each literal of ``dt_to_rule(model)``.

    Literal ``i`` is true where ``X[:, feature] <= threshold`` for the ``i``-th
    ``(feature, threshold)`` pair. A test used by several split nodes is one literal.
    """
    tree = model.tree_
    literal_ids: dict[tuple[int, float], int] = {}
    for node in range(tree.node_count):
        if tree.children_left[node] != -1:
            key = (int(tree.feature[node]), float(tree.threshold[node]))
            literal_ids.setdefault(key, len(literal_ids))
    return list(literal_ids)


def dt_to_rule(model) -> Rule | None:
    """
    The positive-class rule of a fitted ``DecisionTreeClassifier`` with boolean labels.

    Every leaf that predicts the positive class becomes the conjunction of the split
    tests on its path, and the rule is the disjunction of these conjunctions. Going
    left at a split is its literal (see ``dt_literals``); going right is its negation.
    A tree that predicts a constant gives ``None``.
    """
    if True not in model.classes_:
        return None
    positive = list(model.classes_).index(True)
    tree = model.tree_
    literal_ids = {key: i for i, key in enumerate(dt_literals(model))}

    paths = []
    stack = [(0, [])]
    while stack:
        node, conditions = stack.pop()
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            if np.argmax(tree.value[node][0]) == positive:
                if not conditions:
                    return None  # The tree is a single leaf: a constant prediction.
                paths.append(conditions)
            continue
        literal_id = literal_ids[(int(tree.feature[node]), float(tree.threshold[node]))]
        stack.append((left, [*conditions, (literal_id, False)]))
        stack.append((right, [*conditions, (literal_id, True)]))

    terms = [
        _combine(And, [Literal(value=i, negated=neg) for i, neg in conditions])
        for conditions in paths
    ]
    return _combine(Or, terms)


def _ripper_literal(cond) -> tuple[tuple[object, object], bool]:
    """The literal test ``(feature, value)`` of a RIPPER condition, and its negation."""
    if isinstance(cond.val, (bool, np.bool_)):
        # On a binarized column, column=False is the negation of column=True.
        return (cond.feature, True), not bool(cond.val)
    return (cond.feature, cond.val), False


def ripper_literals(model) -> list[tuple[object, object]]:
    """
    The test behind each literal of ``ripper_to_rule(model)``.

    Literal ``i`` is true where ``column == value`` for the ``i``-th
    ``(column, value)`` pair, on the data as RIPPER sees it (after its own binning).
    """
    literal_ids: dict[tuple[object, object], int] = {}
    for ripper_rule in model.ruleset_.rules:
        for cond in ripper_rule.conds:
            literal_ids.setdefault(_ripper_literal(cond)[0], len(literal_ids))
    return list(literal_ids)


def ripper_to_rule(model) -> Rule | None:
    """
    The rule set of a fitted ``wittgenstein.RIPPER`` model, as an hgp-lib rule.

    The rule set is a disjunction of conjunctions, and every condition is one literal
    (see ``ripper_literals``). A condition ``column=False`` on a binarized column is the
    negated literal of ``column=True``. An empty rule set predicts the negative class
    everywhere and gives ``None``.
    """
    literal_ids = {key: i for i, key in enumerate(ripper_literals(model))}
    terms = []
    for ripper_rule in model.ruleset_.rules:
        literals = []
        for cond in ripper_rule.conds:
            key, negated = _ripper_literal(cond)
            literals.append(Literal(value=literal_ids[key], negated=negated))
        term = _combine(And, literals)
        if term is not None:
            terms.append(term)
    return _combine(Or, terms)


def boolxai_to_rule(boolxai_rule) -> Rule | None:
    """
    A BoolXAI rule as an hgp-lib rule with the same tree.

    Literal indices are the column indices of the data BoolXAI was fitted on. ``And``
    and ``Or`` map to their hgp-lib operators. ``AtLeast``, ``AtMost``, and ``Choose``
    have no hgp-lib counterpart and become generic operator nodes, which count the same
    way but cannot be evaluated by hgp-lib. Trivial rules (``Zero``, ``One``) are
    constants: they give ``None`` and count as 0, as in BoolXAI's ``complexity()``.
    """
    name = type(boolxai_rule).__name__
    if name == "Literal":
        return Literal(value=int(boolxai_rule.index), negated=boolxai_rule.negated)
    if not boolxai_rule.subrules:
        return None
    subrules = [boolxai_to_rule(subrule) for subrule in boolxai_rule.subrules]
    subrules = [subrule for subrule in subrules if subrule is not None]
    if not subrules:
        return None
    operator = {"And": And, "Or": Or}.get(name, Rule)
    return operator(
        subrules=subrules, negated=boolxai_rule.negated, copy_subrules=False
    )


# --------------------------------------------------------------------------------------
# Tests. They run only when this file is executed, not when it is imported.
# --------------------------------------------------------------------------------------


class SkipTest(Exception):
    """Raised by a test whose optional dependency is not installed."""


def _require(module: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise SkipTest(f"{module} is not installed in this environment")


def _predict(rule: Rule | None, matrix: np.ndarray) -> np.ndarray:
    """hgp-lib's prediction of a rule; ``None`` predicts the negative class."""
    from hgp_lib.evaluation import predict

    if rule is None:
        return np.zeros(len(matrix), dtype=bool)
    return np.asarray(predict(rule, np.ascontiguousarray(matrix, dtype=bool)))


def _dnf_stats(term_lengths: list[int]) -> tuple[int, int]:
    """Expected size and literals of an Or of Ands with the given numbers of literals."""
    literals = sum(term_lengths)
    ands = sum(length > 1 for length in term_lengths)
    return literals + ands + (len(term_lengths) > 1), literals


def _assert_dnf(rule: Rule) -> None:
    """The rule is a literal, an And of literals, or an Or of those."""
    terms = rule.subrules if isinstance(rule, Or) else [rule]
    for term in terms:
        assert isinstance(term, (And, Literal)), f"unexpected term {term!r}"
        if isinstance(term, And):
            assert all(isinstance(s, Literal) for s in term.subrules), repr(term)


def _real_datasets():
    """A few small datasets: (name, features as DataFrame, boolean labels)."""
    import pandas as pd
    from sklearn.datasets import load_breast_cancer, make_classification, make_moons

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    yield "breast_cancer", X, y.to_numpy() == 0

    X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
    yield "moons", pd.DataFrame(X, columns=["a", "b"]), y == 1

    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=4, weights=[0.7], random_state=0
    )
    yield "classification", pd.DataFrame(X).add_prefix("f"), y == 1

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 4, size=(400, 5))).add_prefix("c")
    y = ((X["c0"] == 1) & (X["c1"] != 2)) | (X["c2"] == 3)
    flip = rng.random(400) < 0.05
    yield "categorical", X, (y.to_numpy() ^ flip)


# ---- rule_stats ----------------------------------------------------------------------


def test_rule_stats_counts_nodes_and_literals():
    rule = And(
        [
            Literal(value=0),
            Or([Literal(value=1, negated=True), Literal(value=2)]),
            Literal(value=3),
        ]
    )
    assert rule_stats(rule) == (6, 4)
    assert rule_stats(rule)[0] == len(rule)
    assert rule_stats(Literal(value=5, negated=True)) == (1, 1)
    assert rule_stats(None) == (0, 0)
    # A negated operator is one node, like a negated literal.
    assert rule_stats(Or([Literal(value=0), Literal(value=1)], negated=True)) == (3, 2)


# ---- Decision tree -------------------------------------------------------------------


def _dt_matrix(model, X) -> np.ndarray:
    """The literal matrix of ``dt_literals``; sklearn compares in float32."""
    values = np.asarray(X, dtype=np.float32)
    tests = dt_literals(model)
    matrix = np.zeros((len(values), max(len(tests), 1)), dtype=bool)
    for i, (feature, threshold) in enumerate(tests):
        matrix[:, i] = values[:, feature] <= threshold
    return matrix


def _assert_dt_matches(model, X) -> Rule | None:
    rule = dt_to_rule(model)
    ours = _predict(rule, _dt_matrix(model, X))
    theirs = model.predict(X).astype(bool)
    assert np.array_equal(ours, theirs), f"{(ours != theirs).sum()} predictions differ"
    return rule


def _positive_leaf_depths(model) -> list[int]:
    tree = model.tree_
    depths = []
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if tree.children_left[node] == -1:
            if model.classes_[np.argmax(tree.value[node][0])]:
                depths.append(depth)
            continue
        stack.append((tree.children_left[node], depth + 1))
        stack.append((tree.children_right[node], depth + 1))
    return depths


def test_dt_conjunction():
    from sklearn.tree import DecisionTreeClassifier

    # y = (x0 == 0) and (x1 == 1): one positive leaf at depth 2.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10, dtype=float)
    y = (X[:, 0] == 0) & (X[:, 1] == 1)
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    rule = _assert_dt_matches(model, X)
    assert isinstance(rule, And) and len(rule.subrules) == 2, repr(rule)
    assert rule_stats(rule) == (3, 2)


def test_dt_xor_reuses_tests():
    from sklearn.tree import DecisionTreeClassifier

    # y = x0 xor x1: two positive leaves at depth 2; both x1 splits use the same test.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10, dtype=float)
    y = (X[:, 0] == 1) != (X[:, 1] == 1)
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    rule = _assert_dt_matches(model, X)
    assert len(dt_literals(model)) == 2, dt_literals(model)
    assert isinstance(rule, Or) and len(rule.subrules) == 2, repr(rule)
    assert rule_stats(rule) == (7, 4)
    # Each conjunction tests both features, one of them negated.
    for term in rule.subrules:
        assert sorted(s.value for s in term.subrules) == [0, 1], repr(term)
        assert [s.negated for s in term.subrules].count(True) == 1, repr(term)


def test_dt_single_split():
    from sklearn.tree import DecisionTreeClassifier

    X = np.arange(20, dtype=float).reshape(-1, 1)
    model = DecisionTreeClassifier(random_state=0).fit(X, X[:, 0] >= 7)
    rule = _assert_dt_matches(model, X)
    # Positive is the right branch of x <= 6.5, so the rule is the negated literal.
    assert dt_literals(model) == [(0, 6.5)]
    assert isinstance(rule, Literal) and rule.negated, repr(rule)
    assert rule_stats(rule) == (1, 1)


def test_dt_constant_trees():
    from sklearn.tree import DecisionTreeClassifier

    X = np.arange(10, dtype=float).reshape(-1, 1)
    for labels in (np.ones(10, dtype=bool), np.zeros(10, dtype=bool)):
        model = DecisionTreeClassifier(random_state=0).fit(X, labels)
        assert dt_to_rule(model) is None
        assert rule_stats(dt_to_rule(model)) == (0, 0)
    # A tree whose leaves all predict the negative class is also constant.
    y = np.array([False] * 8 + [True, False])
    model = DecisionTreeClassifier(random_state=0, max_depth=1).fit(X, y)
    assert not model.predict(X).any()
    assert dt_to_rule(model) is None


def test_dt_matches_predictions_on_real_data():
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    for name, X, y in _real_datasets():
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )
        for max_leaf_nodes in (None, 2, 5, 20):
            model = DecisionTreeClassifier(
                random_state=0, max_leaf_nodes=max_leaf_nodes
            ).fit(X_train, y_train)
            try:
                rule = _assert_dt_matches(model, X_train)
                _assert_dt_matches(model, X_test)  # Also on rows the tree never saw.
                if rule is not None:
                    _assert_dnf(rule)
                assert rule_stats(rule) == _dnf_stats(_positive_leaf_depths(model))
            except AssertionError as e:
                raise AssertionError(
                    f"{name}, max_leaf_nodes={max_leaf_nodes}: {e}"
                ) from e


# ---- RIPPER --------------------------------------------------------------------------


def _ruleset_model(rules):
    """A stand-in for a fitted RIPPER model with a hand-written rule set."""
    _require("wittgenstein")
    from types import SimpleNamespace

    from wittgenstein.base import Cond, Ruleset
    from wittgenstein.base import Rule as RipperRule

    ruleset = Ruleset([RipperRule([Cond(f, v) for f, v in conds]) for conds in rules])
    return SimpleNamespace(ruleset_=ruleset)


def test_ripper_handwritten_ruleset():
    model = _ruleset_model([[("a", 1), ("b", "x-y")], [("c", 3)]])
    rule = ripper_to_rule(model)
    assert repr(rule) == "Or(And(0, 1), 2)", repr(rule)
    assert ripper_literals(model) == [("a", 1), ("b", "x-y"), ("c", 3)]
    assert rule_stats(rule) == (5, 3)


def test_ripper_single_condition():
    rule = ripper_to_rule(_ruleset_model([[("a", 1)]]))
    assert isinstance(rule, Literal) and not rule.negated
    assert rule_stats(rule) == (1, 1)
    rule = ripper_to_rule(_ruleset_model([[("a", 1), ("b", 2)]]))
    assert repr(rule) == "And(0, 1)" and rule_stats(rule) == (3, 2)


def test_ripper_boolean_conditions_negate():
    model = _ruleset_model([[("x", np.True_), ("y", False)], [("x", np.False_)]])
    rule = ripper_to_rule(model)
    # x=False is the negation of literal x, not a new literal.
    assert ripper_literals(model) == [("x", True), ("y", True)]
    assert repr(rule) == "Or(And(0, ~1), ~0)", repr(rule)
    assert rule_stats(rule) == (5, 3)


def test_ripper_empty_ruleset():
    assert ripper_to_rule(_ruleset_model([])) is None
    import contextlib
    import io

    import wittgenstein as lw

    X = np.arange(20, dtype=float).reshape(-1, 2)
    model = lw.RIPPER(random_state=0)
    # wittgenstein prints its "no positive samples" warning to stderr directly.
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        model.fit(X, np.zeros(10, dtype=bool), pos_class=True)
    assert model.ruleset_.rules == []
    assert rule_stats(ripper_to_rule(model)) == (0, 0)


def _ripper_matrix(model, X) -> np.ndarray:
    """The literal matrix of ``ripper_literals`` on X after RIPPER's own preprocessing."""
    from wittgenstein import preprocess

    X_df = preprocess.preprocess_prediction_data(
        {
            "X": X,
            "class_feat": model.class_feat,
            "pos_class": model.pos_class,
            "bin_transformer_": model.bin_transformer_,
            "user_requested_feature_names": None,
            "selected_features_": model.selected_features_,
            "trainset_features_": model.trainset_features_,
            "verbosity": 0,
        }
    )
    tests = ripper_literals(model)
    matrix = np.zeros((len(X_df), max(len(tests), 1)), dtype=bool)
    for i, (feature, value) in enumerate(tests):
        matrix[:, i] = (X_df[feature] == value).to_numpy()
    return matrix


def test_ripper_matches_predictions_on_real_data():
    _require("wittgenstein")
    import wittgenstein as lw
    from sklearn.model_selection import train_test_split

    from hgp_lib.preprocessing import StandardBinarizer

    for name, X, y in _real_datasets():
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )
        binarizer = StandardBinarizer(num_bins=5)
        binarizer.progress_bar = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B_train = binarizer.fit_transform(X_train, y_train)
            B_test = binarizer.transform(X_test)
        for variant, train, test in (
            ("raw", X_train, X_test),
            ("binarized", B_train, B_test),
        ):
            model = lw.RIPPER(random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train, y_train, pos_class=True)
            rule = ripper_to_rule(model)
            try:
                assert model.ruleset_.rules, "expected a non-empty rule set"
                for data in (train, test):
                    ours = _predict(rule, _ripper_matrix(model, data))
                    theirs = np.asarray(model.predict(data), dtype=bool)
                    assert np.array_equal(ours, theirs), (
                        f"{(ours != theirs).sum()} predictions differ"
                    )
                _assert_dnf(rule)
                lengths = [len(r.conds) for r in model.ruleset_.rules]
                assert rule_stats(rule) == _dnf_stats(lengths)
            except AssertionError as e:
                raise AssertionError(f"{name} ({variant}): {e}") from e


# ---- BoolXAI -------------------------------------------------------------------------


def _require_boolxai():
    _require("boolxai")


def _boolxai_nodes(boolxai_rule) -> list:
    return [boolxai_rule] + [
        n for s in boolxai_rule.subrules for n in _boolxai_nodes(s)
    ]


def _assert_same_tree(boolxai_rule, rule: Rule) -> None:
    """Same node kinds, negations, literal indices, and children, recursively."""
    name = type(boolxai_rule).__name__
    if name == "Literal":
        assert isinstance(rule, Literal), f"{boolxai_rule} -> {rule!r}"
        assert (rule.value, rule.negated) == (boolxai_rule.index, boolxai_rule.negated)
        return
    expected = {"And": And, "Or": Or}.get(name, Rule)
    assert type(rule) is expected, f"{boolxai_rule} -> {rule!r}"
    assert rule.negated == boolxai_rule.negated, f"{boolxai_rule} -> {rule!r}"
    children = [s for s in boolxai_rule.subrules if s.complexity() > 0]
    assert len(rule.subrules) == len(children), f"{boolxai_rule} -> {rule!r}"
    for boolxai_child, child in zip(children, rule.subrules):
        _assert_same_tree(boolxai_child, child)


def _assert_boolxai_matches(boolxai_rule, X: np.ndarray) -> Rule | None:
    rule = boolxai_to_rule(boolxai_rule)
    n_literals = sum(
        type(n).__name__ == "Literal" for n in _boolxai_nodes(boolxai_rule)
    )
    assert rule_stats(rule) == (boolxai_rule.complexity(), n_literals), (
        f"{boolxai_rule}: {rule_stats(rule)} vs complexity {boolxai_rule.complexity()}"
    )
    if rule is None:
        return None
    _assert_same_tree(boolxai_rule, rule)
    names = {type(n).__name__ for n in _boolxai_nodes(boolxai_rule)}
    if names <= {"And", "Or", "Literal"}:
        theirs = np.asarray(boolxai_rule.evaluate(X), dtype=bool)
        assert np.array_equal(_predict(rule, X), theirs), (
            f"{boolxai_rule}: predictions differ"
        )
    return rule


def test_boolxai_handwritten_rules():
    _require_boolxai()
    from boolxai import Literal as BLiteral
    from boolxai import Operator

    X = np.random.default_rng(0).random((200, 5)) > 0.5
    rule = _assert_boolxai_matches(
        Operator.And(
            [BLiteral(0), Operator.Or([BLiteral(1, negated=True), BLiteral(2)])]
        ),
        X,
    )
    assert repr(rule) == "And(0, Or(~1, 2))" and rule_stats(rule) == (5, 3)

    rule = _assert_boolxai_matches(
        Operator.Or(
            [BLiteral(3), Operator.And([BLiteral(0), BLiteral(4)])], negated=True
        ),
        X,
    )
    assert repr(rule) == "~Or(3, And(0, 4))" and rule_stats(rule) == (5, 3)

    rule = _assert_boolxai_matches(BLiteral(2, negated=True), X)
    assert repr(rule) == "~2" and rule_stats(rule) == (1, 1)


def test_boolxai_parametrized_operators():
    _require_boolxai()
    from boolxai import Literal as BLiteral
    from boolxai import Operator

    X = np.random.default_rng(0).random((50, 5)) > 0.5
    for operator in (Operator.AtLeast, Operator.AtMost, Operator.Choose):
        boolxai_rule = operator(
            subrules=[
                BLiteral(0),
                BLiteral(1, negated=True),
                Operator.Or([BLiteral(2), BLiteral(3)]),
            ],
            param=1,
            negated=True,
        )
        rule = _assert_boolxai_matches(boolxai_rule, X)
        assert type(rule) is Rule and rule.negated
        assert rule_stats(rule) == (6, 4)


def test_boolxai_trivial_rules():
    _require_boolxai()
    from boolxai import One, Zero

    for trivial in (Zero(), One()):
        assert boolxai_to_rule(trivial) is None
        assert (
            rule_stats(boolxai_to_rule(trivial)) == (0, 0) == (trivial.complexity(), 0)
        )


def test_boolxai_fitted_rules():
    _require_boolxai()
    from boolxai import Operator
    from boolxai.classifiers import RuleClassifier
    from sklearn.preprocessing import KBinsDiscretizer

    rng = np.random.default_rng(0)
    B = rng.random((400, 8)) > 0.5
    datasets = [("synthetic", B, (B[:, 0] & ~B[:, 1]) | B[:, 2])]
    for name, X, y in _real_datasets():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bins = KBinsDiscretizer(
                n_bins=5, encode="onehot-dense", strategy="quantile"
            )
            datasets.append((name, bins.fit_transform(X).astype(bool), y))

    operator_sets = {
        "all operators": None,
        "And/Or only": [Operator.And, Operator.Or],
    }
    checked = 0
    for name, X, y in datasets:
        for label, operators in operator_sets.items():
            for seed in range(2):
                kwargs = {} if operators is None else {"operators": operators}
                model = RuleClassifier(
                    random_state=seed,
                    num_jobs=1,
                    num_iterations=100,
                    num_starts=2,
                    **kwargs,
                )
                model.fit(X, y)
                try:
                    _assert_boolxai_matches(model.best_rule_, X)
                    if (
                        operators is not None
                        and boolxai_to_rule(model.best_rule_) is not None
                    ):
                        ours = _predict(boolxai_to_rule(model.best_rule_), X)
                        theirs = np.asarray(model.predict(X), dtype=bool)
                        assert np.array_equal(ours, theirs), (
                            "classifier predictions differ"
                        )
                except AssertionError as e:
                    raise AssertionError(f"{name}, {label}, seed {seed}: {e}") from e
                checked += 1
    assert checked == len(datasets) * len(operator_sets) * 2


def _run_tests() -> int:
    tests = [
        (name, test)
        for name, test in globals().items()
        if name.startswith("test_") and callable(test)
    ]
    failed = skipped = 0
    for name, test in tests:
        try:
            test()
        except SkipTest as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
        else:
            print(f"PASS  {name}")
    passed = len(tests) - failed - skipped
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_tests() else 0)
