from functools import cmp_to_key

import sympy as sp
from sympy.logic.boolalg import is_cnf, is_dnf


class ThresholdSimplifier:
    """
    Simplifier for Boolean formulas whose atoms are comparisons of
    real-valued variables against numeric thresholds:

        x < c
        x <= c
        x > c
        x >= c

    combined with AND / OR / NOT.

    Internally uses explicit DNF/CNF representations and threshold-aware
    interval reasoning for:
      - contradictory conjunctions
      - redundant atoms
      - DNF absorption
      - CNF subsumption
      - tautological threshold clauses
    """

    # ------------------------------------------------------------------
    # DNF representation
    #
    # FALSE = no terms
    # TRUE  = one empty term
    # ------------------------------------------------------------------

    DNF_FALSE = frozenset()
    DNF_TRUE = frozenset({frozenset()})

    # ------------------------------------------------------------------
    # CNF representation
    #
    # TRUE  = no clauses
    # FALSE = one empty clause
    # ------------------------------------------------------------------

    CNF_TRUE = frozenset()
    CNF_FALSE = frozenset({frozenset()})

    def __init__(self, max_iterations=20):
        self.max_iterations = max_iterations
        self._dnf_cache = {}
        self._cnf_cache = {}
        self._best_cnf = None
        self._best_dnf = None

    # ==================================================================
    # Public API
    # ==================================================================

    def simplify(self, expr):
        self._dnf_cache.clear()
        self._cnf_cache.clear()
        self._best_cnf = None
        self._best_dnf = None

        expr = sp.sympify(expr)

        def simp_dnf(expr):
            dnf = self._to_dnf(expr)
            dnf_simp = self._simplify_dnf(dnf)
            key = self._expr_key(dnf_simp)
            expr_simp = self._dnf_to_expr(dnf_simp)
            if self._best_dnf is None or sp.count_ops(expr_simp) < sp.count_ops(
                self._best_dnf
            ):
                self._best_dnf = expr_simp
            return expr_simp, key

        def simp_cnf(expr):
            cnf = self._to_cnf(expr)
            cnf_simp = self._simplify_cnf(cnf)
            key = self._expr_key(cnf_simp)
            expr_simp = self._cnf_to_expr(cnf_simp)
            if self._best_cnf is None or sp.count_ops(expr_simp) < sp.count_ops(
                self._best_cnf
            ):
                self._best_cnf = expr_simp
            return expr_simp, key

        step1 = simp_cnf
        step2 = simp_dnf

        if is_cnf(expr):
            self._best_cnf = expr
        elif is_dnf(expr):
            self._best_dnf = expr
            step1, step2 = step2, step1
        else:
            expr = sp.to_nnf(expr, simplify=False)

        previous_key = None

        for i in range(self.max_iterations):
            expr_step1, _ = step1(expr)
            expr_step2, key = step2(expr_step1)

            if key == previous_key:
                return {
                    "best_cnf": self._best_cnf,
                    "best_dnf": self._best_dnf,
                }

            previous_key = key
            expr = expr_step2

        return {
            "best_cnf": self._best_cnf,
            "best_dnf": self._best_dnf,
        }

    # ==================================================================
    # Atom handling
    # ==================================================================

    @staticmethod
    def _is_atom(expr):
        return isinstance(
            expr,
            (
                sp.StrictLessThan,
                sp.LessThan,
                sp.StrictGreaterThan,
                sp.GreaterThan,
            ),
        )

    @staticmethod
    def _parts(atom):
        if not ThresholdSimplifier._is_atom(atom):
            raise TypeError(f"Not a threshold atom: {atom}")

        return atom.lhs, atom.rel_op, atom.rhs

    @staticmethod
    def _negate_atom(atom):
        lhs, op, rhs = ThresholdSimplifier._parts(atom)

        if op == "<":
            return lhs >= rhs
        if op == "<=":
            return lhs > rhs
        if op == ">":
            return lhs <= rhs
        if op == ">=":
            return lhs < rhs

        raise ValueError(op)

    # ==================================================================
    # Exact threshold comparison
    # ==================================================================

    @staticmethod
    def _cmp(a, b):
        """
        Compare two SymPy threshold values.

        Returns:
            -1 if a < b
             0 if a == b
             1 if a > b
        """
        if a == b:
            return 0

        if sp.simplify(a < b) is sp.true:
            return -1

        if sp.simplify(a > b) is sp.true:
            return 1

        raise ValueError(f"Cannot compare thresholds {a!r} and {b!r}")

    # ==================================================================
    # Interval representation
    #
    # [lower, lower_closed, upper, upper_closed]
    #
    # None means infinity.
    # ==================================================================

    @staticmethod
    def _empty_interval():
        return [None, False, None, False]

    def _set_lower(self, interval, value, closed):
        old_value = interval[0]

        if old_value is None:
            interval[0] = value
            interval[1] = closed
            return

        c = self._cmp(value, old_value)

        if c > 0:
            interval[0] = value
            interval[1] = closed

        elif c == 0:
            # Strict lower bound wins.
            interval[1] = interval[1] and closed

    def _set_upper(self, interval, value, closed):
        old_value = interval[2]

        if old_value is None:
            interval[2] = value
            interval[3] = closed
            return

        c = self._cmp(value, old_value)

        if c < 0:
            interval[2] = value
            interval[3] = closed

        elif c == 0:
            # Strict upper bound wins.
            interval[3] = interval[3] and closed

    def _valid_interval(self, interval):
        lower, lower_closed, upper, upper_closed = interval

        if lower is None or upper is None:
            return True

        c = self._cmp(lower, upper)

        if c < 0:
            return True

        if c > 0:
            return False

        # Singleton interval is valid only when both ends are closed.
        return lower_closed and upper_closed

    def _interval_from_atoms(self, atoms):
        """
        Convert a conjunction of threshold atoms into:

            variable -> interval

        Return None if the conjunction is contradictory.
        """
        result = {}

        for atom in atoms:
            variable, op, value = self._parts(atom)

            interval = result.setdefault(
                variable,
                self._empty_interval(),
            )

            if op == ">":
                self._set_lower(interval, value, False)

            elif op == ">=":
                self._set_lower(interval, value, True)

            elif op == "<":
                self._set_upper(interval, value, False)

            elif op == "<=":
                self._set_upper(interval, value, True)

            if not self._valid_interval(interval):
                return None

        return result

    # ==================================================================
    # Interval implication
    # ==================================================================

    def _interval_implies_atom(self, interval, atom):
        """
        Does the interval imply the given atom?
        """
        _, op, value = self._parts(atom)

        lower, lower_closed, upper, upper_closed = interval

        if op == ">":
            if lower is None:
                return False

            c = self._cmp(lower, value)

            if c > 0:
                return True

            if c == 0:
                return not lower_closed

            return False

        if op == ">=":
            if lower is None:
                return False

            return self._cmp(lower, value) >= 0

        if op == "<":
            if upper is None:
                return False

            c = self._cmp(upper, value)

            if c < 0:
                return True

            if c == 0:
                return not upper_closed

            return False

        if op == "<=":
            if upper is None:
                return False

            return self._cmp(upper, value) <= 0

        raise ValueError(op)

    def _term_implies_term(self, source, target):
        """
        Exact implication:

            source => target

        where both source and target are conjunctions.
        """
        intervals = self._interval_from_atoms(source)

        # Contradictory source implies everything.
        if intervals is None:
            return True

        for atom in target:
            variable, _, _ = self._parts(atom)

            interval = intervals.get(variable)

            if interval is None:
                return False

            if not self._interval_implies_atom(
                interval,
                atom,
            ):
                return False

        return True

    # ==================================================================
    # DNF term simplification
    # ==================================================================

    def _simplify_term(self, term):
        """
        Reduce a conjunction to its strongest lower/upper bounds.

        Example:

            x >= 3 AND x >= 5 AND x < 10 AND x <= 8

        becomes:

            x >= 5 AND x <= 8

        Returns None for a contradiction.
        """
        intervals = self._interval_from_atoms(term)

        if intervals is None:
            return None

        by_variable = {}

        for atom in term:
            variable, _, _ = self._parts(atom)

            by_variable.setdefault(
                variable,
                [],
            ).append(atom)

        result = set()

        for variable, atoms in by_variable.items():
            lower, lower_closed, upper, upper_closed = intervals[variable]

            # Strongest lower bound.
            if lower is not None:
                candidates = [
                    atom
                    for atom in atoms
                    if (
                        self._parts(atom)[2] == lower
                        and self._parts(atom)[1] in (">", ">=")
                    )
                ]

                if candidates:
                    desired_op = ">=" if lower_closed else ">"

                    chosen = next(
                        (
                            atom
                            for atom in candidates
                            if self._parts(atom)[1] == desired_op
                        ),
                        candidates[0],
                    )

                    result.add(chosen)

            # Strongest upper bound.
            if upper is not None:
                candidates = [
                    atom
                    for atom in atoms
                    if (
                        self._parts(atom)[2] == upper
                        and self._parts(atom)[1] in ("<", "<=")
                    )
                ]

                if candidates:
                    desired_op = "<=" if upper_closed else "<"

                    chosen = next(
                        (
                            atom
                            for atom in candidates
                            if self._parts(atom)[1] == desired_op
                        ),
                        candidates[0],
                    )

                    result.add(chosen)

        return frozenset(result)

    # ==================================================================
    # DNF absorption
    # ==================================================================

    def _absorb_dnf(self, terms):
        """
        Remove term B when:

            B => A

        because:

            A OR B = A
        """
        if len(terms) <= 1:
            return terms

        ordered = sorted(
            terms,
            key=lambda term: (
                len(term),
                tuple(sorted(map(str, term))),
            ),
        )

        kept = []

        for term in ordered:
            absorbed = False

            for other in kept:
                if self._term_implies_term(term, other):
                    absorbed = True
                    break

            if not absorbed:
                kept.append(term)

        return frozenset(kept)

    def _simplify_dnf(self, dnf):
        if dnf == self.DNF_FALSE:
            return dnf

        if self.DNF_TRUE in dnf:
            return self.DNF_TRUE

        terms = set()

        for term in dnf:
            term = self._simplify_term(term)

            if term is None:
                # Contradictory term.
                continue

            if not term:
                # Empty conjunction = TRUE.
                return self.DNF_TRUE

            terms.add(term)

        if not terms:
            return self.DNF_FALSE

        return self._absorb_dnf(frozenset(terms))

    # ==================================================================
    # CNF tautology detection
    # ==================================================================

    def _compare_interval_start(self, a, b):
        a_start = a[0]
        b_start = b[0]

        if a_start is None:
            return -1 if b_start is not None else 0

        if b_start is None:
            return 1

        return self._cmp(a_start, b_start)

    def _union_covers_reals(self, atoms):
        """
        Do these threshold predicates cover all real values?

        Examples:

            x < 3 OR x >= 3       -> True
            x < 3 OR x >= 2       -> True
            x < 3 OR x > 3        -> False
            x >= 3                -> False
        """
        intervals = []

        for atom in atoms:
            _, op, value = self._parts(atom)

            if op == "<":
                intervals.append((None, False, value, False))

            elif op == "<=":
                intervals.append((None, False, value, True))

            elif op == ">":
                intervals.append((value, False, None, False))

            elif op == ">=":
                intervals.append((value, True, None, False))

        if not intervals:
            return False

        intervals.sort(key=cmp_to_key(self._compare_interval_start))

        # To cover all reals, the first interval must start at -inf.
        if intervals[0][0] is not None:
            return False

        current_upper = intervals[0][2]
        current_upper_closed = intervals[0][3]

        # Already reaches +inf.
        if current_upper is None:
            return True

        for (
            lower,
            lower_closed,
            upper,
            upper_closed,
        ) in intervals[1:]:
            if lower is not None:
                c = self._cmp(
                    lower,
                    current_upper,
                )

                # Gap.
                if c > 0:
                    return False

                # Touching at a point:
                # (-inf, 3) U (3, inf)
                # leaves 3 uncovered.
                if c == 0 and not (current_upper_closed or lower_closed):
                    return False

            # This interval reaches +inf.
            if upper is None:
                return True

            c = self._cmp(
                upper,
                current_upper,
            )

            if c > 0:
                current_upper = upper
                current_upper_closed = upper_closed

            elif c == 0:
                current_upper_closed = current_upper_closed or upper_closed

        return False

    def _clause_is_tautology(self, clause):
        """
        An OR clause is tautological if, for at least one variable,
        its threshold regions cover the entire real line.
        """
        by_variable = {}

        for atom in clause:
            variable, _, _ = self._parts(atom)

            by_variable.setdefault(
                variable,
                [],
            ).append(atom)

        return any(self._union_covers_reals(atoms) for atoms in by_variable.values())

    # ==================================================================
    # CNF clause simplification
    # ==================================================================

    def _simplify_clause(self, clause):
        """
        In:

            A OR B

        if A => B, remove A.
        """
        if self._clause_is_tautology(clause):
            return None

        atoms = list(clause)
        keep = set(atoms)

        for a in atoms:
            if a not in keep:
                continue

            for b in atoms:
                if a == b or b not in keep:
                    continue

                if self._term_implies_term(
                    frozenset([a]),
                    frozenset([b]),
                ):
                    keep.discard(a)
                    break

        return frozenset(keep)

    # ==================================================================
    # CNF clause implication/subsumption
    # ==================================================================

    def _clause_implies_clause(self, source, target):
        """
        Exact test:

            source => target

        for OR clauses.

        For every atom a in source:

            a AND NOT(target)

        must be contradictory.
        """
        if not target:
            return False

        negated_target = [self._negate_atom(atom) for atom in target]

        for source_atom in source:
            test_term = frozenset([source_atom] + negated_target)

            # If this is satisfiable, source_atom is a counterexample.
            if self._interval_from_atoms(test_term) is not None:
                return False

        return True

    def _absorb_cnf(self, clauses):
        """
        Remove weaker clauses.

        If:

            A => B

        then:

            A AND B = A

        and B can be removed.
        """
        if len(clauses) <= 1:
            return clauses

        ordered = sorted(
            clauses,
            key=lambda clause: (
                len(clause),
                tuple(sorted(map(str, clause))),
            ),
        )

        kept = set()

        for clause in ordered:
            redundant = False
            other_redundant = set()

            for other in kept:
                # TODO: here, why does it fail only with this?
                if self._clause_implies_clause(
                    other,
                    clause,
                ):
                    redundant = True
                    break

                if self._clause_implies_clause(
                    clause,
                    other,
                ):
                    other_redundant.add(other)

            if not redundant:
                kept.add(clause)
            if len(other_redundant):
                kept.add(clause)
                kept.difference_update(other_redundant)

        return frozenset(kept)

    def _simplify_cnf(self, cnf):
        if cnf == self.CNF_TRUE:
            return cnf

        if self.CNF_FALSE in cnf:
            return self.CNF_FALSE

        clauses = set()

        for clause in cnf:
            clause = self._simplify_clause(clause)

            if clause is None:
                # TRUE clause disappears from an AND.
                continue

            if not clause:
                # Empty OR = FALSE.
                return self.CNF_FALSE

            clauses.add(clause)

        if not clauses:
            return self.CNF_TRUE

        return self._absorb_cnf(frozenset(clauses))

    # ==================================================================
    # DNF construction
    # ==================================================================

    def _to_dnf(self, expr):
        if expr in self._dnf_cache:
            return self._dnf_cache[expr]

        if expr == sp.true:
            result = self.DNF_TRUE

        elif expr == sp.false:
            result = self.DNF_FALSE

        elif self._is_atom(expr):
            result = frozenset([frozenset([expr])])

        elif expr.func is sp.Or:
            result = self.DNF_FALSE

            for arg in expr.args:
                result |= self._to_dnf(arg)

            result = self._simplify_dnf(result)

        elif expr.func is sp.And:
            result = self.DNF_TRUE

            for arg in expr.args:
                result = self._dnf_and(
                    result,
                    self._to_dnf(arg),
                )

                if result == self.DNF_FALSE:
                    break

            result = self._simplify_dnf(result)

        else:
            raise TypeError(f"Unsupported Boolean expression: {expr}")

        self._dnf_cache[expr] = result
        return result

    def _dnf_and(self, left, right):
        if left == self.DNF_FALSE:
            return self.DNF_FALSE

        if right == self.DNF_FALSE:
            return self.DNF_FALSE

        if left == self.DNF_TRUE:
            return right

        if right == self.DNF_TRUE:
            return left

        result = set()

        for a in left:
            for b in right:
                term = self._simplify_term(a | b)

                if term is None:
                    continue

                if not term:
                    return self.DNF_TRUE

                result.add(term)

        if not result:
            return self.DNF_FALSE

        return self._absorb_dnf(frozenset(result))

    # ==================================================================
    # CNF construction
    # ==================================================================

    def _to_cnf(self, expr):
        if expr in self._cnf_cache:
            return self._cnf_cache[expr]

        if expr == sp.true:
            result = self.CNF_TRUE

        elif expr == sp.false:
            result = self.CNF_FALSE

        elif self._is_atom(expr):
            result = frozenset([frozenset([expr])])

        elif expr.func is sp.And:
            result = self.CNF_TRUE

            for arg in expr.args:
                result |= self._to_cnf(arg)

            result = self._simplify_cnf(result)

        elif expr.func is sp.Or:
            # ----------------------------------------------------------
            # CRITICAL:
            #
            # OR in CNF must start from FALSE.
            #
            # Starting from CNF_TRUE would make every OR immediately
            # become TRUE.
            # ----------------------------------------------------------
            result = self.CNF_FALSE

            for arg in expr.args:
                result = self._cnf_or(
                    result,
                    self._to_cnf(arg),
                )

                if result == self.CNF_TRUE:
                    break

            result = self._simplify_cnf(result)

        else:
            raise TypeError(f"Unsupported Boolean expression: {expr}")

        self._cnf_cache[expr] = result
        return result

    def _cnf_or(self, left, right):
        # TRUE OR X = TRUE
        if left == self.CNF_TRUE:
            return self.CNF_TRUE

        if right == self.CNF_TRUE:
            return self.CNF_TRUE

        # FALSE OR X = X
        if left == self.CNF_FALSE:
            return right

        if right == self.CNF_FALSE:
            return left

        result = set()

        # Distribution:
        #
        # (A1 AND A2) OR (B1 AND B2)
        #
        # =
        #
        # (A1 OR B1) AND (A1 OR B2)
        # AND
        # (A2 OR B1) AND (A2 OR B2)
        #
        for a in left:
            for b in right:
                clause = self._simplify_clause(a | b)

                if clause is None:
                    # TRUE clause disappears from AND.
                    continue

                if not clause:
                    return self.CNF_FALSE

                result.add(clause)

        if not result:
            return self.CNF_TRUE

        return self._absorb_cnf(frozenset(result))

    # ==================================================================
    # Conversion back to SymPy
    # ==================================================================

    @staticmethod
    def _and_expr(atoms):
        if not atoms:
            return sp.true

        return sp.And(*sorted(atoms, key=str))

    @staticmethod
    def _or_expr(atoms):
        if not atoms:
            return sp.false

        return sp.Or(*sorted(atoms, key=str))

    def _dnf_to_expr(self, dnf):
        if dnf == self.DNF_FALSE:
            return sp.false

        if dnf == self.DNF_TRUE:
            return sp.true

        terms = [self._and_expr(term) for term in dnf]

        return sp.Or(*sorted(terms, key=str))

    def _cnf_to_expr(self, cnf):
        if cnf == self.CNF_TRUE:
            return sp.true

        if cnf == self.CNF_FALSE:
            return sp.false

        clauses = [self._or_expr(clause) for clause in cnf]

        return sp.And(*sorted(clauses, key=str))

    # ==================================================================
    # Fixed-point key
    # ==================================================================

    @staticmethod
    def _expr_key(dnf):
        return frozenset(frozenset(str(atom) for atom in term) for term in dnf)


def simplify_threshold_formula(expr):
    return ThresholdSimplifier().simplify(expr)
