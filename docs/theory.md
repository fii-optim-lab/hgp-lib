# Theory

This page explains how genetic programming searches for boolean rules, and why an evolutionary search reaches rules that greedy methods such as decision trees cannot.

## The search space

A boolean rule is a tree of `And` and `Or` operators over literals, where each literal tests one binarized feature and may be negated.
With many features and many ways to nest operators, the number of possible rules is enormous and grows fast with rule size.
The space is discrete and highly non-smooth: a small change to a rule, such as flipping one literal, can change its predictions and its score sharply.
There is no gradient to follow, so the search has to be combinatorial.

## Genetic programming

Genetic programming explores this space with a population of candidate rules that improves over generations.

1. Start from an initial population of rules.
2. Score each rule against the data with the fitness function.
3. Select the better rules to act as parents.
4. Produce new rules with crossover, which swaps subtrees between parents, and mutation, which alters a literal or an operator.
5. Repeat for many epochs, keeping track of the best rule found.

Crossover lets the search recombine useful substructure from different rules, and mutation keeps introducing variation so the population does not stall.
Because the search keeps many candidates at once and recombines them, it can cross valleys in the score landscape that a single-path greedy method would get stuck in.

## Comparison with decision trees

A decision tree is built greedily.
At each node it picks the single feature split that maximizes a local criterion, usually information gain or Gini impurity, and then recurses.
This has two consequences.

The choice is greedy and local.
A split that looks best in isolation can lead to a worse tree overall, and the tree never reconsiders an earlier split.
Combinations of features that are only useful together, where neither feature looks good on its own, are easy to miss.

The structure is fixed in form.
A standard tree partitions the space with one axis-aligned test per node, and the path to a leaf is a conjunction of such tests.
Expressing a disjunction means repeating subtrees across branches, which makes the tree large and harder to read.

Genetic programming does not split greedily and is not limited to that shape.
It optimizes the whole rule against the fitness function, evaluates many candidates in parallel, and builds `And` and `Or` freely in one expression.
This lets it find compact rules that mix conjunction and disjunction, including feature interactions that a greedy split would not surface.

## Hierarchical GP

Hierarchical GP adds structure to the search.
Child populations evolve on sampled subsets of features, then their rules are combined into larger rules in a parent population.
Sampling features narrows each child's search space, which makes the sub-search easier, and composing the children builds expressive rules from parts that were each optimized on a focused slice of the problem.

TODO: Replace this section's summary with the formal treatment and refer
to the paper once it is published. Cite the paper that introduces the
hierarchical GP formulation and its theoretical analysis.
