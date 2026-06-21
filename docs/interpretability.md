# Interpretability

HGP produces a boolean rule as its model.
A rule is a logical expression over the input features, for example `And(age < 50, Or(income >= 30k, employed))`.
This is the whole classifier, not an approximation of it.
The same expression that is evaluated to make predictions is the one a person reads.

## Why this matters

Many strong classifiers are opaque.
A gradient-boosted ensemble or a neural network maps inputs to outputs through thousands of parameters, and no single parameter has a clear meaning.
Post-hoc tools such as feature attributions explain a prediction after the fact, but they are approximations of the model, and different tools can disagree.

A boolean rule needs no separate explanation.
The model is the explanation.
You can read it, check it against domain knowledge, and reason about every prediction it makes.

## What you can do with a rule

A trained rule supports several things that opaque models do not.

- Read the decision logic directly, with feature names through `rule.to_str(feature_names)`.
- Audit which features the model relies on, since unused features simply do not appear.
- Trace any single prediction to the literals that fired for that instance.
- Turn the rule into a policy or a specification, because it is already a logical statement.
- Edit or constrain the rule by hand when a domain expert needs to.

## Faithfulness

Interpretability is only useful if the explanation matches the model.
Because the rule is the model, the explanation is exact by construction.
There is no gap between what the classifier computes and what the rule says, which is the gap that post-hoc methods try to estimate.

## Cost of interpretability

Readable models can trade away some raw accuracy on hard problems.
HGP narrows that gap in two ways.
Rule size is bounded, so the search favors expressions a person can still read.
Hierarchical GP composes rules from smaller evolved parts, which reaches more expressive rules without giving up structure.
See [Theory](theory.md) for how the search explores the space of rules, and how this compares to decision trees.
