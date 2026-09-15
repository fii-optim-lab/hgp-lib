import numpy as np
import pytest

from .evaluation_artifacts import LITERAL_COUNTS, NUM_RULES, load_artifact

NUM_SAMPLES = (1_000, 10_000)
SEEDS = {
    (100, 1_000): 3301,
    (100, 10_000): 3302,
    (1_000, 1_000): 3303,
    (1_000, 10_000): 3304,
}

pytestmark = [
    pytest.mark.fast,
    pytest.mark.scenario("evaluation_default"),
    pytest.mark.benchmark(
        group="evaluation_default",
        disable_gc=True,
        warmup=False,
    ),
]


# Evaluates 100 fixed rules against one deterministic boolean matrix.
def evaluate_default(rules, data):
    [rule.evaluate(data) for rule in rules]


@pytest.mark.parametrize(
    "num_literals", LITERAL_COUNTS, ids=lambda value: f"{value:_}-literals"
)
@pytest.mark.parametrize(
    "num_samples", NUM_SAMPLES, ids=lambda value: f"{value:_}-samples"
)
def test_evaluation_default(benchmark, num_literals, num_samples):
    rules = load_artifact(num_literals)
    rng = np.random.default_rng(SEEDS[(num_literals, num_samples)])
    data = rng.integers(
        0,
        2,
        size=(num_samples, num_literals),
        dtype=np.int8,
    ).astype(bool)

    benchmark(evaluate_default, rules, data)

    benchmark.extra_info.update(
        {
            "scenario_id": (
                f"evaluation_default_{num_literals:_}_literals_{num_samples:_}_samples"
            ),
            "case_id": f"{num_literals}-literals-{num_samples}-samples",
            "num_rules": NUM_RULES,
            "num_literals": num_literals,
            "num_samples": num_samples,
        }
    )
