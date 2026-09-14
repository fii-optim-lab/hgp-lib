import numpy as np
import pytest

from hgp_lib.utils.metrics import fast_f1_score

NUM_PREDICTIONS = 100
SIZES = (1_000, 10_000, 100_000, 1_000_000)
SEEDS = {
    1_000: 1301,
    10_000: 1302,
    100_000: 1303,
    1_000_000: 1304,
}

pytestmark = [
    pytest.mark.fast,
    pytest.mark.scenario("scoring_default"),
    pytest.mark.benchmark(
        group="scoring_default",
        disable_gc=True,
        warmup=False,
    ),
]


# Scores 100 predictions with independent sample weights using fast_f1_score.
def score_default(predictions, sample_weights, labels):
    return np.array(
        [
            fast_f1_score(labels, prediction, sample_weight=weights)
            for prediction, weights in zip(predictions, sample_weights)
        ]
    )


@pytest.mark.parametrize("size", SIZES, ids=lambda size: f"{size:_}")
def test_scoring_default(benchmark, size):
    rng = np.random.default_rng(SEEDS[size])
    labels = rng.integers(0, 2, size=size, dtype=np.int8).astype(bool)
    predictions = [
        rng.integers(0, 2, size=size, dtype=np.int8).astype(bool)
        for _ in range(NUM_PREDICTIONS)
    ]
    sample_weights = [
        rng.random(size, dtype=np.float32) for _ in range(NUM_PREDICTIONS)
    ]

    scores = benchmark(score_default, predictions, sample_weights, labels)
    if len(scores) != NUM_PREDICTIONS:
        raise RuntimeError(f"Expected {NUM_PREDICTIONS} scores")

    benchmark.extra_info.update(
        {
            "scenario_id": f"scoring_default_{size:_}",
            "case_id": f"size-{size}",
            "size": size,
            "num_predictions": NUM_PREDICTIONS,
            "test_score": float(np.mean(scores)),
            "test_score_std": float(np.std(scores)),
        }
    )
