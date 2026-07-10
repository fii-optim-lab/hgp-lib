import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.preprocessing import SklearnBinarizer, StandardBinarizer
from hgp_lib.utils.metrics import fast_f1_score


def _make_dataset(n: int = 60):
    labels = np.array([False, True] * (n // 2))
    rng = np.random.RandomState(0)
    data = pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, n) + labels * 0.5,
            "f2": rng.normal(size=n),
        }
    )
    return data, labels


def _run(binarizer):
    data, labels = _make_dataset()
    gp_config = BooleanGPConfig(score_fn=fast_f1_score)
    trainer_config = TrainerConfig(
        gp_config=gp_config, num_epochs=20, progress_bar=False
    )
    config = BenchmarkerConfig(
        data=data,
        labels=labels,
        trainer_config=trainer_config,
        binarizer=binarizer,
        num_runs=1,
        n_folds=2,
        n_jobs=1,
        show_run_progress=False,
        show_fold_progress=False,
        show_epoch_progress=False,
    )
    return GPBenchmarker(config).fit()


class TestBenchmarkerBinarizers(unittest.TestCase):
    def test_benchmarker_with_standard_binarizer(self):
        result = _run(StandardBinarizer(num_bins=3))
        self.assertEqual(len(result.test_scores), 1)
        self.assertGreater(len(result.best_run.feature_names), 0)

    def test_benchmarker_with_sklearn_kbins_binarizer(self):
        binarizer = SklearnBinarizer(
            KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform")
        )
        result = _run(binarizer)
        self.assertEqual(len(result.test_scores), 1)
        # Feature names come from the discretizer, one per bin per feature.
        self.assertGreater(len(result.best_run.feature_names), 0)

    def test_benchmarker_default_binarizer_is_standard(self):
        data, labels = _make_dataset()
        gp_config = BooleanGPConfig(score_fn=fast_f1_score)
        trainer_config = TrainerConfig(
            gp_config=gp_config, num_epochs=10, progress_bar=False
        )
        config = BenchmarkerConfig(
            data=data,
            labels=labels,
            trainer_config=trainer_config,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            show_run_progress=False,
            show_fold_progress=False,
            show_epoch_progress=False,
        )
        benchmarker = GPBenchmarker(config)
        self.assertIsInstance(benchmarker.config.binarizer, StandardBinarizer)


if __name__ == "__main__":
    unittest.main()
