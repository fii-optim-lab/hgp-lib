import random

import numpy as np

from hgp_lib import BooleanGPConfig, BooleanRuleClassifier, TrainerConfig
from hgp_lib.utils.metrics import fast_f1_score
from .data import DATASET_NAMES, N_SPLITS, load_fold


class HGPBenchmark:
    params = (
        DATASET_NAMES,
        list(range(N_SPLITS)),
    )
    param_names = (
        "dataset",
        "fold",
    )

    number = 1
    repeat = 1
    warmup_time = 0
    timeout = 3600

    def setup(self, dataset, fold):
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = load_fold(dataset, fold)

        self.config = TrainerConfig(
            gp_config=BooleanGPConfig(),
            num_epochs=1000,
            val_every=100,
        )

    def create_classifier(self, fold):
        np.random.seed(fold)
        random.seed(fold)

        return BooleanRuleClassifier(self.config)

    def time_fit(self, dataset, fold):
        clf = self.create_classifier(fold)

        clf.fit(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        )

    def track_test_score(self, dataset, fold):
        clf = self.create_classifier(fold)

        clf.fit(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        )
        predictions = clf.predict(self.X_test)
        return float(fast_f1_score(self.y_test, predictions))