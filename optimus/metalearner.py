import json
import os
import openml
from lightgbm import LGBMRegressor
from sklearn.feature_selection import VarianceThreshold

from optimus.run_loader import RunLoader
import pandas as pd
from arbok.param_preprocessor import ParamPreprocessor


class MetaLearner:
    def __init__(self, cache_dir="cache", metric="predictive_accuracy"):
        self.cache_dir = cache_dir
        self.metric = metric
        self.model = LGBMRegressor(n_estimators=500, num_leaves=16, learning_rate=0.05, min_child_samples=1, verbose=-1)
        self.preprocessor = ParamPreprocessor()

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def download_flow(self, flow_id, tasks=None, metric="predictive_accuracy", max_per_task=5000):
        file = os.path.join(self.cache_dir, f"runs-{flow_id}.csv")

        if os.path.exists(file):
            with open(file, "r+") as f:
                return pd.read_csv(f, index_col=0)

        if tasks is None:
            tasks = RunLoader.get_cc18_benchmarking_suite()

        frame = RunLoader.load_tasks(tasks, flow_id, metric=metric, max_per_task=max_per_task)
        frame.to_csv(file)
        return frame

    def convert(self, frame, metric="predictive_accuracy"):
        copied = frame.copy()
        datasets = copied.pop("data_id")
        tasks = copied.pop("task_id")
        runs = copied.pop("run_id")
        y = copied.pop(metric)

        # Drop columns we can not use
        num_unique = copied.nunique()
        for c in copied.columns:
            if (copied.dtypes[c] == object and MetaLearner.is_json(copied[c][0])) or num_unique[c] <= 1:
                print("Deleted", c)
                del copied[c]

        X = pd.get_dummies(copied)
        return X, y, datasets

    @staticmethod
    def is_json(myjson):
        try:
            json_object = json.loads(myjson)
        except ValueError as e:
            return False
        return True

ml = MetaLearner()
frame = ml.download_flow(6969)
X, y, groups = ml.convert(frame)
print()