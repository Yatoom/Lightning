import json
import os
import requests
import hashlib

import openml
from lightgbm import LGBMRegressor
import numpy as np
from tqdm import tqdm

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

    def download_runs(self, flow_id, tasks=None, metric=None, max_per_task=5000):
        # Set metric
        if metric is None:
            metric = self.metric

        # Define path
        file = os.path.join(self.cache_dir, f"runs-{flow_id}.csv")

        # Load from cache if it exists
        if os.path.exists(file):
            with open(file, "r+") as f:
                return pd.read_csv(f, index_col=0)

        # Set tasks
        if tasks is None:
            tasks, _ = RunLoader.get_cc18_benchmarking_suite()

        # Load run-evaluations and save to cache
        frame = RunLoader.load_tasks(tasks, flow_id, metric=metric, max_per_task=max_per_task)
        frame.to_csv(file)
        return frame

    def convert_runs_to_features(self, frame, metric=None):
        if metric is None:
            metric = self.metric
        return RunLoader.convert_runs_to_features(frame, metric)

    def download_meta_features(self, groups=None):

        # Set tasks
        if groups is None:
            _, groups = RunLoader.get_cc18_benchmarking_suite()

        # Make a short hash for the downloaded file
        groups = np.unique(groups)
        hash = hashlib.md5(str.encode("".join([str(i) for i in groups]))).hexdigest()[:8]
        file = os.path.join(self.cache_dir, f"metafeatures-{hash}.csv")

        # Load file from cache
        if os.path.exists(file):
            with open(file, "r+") as f:
                return pd.read_csv(f, index_col=0)

        # Download qualities
        frame = RunLoader.load_meta_features(groups)

        # Write to file
        frame.to_csv(file)

        return frame

ml = MetaLearner()
ml.download_meta_features()