import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arbok.param_preprocessor import ParamPreprocessor
from lightgbm import LGBMRegressor, plot_importance

from optimus.run_loader import RunLoader

plt.style.use("seaborn")


class MetaLearner:
    def __init__(self, cache_dir="cache", metric="predictive_accuracy"):
        self.cache_dir = cache_dir
        self.metric = metric
        self.model = LGBMRegressor(n_estimators=500, num_leaves=16, learning_rate=0.05, min_child_samples=1, verbose=-1)
        self.preprocessor = ParamPreprocessor()

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def train(self, X, y):
        self.model.fit(X, y)

    def plot_importance(self, **kwargs):
        plot_importance(self.model, **kwargs)
        plt.show()

    @staticmethod
    def plot_correlation(X, y, method="pearson"):
        correlation = X.copy()
        correlation["y"] = y
        ax = plt.gca()
        plot = correlation.corr(method=method)["y"].sort_values().drop(["y"])
        plot = plot.plot.barh(ax=ax, color="#4C72B0")
        ax.set_title("Correlation")
        ax.set_xlabel(f"Correlation ({method})")
        ax.set_ylabel("Features")
        x_width_1 = plot.axes.viewLim.x1
        x_width_0 = plot.axes.viewLim.x0
        ax.set_xlim(x_width_0 - x_width_1 / 10, x_width_1 + x_width_1 / 10)
        for rect in plot.containers[0]:
            width = rect.get_width()
            x_width = plot.axes.viewLim.x1
            ha = "right" if width < 0 else "left"
            offset = -x_width / 100 if width < 0 else x_width / 100
            ax.text(x=width + offset, y=rect.xy[1] + 0.05, s=f"{width:.2}", color="black", ha=ha, va="bottom")
        plt.show()

    def std_over_group_means(self, X, groups):
        X["groups"] = groups
        ax = plt.gca()
        mean_per_group = X.groupby("groups").mean()
        variances = (mean_per_group / X.drop("groups", axis=1).mean()).std()
        variances = variances.sort_values()
        plot = variances.plot.barh(ax=ax, color="#4C72B0")
        ax.set_title("Standard deviation over normalized group means")
        ax.set_xlabel("$\sigma(\mu_{group} / \mu)$")
        ax.set_ylabel("Features")
        x_width = plot.axes.viewLim.x1
        ax.set_xlim(0, x_width + x_width / 10)
        for rect in plot.containers[0]:
            width = rect.get_width()

            ax.text(x=width + x_width / 100, y=rect.xy[1] + 0.05, s=f"{width:.3}", color="black", va="bottom")
        plt.show()

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

    def download_meta_features(self, datasets=None):

        # Set tasks
        if datasets is None:
            _, datasets = RunLoader.get_cc18_benchmarking_suite()

        # Make a short hash for the downloaded file
        datasets = np.unique(datasets)
        hash = hashlib.md5(str.encode("".join([str(i) for i in datasets]))).hexdigest()[:8]
        file = os.path.join(self.cache_dir, f"metafeatures-{hash}.csv")

        # Load file from cache
        if os.path.exists(file):
            with open(file, "r+") as f:
                return pd.read_csv(f, index_col=0)

        # Download qualities
        frame = RunLoader.load_meta_features(datasets)

        # Write to file
        frame.to_csv(file)

        return frame


ml = MetaLearner()
meta_features = ml.download_meta_features()
runs = ml.download_runs(6969)
X, y, groups = ml.convert_runs_to_features(runs)
ml.train(X, y)
ml.plot_importance()
ml.std_over_group_means(X, groups)
ml.plot_correlation(X, y)
