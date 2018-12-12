import hashlib

from lightgbm import LGBMRegressor
from sklearn import clone
from sklearn.model_selection import cross_val_score, ParameterSampler
import pandas as pd
import numpy as np
from tqdm import tqdm


class Optimizer:
    def __init__(self, pipeline):
        self.model = LGBMRegressor(num_leaves=8, min_child_samples=1, min_data_in_bin=1, verbose=-1, objective="quantile")
        self.observed_X = []
        self.observed_y = []
        self.observed_hashes = []
        self.pipeline = pipeline
        self.X = None
        self.y = None
        self.cv = None
        self.metric = None
        self.grid = None
        self.n_draws = None

    def setup(self, grid, X, y, cv, metric="accuracy", n_draws=500):
        self.grid = grid
        self.n_draws = n_draws
        self.X = X
        self.y = y
        self.cv = cv
        self.metric = metric

    def seed(self, configurations):
        for config in tqdm(configurations):
            self.observe(config)

    def loop(self, n_iter=150):
        for n in range(n_iter):
            sample = self.suggest()
            score = self.observe(sample)
            print(f"{score}/{np.max(self.observed_y)}")

    def observe(self, params):
        params = {i: None if isinstance(j, np.float) and np.isnan(j) else j for i, j in params.items()}
        pipeline = clone(self.pipeline).set_params(**params)
        scores = cross_val_score(estimator=pipeline, X=self.X, y=self.y, cv=self.cv, scoring=self.metric)
        score = np.mean(scores)
        self.observed_X.append(params)
        self.observed_y.append(score)
        self.observed_hashes.append(hashlib.md5(str(params).encode()).hexdigest()[:8])
        return score

    def clean(self, X):
        return pd.get_dummies(X).astype(float)

    def suggest(self):
        samples = ParameterSampler(self.grid, 500)
        samples = list(samples)
        samples = np.array(samples)

        # Remove already observed
        hashes = [hashlib.md5(str.encode(str(i))).hexdigest()[:8] for i in samples]
        for hash in self.observed_hashes:
            selection = np.array(hashes) != np.array(hash)
            hashes = np.array(hashes)[selection]
            samples = samples[selection]

        # Clean samples and observed samples together
        frame = pd.concat([pd.DataFrame(self.observed_X), pd.DataFrame(samples.tolist())], sort=True)
        frame = self.clean(frame)

        # Extract
        num_observed = len(self.observed_hashes)
        X_ = frame.iloc[:num_observed]
        samples_ = frame.iloc[num_observed:]

        # Train model
        self.model.fit(X_, self.observed_y)
        predicted = self.model.predict(samples_)

        return samples[np.argmax(predicted)]
