import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
plt.style.use("seaborn")

frame = pd.read_csv("cache/runs-6969.csv", index_col=0)

# Plot kde
ax = plt.gca()
groups = np.array(frame["data_id"])
u_groups = np.unique(groups)[:1]
frame = frame[[i in u_groups for i in frame["data_id"]]]
frame.groupby("data_id")["predictive_accuracy"].plot.kde(ax=ax)
frame[frame["predictive_accuracy"] > 0.995].groupby("data_id")["predictive_accuracy"].plot.kde(ax=ax)
plt.show()

# Plot kde after standardizing
groups = np.array(frame["data_id"])
u_groups = np.unique(groups)
performance = np.array(frame["predictive_accuracy"])

standardized_performance = np.zeros_like(performance)
for g in u_groups:
    indices = groups == g
    standardized_performance[indices] = StandardScaler().fit_transform(X=performance[indices].reshape(-1, 1)).reshape(-1)
f = pd.DataFrame()
f["y"] = standardized_performance
f["g"] = groups
f.groupby("g")["y"].plot.kde()
plt.show()


print()