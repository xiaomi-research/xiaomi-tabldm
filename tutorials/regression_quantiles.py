"""Regression quantiles with TabLDM
==================================

This example demonstrates mean predictions and predictive quantiles on a
synthetic heteroscedastic regression task.
"""

from pathlib import Path
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabldm import TabLDMRegressor


rng = np.random.default_rng(42)
n_samples = 500
x = rng.uniform(-3, 3, size=n_samples)
X = np.column_stack([x, np.sin(x), x**2 / 10])
mean = np.sin(x) + 0.1 * x
noise_scale = 0.03 + 0.12 * np.exp(-((x - 0.5) ** 2) / 0.5)
y = rng.normal(mean, noise_scale)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

regressor = TabLDMRegressor(
    n_estimators=2,
    device="cpu",
    model_path=os.environ.get("TABLDM_REG_CKPT"),
    checkpoint_version="checkpoints/reg_stage3_moe1_step-10000.ckpt",
)
regressor.fit(X_train, y_train)

mean_prediction = regressor.predict(X_test)
quantiles = regressor.predict(
    X_test,
    output_type="quantiles",
    alphas=[0.1, 0.5, 0.9],
)
coverage = np.mean(
    (y_test >= quantiles[:, 0]) & (y_test <= quantiles[:, 2])
)

print(f"Mean prediction shape: {mean_prediction.shape}")
print(f"Quantile prediction shape: {quantiles.shape}")
print(f"Observed 80% interval coverage: {coverage:.3f}")
