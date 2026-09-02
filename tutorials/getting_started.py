"""Getting started with TabLDM
==============================

This example demonstrates TabLDM's scikit-learn compatible API for
classification and regression. ``fit`` prepares the in-context examples and
loads the pretrained model; it does not update model weights.
"""

from pathlib import Path
import os
import sys

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabldm import TabLDMClassifier, TabLDMRegressor


X, y = make_classification(
    n_samples=300,
    n_features=10,
    n_informative=5,
    random_state=42,
)

classifier = TabLDMClassifier(
    n_estimators=2,
    device="cpu",
    model_path=os.environ.get("TABLDM_CLF_CKPT"),
    checkpoint_version="checkpoints/clf_default.ckpt",
)
classification_scores = cross_val_score(
    classifier,
    X,
    y,
    cv=3,
    scoring="accuracy",
)
print(
    "Classification accuracy: "
    f"{classification_scores.mean():.3f} ± {classification_scores.std():.3f}"
)

X, y = make_regression(
    n_samples=300,
    n_features=10,
    n_informative=5,
    noise=0.5,
    random_state=42,
)

regressor = TabLDMRegressor(
    n_estimators=2,
    device="cpu",
    model_path=os.environ.get("TABLDM_REG_CKPT"),
    checkpoint_version="checkpoints/reg_default.ckpt",
)
regression_scores = cross_val_score(
    regressor,
    X,
    y,
    cv=3,
    scoring="r2",
)
print(
    "Regression R²: "
    f"{regression_scores.mean():.3f} ± {regression_scores.std():.3f}"
)
