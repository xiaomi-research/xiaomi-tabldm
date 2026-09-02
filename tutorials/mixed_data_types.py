"""Mixed data types with TabLDM
===============================

TabLDM can preprocess pandas DataFrames containing numeric, categorical,
boolean, and missing values. Categorical and string columns should be passed as
DataFrame columns so their dtypes can be detected.
"""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabldm import TabLDMClassifier


rng = np.random.default_rng(42)
n_samples = 400

income = rng.normal(60000, 15000, size=n_samples).clip(10000, None)
age = rng.integers(18, 75, size=n_samples)
city = rng.choice(["Beijing", "Shanghai", "Shenzhen", "Hangzhou"], size=n_samples)
is_member = rng.choice([True, False], size=n_samples, p=[0.35, 0.65])

score = 0.00003 * (income - 55000) + 0.01 * (age - 35)
score += np.where(is_member, 0.35, 0.0)
score += np.where(city == "Beijing", 0.2, 0.0)
probability = 1 / (1 + np.exp(-score))
y = rng.binomial(1, probability)

data = pd.DataFrame(
    {
        "income": income,
        "age": age,
        "city": city,
        "is_member": is_member,
    }
)
data.loc[rng.choice(n_samples, 20, replace=False), "income"] = np.nan

X_train, X_test, y_train, y_test = train_test_split(
    data,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

classifier = TabLDMClassifier(
    n_estimators=2,
    device="cpu",
    model_path=os.environ.get("TABLDM_CLF_CKPT"),
    checkpoint_version="checkpoints/clf_default.ckpt",
)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
