"""Repeated inference with TabLDM KV cache
=========================================

When the same training context is used for multiple prediction calls, enable
``kv_cache=True``. TabLDM builds the cache during ``fit`` and reuses it in each
subsequent ``predict`` call.
"""

from pathlib import Path
import os
import sys
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabldm import TabLDMClassifier


X, y = make_classification(
    n_samples=300,
    n_features=10,
    n_informative=5,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

classifier = TabLDMClassifier(
    n_estimators=2,
    kv_cache=True,
    device="cpu",
    model_path=os.environ.get("TABLDM_CLF_CKPT"),
    checkpoint_version="checkpoints/clf_stage3_moe1_step-10000.ckpt",
)
classifier.fit(X_train, y_train)

for batch_index, batch in enumerate(np.array_split(X_test, 3), start=1):
    start = time.perf_counter()
    predictions = classifier.predict(batch)
    elapsed = time.perf_counter() - start
    print(
        f"Batch {batch_index}: shape={predictions.shape}, "
        f"elapsed={elapsed:.3f}s"
    )

print("All prediction calls reused the cache built during fit.")
