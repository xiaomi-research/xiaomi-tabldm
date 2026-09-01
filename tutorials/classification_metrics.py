"""Classification metrics with TabLDM
====================================

This example shows how to obtain labels, class probabilities, and common
classification metrics from a fitted TabLDM classifier.
"""

from pathlib import Path
import os
import sys

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabldm import TabLDMClassifier


X, y = make_classification(
    n_samples=600,
    n_features=12,
    n_informative=6,
    weights=[0.7, 0.3],
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

classifier = TabLDMClassifier(
    n_estimators=2,
    device="cpu",
    model_path=os.environ.get("TABLDM_CLF_CKPT"),
    checkpoint_version="checkpoints/clf_stage3_moe1_step-10000.ckpt",
)
classifier.fit(X_train, y_train)

predicted_labels = classifier.predict(X_test)
predicted_probabilities = classifier.predict_proba(X_test)

print(f"Classes: {classifier.classes_}")
print(f"Label shape: {predicted_labels.shape}")
print(f"Probability shape: {predicted_probabilities.shape}")
print(f"Accuracy: {accuracy_score(y_test, predicted_labels):.3f}")
print(f"F1 score: {f1_score(y_test, predicted_labels):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, predicted_probabilities[:, 1]):.3f}")
