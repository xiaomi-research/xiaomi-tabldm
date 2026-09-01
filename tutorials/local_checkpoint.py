"""Using a local TabLDM checkpoint
=================================

This example shows how to load local classifier and regressor checkpoints from
``TABLDM_CLF_CKPT`` and ``TABLDM_REG_CKPT``. Passing ``--ckpt`` overrides the
environment variables and preserves the original classifier-only behavior.

Usage::

    export TABLDM_CLF_CKPT=/path/to/classifier.ckpt
    export TABLDM_REG_CKPT=/path/to/regressor.ckpt
    python tutorials/local_checkpoint.py

    python tutorials/local_checkpoint.py --ckpt /path/to/clf.ckpt
    CUDA_VISIBLE_DEVICES=0 python tutorials/local_checkpoint.py \\
        --ckpt /path/to/clf.ckpt --device cuda
"""

from pathlib import Path
import argparse
import os
import sys

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabldm import TabLDMClassifier, TabLDMRegressor


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--ckpt",
    default=None,
    help="Classifier checkpoint path; overrides TABLDM_CLF_CKPT and TABLDM_REG_CKPT",
)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--n-estimators", type=int, default=2)
args = parser.parse_args()

classifier_checkpoint = args.ckpt or os.environ.get("TABLDM_CLF_CKPT")
regressor_checkpoint = None if args.ckpt else os.environ.get("TABLDM_REG_CKPT")

if classifier_checkpoint is None and regressor_checkpoint is None:
    parser.error(
        "No checkpoint configured. Set TABLDM_CLF_CKPT and/or "
        "TABLDM_REG_CKPT, or pass --ckpt."
    )

if classifier_checkpoint is not None:
    classifier_checkpoint = Path(classifier_checkpoint).expanduser()
    if not classifier_checkpoint.is_file():
        parser.error(f"Classifier checkpoint does not exist: {classifier_checkpoint}")

    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
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
        n_estimators=args.n_estimators,
        model_path=classifier_checkpoint,
        device=args.device,
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    source = "--ckpt" if args.ckpt else "TABLDM_CLF_CKPT"
    print(f"Classifier checkpoint source: {source}")
    print(f"Checkpoint: {classifier_checkpoint.resolve()}")
    print(f"Device: {classifier.device_}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")

if regressor_checkpoint is not None:
    regressor_checkpoint = Path(regressor_checkpoint).expanduser()
    if not regressor_checkpoint.is_file():
        parser.error(f"Regressor checkpoint does not exist: {regressor_checkpoint}")

    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=5,
        noise=0.5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    regressor = TabLDMRegressor(
        n_estimators=args.n_estimators,
        model_path=regressor_checkpoint,
        device=args.device,
    )
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("Regressor checkpoint source: TABLDM_REG_CKPT")
    print(f"Checkpoint: {regressor_checkpoint.resolve()}")
    print(f"Device: {regressor.device_}")
    print(f"Test R²: {r2_score(y_test, predictions):.3f}")
