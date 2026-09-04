# Copyright (C) 2026 Xiaomi Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone tests for the inference-only ``tabldm`` package.

Run from the package root::

    cd TabLDM
    pytest tests/test_infer_package.py -v

These tests exercise the default-mode inference path (load checkpoint →
fit → predict) for the MoE1 classifier and regressor. Checkpoints are
looked up in the checkpoint directory (override via ``TABLDM_CKPT_DIR``).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure this package's own ``tabldm`` shadows any globally-installed copy.
_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import tabldm  # noqa: E402

if not str(tabldm.__file__).startswith(str(_PKG_DIR)):
    pytest.skip(
        f"imported tabldm from {tabldm.__file__!s}, not from {_PKG_DIR!s}",
        allow_module_level=True,
    )


# --- checkpoint discovery (sibling training project) ---------------------

def _latest_ckpt(pattern: str) -> pathlib.Path | None:
    base = pathlib.Path(os.environ.get("TABLDM_CKPT_DIR", _PKG_DIR.parent / "checkpoints"))
    ckpts = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


CLF_CKPT = _latest_ckpt("clf_default.ckpt")
REG_CKPT = _latest_ckpt("reg_default.ckpt")

# --- fixtures -------------------------------------------------------------

@pytest.fixture
def clf_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=60)
    return X, y


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    w = rng.standard_normal(8)
    y = X @ w + 0.1 * rng.standard_normal(60)
    return X, y


def _clf_kwargs(ckpt):
    return dict(
        n_estimators=1, model_path=str(ckpt), allow_auto_download=False,
        device="cpu", use_amp=False, use_fa3=False, verbose=False, n_jobs=1,
    )


def _reg_kwargs(ckpt):
    return dict(
        n_estimators=1, model_path=str(ckpt), allow_auto_download=False,
        device="cpu", use_amp=False, use_fa3=False, verbose=False, n_jobs=1,
    )


# --- tests ----------------------------------------------------------------

def test_imports():
    """All public MoE1 symbols are importable and the version matches."""
    assert tabldm.__version__ == "0.1.0"
    expected = {
        "TabLDMClassifier",
        "TabLDMRegressor",
        "InferenceConfig",
    }
    assert set(tabldm.__all__) == expected
    from tabldm import (
        TabLDMClassifier,
        TabLDMRegressor,
        InferenceConfig,
    )
    assert callable(TabLDMClassifier)
    assert callable(TabLDMRegressor)
    assert InferenceConfig() is not None


def test_transform_to_numerical_preserves_numeric_first_order():
    """Mixed DataFrames keep TALENT's numeric-before-categorical order."""
    from tabldm._sklearn.preprocessing import TransformToNumerical

    X = pd.DataFrame(
        {
            "num_0": [1.0, 2.0, 3.0],
            "cat_0": ["b", "a", "c"],
            "num_1": [4.0, 5.0, 6.0],
            "cat_1": ["y", "x", "z"],
        }
    )
    transformed = TransformToNumerical().fit_transform(X)

    assert transformed.shape == (3, 4)
    assert np.array_equal(transformed[:, :2], X[["num_0", "num_1"]].to_numpy())
    assert np.array_equal(transformed[:, 2], [1.0, 0.0, 2.0])
    assert np.array_equal(transformed[:, 3], [1.0, 0.0, 2.0])


def test_ensemble_randomizes_categorical_codes_and_class_codes():
    from tabldm._sklearn.preprocessing import EnsembleGenerator

    rows = np.arange(24, dtype=np.float64)
    X = np.column_stack((rows, rows % 3, rows % 2))
    y = np.asarray(rows, dtype=np.int64) % 3
    generator = EnsembleGenerator(
        classification=True,
        n_estimators=4,
        norm_methods=["none"],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        random_state=7,
        cat_random_encode=True,
        categorical_indices=[1, 2],
    )

    generator.fit(X, y)
    mappings = [
        mapping
        for member_mappings in generator.category_code_mappings_.values()
        for mapping in member_mappings
    ]
    class_patterns = [
        pattern
        for patterns in generator.class_shuffles_.values()
        for pattern in patterns
    ]

    assert len(mappings) == len(class_patterns) == 4
    for mapping in mappings:
        for categories, codes in mapping.values():
            assert np.array_equal(np.sort(codes), np.arange(categories.size))
    for pattern in class_patterns:
        assert np.array_equal(np.sort(pattern), np.arange(y.max() + 1))

    train = generator.transform(None, mode="train")
    test = generator.transform(np.array([[24.0, 0.0, 1.0]]), mode="test")
    both = generator.transform(np.array([[24.0, 0.0, 1.0]]), mode="both")
    assert train["none"][0].shape == (4, 24, 3)
    assert train["none"][1].shape == (4, 24)
    assert test["none"][0].shape == (4, 1, 3)
    np.testing.assert_array_equal(train["none"][1], both["none"][1])
    np.testing.assert_allclose(both["none"][0][:, :24], train["none"][0])

    repeat = EnsembleGenerator(
        classification=True,
        n_estimators=4,
        norm_methods=["none"],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        random_state=7,
        cat_random_encode=True,
        categorical_indices=[1, 2],
    )
    repeat.fit(X, y)
    for method in generator.category_code_mappings_:
        for expected, actual in zip(
            generator.category_code_mappings_[method],
            repeat.category_code_mappings_[method],
        ):
            for column in expected:
                expected_categories, expected_codes = expected[column]
                actual_categories, actual_codes = actual[column]
                np.testing.assert_array_equal(expected_categories, actual_categories)
                np.testing.assert_array_equal(expected_codes, actual_codes)
    for method in generator.class_shuffles_:
        for expected, actual in zip(
            generator.class_shuffles_[method], repeat.class_shuffles_[method]
        ):
            np.testing.assert_array_equal(expected, actual)


def test_ensemble_random_class_code_is_used_for_one_member():
    from tabldm._sklearn.preprocessing import EnsembleGenerator

    X = np.column_stack((np.arange(20, dtype=np.float64), np.arange(20) % 2))
    y = np.arange(20, dtype=np.int64) % 3
    generator = EnsembleGenerator(
        classification=True,
        n_estimators=1,
        norm_methods=["none"],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        random_state=3,
        cat_random_encode=True,
        categorical_indices=[1],
    )

    generator.fit(X, y)
    pattern = generator.class_shuffles_["none"][0]
    assert np.array_equal(np.sort(pattern), np.arange(3))
    assert not np.array_equal(pattern, np.arange(3))


@pytest.mark.skipif(CLF_CKPT is None, reason="no clf MoE1 checkpoint")
@pytest.mark.parametrize("n_classes", [2, 5])
def test_classifier_inference(n_classes):
    """MoE1 classifier produces correctly-shaped, normalized probabilities."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    y = rng.integers(0, n_classes, size=60)
    from tabldm import TabLDMClassifier
    clf = TabLDMClassifier(**_clf_kwargs(CLF_CKPT))
    clf.fit(X[:40], y[:40])
    proba = clf.predict_proba(X[40:])
    pred = clf.predict(X[40:])
    assert proba.shape == (20, n_classes)
    assert pred.shape == (20,)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert set(np.unique(pred)).issubset(set(range(n_classes)))


@pytest.mark.skipif(REG_CKPT is None, reason="no reg MoE1 checkpoint")
def test_regressor_inference(reg_data):
    """MoE1 regressor produces finite predictions of the right shape."""
    X, y = reg_data
    from tabldm import TabLDMRegressor
    reg = TabLDMRegressor(**_reg_kwargs(REG_CKPT))
    reg.fit(X[:40], y[:40])
    pred = reg.predict(X[40:])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


@pytest.mark.skipif(CLF_CKPT is None, reason="no clf MoE1 checkpoint")
def test_save_load_roundtrip(clf_data, tmp_path):
    """A self-contained save/load reproduces predictions exactly."""
    X, y = clf_data
    from tabldm import TabLDMClassifier
    clf = TabLDMClassifier(**_clf_kwargs(CLF_CKPT))
    clf.fit(X[:40], y[:40])
    pred_before = clf.predict(X[40:])

    pkl = tmp_path / "clf.pkl"
    clf.save(pkl, save_model_weights=True, save_training_data=True, save_kv_cache=False)
    loaded = TabLDMClassifier.load(pkl)
    pred_after = loaded.predict(X[40:])
    assert np.array_equal(pred_before, pred_after)
