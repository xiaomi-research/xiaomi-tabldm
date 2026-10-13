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
"""Tests for the ``time_limit`` fit budget.

Two layers:

* **Mechanics** (no checkpoint needed): ``_FitBudget`` itself and the
  batch-level early-stop in ``_forward_candidates`` / ``_batch_forward``,
  exercised against a stub model.  These pin the three docstring contracts —
  unlimited is bit-identical, overshoot is bounded to one batch, truncation is
  a prefix of the canonical estimator order.
* **Integration** (checkpoint needed, skipped otherwise): ``fit`` with
  ``time_limit=None`` vs a generous limit must agree exactly, and a tiny limit
  must still return a usable model.

Run from the package root::

    pytest tests/test_time_limit.py -v
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pytest
import torch

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

from tabldm._sklearn.base import _FitBudget, _reserve  # noqa: E402


# --- _FitBudget -----------------------------------------------------------

def test_budget_unlimited_never_expires():
    b = _FitBudget(None)
    assert b.unlimited
    assert not b.expired()
    assert b.remaining is None
    assert b.time_limit is None


def test_budget_zero_and_negative_are_expired_not_error():
    for limit in (0, -3.0):
        b = _FitBudget(limit)
        assert b.time_limit == 0.0
        assert b.expired()


def test_budget_reserve_scales_and_floors():
    assert _reserve(10.0) == 5.0          # floor wins
    assert _reserve(1000.0) == 20.0       # 2% wins
    b = _FitBudget(100.0, reserve=_reserve(100.0))
    assert not b.expired()
    # reserve is held back from the deadline, not from ``time_limit``
    assert b.remaining is not None and 0.0 < b.remaining < 100.0


def test_budget_expires_after_deadline():
    b = _FitBudget(0.05, reserve=0.0)
    assert not b.expired()
    time.sleep(0.06)
    assert b.expired()


# --- stub-backed forward helpers -----------------------------------------

class _StubModel:
    """Returns a deterministic per-estimator output so truncation is checkable.

    Output shape ``(n_est, 4, 3)`` with ``out[i] = i`` filled in, matching the
    real model's ``(estimator, test, target)`` layout.
    """

    def __call__(self, X=None, y_train=None, X_test=None, **kwargs):
        n_est = int(X.shape[0]) if X is not None else int(X_test.shape[0])
        out = torch.zeros(n_est, 4, 3)
        out[:, :, 0] = torch.arange(n_est, dtype=torch.float32)[:, None]
        return out

    def predict_stats(self, X, y, output_type=None, alphas=None, inference_config=None):
        n_est = int(X.shape[0])
        out = torch.zeros(n_est, 4, 3)
        out[:, :, 0] = torch.arange(n_est, dtype=torch.float32)[:, None]
        if output_type == ["mean"] or output_type == "mean":
            return out[:, :, 0]
        return {key: out[:, :, 0] for key in output_type}


class _StubRegModel:
    """Regressor-style stub: ``predict_stats`` returns per-estimator means."""

    def predict_stats(self, X, y, output_type=None, alphas=None, inference_config=None):
        n_est = int(X.shape[0])
        out = torch.zeros(n_est, 4)
        out[:, 0] = torch.arange(n_est, dtype=torch.float32)
        if output_type == ["mean"] or output_type == "mean":
            return out
        return {key: out for key in output_type}


def _clf_stub():
    from tabldm._sklearn.classifier import TabLDMClassifier

    est = TabLDMClassifier(n_estimators=8, device="cpu", verbose=False)
    est.model_ = _StubModel()
    est.device_ = torch.device("cpu")
    est.batch_size_ = 2
    est.softmax_temperature = 0.9
    est.inference_config_ = None
    return est


def _reg_stub():
    from tabldm._sklearn.regressor import TabLDMRegressor

    est = TabLDMRegressor(n_estimators=8, device="cpu", verbose=False)
    est.model_ = _StubRegModel()
    est.device_ = torch.device("cpu")
    est.batch_size_ = 2
    est.softmax_temperature = 0.9
    est.inference_config_ = None
    return est


def _fake_inputs(n_est=8, n_rows=6, n_feat=4):
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((n_est, n_rows, n_feat)).astype(np.float32)
    ys = rng.standard_normal((n_est, n_rows)).astype(np.float32)
    return Xs, ys


def test_forward_candidates_unlimited_is_complete():
    est = _clf_stub()
    Xs, ys = _fake_inputs()
    probs = est._forward_candidates(Xs, ys, feat_indices=None, budget=None)
    assert probs.shape[0] == 8
    assert np.isfinite(probs).all()


def test_forward_candidates_expired_fills_nan_suffix():
    est = _clf_stub()
    Xs, ys = _fake_inputs(n_est=8)
    budget = _FitBudget(None)
    budget._deadline = time.monotonic() - 1.0  # already spent
    # floor: the very first batch still runs
    probs = est._forward_candidates(Xs, ys, feat_indices=None, budget=budget)
    assert probs.shape[0] == 8
    assert budget.n_batches_done == 1
    assert budget.n_truncated == 6
    # prefix finite, tail NaN — truncation is a suffix of the canonical order
    assert np.isfinite(probs[:2]).all()
    assert np.isnan(probs[2:]).all()


def test_forward_candidates_floor_keeps_at_least_one_batch():
    est = _clf_stub()
    Xs, ys = _fake_inputs(n_est=8)
    budget = _FitBudget(0.0)  # expired from the start
    probs = est._forward_candidates(Xs, ys, feat_indices=None, budget=budget)
    assert np.isfinite(probs).any(), "floor must keep at least one batch"
    assert budget.n_batches_done == 1


def test_forward_candidates_second_call_short_circuits_to_nan():
    """Once the floor is spent, later calls must not forward anything."""
    est = _clf_stub()
    Xs, ys = _fake_inputs(n_est=8)
    budget = _FitBudget(0.0)
    first = est._forward_candidates(Xs, ys, feat_indices=None, budget=budget)
    assert np.isfinite(first).any()
    done_after_first = budget.n_batches_done

    second = est._forward_candidates(Xs, ys, feat_indices=None, budget=budget)
    assert budget.n_batches_done == done_after_first, "no further batches may run"
    assert np.isnan(second).all()
    assert second.shape[0] == 8
    assert budget.n_truncated == 6 + 8


def test_forward_candidates_mid_call_truncation_is_prefix():
    """Expire after exactly one batch: only the tail is NaN."""
    est = _clf_stub()
    Xs, ys = _fake_inputs(n_est=8)
    budget = _FitBudget(None)
    # Expire only once one batch has been counted, so the first batch runs and
    # the rest are cut — the truncation must be a suffix of the estimator axis.
    budget.expired = lambda: budget.n_batches_done >= 1
    probs = est._forward_candidates(Xs, ys, feat_indices=None, budget=budget)
    assert budget.n_batches_done == 1
    assert np.isfinite(probs[:2]).all()
    assert np.isnan(probs[2:]).all()
    assert probs.shape[0] == 8


def test_batch_forward_reg_expired_fills_nan_suffix():
    est = _reg_stub()
    Xs, ys = _fake_inputs(n_est=8)
    budget = _FitBudget(0.0)
    out = est._batch_forward(Xs, ys, output_type=["mean"], budget=budget)
    assert out.shape[0] == 8
    assert np.isfinite(out[:2]).all()
    assert np.isnan(out[2:]).all()
    assert budget.n_batches_done == 1


def test_batch_forward_reg_unlimited_unchanged():
    est = _reg_stub()
    Xs, ys = _fake_inputs(n_est=8)
    out = est._batch_forward(Xs, ys, output_type=["mean"], budget=None)
    assert out.shape[0] == 8
    assert np.isfinite(out).all()


def test_batch_forward_reg_multi_output_truncation():
    est = _reg_stub()
    Xs, ys = _fake_inputs(n_est=8)
    budget = _FitBudget(0.0)
    out = est._batch_forward(Xs, ys, output_type=["mean", "quantile"], budget=budget)
    assert set(out) == {"mean", "quantile"}
    for key in ("mean", "quantile"):
        assert out[key].shape[0] == 8
        assert np.isnan(out[key][2:]).all()


# --- budget scoping (tag="val" vs tag="predict") -------------------------

class _StubGenerator:
    """Minimal ``EnsembleGenerator`` stand-in for ``_collect_candidate_probs``."""

    def __init__(self, n_est=8, n_rows=6, n_feat=4, seed=0):
        rng = np.random.default_rng(seed)
        self._Xs = rng.standard_normal((n_est, n_rows, n_feat)).astype(np.float32)
        self._ys = rng.standard_normal((n_est, n_rows)).astype(np.float32)

    def transform(self, X, mode="both"):
        return {"none": (self._Xs, self._ys)}


def test_collect_candidate_probs_predict_ignores_budget():
    """tag="predict" must run the full pool even when the fit budget is spent."""
    est = _clf_stub()
    est._fit_budget_ = _FitBudget(0.0)  # spent
    est._current_fold_label_ = ""
    gen = _StubGenerator()
    probs, valid_mask, names = est._collect_candidate_probs(
        np.zeros((4, 4), dtype=np.float32), gen, None, tag="predict"
    )
    assert valid_mask.shape[0] == 8
    assert valid_mask.all(), "predict must not drop candidates for the fit budget"
    assert len(names) == 8
    assert probs.shape[0] == 8
    assert est._fit_budget_.n_truncated == 0


def test_collect_candidate_probs_val_honours_budget():
    """tag="val" truncates and records the drop in the valid-candidate mask."""
    est = _clf_stub()
    est._fit_budget_ = _FitBudget(0.0)  # spent
    est._current_fold_label_ = ""
    gen = _StubGenerator()
    probs, valid_mask, names = est._collect_candidate_probs(
        np.zeros((4, 4), dtype=np.float32), gen, None, tag="val"
    )
    assert valid_mask.shape[0] == 8
    assert len(names) == 8, "names cover the planned pool so the mask stays aligned"
    # prefix kept, suffix dropped — contract 3
    assert valid_mask[:2].all()
    assert not valid_mask[2:].any()
    assert probs.shape[0] == 2
    assert est._fit_budget_.n_truncated == 6


def test_collect_candidate_probs_val_unlimited_unchanged():
    """tag="val" with no budget is the untouched path."""
    est = _clf_stub()
    est._current_fold_label_ = ""
    gen = _StubGenerator()
    probs, valid_mask, names = est._collect_candidate_probs(
        np.zeros((4, 4), dtype=np.float32), gen, None, tag="val"
    )
    assert valid_mask.shape[0] == 8
    assert valid_mask.all()
    assert len(names) == 8
    assert probs.shape[0] == 8


# --- holdout gate --------------------------------------------------------

def _nnls_ready_stub():
    est = _clf_stub()
    est.validation = True
    est.k_fold = False
    est.shrink_lambda = 0.5  # force lam > 0 so the holdout is actually wanted
    est.enable_calibration = False
    est.n_classes_ = 2
    return est


def test_fit_nnls_skips_holdout_when_budget_spent():
    """A spent budget falls through to the pure-group-prior path, no holdout."""
    est = _nnls_ready_stub()
    est._fit_budget_ = _FitBudget(0.0)  # spent before the holdout would start
    X = np.zeros((60, 4), dtype=np.float32)
    y = np.array([0, 1] * 30)

    called = {"collect": False}

    def _boom(*args, **kwargs):
        called["collect"] = True
        raise AssertionError("holdout collection must not run")

    est._single_val_probs = _boom
    est._fit_nnls_weights(X, y)

    assert not called["collect"]
    assert est.shrink_lambda_ == 0.0
    assert est.nnls_valid_candidate_mask_ is None


def test_fit_nnls_runs_holdout_when_budget_unlimited():
    est = _nnls_ready_stub()
    est._fit_budget_ = _FitBudget(None)  # unlimited
    X = np.zeros((60, 4), dtype=np.float32)
    y = np.array([0, 1] * 30)

    called = {"collect": False}

    def _collect(X_, y_):
        # Sentinel: proves the holdout was entered. The surrounding
        # ``except Exception`` in ``_fit_nnls_weights`` swallows it and falls
        # back to the group prior, which is all this test needs to avoid
        # depending on the NNLS solver.
        called["collect"] = True
        raise RuntimeError("sentinel")

    est._single_val_probs = _collect
    est._fit_nnls_weights(X, y)

    assert called["collect"], "an unspent budget must not skip the holdout"


# --- integration (needs a checkpoint) ------------------------------------

def _latest_ckpt(pattern: str) -> pathlib.Path | None:
    base = pathlib.Path(os.environ.get("TABLDM_CKPT_DIR", _PKG_DIR.parent / "checkpoints"))
    ckpts = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


CLF_CKPT = _latest_ckpt("clf_default.ckpt")
REG_CKPT = _latest_ckpt("reg_default.ckpt")


def _clf_kwargs(ckpt):
    return dict(
        n_estimators=4, model_path=str(ckpt), allow_auto_download=False,
        device="cpu", use_amp=False, use_fa3=False, verbose=False, n_jobs=1,
    )


def _reg_kwargs(ckpt):
    return dict(
        n_estimators=4, model_path=str(ckpt), allow_auto_download=False,
        device="cpu", use_amp=False, use_fa3=False, verbose=False, n_jobs=1,
    )


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=40)
    return X, y


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    w = rng.standard_normal(6)
    y = X @ w + 0.1 * rng.standard_normal(40)
    return X, y


@pytest.mark.skipif(CLF_CKPT is None, reason="no clf MoE1 checkpoint")
def test_clf_time_limit_none_matches_generous_limit(clf_data):
    """Contract 1: an unspent budget changes nothing."""
    from tabldm import TabLDMClassifier

    X, y = clf_data
    a = TabLDMClassifier(**_clf_kwargs(CLF_CKPT))
    a.fit(X, y)
    pa = a.predict_proba(X)

    b = TabLDMClassifier(**_clf_kwargs(CLF_CKPT))
    b.fit(X, y, time_limit=3600.0)
    pb = b.predict_proba(X)

    assert pa.shape == pb.shape
    np.testing.assert_array_equal(pa, pb)


@pytest.mark.skipif(CLF_CKPT is None, reason="no clf MoE1 checkpoint")
def test_clf_tiny_time_limit_still_predicts(clf_data):
    """Floor path: a spent budget still yields a usable model."""
    from tabldm import TabLDMClassifier

    X, y = clf_data
    clf = TabLDMClassifier(**_clf_kwargs(CLF_CKPT))
    clf.fit(X, y, time_limit=0.0)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert 0.0 <= clf.shrink_lambda_ <= 1.0
    assert clf.group_prior_ and all(v >= 0.0 for v in clf.group_prior_.values())


@pytest.mark.skipif(REG_CKPT is None, reason="no reg MoE1 checkpoint")
def test_reg_time_limit_none_matches_generous_limit(reg_data):
    from tabldm import TabLDMRegressor

    X, y = reg_data
    a = TabLDMRegressor(**_reg_kwargs(REG_CKPT))
    a.fit(X, y)
    pa = a.predict(X)

    b = TabLDMRegressor(**_reg_kwargs(REG_CKPT))
    b.fit(X, y, time_limit=3600.0)
    pb = b.predict(X)

    np.testing.assert_array_equal(pa, pb)


@pytest.mark.skipif(REG_CKPT is None, reason="no reg MoE1 checkpoint")
def test_reg_tiny_time_limit_still_predicts(reg_data):
    from tabldm import TabLDMRegressor

    X, y = reg_data
    reg = TabLDMRegressor(**_reg_kwargs(REG_CKPT))
    reg.fit(X, y, time_limit=0.0)
    pred = reg.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(pred).all()
