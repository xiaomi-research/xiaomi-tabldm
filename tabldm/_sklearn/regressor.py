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
"""TabLDM Regressor with inference enhancement methods.

This module provides ``TabLDMRegressor``, the public sklearn-style
estimator for TabLDM in-context tabular regression, with multi-group
candidate ensembling, NNLS weight learning, SVD+cross feature
augmentation, high-kurtosis target transforms, and other inference-time
enhancements ported from the MiTabAttnResEnsembleRegressor reference
implementation.

All log messages use the ``[TabLDM:...]`` prefix for consistency with the
Xiaomi-TabLDM project conventions.

Usage::

    from tabldm import TabLDMRegressor

    model = TabLDMRegressor(
        model_path="path/to/regressor_checkpoint.ckpt",
        enhance_candidates=True,
        n_estimators=32,
        n_quantile_estimators=16,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

from __future__ import annotations

import itertools
import math
import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from scipy.optimize import nnls as _scipy_nnls
from scipy.stats import kurtosis as _scipy_kurtosis
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .base import TabLDMBaseEstimator
from .preprocessing import (
    EnsembleGenerator,
    TransformToNumerical,
    UniqueFeatureFilter,
    PipelineEnsemble,
    default_regressor_pipeline_specs,
)
from .sklearn_utils import _moe_load_mismatch, _num_samples, validate_data

from tabldm import InferenceConfig
from tabldm._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tabldm._model.embedding_dual_stream import ColEmbeddingDualStream
from tabldm._model.kv_cache import TabLDMCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ensemble_mean(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """NaN-aware mean across ensemble members.

    Falls back to nanmean when any estimator produces NaN/Inf.
    Samples where all estimators are NaN get filled with 0 (= predict the
    y_train mean in z-space) and a warning is emitted.
    """
    bad_mask = ~np.isfinite(arr)
    if not bad_mask.any():
        return np.mean(arr, axis=axis)
    n_bad = int(bad_mask.sum())
    safe = np.where(bad_mask, np.nan, arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(safe, axis=axis)
    all_nan = ~np.isfinite(out)
    n_all_nan = int(all_nan.sum())
    if n_all_nan:
        out = np.where(all_nan, 0.0, out)
    warnings.warn(
        f"Ensemble produced {n_bad} NaN/Inf logits; recovered with nanmean "
        f"({n_all_nan} samples had all-NaN estimators, replaced with 0=z-space mean)"
    )
    return out


# ---------------------------------------------------------------------------
# Enhanced Regressor
# ---------------------------------------------------------------------------

class TabLDMRegressor(RegressorMixin, TabLDMBaseEstimator):
    """TabLDM Regressor with inference enhancement.

    Supports multi-group candidate ensembling, NNLS weight learning,
    SVD+cross feature augmentation, high-kurtosis target transforms, and
    other inference-time enhancements ported from the
    MiTabAttnResEnsembleRegressor reference implementation. When
    ``enhance_candidates=False`` (default), these enhancements are skipped
    and the estimator runs the plain single-group inference path.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of estimators for the main ensemble group.

    norm_methods : str or list[str] or None, default=None
        Normalization methods for the main group.

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection.

    batch_size : int, "auto", or None, default=4
        Batch size for inference. ``"auto"`` picks a value based on
        ``n_samples_in_ * n_features_in_`` to reduce CUDA memory pressure on
        large datasets (<=1M cells -> 8, <=2M -> 4, <=5M -> 2, else 1).

    kv_cache : bool or str, default=False
        KV cache mode. Not compatible with ``enhance_candidates=True``.

    model_path : Optional[str or Path], default=None
        Path to the pre-trained model checkpoint.

    allow_auto_download : bool, default=True
        Allow automatic download from Hugging Face Hub.

    checkpoint_version : str
        Checkpoint version identifier.

    device : Optional[str or torch.device], default=None
        Device for inference.

    use_amp : bool or "auto", default="auto"
        Automatic mixed precision control.

    use_fa3 : bool or "auto", default="auto"
        Flash Attention 3 control.

    offload_mode : str or bool, default='auto'
        Column embedding offload mode.

    disk_offload_dir : Optional[str], default=None
        Directory for disk offloading.

    random_state : int or None, default=42
        Random seed.

    n_jobs : Optional[int], default=None
        Number of threads for CPU inference.

    verbose : bool, default=False
        Print detailed information.

    inference_config : Optional[InferenceConfig | Dict], default=None
        Fine-grained inference configuration.

    enhance_candidates : bool, default=False
        Master switch for inference enhancement. When False, runs the plain
        single-group inference path with no ensembling enhancements.

    n_quantile_estimators : int, default=16
        Number of quantile-safe candidates.

    use_cross_feature : bool, default=True
        Append SVD + cross features to odd-indexed main candidates.

    validation : bool, default=True
        Enable NNLS weight learning via validation.

    k_fold : bool, default=True
        Use K-Fold OOF for NNLS.

    n_splits : int, default=5
        Number of OOF folds.

    foundation_rate : float, default=0.25
        Blend ratio for foundation (equal-weight) prediction.

    max_num_features : Optional[int], default=500
        Max features before triggering per-estimator sampling.

    enable_high_kurtosis_target_ensemble : bool, default=True
        Enable high-kurtosis target transform ensemble.

    high_kurtosis_threshold : float, default=10.0
        Excess kurtosis threshold to trigger HK ensemble.

    high_kurtosis_n_estimators : int, default=8
        Number of HK candidates (split evenly between asinh and Yeo-Johnson).

    adaptive_enhancement : bool, default=True
        When ``enhance_candidates=True``, adapt the enhancement configuration
        from the effective post-filter feature count, training sample count,
        and target kurtosis. This uses the documented no-K-Fold policy. Set to
        False to preserve all manually supplied enhancement parameters.

    fixed_candidate_names : list[str] or None, default=None
        Names of the only base candidates to fuse for a documented fixed plan.
        Supported names are ``default[i]``, ``quantile[i]``, and the legacy
        positional alias ``svd+cross[i]``. A fixed plan skips holdout/NNLS and
        is valid only for the low-sample, low-dimensional adaptive route.

    fixed_candidate_weights : list[float] or "equal" or None, default=None
        Weights paired with ``fixed_candidate_names``. ``"equal"`` (or None)
        gives each named candidate equal weight. Numeric weights must be
        finite, non-negative, and sum to a positive value.
    """

    def __init__(
        self,
        # -- base parameters --
        n_estimators: int = 8,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: Optional[int | str] = 4,
        kv_cache: bool | str = False,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "checkpoints/reg_default.ckpt",
        device: Optional[str | torch.device] = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
        # -- enhancement parameters --
        enhance_candidates: bool = True,
        n_quantile_estimators: int = 16,
        use_cross_feature: bool = True,
        validation: bool = True,
        k_fold: bool = True,
        n_splits: int = 5,
        foundation_rate: float = 0.25,
        max_num_features: Optional[int] = 300,
        enable_high_kurtosis_target_ensemble: bool = True,
        high_kurtosis_threshold: float = 10.0,
        high_kurtosis_n_estimators: int = 8,
        adaptive_enhancement: bool = True,
        fixed_candidate_names: Optional[List[str]] = None,
        fixed_candidate_weights: Optional[List[float] | Literal["equal"]] = None,
        pipeline_specs=None,
        validation_size: float = 0.2,
    ):
        # base
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config
        # enhancement
        self.enhance_candidates = enhance_candidates
        self.n_quantile_estimators = n_quantile_estimators
        self.use_cross_feature = use_cross_feature
        self.validation = validation
        self.k_fold = k_fold
        self.n_splits = n_splits
        self.foundation_rate = foundation_rate
        self.max_num_features = max_num_features
        self.enable_high_kurtosis_target_ensemble = enable_high_kurtosis_target_ensemble
        self.high_kurtosis_threshold = high_kurtosis_threshold
        self.high_kurtosis_n_estimators = high_kurtosis_n_estimators
        self.adaptive_enhancement = adaptive_enhancement
        self.fixed_candidate_names = fixed_candidate_names
        self.fixed_candidate_weights = fixed_candidate_weights
        self.pipeline_specs = pipeline_specs
        self.validation_size = validation_size

    # ==================================================================
    # Model loading (MoE architecture)
    # ==================================================================

    def _load_model(self) -> None:
        """Load a MoE model from checkpoint.

        Builds a ``TabLDMSparseMoE`` model with a ``ColEmbeddingDualStream``
        column embedder and drops the frozen dense FFN from MoE layers.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        repo_id = "occams/Xiaomi-TabLDM"
        filename = self.checkpoint_version

        if self.model_path is None:
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(
                        f"Checkpoint '{filename}' not cached.\n"
                        f" Downloading from Hugging Face Hub ({repo_id}).\n"
                    )
                    model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    )
            checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                if self.allow_auto_download:
                    print(
                        f"Checkpoint not found at '{model_path_}'.\n"
                        f"Downloading '{filename}' from Hugging Face Hub ({repo_id}) to this location.\n"
                    )
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                        f"'{filename}' from Hugging Face Hub ({repo_id})."
                    )

        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError("The checkpoint must contain 'config' and 'state_dict'.")

        self.model_path_ = model_path_
        config = dict(checkpoint["config"])
        self.model_config_ = config

        # DualStream args
        dual_stream_cfg = checkpoint.get("dual_stream_config", {})
        global_dilation = config.get("global_dilation", dual_stream_cfg.get("global_dilation", "adaptive"))
        global_max_span = config.get("global_max_span", dual_stream_cfg.get("global_max_span", 32))

        parent_config = {
            key: value
            for key, value in config.items()
            if key not in ("global_dilation", "global_max_span")
            and not key.startswith("icl_moe_")
        }
        for legacy_name, native_name in {
            "icl_moe_num_experts": "moe_num_experts",
            "icl_moe_top_k": "moe_top_k",
            "icl_moe_num_shared_experts": "moe_num_shared_experts",
            "icl_moe_layers": "moe_layers",
            "icl_moe_router_z_loss_coef": "moe_router_z_loss_coef",
            "icl_moe_load_balance_loss_coef": "moe_load_balance_loss_coef",
            "icl_moe_router_jitter": "moe_router_jitter",
            "icl_moe_router_weight_mode": "moe_router_weight_mode",
            "icl_moe_expert_init_noise": "moe_expert_init_noise",
            "icl_moe_init_from_dense": "moe_init_from_dense",
        }.items():
            if legacy_name in config:
                parent_config[native_name] = config[legacy_name]

        self.model_ = TabLDMSparseMoE(**parent_config)

        ff_factor = config.get("ff_factor", 2)
        embed_dim = config.get("embed_dim", 128)
        max_classes = config.get("max_classes", 0)

        self.model_.col_embedder = ColEmbeddingDualStream(
            embed_dim=embed_dim,
            num_blocks=config.get("col_num_blocks", 3),
            nhead=config.get("col_nhead", 8),
            dim_feedforward=embed_dim * ff_factor,
            num_inds=config.get("col_num_inds", 128),
            dropout=config.get("dropout", 0.0),
            activation=config.get("activation", "gelu"),
            norm_first=config.get("norm_first", True),
            bias_free_ln=config.get("bias_free_ln", True),
            affine=config.get("col_affine", False),
            feature_group=config.get("col_feature_group", "same"),
            feature_group_size=config.get("col_feature_group_size", 3),
            global_dilation=global_dilation,
            global_max_span=global_max_span,
            target_aware=config.get("col_target_aware", True),
            max_classes=max_classes,
            reserve_cls_tokens=config.get("row_num_cls", 4),
            ssmax=config.get("col_ssmax", False),
            zero_init=config.get("zero_init", False),
            mixed_radix_ensemble=True,
            recompute=False,
        )

        self.model_.drop_dense_ffn()
        state_dict = checkpoint["state_dict"]
        try:
            missing, unexpected = self.model_.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Inference requires an exact current AttnRes/RMSNorm/MoE checkpoint; "
                "legacy checkpoints must be converted during training first."
            ) from exc
        bad_missing, bad_unexpected = _moe_load_mismatch(
            set(self.model_.state_dict().keys()), missing, unexpected
        )
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Inference requires an exact current AttnRes/RMSNorm/MoE checkpoint; "
                "legacy checkpoints must be converted during training first."
            )
        self.model_.eval()

    # ==================================================================
    # High-kurtosis target transform helpers
    # ==================================================================

    def _check_high_kurtosis(self, y: np.ndarray) -> tuple:
        """Return (triggered: bool, kurtosis_value: float).

        Computes excess kurtosis on the full y_train after filtering NaN/Inf.
        Does NOT trigger when:
        - enable_high_kurtosis_target_ensemble is False
        - fewer than 10 valid samples
        - y is near-constant (std < 1e-8)
        - kurtosis computation fails
        """
        if not self.enable_high_kurtosis_target_ensemble:
            return False, float("nan")
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(y_flat)
        y_valid = y_flat[mask]
        if len(y_valid) < 10:
            print("[TabLDM:hk] skipped: too few valid samples")
            return False, float("nan")
        if y_valid.std() < 1e-8:
            print("[TabLDM:hk] skipped: y is near-constant")
            return False, float("nan")
        try:
            kurt = float(_scipy_kurtosis(y_valid, fisher=True, bias=False))
        except Exception:
            print("[TabLDM:hk] skipped: kurtosis computation failed")
            return False, float("nan")
        if not np.isfinite(kurt):
            print("[TabLDM:hk] skipped: kurtosis is not finite")
            return False, float("nan")
        triggered = kurt > self.high_kurtosis_threshold
        print(
            f"[TabLDM:hk] kurtosis={kurt:.4f}, threshold={self.high_kurtosis_threshold}, "
            f"triggered={triggered}"
        )
        return triggered, kurt

    def _fit_asinh_transformer(self, y: np.ndarray) -> dict:
        """Fit and return asinh transform params from y (original scale)."""
        y_f = np.asarray(y, dtype=np.float64).ravel()
        center = float(np.median(y_f))
        q75, q25 = np.percentile(y_f, [75, 25])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = float(y_f.std())
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return {"center": center, "scale": scale}

    def _asinh_transform(self, y: np.ndarray, params: dict) -> np.ndarray:
        return np.arcsinh((y - params["center"]) / params["scale"]).astype(np.float32)

    def _asinh_inverse(self, z: np.ndarray, params: dict) -> np.ndarray:
        return (params["center"] + params["scale"] * np.sinh(z)).astype(np.float64)

    def _make_hk_generators(self, X_tr: np.ndarray, y_tr_orig: np.ndarray, seed_offset: int = 0):
        """Fit HK ensemble generators for one split.

        Returns list of dicts, each with keys:
          'kind'        : 'asinh' or 'yeojohnson'
          'gen'         : fitted EnsembleGenerator  (y fed in transformed scale)
          'asinh_params': dict (only for kind='asinh')
          'yj_pt'       : PowerTransformer (only for kind='yeojohnson')
          'seed'        : int
        """
        n_hk = self.high_kurtosis_n_estimators  # 8
        n_asinh = n_hk // 2    # 4
        n_yj = n_hk - n_asinh  # 4
        base_seed = (self.random_state if self.random_state is not None else 0) + seed_offset

        norm_methods = self.norm_methods or ["none", "power"]
        results = []

        for i in range(n_asinh):
            seed = base_seed + 1000 + i
            params = self._fit_asinh_transformer(y_tr_orig)
            try:
                y_tr_t = self._asinh_transform(y_tr_orig, params)
                ss = StandardScaler()
                y_tr_ts = ss.fit_transform(y_tr_t.reshape(-1, 1)).flatten().astype(np.float32)
                gen = EnsembleGenerator(
                    classification=False,
                    n_estimators=1,
                    norm_methods=[norm_methods[i % len(norm_methods)]],
                    feat_shuffle_method=self.feat_shuffle_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=seed,
                )
                gen.fit(X_tr, y_tr_ts)
                results.append({
                    "kind": "asinh",
                    "gen": gen,
                    "asinh_params": params,
                    "asinh_ss": ss,
                    "yj_pt": None,
                    "seed": seed,
                })
                print(f"[TabLDM:hk] asinh[{i}] seed={seed} center={params['center']:.4f} scale={params['scale']:.4f}")
            except Exception as e:
                print(f"[TabLDM:hk] asinh[{i}] failed: {e}, skipping")

        for i in range(n_yj):
            seed = base_seed + 2000 + i
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                y_tr_t = pt.fit_transform(y_tr_orig.reshape(-1, 1)).flatten().astype(np.float32)
                gen = EnsembleGenerator(
                    classification=False,
                    n_estimators=1,
                    norm_methods=[norm_methods[i % len(norm_methods)]],
                    feat_shuffle_method=self.feat_shuffle_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=seed,
                )
                gen.fit(X_tr, y_tr_t)
                results.append({
                    "kind": "yeojohnson",
                    "gen": gen,
                    "asinh_params": None,
                    "asinh_ss": None,
                    "yj_pt": pt,
                    "seed": seed,
                })
                print(f"[TabLDM:hk] yj[{i}] seed={seed}")
            except Exception as e:
                print(f"[TabLDM:hk] yj[{i}] failed: {e}, skipping")

        return results

    def _hk_inverse(self, z: np.ndarray, hk_info: dict) -> np.ndarray:
        """Inverse transform HK estimator predictions back to original scale."""
        kind = hk_info["kind"]
        if kind == "asinh":
            ss = hk_info["asinh_ss"]
            params = hk_info["asinh_params"]
            z_unscaled = ss.inverse_transform(z.reshape(-1, 1)).flatten()
            return self._asinh_inverse(z_unscaled, params)
        else:
            pt = hk_info["yj_pt"]
            return pt.inverse_transform(z.reshape(-1, 1)).flatten().astype(np.float64)

    def _collect_hk_val_predictions_orig(
        self,
        X_val: np.ndarray,
        hk_infos: list,
    ) -> np.ndarray:
        """Run each HK estimator on val set, return (n_hk_valid, n_val) in ORIGINAL scale."""
        output_type = ["mean"]
        all_preds = []
        for hk in hk_infos:
            gen = hk["gen"]
            try:
                data = gen.transform(X_val, mode="both")
                for Xs, ys in data.values():
                    bout = self._batch_forward(Xs, ys, output_type=output_type)
                    p = bout if not isinstance(bout, dict) else bout["mean"]
                    p_orig = self._hk_inverse(p[0], hk)
                    if not np.all(np.isfinite(p_orig)):
                        print(f"[TabLDM:hk] {hk['kind']} seed={hk['seed']}: val preds contain NaN/Inf, skipping")
                        continue
                    all_preds.append(p_orig[np.newaxis, :])
            except Exception as e:
                print(f"[TabLDM:hk] {hk['kind']} seed={hk['seed']}: val prediction failed: {e}, skipping")
        if all_preds:
            return np.concatenate(all_preds, axis=0)
        return np.empty((0, X_val.shape[0]), dtype=np.float64)

    def _collect_hk_test_predictions_orig(
        self,
        X: np.ndarray,
        hk_infos: list,
    ) -> np.ndarray:
        """Run each HK estimator on test X, return (n_hk_valid, n_test) in ORIGINAL scale."""
        output_type = ["mean"]
        all_preds = []
        for hk in hk_infos:
            gen = hk["gen"]
            try:
                data = gen.transform(X, mode="both")
                for Xs, ys in data.values():
                    bout = self._batch_forward(Xs, ys, output_type=output_type)
                    p = bout if not isinstance(bout, dict) else bout["mean"]
                    p_orig = self._hk_inverse(p[0], hk)
                    if not np.all(np.isfinite(p_orig)):
                        print(f"[TabLDM:hk] {hk['kind']} seed={hk['seed']}: test preds contain NaN/Inf, skipping")
                        continue
                    all_preds.append(p_orig[np.newaxis, :])
            except Exception as e:
                print(f"[TabLDM:hk] {hk['kind']} seed={hk['seed']}: test prediction failed: {e}, skipping")
        if all_preds:
            return np.concatenate(all_preds, axis=0)
        return np.empty((0,), dtype=np.float64)

    # ==================================================================
    # Validation routing
    # ==================================================================

    @staticmethod
    def _route_validation(validation: bool, k_fold: bool, n_train: int, n_val: int) -> str:
        """Decide which validation path fit() takes. Returns one of:
        'default', 'single_validation', 'kfold'.

        Decision table:
          | k_fold | condition       | path                     |
          |--------|-----------------|--------------------------|
          | False  | n_train < 1000  | default                  |
          | False  | n_train >= 1000 | single_validation        |
          | True   | n_val > 2000    | single_validation        |
          | True   | n_val <= 2000   | kfold                    |
        """
        if not validation:
            return "default"
        if not k_fold:
            return "single_validation" if n_train >= 1000 else "default"
        return "single_validation" if n_val > 2000 else "kfold"

    @staticmethod
    def _adaptive_enhancement_config(
        n_train: int, n_features: int, has_fixed_plan: bool = False
    ) -> dict[str, Any]:
        """Return the documented no-K-Fold regression route.

        The fixed route is selected only when the caller supplies a named
        candidate plan.  Dataset identity is intentionally not inferred here;
        the evaluation runner owns the dataset-specific candidate table.
        """
        config = {
            "k_fold": False,
            "use_cross_feature": False,
            "foundation_rate": 0.0,
            "n_estimators": 8,
        }
        if n_features > 100:
            return {
                **config,
                "branch": "nnls_high_dimensional",
                "validation": True,
                "n_quantile_estimators": 16,
            }
        if n_train < 1_000 and n_features <= 10:
            return {
                **config,
                "branch": "fixed" if has_fixed_plan else "fixed_plan_required",
                "validation": False,
                "n_quantile_estimators": 8,
            }
        if n_train < 5_000:
            return {
                **config,
                "branch": "nnls_medium",
                "validation": True,
                "n_quantile_estimators": 8,
            }
        return {
            **config,
            "branch": "nnls_large",
            "validation": True,
            "n_quantile_estimators": 16,
        }

    @staticmethod
    def _parse_candidate_name(name: str) -> tuple[str, int]:
        """Parse a fixed-plan candidate name into group and local index."""
        if not isinstance(name, str):
            raise TypeError("fixed_candidate_names must contain strings")
        match = re.fullmatch(r"(default|quantile|svd\+cross)\[(\d+)\]", name)
        if match is None:
            raise ValueError(
                f"Invalid fixed candidate name {name!r}; expected default[i], "
                "quantile[i], or svd+cross[i]."
            )
        return match.group(1), int(match.group(2))

    def _resolve_fixed_candidate_plan(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Resolve named fixed candidates to global base-candidate indices."""
        names = self.fixed_candidate_names
        if names is None:
            return None
        if not names:
            raise ValueError("fixed_candidate_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("fixed_candidate_names must not contain duplicates")

        indices = []
        for name in names:
            group, local_idx = self._parse_candidate_name(name)
            if group in {"default", "svd+cross"}:
                if local_idx >= self.n_estimators:
                    raise ValueError(
                        f"Fixed candidate {name!r} is unavailable: main group has "
                        f"{self.n_estimators} candidates."
                    )
                indices.append(local_idx)
            else:
                if local_idx >= self.n_quantile_estimators:
                    raise ValueError(
                        f"Fixed candidate {name!r} is unavailable: quantile group has "
                        f"{self.n_quantile_estimators} candidates."
                    )
                indices.append(self.n_estimators + local_idx)

        weights = self.fixed_candidate_weights
        if weights is None or (isinstance(weights, str) and weights == "equal"):
            resolved_weights = np.ones(len(indices), dtype=np.float64)
        else:
            if isinstance(weights, str) or len(weights) != len(indices):
                raise ValueError(
                    "fixed_candidate_weights must be 'equal' or have one value per candidate"
                )
            resolved_weights = np.asarray(weights, dtype=np.float64)
            if (
                not np.isfinite(resolved_weights).all()
                or (resolved_weights < 0).any()
                or not float(resolved_weights.sum()) > 0
            ):
                raise ValueError(
                    "fixed_candidate_weights must be finite, non-negative, and have a positive sum"
                )
        resolved_weights /= resolved_weights.sum()
        return np.asarray(indices, dtype=np.int64), resolved_weights

    def _configure_adaptive_enhancement(
        self, n_train: int, n_features: int, y: np.ndarray
    ) -> tuple[bool, float]:
        """Apply the documented route and return the HK status."""
        config = self._adaptive_enhancement_config(
            n_train, n_features, has_fixed_plan=self.fixed_candidate_names is not None
        )
        if config["branch"] == "fixed_plan_required":
            raise ValueError(
                "Adaptive regression enhancement requires fixed_candidate_names for "
                "datasets with n_train < 1000 and n_features <= 10."
            )
        if self.fixed_candidate_names is not None and config["branch"] != "fixed":
            raise ValueError(
                "fixed_candidate_names is supported only by the low-sample, "
                "low-dimensional adaptive route."
            )
        if self.fixed_candidate_weights is not None and self.fixed_candidate_names is None:
            raise ValueError(
                "fixed_candidate_weights requires fixed_candidate_names."
            )
        for name, value in config.items():
            if name != "branch":
                setattr(self, name, value)

        self.enable_high_kurtosis_target_ensemble = True
        self.high_kurtosis_threshold = 10.0
        self.high_kurtosis_n_estimators = 8
        triggered, kurtosis_value = self._check_high_kurtosis(y)
        self.adaptive_enhancement_config_ = {
            **config,
            "n_train": n_train,
            "n_features": n_features,
            "target_kurtosis": kurtosis_value,
            "high_kurtosis_triggered": triggered,
            "high_kurtosis_threshold": self.high_kurtosis_threshold,
            "high_kurtosis_n_estimators": self.high_kurtosis_n_estimators,
            "fixed_candidate_names": None if self.fixed_candidate_names is None else list(self.fixed_candidate_names),
        }
        print(
            f"[TabLDM:enhance] adaptive routing: branch={config['branch']}, "
            f"n_train={n_train}, n_features={n_features}, "
            f"validation={self.validation}, main={self.n_estimators}, "
            f"quantile={self.n_quantile_estimators}, foundation_rate={self.foundation_rate}, "
            f"k_fold={self.k_fold}, use_cross_feature={self.use_cross_feature}"
        )
        return triggered, kurtosis_value


    def _fit_svd_cross(self, X_raw: np.ndarray, attr_prefix: str = "") -> None:
        """Fit SVD pipeline and build crossing pool on unique-filtered raw training features.

        Parameters
        ----------
        X_raw : np.ndarray
            Raw training features after unique filtering (before normalization).
        attr_prefix : str, default=""
            Prefix for attribute names. Use ``""`` for the main group (default).
        """
        rng = np.random.default_rng(self.random_state)
        n_train, n_orig = X_raw.shape

        is_cat = np.array(
            [np.issubdtype(X_raw[:, c].dtype, np.integer) or len(np.unique(X_raw[:, c])) <= 10
             for c in range(n_orig)],
            dtype=bool,
        )
        num_idx = np.where(~is_cat)[0]
        cat_idx = np.where(is_cat)[0]

        k = max(1, int(math.sqrt(n_orig)))
        setattr(self, f"{attr_prefix}_k_", k)

        # ---- Crossing pool (numerical cols only) ----
        if not self.use_cross_feature:
            setattr(self, f"{attr_prefix}_cross_pool_", [])
            setattr(self, f"{attr_prefix}_cross_scaler_", None)
            setattr(self, f"{attr_prefix}_cross_pool_train_", np.empty((n_train, 0), dtype=np.float32))
        elif len(num_idx) >= 2:
            all_pairs = list(itertools.combinations(num_idx.tolist(), 2))
            cross_pool_size = min(16 * k, len(all_pairs))
            pool_sel = rng.choice(len(all_pairs), size=cross_pool_size, replace=False)
            cross_pool = [all_pairs[i] for i in pool_sel]
            setattr(self, f"{attr_prefix}_cross_pool_", cross_pool)
            cross_mat = np.stack(
                [X_raw[:, i].astype(float) * X_raw[:, j].astype(float) for i, j in cross_pool],
                axis=1,
            )
            cross_scaler = StandardScaler()
            setattr(self, f"{attr_prefix}_cross_scaler_", cross_scaler)
            setattr(self, f"{attr_prefix}_cross_pool_train_",
                    cross_scaler.fit_transform(cross_mat).clip(-100, 100))
        else:
            setattr(self, f"{attr_prefix}_cross_pool_", [])
            setattr(self, f"{attr_prefix}_cross_scaler_", None)
            setattr(self, f"{attr_prefix}_cross_pool_train_", np.empty((n_train, 0), dtype=np.float32))

        # Per-estimator crossing selections for 16 augmented slots
        cross_pool = getattr(self, f"{attr_prefix}_cross_pool_")
        cross_selections = []
        for _ in range(16):
            n_sel = min(k, len(cross_pool))
            if n_sel > 0:
                sel = rng.choice(len(cross_pool), size=n_sel, replace=False).tolist()
            else:
                sel = []
            cross_selections.append(sel)
        setattr(self, f"{attr_prefix}_cross_selections_", cross_selections)

        # ---- SVD pool ----
        if len(cat_idx) > 0 and len(num_idx) > 0:
            svd_pre = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx.tolist()),
                 ("num", StandardScaler(), num_idx.tolist())],
                remainder="drop",
            )
        elif len(cat_idx) > 0:
            svd_pre = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx.tolist())],
                remainder="drop",
            )
        else:
            svd_pre = ColumnTransformer(
                [("num", StandardScaler(), num_idx.tolist())],
                remainder="drop",
            )

        X_pre = svd_pre.fit_transform(X_raw)
        n_preprocessed = X_pre.shape[1]
        svd_pool_size = min(16 * k, min(n_train, n_preprocessed) - 1)

        if svd_pool_size < 1:
            setattr(self, f"{attr_prefix}svd_", None)
            setattr(self, f"{attr_prefix}_svd_pre_", None)
            setattr(self, f"{attr_prefix}_X_train_svd_", np.empty((n_train, 0), dtype=np.float32))
            setattr(self, f"{attr_prefix}_svd_selections_", [[] for _ in range(16)])
            return

        svd = TruncatedSVD(n_components=svd_pool_size, random_state=self.random_state)
        setattr(self, f"{attr_prefix}_svd_pre_", svd_pre)
        setattr(self, f"{attr_prefix}svd_", svd)
        setattr(self, f"{attr_prefix}_X_train_svd_",
                svd.fit_transform(X_pre).clip(-100, 100).astype(np.float32))

        svd_selections = []
        for _ in range(16):
            n_sel = min(k, svd_pool_size)
            sel = rng.choice(svd_pool_size, size=n_sel, replace=False).tolist()
            svd_selections.append(sel)
        setattr(self, f"{attr_prefix}_svd_selections_", svd_selections)

    def _augment_estimator(
        self,
        X_preprocessed: np.ndarray,
        raw_X: np.ndarray,
        odd_local: int,
        is_train: bool,
        attr_prefix: str = "",
    ) -> np.ndarray:
        """Append selected cross and SVD features to preprocessed features."""
        parts = [X_preprocessed]

        cross_selections = getattr(self, f"{attr_prefix}_cross_selections_")
        cross_sel = cross_selections[odd_local]
        if cross_sel:
            cross_pool = getattr(self, f"{attr_prefix}_cross_pool_")
            cross_pool_train = getattr(self, f"{attr_prefix}_cross_pool_train_")
            cross_scaler = getattr(self, f"{attr_prefix}_cross_scaler_")
            if is_train:
                cross_feats = cross_pool_train[:, cross_sel].astype(np.float32)
            else:
                all_cross_mat = np.stack(
                    [raw_X[:, i].astype(float) * raw_X[:, j].astype(float)
                     for i, j in cross_pool],
                    axis=1,
                )
                all_cross_scaled = cross_scaler.transform(all_cross_mat).clip(-100, 100)
                cross_feats = all_cross_scaled[:, cross_sel].astype(np.float32)
            parts.append(cross_feats)

        svd_selections = getattr(self, f"{attr_prefix}_svd_selections_")
        svd_sel = svd_selections[odd_local]
        svd = getattr(self, f"{attr_prefix}svd_")
        if svd_sel and svd is not None:
            svd_pre = getattr(self, f"{attr_prefix}_svd_pre_")
            X_train_svd = getattr(self, f"{attr_prefix}_X_train_svd_")
            if is_train:
                svd_feats = X_train_svd[:, svd_sel]
            else:
                svd_feats = svd.transform(svd_pre.transform(raw_X))[:, svd_sel].clip(-100, 100).astype(np.float32)
            parts.append(svd_feats)

        return np.concatenate(parts, axis=1) if len(parts) > 1 else X_preprocessed

    # ==================================================================
    # Forward helpers
    # ==================================================================

    def _batch_forward(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        output_type: str | list[str] = "mean",
        alphas: Optional[List[float]] = None,
        feat_indices: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes in batches to manage memory efficiently."""
        if feat_indices is not None:
            Xs = np.stack(
                [Xs[i][:, feat_indices[i]] for i in range(Xs.shape[0])], axis=0
            )

        batch_size = self.batch_size_ or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {key: [] for key in output_type}

        for X_batch, y_batch in zip(Xs, ys):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)

            with torch.no_grad():
                out = self.model_.predict_stats(
                    X_batch,
                    y_batch,
                    output_type=output_type,
                    alphas=alphas,
                    inference_config=self.inference_config_,
                )
                if isinstance(out, dict):
                    for key in output_type:
                        results[key].append(out[key].float().cpu().numpy())
                else:
                    results[output_type[0]].append(out.float().cpu().numpy())

        for key in results:
            results[key] = np.concatenate(results[key], axis=0)

        if len(output_type) == 1:
            return results[output_type[0]]
        return results

    def _batch_forward_with_cache(
        self,
        Xs: np.ndarray,
        kv_cache: TabLDMCache,
        output_type: str | list[str] = "mean",
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes using a pre-computed KV cache."""
        n_total = Xs.shape[0]
        batch_size = self.batch_size_ or n_total
        n_batches = int(np.ceil(n_total / batch_size))
        Xs_split = np.array_split(Xs, n_batches)

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {key: [] for key in output_type}

        offset = 0
        for X_batch in Xs_split:
            bs = X_batch.shape[0]
            cache_subset = kv_cache.slice_batch(offset, offset + bs)
            offset += bs

            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            with torch.no_grad():
                out = self.model_.predict_stats_with_cache(
                    X_test=X_batch,
                    output_type=output_type,
                    alphas=alphas,
                    cache=cache_subset,
                    inference_config=self.inference_config_,
                )
                if isinstance(out, dict):
                    for key in output_type:
                        results[key].append(out[key].float().cpu().numpy())
                else:
                    results[output_type[0]].append(out.float().cpu().numpy())

        for key in results:
            results[key] = np.concatenate(results[key], axis=0)

        if len(output_type) == 1:
            return results[output_type[0]]
        return results

    # ==================================================================
    # Generator fitting
    # ==================================================================

    def _fit_full_generators(self, X: np.ndarray, y_scaled: np.ndarray) -> None:
        """Fit ensemble_generator_, quantile_ensemble_generator_, and HK generators."""
        norm_methods = self.norm_methods or ["none", "power"]

        # Main EnsembleGenerator
        self.ensemble_generator_ = EnsembleGenerator(
            classification=False,
            n_estimators=self.n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y_scaled)

        # SVD + crossing augmentation for odd estimators in main group
        self.svd_ = None
        if self.n_estimators == 32 and sorted(norm_methods) == ["none", "power"]:
            self._fit_svd_cross(self.ensemble_generator_.X_)

        # Quantile-only EnsembleGenerator
        self.quantile_ensemble_generator_ = None
        if self.n_quantile_estimators > 0:
            self.quantile_ensemble_generator_ = EnsembleGenerator(
                classification=False,
                n_estimators=self.n_quantile_estimators,
                norm_methods=["quantile"],
                feat_shuffle_method=self.feat_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            self.quantile_ensemble_generator_.fit(X, y_scaled)

        print(
            f"[TabLDM:enhance] Estimator groups: main={self.n_estimators}, "
            f"quantile={self.n_quantile_estimators}, "
            f"total={self.n_estimators + self.n_quantile_estimators}"
        )

        # HK full-train generators
        self.hk_generators_ = []
        if self.hk_triggered_:
            y_orig_full = self.y_scaler_.inverse_transform(
                y_scaled.reshape(-1, 1)
            ).flatten()
            self.hk_generators_ = self._make_hk_generators(X, y_orig_full, seed_offset=0)
            print(f"[TabLDM:hk] full-train HK generators fitted: {len(self.hk_generators_)}")

    def _make_and_fit_generators(self, X_tr: np.ndarray, y_tr: np.ndarray):
        """Fit validation generators in the full-train feature space."""
        main_frozen_filter = getattr(self.ensemble_generator_, "unique_filter_", None)
        main_gen = EnsembleGenerator(
            classification=False,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        main_gen.fit(X_tr, y_tr, frozen_unique_filter=main_frozen_filter)

        # Save and wire SVD/cross state
        svd_attrs = ["svd_", "_svd_pre_", "_X_train_svd_", "_svd_selections_",
                     "_cross_pool_", "_cross_scaler_", "_cross_pool_train_",
                     "_cross_selections_", "_k_"]
        saved_svd = {a: getattr(self, a, None) for a in svd_attrs}

        val_svd = None
        if self.n_estimators == 32 and sorted(self.norm_methods or ["none", "power"]) == ["none", "power"]:
            self._fit_svd_cross(main_gen.X_)
            val_svd = self.svd_
        else:
            self.svd_ = None

        q_gen = None
        if self.n_quantile_estimators > 0:
            q_frozen_filter = getattr(
                getattr(self, "quantile_ensemble_generator_", None),
                "unique_filter_",
                None,
            )
            q_gen = EnsembleGenerator(
                classification=False,
                n_estimators=self.n_quantile_estimators,
                norm_methods=["quantile"],
                feat_shuffle_method=self.feat_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            q_gen.fit(X_tr, y_tr, frozen_unique_filter=q_frozen_filter)

        return {
            "main_gen": main_gen,
            "val_svd": val_svd,
            "q_gen": q_gen,
            "saved_svd": saved_svd,
        }

    def _collect_val_predictions(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        val_gen_info: dict,
    ) -> np.ndarray:
        """Run all estimators on the validation set and return (E, n_val) predictions."""
        main_gen: EnsembleGenerator = val_gen_info["main_gen"]
        q_gen = val_gen_info["q_gen"]
        output_type = ["mean"]

        all_preds = []
        fsi = getattr(self, "feat_sample_indices_", None)

        # ---- main group ----
        use_augment = (
            val_gen_info["val_svd"] is not None
            and self.n_estimators == 32
        )
        if use_augment:
            data = main_gen.transform(X_val, mode="both")
            X_raw_val = main_gen.unique_filter_.transform(X_val)
            X_raw_train = main_gen.X_

            global_est_idx = 0
            odd_local = 0
            for norm_method, (Xs_both, ys) in data.items():
                n_est_this_method = Xs_both.shape[0]
                n_train_rows = ys.shape[1]
                even_idxs, odd_idxs, odd_locals = [], [], []
                even_global, odd_global = [], []
                for local in range(n_est_this_method):
                    if global_est_idx % 2 == 0:
                        even_idxs.append(local)
                        even_global.append(global_est_idx)
                    else:
                        odd_idxs.append(local)
                        odd_locals.append(odd_local)
                        odd_global.append(global_est_idx)
                        odd_local += 1
                    global_est_idx += 1

                per_est_preds = {}

                if even_idxs:
                    fi = [fsi[g] for g in even_global] if fsi is not None else None
                    bout = self._batch_forward(
                        Xs_both[even_idxs], ys[even_idxs], output_type=output_type, feat_indices=fi
                    )
                    preds_val = bout if not isinstance(bout, dict) else bout["mean"]
                    for i, li in enumerate(even_idxs):
                        per_est_preds[li] = preds_val[i]

                if odd_idxs:
                    Xs_odd_list = []
                    for li, ol in zip(odd_idxs, odd_locals):
                        x_both_i = Xs_both[li]
                        x_tr = self._augment_estimator(
                            x_both_i[:n_train_rows], X_raw_train, ol, is_train=True)
                        x_te = self._augment_estimator(
                            x_both_i[n_train_rows:], X_raw_val, ol, is_train=False)
                        Xs_odd_list.append(np.concatenate([x_tr, x_te], axis=0))
                    Xs_odd = np.stack(Xs_odd_list, axis=0)
                    fi = [fsi[g] for g in odd_global] if fsi is not None else None
                    bout = self._batch_forward(
                        Xs_odd, ys[odd_idxs], output_type=output_type, feat_indices=fi
                    )
                    preds_val = bout if not isinstance(bout, dict) else bout["mean"]
                    for i, li in enumerate(odd_idxs):
                        per_est_preds[li] = preds_val[i]

                stacked = np.stack(
                    [per_est_preds[li] for li in range(n_est_this_method)], axis=0
                )
                all_preds.append(stacked)
        else:
            data = main_gen.transform(X_val, mode="both")
            _grp_offset = 0
            for norm_method, (Xs, ys) in data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_grp_offset + i] for i in range(n_est)] if fsi is not None else None
                bout = self._batch_forward(Xs, ys, output_type=output_type, feat_indices=fi)
                preds_val = bout if not isinstance(bout, dict) else bout["mean"]
                all_preds.append(preds_val)
                _grp_offset += n_est

        # ---- quantile group ----
        if q_gen is not None:
            q_data = q_gen.transform(X_val, mode="both")
            _q_offset = self.n_estimators
            for norm_method, (Xs, ys) in q_data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_q_offset + i] for i in range(n_est)] if fsi is not None else None
                bout = self._batch_forward(Xs, ys, output_type=output_type, feat_indices=fi)
                preds_val = bout if not isinstance(bout, dict) else bout["mean"]
                all_preds.append(preds_val)
                _q_offset += n_est

        # Restore SVD attributes on self to their pre-validation state
        for attr, val in val_gen_info["saved_svd"].items():
            if val is None and hasattr(self, attr):
                delattr(self, attr)
            elif val is not None:
                setattr(self, attr, val)

        return np.concatenate(all_preds, axis=0)

    def _candidate_name(self, idx: int, n_main: int) -> str:
        """Human-readable candidate label for NNLS weight logging."""
        if idx < n_main:
            uses_svd_cross = (
                getattr(self, "svd_", None) is not None
                and n_main == 32
            )
            if uses_svd_cross and idx % 2:
                return f"svd+cross[{idx}]"
            return f"default[{idx}]"
        n_q = self.n_quantile_estimators if getattr(self, "n_quantile_estimators", 0) > 0 else 0
        if idx < n_main + n_q:
            return f"quantile[{idx - n_main}]"
        hk_idx = idx - n_main - n_q
        hk_infos = getattr(self, "hk_generators_", [])
        if hk_idx < len(hk_infos):
            h = hk_infos[hk_idx]
            return f"hk_{h['kind']}[{hk_idx}]"
        return f"hk[{hk_idx}]"

    def _build_kv_cache(self) -> None:
        """Pre-compute KV caches for training data across all ensemble batches."""

        def _cache_generator(generator):
            train_data = generator.transform(X=None, mode="train")
            kv_cache_dict = OrderedDict()
            for norm_method, (Xs, ys) in train_data.items():
                batch_size = self.batch_size_ or Xs.shape[0]
                n_batches = int(np.ceil(Xs.shape[0] / batch_size))
                Xs_split = np.array_split(Xs, n_batches)
                ys_split = np.array_split(ys, n_batches)

                caches = []
                for X_batch, y_batch in zip(Xs_split, ys_split):
                    X_batch = torch.from_numpy(X_batch).float().to(self.device_)
                    y_batch = torch.from_numpy(y_batch).float().to(self.device_)
                    with torch.no_grad():
                        self.model_.predict_stats_with_cache(
                            X_train=X_batch,
                            y_train=y_batch,
                            use_cache=False,
                            store_cache=True,
                            cache_mode=self.cache_mode_,
                            inference_config=self.inference_config_,
                        )
                    caches.append(self.model_._cache)
                    self.model_.clear_cache()

                kv_cache_dict[norm_method] = TabLDMCache.concat(caches)
            return kv_cache_dict

        self.model_kv_cache_ = _cache_generator(self.ensemble_generator_)

        # Build KV cache for quantile group
        self.quantile_kv_cache_ = None
        if self.quantile_ensemble_generator_ is not None:
            self.quantile_kv_cache_ = _cache_generator(self.quantile_ensemble_generator_)

    # ==================================================================
    # Feature sampling
    # ==================================================================

    def _apply_feat_sample(self, X: np.ndarray, est_idx: int) -> np.ndarray:
        """Return X with selected columns if feature sampling is active for est_idx."""
        if self.feat_sample_indices_ is None or est_idx >= len(self.feat_sample_indices_):
            return X
        return X[:, self.feat_sample_indices_[est_idx]]

    def _fit_pipeline_enhancement(self, X: np.ndarray, y: np.ndarray, y_scaled: np.ndarray) -> None:
        """Fit LimiX V2 regression views and one holdout NNLS ensemble."""
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be in (0, 1)")
        self.pipeline_specs_ = tuple(self.pipeline_specs or default_regressor_pipeline_specs())
        categorical_indices = getattr(self.X_encoder_, "categorical_indices_", [])
        self.pipeline_member_names_ = [spec.name for spec in self.pipeline_specs_]
        self.pipeline_validation_audit_ = []
        self.pipeline_failed_members_ = []
        valid_indices = list(range(len(self.pipeline_specs_)))
        weights = None

        if self.validation:
            try:
                X_tr, X_val, y_tr_scaled, _, _, y_val = train_test_split(
                    X, y_scaled, y, test_size=self.validation_size, shuffle=True,
                    random_state=self.random_state,
                )
                holdout = PipelineEnsemble(
                    classification=False, specs=self.pipeline_specs_, categorical_indices=categorical_indices,
                    random_state=self.random_state,
                ).fit(X_tr, y_tr_scaled)
                predictions, successful = [], []
                for index, member in zip(holdout.member_indices_, holdout.members_):
                    try:
                        prediction = self._pipeline_member_predictions(member, X_val)
                        prediction = self.y_scaler_.inverse_transform(prediction.reshape(-1, 1)).ravel()
                        if prediction.shape != y_val.shape or not np.isfinite(prediction).all():
                            raise ValueError(f"invalid prediction shape or values: {prediction.shape}")
                    except Exception as exc:
                        self.pipeline_failed_members_.append({"index": index, "name": member.spec.name, "stage": "validation", "reason": repr(exc)})
                        continue
                    predictions.append(prediction)
                    successful.append(index)
                self.pipeline_failed_members_.extend(holdout.failed_members_)
                if successful:
                    valid_indices = successful
                    matrix = np.column_stack(predictions)
                    try:
                        raw_weights, _ = _scipy_nnls(matrix, y_val)
                        if np.isfinite(raw_weights).all() and raw_weights.sum() > 0:
                            weights = raw_weights / raw_weights.sum()
                        else:
                            raise ValueError("degenerate NNLS result")
                    except Exception as exc:
                        weights = np.full(len(successful), 1 / len(successful), dtype=np.float64)
                        self.pipeline_validation_audit_.append({"status": "equal_weight_fallback", "reason": repr(exc)})
                    self.pipeline_validation_audit_.append({"status": "nnls", "n_validation": len(y_val), "valid_member_indices": successful})
                else:
                    valid_indices = []
                    self.pipeline_validation_audit_.append({"status": "no_valid_members", "reason": "all holdout members failed"})
            except Exception as exc:
                self.pipeline_validation_audit_.append({"status": "split_failed", "reason": repr(exc)})

        full = PipelineEnsemble(
            classification=False, specs=self.pipeline_specs_, categorical_indices=categorical_indices,
            random_state=self.random_state,
        ).fit(X, y_scaled, member_indices=valid_indices)
        self.pipeline_failed_members_.extend(full.failed_members_)
        self.pipeline_members_ = full.members_
        self.nnls_valid_member_indices_ = full.member_indices_
        if not self.pipeline_members_:
            raise RuntimeError("All LimiX pipeline members failed during full-data fitting.")
        if weights is None or len(weights) != len(self.pipeline_members_):
            weights = np.full(len(self.pipeline_members_), 1 / len(self.pipeline_members_), dtype=np.float64)
        self.nnls_weights_ = weights
        self.ensemble_generator_ = None
        self.hk_triggered_ = False
        self.feat_sample_indices_ = None

    def _pipeline_member_predictions(self, member, X: np.ndarray) -> np.ndarray:
        X_both = np.concatenate([member.X_train_, member.transform(X)], axis=0)[None, ...]
        y_train = np.asarray(member.y_train_, dtype=np.float32)[None, ...]
        output = self._batch_forward(X_both, y_train, output_type="mean")
        return np.asarray(output)[0]


    def fit(self, X: np.ndarray, y: np.ndarray, kv_cache: bool | str = False) -> "TabLDMRegressor":
        """Fit the regressor to training data.

        When ``enhance_candidates=True``, fits multiple candidate generators
        and optionally learns NNLS ensemble weights via validation/OOF.
        When ``enhance_candidates=False``, runs the plain single-group
        inference path.
        """
        if y is None:
            raise ValueError("This regressor requires y to be passed, but the target y is None.")

        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 2 and y.shape[1] == 1:
            from sklearn.exceptions import DataConversionWarning
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. Please change "
                "the shape of y to (n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
            y = y.ravel()

        # Device setup
        self._resolve_device()
        self.n_samples_in_ = _num_samples(X)
        self._build_inference_config()

        # Load model
        self._load_model()
        self.model_.to(self.device_)

        # Scale target values
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).flatten()

        # Transform input features
        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)

        # Initialize enhancement state
        self.nnls_weights_ = None

        if self.enhance_candidates:
            if self.kv_cache:
                raise ValueError(
                    "kv_cache is not supported together with enhance_candidates=True. "
                    "Disable one of them."
                )
            self._fit_pipeline_enhancement(X, y, y_scaled)
            self.n_features_in_ = X.shape[1]
        else:
            # ---- Non-enhanced path ----
            self.hk_triggered_ = False
            self.feat_sample_indices_ = None
            self.ensemble_generator_ = EnsembleGenerator(
                classification=False,
                n_estimators=self.n_estimators,
                norm_methods=self.norm_methods or ["none", "power"],
                feat_shuffle_method=self.feat_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            self.ensemble_generator_.fit(X, y_scaled)

        # KV cache (only for non-enhanced path)
        self.model_kv_cache_ = None
        if not self.enhance_candidates and kv_cache:
            if kv_cache is True or kv_cache == "kv":
                self.cache_mode_ = "kv"
            elif kv_cache == "repr":
                self.cache_mode_ = "repr"
            else:
                raise ValueError(f"Invalid kv_cache value '{kv_cache}'. Expected False, True, 'kv', or 'repr'.")
            self._build_kv_cache()

        return self

    # ==================================================================
    # predict()
    # ==================================================================

    def predict(
        self, X: np.ndarray, output_type: str | list[str] = "mean", alphas: Optional[List[float]] = None
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict target values for test samples.

        When ``enhance_candidates=True`` (and ``fit()`` was called with that flag),
        applies the full multi-group ensemble pipeline with NNLS weighting.
        Otherwise, runs the plain single-group predict path.
        """
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        # Check if prediction is possible
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        has_training_data = (
            hasattr(self, "ensemble_generator_") and getattr(self.ensemble_generator_, "X_", None) is not None
        )
        has_pipeline_training_data = bool(getattr(self, "pipeline_members_", []))
        if not has_kv_cache and not has_training_data and not has_pipeline_training_data:
            raise RuntimeError(
                "Cannot predict: this estimator was saved without training data and has no KV cache. "
                "Predictions require either cached KV projections or the original training data. "
                "Re-fit the estimator or load from a file saved with save_training_data=True or "
                "save_kv_cache=True."
            )

        if self.n_jobs is not None:
            assert self.n_jobs != 0
            import multiprocessing as mp
            old_n_threads = torch.get_num_threads()
            n_logical_cores = mp.cpu_count()
            if self.n_jobs > 0:
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"TabLDM got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used."
                    )
                n_threads = max(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, mp.cpu_count() + 1 + self.n_jobs)
            torch.set_num_threads(n_threads)

        X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
        X = self.X_encoder_.transform(X)

        output_type = [output_type] if isinstance(output_type, str) else list(output_type)

        if not getattr(self, "enhance_candidates", False):
            # ---- Non-enhanced path ----
            if has_kv_cache:
                test_data = self.ensemble_generator_.transform(X, mode="test")
                results = {key: [] for key in output_type}
                for norm_method, (Xs_test,) in test_data.items():
                    kv_cache = self.model_kv_cache_[norm_method]
                    batch_out = self._batch_forward_with_cache(Xs_test, kv_cache, output_type=output_type, alphas=alphas)
                    if isinstance(batch_out, dict):
                        for key in output_type:
                            results[key].append(batch_out[key])
                    else:
                        results[output_type[0]].append(batch_out)
            else:
                data = self.ensemble_generator_.transform(X, mode="both")
                results = {key: [] for key in output_type}
                for Xs, ys in data.values():
                    batch_out = self._batch_forward(Xs, ys, output_type=output_type, alphas=alphas)
                    if isinstance(batch_out, dict):
                        for key in output_type:
                            results[key].append(batch_out[key])
                    else:
                        results[output_type[0]].append(batch_out)

            final_results = {}
            for key in output_type:
                arr = np.concatenate(results[key], axis=0)
                n_estimators = arr.shape[0]
                n_samples = arr.shape[1]
                if arr.ndim == 2:
                    arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples)
                    final_results[key] = np.mean(arr, axis=0)
                else:
                    n_quantiles = arr.shape[2]
                    arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples, n_quantiles)
                    final_results[key] = np.mean(arr, axis=0)

            if self.n_jobs is not None:
                torch.set_num_threads(old_n_threads)
            if len(output_type) == 1:
                return final_results[output_type[0]]
            return final_results

        # ---- Enhanced path ----
        if output_type != ["mean"]:
            raise ValueError("The LimiX pipeline ensemble supports only output_type='mean'.")
        predictions, weights = [], []
        for member, weight in zip(self.pipeline_members_, self.nnls_weights_):
            try:
                prediction = self._pipeline_member_predictions(member, X)
                prediction = self.y_scaler_.inverse_transform(prediction.reshape(-1, 1)).ravel()
                if not np.isfinite(prediction).all():
                    raise ValueError("prediction contains non-finite values")
            except Exception as exc:
                warnings.warn(f"Skipping pipeline member {member.spec.name}: {exc}", UserWarning, stacklevel=2)
                continue
            predictions.append(prediction)
            weights.append(weight)
        if not predictions:
            raise RuntimeError("All retained LimiX pipeline members failed during prediction.")
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        final_prediction = np.average(np.stack(predictions), axis=0, weights=weights)
        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)
        return final_prediction


    # ==================================================================
    # Pickle deserialization
    # ==================================================================

    def __setstate__(self, state):
        """Customize pickle deserialization to reconstruct the MoE model."""
        from .base import _check_version_compatibility

        metadata = state.pop("_persistence_metadata", None)
        model_state_dict = state.pop("_model_state_dict", None)

        self.__dict__.update(state)

        if metadata:
            _check_version_compatibility(metadata)

        if "n_features_in_" not in state:
            return

        self._resolve_device()

        if model_state_dict is not None and hasattr(self, "model_config_"):
            config = self.model_config_
            dual_stream_cfg = {}
            global_dilation = config.get("global_dilation", dual_stream_cfg.get("global_dilation", "adaptive"))
            global_max_span = config.get("global_max_span", dual_stream_cfg.get("global_max_span", 32))

            parent_config = {
                key: value
                for key, value in config.items()
                if key not in ("global_dilation", "global_max_span")
                and not key.startswith("icl_moe_")
            }
            for legacy_name, native_name in {
                "icl_moe_num_experts": "moe_num_experts",
                "icl_moe_top_k": "moe_top_k",
                "icl_moe_num_shared_experts": "moe_num_shared_experts",
                "icl_moe_layers": "moe_layers",
                "icl_moe_router_z_loss_coef": "moe_router_z_loss_coef",
                "icl_moe_load_balance_loss_coef": "moe_load_balance_loss_coef",
                "icl_moe_router_jitter": "moe_router_jitter",
                "icl_moe_router_weight_mode": "moe_router_weight_mode",
                "icl_moe_expert_init_noise": "moe_expert_init_noise",
                "icl_moe_init_from_dense": "moe_init_from_dense",
            }.items():
                if legacy_name in config:
                    parent_config[native_name] = config[legacy_name]

            self.model_ = TabLDMSparseMoE(**parent_config)

            ff_factor = config.get("ff_factor", 2)
            embed_dim = config.get("embed_dim", 128)
            max_classes = config.get("max_classes", 0)

            self.model_.col_embedder = ColEmbeddingDualStream(
                embed_dim=embed_dim,
                num_blocks=config.get("col_num_blocks", 3),
                nhead=config.get("col_nhead", 8),
                dim_feedforward=embed_dim * ff_factor,
                num_inds=config.get("col_num_inds", 128),
                dropout=config.get("dropout", 0.0),
                activation=config.get("activation", "gelu"),
                norm_first=config.get("norm_first", True),
                bias_free_ln=config.get("bias_free_ln", True),
                affine=config.get("col_affine", False),
                feature_group=config.get("col_feature_group", "same"),
                feature_group_size=config.get("col_feature_group_size", 3),
                global_dilation=global_dilation,
                global_max_span=global_max_span,
                target_aware=config.get("col_target_aware", True),
                max_classes=max_classes,
                reserve_cls_tokens=config.get("row_num_cls", 4),
                ssmax=config.get("col_ssmax", False),
                zero_init=config.get("zero_init", False),
                mixed_radix_ensemble=True,
                recompute=False,
            )

            self.model_.drop_dense_ffn()
            try:
                missing, unexpected = self.model_.load_state_dict(model_state_dict, strict=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to load saved MoE model weights; file may be incompatible."
                ) from exc
            bad_missing, bad_unexpected = _moe_load_mismatch(
                set(self.model_.state_dict().keys()), missing, unexpected
            )
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    "Failed to load saved MoE model weights; "
                    f"missing={bad_missing}, unexpected={bad_unexpected}"
                )
            self.model_.eval()
        else:
            self._load_model()

        self.model_.to(self.device_)
        self._build_inference_config()
        self._move_cache_to_device()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


__all__ = [
    "TabLDMRegressor",
]
