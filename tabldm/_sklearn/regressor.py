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
estimator for TabLDM in-context tabular regression.

Inference enhancement follows the **v3.1 two-level shrinkage rules** (a
revision of ``docs/tabldm_regression_no_kfold_recommendations.md``,
recalibrated against TabArena regression holdout statistics):

* Rules decide only the **group mixing ratios** ``pi`` (main / quantile /
  hk); candidate slots are never filtered by name or index.  The v2 rule
  table is kept but recalibrated (R3/R4 quantile mass lowered, R2 uses
  its own measured split instead of falling through to R5).
* Within each group all candidates are averaged with equal weight
  (HK group splits mass asinh : yeo-johnson = 2 : 1).
* K-Fold is **completely disabled**.  A single holdout supplies two
  shrinkage targets and the final prediction blends all three terms::

      pi_hold     = group masses of the slot-level NNLS   (smooth, 2-3 numbers)
      holdout_pred = (1 - rho) * mix(pi_hold) + rho * slot_nnls_pred
      pred         = (1 - lambda) * mix(pi_0) + lambda * holdout_pred

      lambda = n_val / (n_val + shrink_kappa)   # when shrink_lambda=None
      rho    = slot_blend_rate                  # default 0.3

  The group-mass part is the stable signal (v2 §0); the slot-level part
  only restores within-group reweighting and is capped to ``rho`` of the
  holdout budget.

All log messages use the ``[TabLDM:...]`` prefix for consistency with the
Xiaomi-TabLDM project conventions.

Usage::

    from tabldm import TabLDMRegressor

    model = TabLDMRegressor(
        model_path="path/to/regressor_checkpoint.ckpt",
        enhance_candidates=True,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.optimize import nnls as _scipy_nnls
from scipy.stats import (
    kurtosis as _scipy_kurtosis,
    skew as _scipy_skew,
)
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from tabldm._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tabldm._model.embedding_dual_stream import ColEmbeddingDualStream
from tabldm._model.kv_cache import TabLDMCache

from .base import TabLDMBaseEstimator
from .preprocessing import EnsembleGenerator, TransformToNumerical
from .sklearn_utils import _moe_load_mismatch, _num_samples, validate_data

if TYPE_CHECKING:
    from tabldm import InferenceConfig

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
        f"({n_all_nan} samples had all-NaN estimators, replaced with 0=z-space mean)",
        stacklevel=2,
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
        Number of quantile-safe candidates (effective size is
        ``min(n_quantile_estimators, n_features)`` inside
        ``EnsembleGenerator``).

    use_cross_feature : bool, default=False
        Append SVD + cross features to odd-indexed main candidates. Only
        takes effect when ``n_estimators == 32``; default False so the
        candidate names stay honest (v2 §3.2).

    validation : bool, default=True
        When False, the holdout is skipped entirely and the pure group
        prior ``pi_0`` is used (no ``pi_hold`` diagnostics either).

    k_fold : bool, default=False
        **Completely disabled** (v2 §3.2). The parameter is retained only
        for backward compatibility and is ignored; a DeprecationWarning is
        raised when set to True.

    n_splits : int, default=5
        Deprecated. No longer used (K-Fold is disabled).

    foundation_rate : float, default=0.25
        Deprecated. Replaced by ``shrink_lambda`` / ``shrink_kappa`` /
        ``slot_blend_rate`` (v3.1 §3.6).

    max_num_features : Optional[int], default=300
        Max features before triggering per-estimator sampling. Must be
        kept: the 300-dim sampling is the source of main-group diversity
        for high-dimensional data (v2 §3.2).

    enable_high_kurtosis_target_ensemble : bool, default=True
        Enable high-kurtosis target transform ensemble.

    high_kurtosis_threshold : float, default=10.0
        Excess kurtosis threshold to trigger HK ensemble.

    high_kurtosis_n_estimators : int, default=8
        Number of HK candidates (split evenly between asinh and Yeo-Johnson).

    group_prior : Optional[dict[str, float]], default=None
        Explicit group mixing ratios ``pi_0``, e.g.
        ``{'main': 0.55, 'quantile': 0.25, 'hk': 0.20}``. When None, the
        ratios are auto-routed from dataset statistics per v3 §3.3-3.4.
        Groups absent from the dict get weight 0. Values are normalized
        to sum to 1.

    hk_kind_ratio : tuple[float, float], default=(2.0, 1.0)
        Within-HK-group mass split ``(asinh, yeo-johnson)``. The 4 asinh
        slots share ``ratio[0] / sum(ratio)`` of the HK mass and the 4
        yeo-johnson slots share ``ratio[1] / sum(ratio)`` (v2 §3.5).

    shrink_lambda : float or None, default=None
        Shrinkage ratio for blending the holdout term into the group
        prior (v3.1 §3.6):
        ``pred = (1 - lambda) * mix(pi_0) + lambda * holdout_pred``.
        ``None`` (default) picks ``lambda = n_val / (n_val + shrink_kappa)``
        automatically — the more holdout evidence, the more the holdout
        is trusted.  An explicit float in ``[0, 1]`` overrides this
        (``0`` = pure group prior, ``1.0`` = pure holdout term).  Ignored
        when ``validation=False``.

    shrink_kappa : float, default=750.0
        Half-weight holdout size for the automatic ``lambda`` above:
        at ``n_val == shrink_kappa`` the holdout and the prior count
        equally.  Only used when ``shrink_lambda is None``.

    slot_blend_rate : float, default=0.3
        Split of the *holdout* budget between the stable group-level
        target and the full slot-level NNLS prediction (v3.1 §3.6):
        ``holdout_pred = (1 - rho) * mix(pi_hold) + rho * slot_nnls``
        with ``rho = slot_blend_rate``.  ``0`` = group masses only
        (v3.0 behaviour, which lost within-group reweighting), ``1`` =
        pure sparse-NNLS prediction.  ``0.3`` keeps most of the weight on
        the stable signal while restoring the within-group detail.
    """

    def __init__(
        self,
        # -- base parameters --
        n_estimators: int = 8,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: int | str | None = 4,
        kv_cache: bool | str = False,
        model_path: str | Path | None = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "checkpoints/reg_default.ckpt",
        device: str | torch.device | None = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: str | None = None,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        verbose: bool = False,
        inference_config: InferenceConfig | dict | None = None,
        # -- enhancement parameters --
        enhance_candidates: bool = True,
        n_quantile_estimators: int = 16,
        use_cross_feature: bool = False,
        validation: bool = True,
        k_fold: bool = False,
        n_splits: int = 1,
        foundation_rate: float = 0.25,
        max_num_features: int | None = 300,
        enable_high_kurtosis_target_ensemble: bool = True,
        high_kurtosis_threshold: float = 10.0,
        high_kurtosis_n_estimators: int = 8,
        # -- v2/v3 group-prior parameters --
        group_prior: dict[str, float] | None = None,
        hk_kind_ratio: tuple[float, float] = (2.0, 1.0),
        shrink_lambda: float | None = None,
        shrink_kappa: float = 750.0,
        slot_blend_rate: float = 0.3,
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
        # v2/v3 group-prior parameters
        self.group_prior = group_prior
        self.hk_kind_ratio = hk_kind_ratio
        self.shrink_lambda = shrink_lambda
        self.shrink_kappa = shrink_kappa
        self.slot_blend_rate = slot_blend_rate

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
                    print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                    model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    ) from None
            checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            elif self.allow_auto_download:
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
            if key not in ("global_dilation", "global_max_span") and not key.startswith("icl_moe_")
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
        bad_missing, bad_unexpected = _moe_load_mismatch(set(self.model_.state_dict().keys()), missing, unexpected)
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
        print(f"[TabLDM:hk] kurtosis={kurt:.4f}, threshold={self.high_kurtosis_threshold}, triggered={triggered}")
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
        n_asinh = n_hk // 2  # 4
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
                results.append(
                    {
                        "kind": "asinh",
                        "gen": gen,
                        "asinh_params": params,
                        "asinh_ss": ss,
                        "yj_pt": None,
                        "seed": seed,
                    }
                )
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
                results.append(
                    {
                        "kind": "yeojohnson",
                        "gen": gen,
                        "asinh_params": None,
                        "asinh_ss": None,
                        "yj_pt": pt,
                        "seed": seed,
                    }
                )
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
        pt = hk_info["yj_pt"]
        return pt.inverse_transform(z.reshape(-1, 1)).flatten().astype(np.float64)

    def _collect_hk_val_predictions_orig(
        self,
        X_val: np.ndarray,
        hk_infos: list,
    ) -> tuple[np.ndarray, list[str]]:
        """Run each HK estimator on val set.

        Returns ``(preds, kinds)``: preds has shape ``(n_hk_valid, n_val)``
        in ORIGINAL scale and ``kinds`` lists the surviving estimators'
        kinds in row order.  ``kinds`` must travel with the rows — when an
        estimator is skipped the remaining rows shift, and re-deriving
        names by position would mis-assign the asinh / yeo-johnson split.
        """
        output_type = ["mean"]
        all_preds = []
        all_kinds = []
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
                    all_kinds.append(hk["kind"])
            except Exception as e:
                print(f"[TabLDM:hk] {hk['kind']} seed={hk['seed']}: val prediction failed: {e}, skipping")
        if all_preds:
            return np.concatenate(all_preds, axis=0), all_kinds
        return np.empty((0, X_val.shape[0]), dtype=np.float64), []

    def _collect_hk_test_predictions_orig(
        self,
        X: np.ndarray,
        hk_infos: list,
    ) -> tuple[np.ndarray, list[str]]:
        """Run each HK estimator on test X.

        Returns ``(preds, kinds)`` like ``_collect_hk_val_predictions_orig``.
        """
        output_type = ["mean"]
        all_preds = []
        all_kinds = []
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
                    all_kinds.append(hk["kind"])
            except Exception as e:
                print(f"[TabLDM:hk] {hk['kind']} seed={hk['seed']}: test prediction failed: {e}, skipping")
        if all_preds:
            return np.concatenate(all_preds, axis=0), all_kinds
        return np.empty((0, X.shape[0]), dtype=np.float64), []

    # ==================================================================
    # Validation routing (v2: K-Fold completely disabled)
    # ==================================================================

    @staticmethod
    def _route_validation(validation: bool, k_fold: bool, n_train: int) -> str:
        """Decide which validation path fit() takes. Returns one of:
        'default', 'single_validation'.

        K-Fold is **completely disabled** (v2 §3.2 / §4.6).  The only
        remaining question is whether a single holdout is worth running at
        all.  Small training sets no longer skip it (v3): the holdout
        solution is *group-level* (2-3 unknowns) and the automatic
        ``shrink_lambda`` already decays with ``n_val``, so a small
        holdout contributes a small nudge instead of being thrown away.
        The floor ``n_train >= 50`` merely keeps ``train_test_split`` and
        the group solve well-defined.

          | condition                  | path              |
          |----------------------------|-------------------|
          | validation=False           | default           |
          | k_fold=True (ignored)      | *warn*            |
          | n_train < 50               | default           |
          | n_train >= 50              | single_validation |

        ``k_fold`` is accepted for backward compatibility only and never
        routes to a K-Fold path.
        """
        if k_fold:
            warnings.warn(
                "k_fold is deprecated and completely disabled in TabLDMRegressor "
                "(v2 no-K-Fold rules). The parameter is ignored; a single holdout "
                "is used at most for shrinkage. Pass k_fold=False to silence this "
                "warning.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not validation:
            return "default"
        return "single_validation" if n_train >= 50 else "default"

    # ==================================================================
    # v2 group-prior routing and prediction
    # ==================================================================

    @staticmethod
    def _route_group_prior(
        n_train: int,
        n_features: int,
        heavy_tail: bool,
        feat_sampled: bool,
    ) -> dict[str, float]:
        """Compute group mixing ratios ``pi_0`` from dataset statistics (v3 §3.4).

        Routing inputs are computed from the training set actually received
        by this outer split — ``random_state`` is never used as a routing
        condition.

        The v2 table is recalibrated against TabArena regression holdout
        group masses (the slot-level NNLS ``w_quantile`` logged per split):

        ===========  =========  ==============  =========================================
        rule         v2 ``pi_q``  holdout ``w_q``  v3 ``pi_q`` (share of non-HK mass)
        ===========  =========  ==============  =========================================
        R1 high_dim      0.05          0.00       0.05   (unchanged)
        R2 heavy_tail    0.32*         0.19       0.25 absolute (0.3125 of remaining)
        R3 low_dim       0.15          ~0.00-0.09 0.05
        R4 large_n       0.65          0.18-0.39  0.35
        R5 default       0.40          0.30-0.45  0.40   (unchanged)
        ===========  =========  ==============  =========================================

        (*) v2's stacking formula dropped R2 into the R5 branch and gave
        miami 0.48/0.32/0.20 instead of the intended 0.55/0.25/0.20 —
        R2 now owns its measured main:quantile split (0.55:0.25) and only
        falls through to the chain below when combined with other rules.

        Parameters
        ----------
        n_train : int
            Number of training samples.
        n_features : int
            Effective (encoded) feature count.
        heavy_tail : bool
            Target excess kurtosis > 10.
        feat_sampled : bool
            Whether the 300-dim per-estimator feature sampling triggered.

        Returns:
        -------
        dict
            ``{'main': ..., 'quantile': ..., 'hk': ...}`` — 'hk' is only
            present when ``heavy_tail`` is True. Values sum to 1.
        """
        high_dim = n_features > 100 or feat_sampled
        low_dim = n_features <= 5
        large_n = n_train >= 10000

        pi_hk = 0.20 if heavy_tail else 0.0
        remaining = 1.0 - pi_hk

        # Quantile share of the non-HK mass, by priority
        # high_dim > heavy_tail > low_dim > large_n > default.
        if high_dim:
            q_share = 0.05
        elif heavy_tail:
            q_share = 0.25 / 0.80  # R2's measured 0.55 : 0.25 split
        elif low_dim:
            q_share = 0.05
        elif large_n:
            q_share = 0.35
        else:
            q_share = 0.40

        pi_quantile = q_share * remaining
        pi_main = remaining - pi_quantile

        prior: dict[str, float] = {"main": pi_main, "quantile": pi_quantile}
        if pi_hk > 0.0:
            prior["hk"] = pi_hk
        return prior

    @staticmethod
    def _candidate_group(name: str) -> str:
        """Resolve group membership from candidate name prefix (v2 §3.8).

        Group membership **must** be parsed by name prefix rather than by
        global integer index: whether HK is triggered and how large the
        quantile pool is both shift the index layout (v2 §1.4).

        Returns one of ``'main'``, ``'quantile'``, ``'hk_asinh'``,
        ``'hk_yj'``, or ``'hk_other'``.
        """
        if name.startswith("hk_asinh"):
            return "hk_asinh"
        if name.startswith(("hk_yeojohnson", "hk_yj")):
            return "hk_yj"
        if name.startswith("hk"):
            return "hk_other"
        if name.startswith("quantile"):
            return "quantile"
        if name.startswith(("svd+cross", "default")):
            return "main"
        return "main"

    def _group_means(self, arr: np.ndarray, names: list[str]) -> dict[str, np.ndarray | None]:
        """Per-group mean predictions (v2 §3.5).

        Within every group all candidates are averaged with equal weight
        (candidate slots are never filtered).  The HK group splits its
        mass between asinh and yeo-johnson according to
        ``hk_kind_ratio``.

        Parameters
        ----------
        arr : np.ndarray, shape (n_candidates, n_samples)
            Candidate predictions in **original** target scale.
        names : list[str]
            Candidate labels, one per row of ``arr``.  Group membership is
            parsed from these (v2 §3.8) — never from integer slices.

        Returns:
        -------
        dict
            ``{'main': ..., 'quantile': ..., 'hk': ...}``; ``None`` for
            groups with no row in ``arr``.
        """
        groups = [self._candidate_group(n) for n in names]

        def _gmean(pred_group: str) -> np.ndarray | None:
            mask = np.array([g == pred_group for g in groups])
            if not mask.any():
                return None
            return _safe_ensemble_mean(arr[mask], axis=0)

        main_pred = _gmean("main")
        q_pred = _gmean("quantile")

        # HK group with asinh : yeo-johnson = hk_kind_ratio (v2 §3.5)
        hk_asinh_ratio, hk_yj_ratio = self.hk_kind_ratio
        hk_parts: list[np.ndarray] = []
        hk_weights: list[float] = []
        for pred_group, ratio in (("hk_asinh", hk_asinh_ratio), ("hk_yj", hk_yj_ratio), ("hk_other", 1.0)):
            part = _gmean(pred_group)
            if part is not None:
                hk_parts.append(part)
                hk_weights.append(ratio)
        hk_pred = None
        if hk_parts:
            w_sum = float(sum(hk_weights))
            if w_sum <= 0:
                w_sum = float(len(hk_weights))
                hk_weights = [1.0] * len(hk_weights)
            hk_pred = sum(w * p for w, p in zip(hk_weights, hk_parts, strict=False)) / w_sum

        return {"main": main_pred, "quantile": q_pred, "hk": hk_pred}

    def _slot_group_mass(self, weights: np.ndarray, names: list[str]) -> dict[str, float]:
        """Aggregate a slot-level NNLS weight vector to group masses (v3.1 §3.6).

        This is the smooth, non-cornering holdout target ``pi_hold``.  It is
        deliberately *not* a separate NNLS on 2-3 group means: with highly
        correlated group-mean columns that solve collapses to a vertex
        (observed ``pi_hold = 1.0/0.0/0.0`` on 5 of 13 TabArena datasets,
        which zeroed miami's HK mass and cut wine/healthcare's quantile
        mass).  The stable quantity is the *group mass of the slot-level
        solution* — exactly what v2 §0 calls the signal.

        Returns:
        -------
        dict
            ``{'main': ..., 'quantile': ..., 'hk': ...}`` subset with values
            summing to 1 (groups with zero mass are dropped).  Empty dict if
            ``weights`` is empty or degenerate.
        """
        totals = {"main": 0.0, "quantile": 0.0, "hk": 0.0}
        for nm, w in zip(names, weights, strict=False):
            group = self._candidate_group(nm)
            key = "hk" if group.startswith("hk") else group
            if key not in totals:
                key = "main"
            totals[key] += float(w)
        total = float(sum(totals.values()))
        if not np.isfinite(total) or total <= 0:
            return {}
        return {k: v / total for k, v in totals.items() if v > 0}

    def _group_prior_prediction(
        self,
        arr: np.ndarray,
        names: list[str] | None = None,
        pi: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Combine candidate predictions with a group mixing vector (v2 §3.5).

        Parameters
        ----------
        arr : np.ndarray, shape (n_candidates, ...)
            Candidate predictions in **original** target scale (2-d, or 3-d
            for multi-quantile output — the mixture is taken over axis 0).
        names : list[str], optional
            Candidate labels for the rows of ``arr``.  Must be supplied
            whenever rows have been dropped (NaN filtering) — rebuilding
            names from row positions would then mis-assign groups.
        pi : dict[str, float], optional
            Mixing ratios to use.  Defaults to ``self.group_prior_`` (pi_0).

        Returns:
        -------
        np.ndarray
            The group-mixture ensemble prediction (``arr.shape[1:]``).
        """
        if names is None or len(names) != arr.shape[0]:
            names = [self._candidate_name(i) for i in range(arr.shape[0])]
        gmeans = self._group_means(arr, names)

        if pi is None:
            pi = self.group_prior_
        parts: list[np.ndarray] = []
        weights: list[float] = []
        for g in ("main", "quantile", "hk"):
            pred_g = gmeans.get(g)
            w_g = pi.get(g, 0.0)
            if pred_g is not None and w_g > 0.0:
                parts.append(pred_g)
                weights.append(w_g)

        if not parts:
            # No group matched the prior — fall back to all-candidate mean.
            return _safe_ensemble_mean(arr, axis=0)

        total_w = float(sum(weights))
        if total_w <= 0:
            return _safe_ensemble_mean(arr, axis=0)
        return sum(w * p for w, p in zip(weights, parts, strict=False)) / total_w

    def _combine_predictions(self, arr: np.ndarray, names: list[str]) -> np.ndarray:
        """Two-level holdout shrinkage (v3.1 §3.6)::

            holdout_pred = (1 - rho) * mix(pi_hold) + rho * slot_nnls_pred
            pred = (1 - lambda) * mix(pi_0) + lambda * holdout_pred

        ``mix(pi_hold)`` carries the stable group-mass correction and
        ``slot_nnls_pred`` only restores within-group reweighting (capped to
        ``rho`` of the holdout budget, since the sparse slot support is the
        noisy part — v2 §2.7).  Slot weights are looked up **by candidate
        name**, so rows dropped on either the holdout or the test side
        (NaN filtering / skipped HK estimators) cannot shift them.

        Returns:
        -------
        np.ndarray
            Combined prediction, shape ``arr.shape[1:]``.
        """
        mix0 = self._group_prior_prediction(arr, names)
        lam = float(getattr(self, "shrink_lambda_", 0.0) or 0.0)
        if lam <= 0.0:
            return mix0

        rho = float(self.slot_blend_rate)
        pi_hold = getattr(self, "group_prior_holdout_", None)
        w_map = getattr(self, "nnls_weight_map_", None)

        parts: list[np.ndarray] = []
        weights: list[float] = []
        if pi_hold:
            parts.append(self._group_prior_prediction(arr, names, pi=pi_hold))
            weights.append(1.0 - rho)
        if w_map:
            w_aligned = np.array([w_map.get(nm, 0.0) for nm in names], dtype=np.float64)
            w_sum = float(w_aligned.sum())
            if w_sum > 0:
                parts.append(np.tensordot(w_aligned / w_sum, arr, axes=(0, 0)))
                weights.append(rho if pi_hold else 1.0)
        if not parts:
            return mix0

        w_total = float(sum(weights))
        holdout_pred = sum(w * p for w, p in zip(weights, parts, strict=False)) / w_total
        return (1.0 - lam) * mix0 + lam * holdout_pred

    def _compute_feature_stats(self, X: np.ndarray) -> None:
        """Record feature-side summary statistics (v2 §4.5).

        Target kurtosis alone can only route the HK branch — it has no
        predictive power over the quantile-group preference (v2 §2.8).  The
        quantile group's payoff comes from the *feature* side (heavy tails /
        outliers in the feature columns), so each task must at least log
        ``n_num``, ``n_cat``, mean skewness of numeric columns, mean excess
        kurtosis of numeric columns, and the outlier rate (share of |z| > 4
        entries).  Stored on ``self`` for downstream analysis.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.size == 0:
            self.n_num_ = 0
            self.n_cat_ = 0
            self.feat_skew_mean_ = float("nan")
            self.feat_kurt_mean_ = float("nan")
            self.feat_outlier_rate_ = float("nan")
            return

        n_cols = X.shape[1]
        is_num = np.zeros(n_cols, dtype=bool)
        n_unique = np.zeros(n_cols, dtype=np.int64)
        for c in range(n_cols):
            col = X[:, c]
            col = col[np.isfinite(col)]
            n_unique[c] = len(np.unique(col)) if col.size else 0
            # A column is treated as numeric when it has more than 10 distinct
            # finite values (mirrors `_fit_svd_cross`'s cat/num split).
            is_num[c] = n_unique[c] > 10

        self.n_num_ = int(is_num.sum())
        self.n_cat_ = int(n_cols - is_num.sum())

        skews: list[float] = []
        kurts: list[float] = []
        outlier_rates: list[float] = []
        for c in np.where(is_num)[0]:
            col = X[:, c]
            col = col[np.isfinite(col)]
            if col.size < 3 or col.std() < 1e-12:
                continue
            try:
                skews.append(float(_scipy_skew(col, bias=False)))
            except Exception:  # noqa: S110 — optional stat, skip on failure
                pass
            try:
                kurts.append(float(_scipy_kurtosis(col, fisher=True, bias=False)))
            except Exception:  # noqa: S110 — optional stat, skip on failure
                pass
            z = (col - col.mean()) / col.std()
            outlier_rates.append(float(np.mean(np.abs(z) > 4.0)))

        self.feat_skew_mean_ = float(np.mean(skews)) if skews else float("nan")
        self.feat_kurt_mean_ = float(np.mean(kurts)) if kurts else float("nan")
        self.feat_outlier_rate_ = float(np.mean(outlier_rates)) if outlier_rates else float("nan")

    # ==================================================================
    # SVD + cross feature fitting
    # ==================================================================

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
            [np.issubdtype(X_raw[:, c].dtype, np.integer) or len(np.unique(X_raw[:, c])) <= 10 for c in range(n_orig)],
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
            setattr(self, f"{attr_prefix}_cross_pool_train_", cross_scaler.fit_transform(cross_mat).clip(-100, 100))
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
                [
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx.tolist()),
                    ("num", StandardScaler(), num_idx.tolist()),
                ],
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
        setattr(self, f"{attr_prefix}_X_train_svd_", svd.fit_transform(X_pre).clip(-100, 100).astype(np.float32))

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
                    [raw_X[:, i].astype(float) * raw_X[:, j].astype(float) for i, j in cross_pool],
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
        alphas: list[float] | None = None,
        feat_indices: list[np.ndarray] | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes in batches to manage memory efficiently."""
        if feat_indices is not None:
            Xs = np.stack([Xs[i][:, feat_indices[i]] for i in range(Xs.shape[0])], axis=0)

        batch_size = self.batch_size_ or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {key: [] for key in output_type}

        for X_batch, y_batch in zip(Xs, ys, strict=False):
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
        alphas: list[float] | None = None,
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
        # Freeze the REAL row count: EnsembleGenerator truncates to the
        # shuffle patterns it can produce, so this is
        # min(n_estimators, n_features * n_norm_methods) with the default
        # latin shuffler — not necessarily n_estimators.
        self.n_main_actual_ = sum(len(v) for v in self.ensemble_generator_.ensemble_configs_.values())

        # SVD + crossing augmentation for odd estimators in main group
        self.svd_ = None
        if self.n_estimators == 32 and sorted(norm_methods) == ["none", "power"]:
            self._fit_svd_cross(self.ensemble_generator_.X_)

        # Quantile-only EnsembleGenerator
        self.quantile_ensemble_generator_ = None
        self.n_quantile_actual_ = 0
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
            self.n_quantile_actual_ = sum(len(v) for v in self.quantile_ensemble_generator_.ensemble_configs_.values())

        # Base-row name labels (main + quantile).  HK rows are appended at
        # predict time with names built from the surviving estimators'
        # kinds, so a skipped HK estimator cannot shift the asinh/yj split.
        self.candidate_names_ = [self._candidate_name(i) for i in range(self.n_main_actual_ + self.n_quantile_actual_)]

        print(
            f"[TabLDM:enhance] Estimator groups: main={self.n_main_actual_} "
            f"(requested {self.n_estimators}), "
            f"quantile={self.n_quantile_actual_} (requested {self.n_quantile_estimators}), "
            f"total={self.n_main_actual_ + self.n_quantile_actual_}"
        )

        # HK full-train generators
        self.hk_generators_ = []
        if self.hk_triggered_:
            y_orig_full = self.y_scaler_.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            self.hk_generators_ = self._make_hk_generators(X, y_orig_full, seed_offset=0)
            print(f"[TabLDM:hk] full-train HK generators fitted: {len(self.hk_generators_)}")

    def _make_and_fit_generators(self, X_tr: np.ndarray, y_tr: np.ndarray):
        """Fit all generators on a (possibly reduced) training split.

        The full-train ``unique_filter_``s are frozen into the split-local
        generators so every generator output shares one feature axis (required
        for ``feat_sample_indices_`` to address the same columns everywhere).
        """
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
        svd_attrs = [
            "svd_",
            "_svd_pre_",
            "_X_train_svd_",
            "_svd_selections_",
            "_cross_pool_",
            "_cross_scaler_",
            "_cross_pool_train_",
            "_cross_selections_",
            "_k_",
        ]
        saved_svd = {a: getattr(self, a, None) for a in svd_attrs}

        val_svd = None
        if self.n_estimators == 32 and sorted(self.norm_methods or ["none", "power"]) == ["none", "power"]:
            self._fit_svd_cross(main_gen.X_)
            val_svd = self.svd_
        else:
            self.svd_ = None

        q_gen = None
        if self.n_quantile_estimators > 0:
            q_frozen_filter = getattr(getattr(self, "quantile_ensemble_generator_", None), "unique_filter_", None)
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
        use_augment = val_gen_info["val_svd"] is not None and self.n_estimators == 32
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
                    for li, ol in zip(odd_idxs, odd_locals, strict=False):
                        x_both_i = Xs_both[li]
                        x_tr = self._augment_estimator(x_both_i[:n_train_rows], X_raw_train, ol, is_train=True)
                        x_te = self._augment_estimator(x_both_i[n_train_rows:], X_raw_val, ol, is_train=False)
                        Xs_odd_list.append(np.concatenate([x_tr, x_te], axis=0))
                    Xs_odd = np.stack(Xs_odd_list, axis=0)
                    fi = [fsi[g] for g in odd_global] if fsi is not None else None
                    bout = self._batch_forward(Xs_odd, ys[odd_idxs], output_type=output_type, feat_indices=fi)
                    preds_val = bout if not isinstance(bout, dict) else bout["mean"]
                    for i, li in enumerate(odd_idxs):
                        per_est_preds[li] = preds_val[i]

                stacked = np.stack([per_est_preds[li] for li in range(n_est_this_method)], axis=0)
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
            _q_offset = getattr(self, "n_main_actual_", self.n_estimators)
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

    def _candidate_name(self, idx: int) -> str:
        """Human-readable candidate label for logging and group parsing.

        Labels are derived from the **actual** generator row counts
        (``n_main_actual_`` / ``n_quantile_actual_``), never from the
        requested ``n_estimators`` / ``n_quantile_estimators``.
        ``EnsembleGenerator`` truncates to the shuffle patterns it can
        produce — with the default ``feat_shuffle_method='latin'`` that is
        ``min(requested, n_features * n_norm_methods)`` rows for the main
        group and ``min(requested, n_features)`` for the quantile group —
        so the requested counts overstate the matrix.  Using them shifts
        every index past the real main block: quantile rows get labelled
        ``default[i]`` (the whole matrix collapses into one group and the
        group prior silently degrades to a flat mean) and HK rows shift by
        the shortfall (v2 §1.4).  ``fit()`` freezes the real counts from
        the fitted generators before any name is built.

        The ``svd+cross`` label is only used when the SVD+cross augmentation
        is actually active (``use_augment``, i.e. ``n_estimators == 32``);
        otherwise every main candidate is a plain feature candidate and is
        labelled ``default[i]`` so the name matches the construction
        (v2 §1.3 / §4.4).
        """
        n_main = int(getattr(self, "n_main_actual_", self.n_estimators))
        n_q = int(getattr(self, "n_quantile_actual_", self.n_quantile_estimators or 0))
        if idx < n_main:
            use_augment = getattr(self, "svd_", None) is not None and self.n_estimators == 32
            if use_augment and idx % 2 == 1:
                return f"svd+cross[{idx}]"
            return f"default[{idx}]"
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
                for X_batch, y_batch in zip(Xs_split, ys_split, strict=False):
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
        """Return X with selected columns if feature sampling is active for est_idx.

        ``feat_sample_indices_`` addresses generator-output columns (post
        UniqueFeatureFilter), so ``X`` must be a generator-transformed matrix.
        """
        if self.feat_sample_indices_ is None or est_idx >= len(self.feat_sample_indices_):
            return X
        return X[:, self.feat_sample_indices_[est_idx]]

    # ==================================================================
    # fit()
    # ==================================================================

    def fit(self, X: np.ndarray, y: np.ndarray, kv_cache: bool | str = False) -> TabLDMRegressor:
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
                    "kv_cache is not supported together with enhance_candidates=True. Disable one of them."
                )

            # Kurtosis check (uses original-scale y before StandardScaler)
            self.hk_triggered_ = False
            self.hk_kurtosis_ = float("nan")
            self.hk_triggered_, self.hk_kurtosis_ = self._check_high_kurtosis(y)

            if self.k_fold:
                warnings.warn(
                    "k_fold is deprecated and completely disabled in TabLDMRegressor "
                    "(v2 no-K-Fold rules). The parameter is ignored; a single holdout "
                    "is used at most for shrinkage. Pass k_fold=False to silence this "
                    "warning.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if abs(self.foundation_rate - 0.25) > 1e-12:
                warnings.warn(
                    "foundation_rate is deprecated in favor of shrink_lambda "
                    "(v2 §3.6). The foundation is now the group-prior prediction, "
                    "not the all-candidate equal-weight mean. "
                    f"Got foundation_rate={self.foundation_rate}; this value is ignored.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if self.shrink_lambda is not None and not (0.0 <= self.shrink_lambda <= 1.0):
                raise ValueError(f"shrink_lambda must be in [0, 1] or None (auto), got {self.shrink_lambda}")
            if self.shrink_kappa <= 0:
                raise ValueError(f"shrink_kappa must be positive, got {self.shrink_kappa}")
            if not (0.0 <= self.slot_blend_rate <= 1.0):
                raise ValueError(f"slot_blend_rate must be in [0, 1], got {self.slot_blend_rate}")
            if not self.validation and self.shrink_lambda:
                print(
                    f"[TabLDM:enhance] validation=False: holdout is skipped and "
                    f"shrink_lambda={self.shrink_lambda} is ignored; pure group prior will be used."
                )

            print(f"[TabLDM:enhance] use_cross_feature={self.use_cross_feature}")
            print(
                f"[TabLDM:enhance] v3.1 group-prior rules active: "
                f"shrink_lambda={'auto' if self.shrink_lambda is None else self.shrink_lambda}, "
                f"shrink_kappa={self.shrink_kappa}, slot_blend_rate={self.slot_blend_rate}, "
                f"hk_kind_ratio={self.hk_kind_ratio}"
            )

            # Fit generators
            self._fit_full_generators(X, y_scaled)

            # Feature sampling: build per-estimator column indices for ALL
            # estimator groups ([main | quantile | hk]) after the generators are
            # fitted, using the generator's actual n_features_in_ (post
            # UniqueFeatureFilter). Sampling against the raw X width would
            # produce out-of-bounds / misaligned column indices whenever
            # UniqueFeatureFilter drops constant columns (mirrors classifier).
            _n_hk_est = self.high_kurtosis_n_estimators if self.hk_triggered_ else 0
            n_total_estimators = self.n_main_actual_ + self.n_quantile_actual_ + _n_hk_est
            n_features = self.ensemble_generator_.n_features_in_
            self.feat_sample_indices_ = None
            if self.max_num_features is not None and n_features > self.max_num_features:
                seed_base = self.random_state if self.random_state is not None else 0
                self.feat_sample_indices_ = [
                    np.sort(
                        np.random.default_rng(seed_base + est_idx).choice(
                            n_features, size=self.max_num_features, replace=False
                        )
                    )
                    for est_idx in range(n_total_estimators)
                ]
                print(
                    f"[TabLDM:enhance] feature sampling triggered: "
                    f"n_features={n_features} -> max_num_features={self.max_num_features}, "
                    f"n_estimators={n_total_estimators}"
                )
            else:
                print(
                    f"[TabLDM:enhance] feature sampling off: n_features={n_features}, "
                    f"max_num_features={self.max_num_features}"
                )

            # Feature-side statistics for routing analysis (§4.5)
            self._compute_feature_stats(X)

            # ---- v3 group-prior routing (§3.3-3.4, recalibrated) ----
            n_full = X.shape[0]
            n_features_eff = X.shape[1]
            heavy_tail = bool(self.hk_triggered_)
            feat_sampled = self.feat_sample_indices_ is not None
            n_main_actual = self.n_main_actual_
            n_q_actual = self.n_quantile_actual_

            if self.group_prior is not None:
                prior = {str(k): float(v) for k, v in self.group_prior.items()}
            else:
                prior = self._route_group_prior(
                    n_full,
                    n_features_eff,
                    heavy_tail,
                    feat_sampled,
                )
            prior_sum = float(sum(prior.values()))
            if prior_sum <= 0:
                raise ValueError(f"group_prior values must sum to a positive value, got {prior}")
            pi_0 = {k: v / prior_sum for k, v in prior.items()}
            self.group_prior_initial_ = dict(pi_0)
            self.group_prior_holdout_ = None
            self.group_prior_ = dict(pi_0)  # pi_0: the mix(pi_0) term of _combine_predictions
            self.shrink_lambda_ = 0.0

            rule_tag = (
                "explicit"
                if self.group_prior is not None
                else (
                    "R1 high_dim"
                    if feat_sampled or n_features_eff > 100
                    else "R2 heavy_tail"
                    if heavy_tail
                    else "R3 low_dim"
                    if n_features_eff <= 5
                    else "R4 large_n"
                    if n_full >= 10000
                    else "R5 default"
                )
            )
            if heavy_tail and self.group_prior is not None:
                rule_tag = "explicit (heavy_tail)"
            elif heavy_tail and not rule_tag.startswith("R2"):
                rule_tag = "R2 heavy_tail + " + rule_tag
            print(
                f"[TabLDM:enhance] routing: n_train={n_full}, n_features={n_features_eff}, "
                f"n_main={n_main_actual}, n_q={n_q_actual}, heavy_tail={heavy_tail}, "
                f"feat_sampled={feat_sampled}, rule={rule_tag}"
            )
            print(
                f"[TabLDM:enhance] group prior pi_0: main={pi_0.get('main', 0.0):.4f}, "
                f"quantile={pi_0.get('quantile', 0.0):.4f}, "
                f"hk={pi_0.get('hk', 0.0):.4f}"
            )
            print(
                f"[TabLDM:enhance] feature-side stats: n_num={self.n_num_}, n_cat={self.n_cat_}, "
                f"feat_skew_mean={self.feat_skew_mean_:.4f}, "
                f"feat_kurt_mean={self.feat_kurt_mean_:.4f}, "
                f"feat_outlier_rate={self.feat_outlier_rate_:.4f}"
            )

            # ---- single-holdout group-level shrinkage (§3.6, v3) ----
            # K-Fold is completely disabled.  The holdout learns a
            # *group-level* weight vector (2-3 unknowns) used purely as a
            # shrinkage term towards pi_0 — never as a candidate selector
            # (v2 §3.1 / §2.7).  Slot-level NNLS is still computed, but
            # only for the diagnostic log lines below.
            self.nnls_weights_ = None
            self.nnls_weight_map_ = None
            self.nan_valid_mask_ = None
            self.nan_valid_idx_ = None
            n_val = math.ceil(0.2 * n_full)
            route = self._route_validation(self.validation, self.k_fold, n_full)
            print(
                f"[TabLDM:enhance] shrinkage path: k_fold=disabled, n_train={n_full}, "
                f"n_val={n_val}, path={route}, shrink_lambda="
                f"{'auto' if self.shrink_lambda is None else self.shrink_lambda}, "
                f"shrink_kappa={self.shrink_kappa}"
            )

            if route != "single_validation":
                print("[TabLDM:enhance] no holdout: using pure group prior pi_0")
            else:
                X_tr, X_val, y_tr_scaled, y_val_scaled = train_test_split(
                    X,
                    y_scaled,
                    test_size=0.2,
                    shuffle=True,
                    random_state=self.random_state,
                )
                y_val_orig = self.y_scaler_.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
                y_tr_orig_sv = self.y_scaler_.inverse_transform(y_tr_scaled.reshape(-1, 1)).flatten()
                print(f"[TabLDM:enhance] validation split: n_train={X_tr.shape[0]}, n_val={X_val.shape[0]}")
                val_gen = self._make_and_fit_generators(X_tr, y_tr_scaled)
                val_preds_scaled = self._collect_val_predictions(X_tr, y_tr_scaled, X_val, val_gen)
                # Detect and skip NaN estimators
                nan_per_est = np.isnan(val_preds_scaled).any(axis=1)
                n_nan_est = int(nan_per_est.sum())
                if n_nan_est > 0:
                    valid_mask = ~nan_per_est
                    nan_indices = np.where(nan_per_est)[0].tolist()
                    val_preds_scaled = val_preds_scaled[valid_mask]
                    print(
                        f"[TabLDM:enhance] WARNING: {n_nan_est}/{len(nan_per_est)} estimators "
                        f"produced NaN and were SKIPPED: indices={nan_indices}"
                    )
                else:
                    valid_mask = np.ones(val_preds_scaled.shape[0], dtype=bool)
                self.nan_valid_mask_ = valid_mask
                # Map from the filtered (valid-only) weight index back to the
                # global candidate index so names/log sums stay aligned (§4.2).
                valid_idx = np.where(valid_mask)[0]
                self.nan_valid_idx_ = valid_idx

                val_preds_orig = self.y_scaler_.inverse_transform(val_preds_scaled.reshape(-1, 1)).reshape(
                    val_preds_scaled.shape
                )
                if not np.isfinite(val_preds_orig).all():
                    n_bad = int((~np.isfinite(val_preds_orig)).sum())
                    val_preds_orig = np.nan_to_num(val_preds_orig, nan=0.0, posinf=0.0, neginf=0.0)
                    print(
                        f"[TabLDM:enhance] WARNING: val inverse transform produced {n_bad} "
                        f"NaN/Inf values, replaced with 0.0"
                    )

                # HK single-validation predictions (kinds travel with rows)
                sv_hk_kinds: list[str] = []
                if self.hk_triggered_:
                    sv_hk_infos = self._make_hk_generators(X_tr, y_tr_orig_sv, seed_offset=0)
                    sv_hk_preds, sv_hk_kinds = self._collect_hk_val_predictions_orig(X_val, sv_hk_infos)
                    if sv_hk_preds.shape[0] > 0:
                        val_preds_orig = np.concatenate([val_preds_orig, sv_hk_preds], axis=0)
                        print(
                            f"[TabLDM:hk] appended {sv_hk_preds.shape[0]} HK val rows; total={val_preds_orig.shape[0]}"
                        )

                print(f"[TabLDM:enhance] val_predictions shape={val_preds_orig.shape}")

                names_w = [self._candidate_name(int(gi)) for gi in valid_idx]
                names_w += [f"hk_{kind}[{i}]" for i, kind in enumerate(sv_hk_kinds)]

                # ---- slot-level NNLS: diagnostics only (§2.7: support is noise) ----
                raw_weights, _ = _scipy_nnls(val_preds_orig.T, y_val_orig)
                w_sum = raw_weights.sum()
                n_nonzero = int((raw_weights > 0).sum())
                if w_sum > 0:
                    weights = raw_weights / w_sum
                else:
                    weights = np.ones(val_preds_orig.shape[0]) / val_preds_orig.shape[0]
                    w_sum = 1.0

                # Log group-level weight sums aggregated by candidate-name
                # prefix (§1.7 / §4.1: the old integer-slice `_group_sum` mixed
                # up relative and absolute indices and made w_default ==
                # w_svd/cross always).  HK rows appended above sit after the
                # base candidates and are named via their recorded kinds.

                def _sum_prefixes(*prefixes: str) -> float:
                    total = 0.0
                    for nm, w in zip(names_w, weights, strict=False):
                        if any(nm.startswith(p) for p in prefixes):
                            total += float(w)
                    return total

                w_default = _sum_prefixes("default")
                w_svd = _sum_prefixes("svd+cross")
                w_q = _sum_prefixes("quantile")
                w_hk = _sum_prefixes("hk")
                print(
                    f"[TabLDM:enhance] NNLS (diagnostic): n_nonzero={n_nonzero}/{len(weights)}, "
                    f"weights.sum()={w_sum:.4f}, "
                    f"w_main={w_default + w_svd:.4f} "
                    f"(w_default={w_default:.4f}, w_svd/cross={w_svd:.4f}), "
                    f"w_quantile={w_q:.4f}, w_hk={w_hk:.4f}"
                )
                # §4.2: index the *global* candidate name, not the filtered
                # weight position (wrong the moment NaN filtering drops one).
                nonzero_str = ", ".join(f"{nm}={w:.4f}" for nm, w in zip(names_w, weights, strict=False) if w > 0)
                print(f"[TabLDM:enhance] NNLS nonzero weights: {nonzero_str}")
                self.nnls_weights_ = weights
                self.nnls_weight_map_ = {nm: float(w) for nm, w in zip(names_w, weights, strict=False) if w > 0}

                # ---- holdout shrinkage targets (v3.1 §3.6) ----
                # pi_hold = the *group masses* of the slot-level NNLS (smooth,
                # never cornering).  The slot weights themselves only feed the
                # rho-blend in _combine_predictions (within-group reweighting).
                pi_hold = self._slot_group_mass(weights, names_w)
                if not pi_hold:
                    pi_hold = dict(pi_0)
                n_val_rows = int(y_val_orig.shape[0])
                lam = (
                    float(self.shrink_lambda)
                    if self.shrink_lambda is not None
                    else n_val_rows / (n_val_rows + self.shrink_kappa)
                )
                lam = min(max(lam, 0.0), 1.0)
                self.group_prior_holdout_ = dict(pi_hold)
                self.shrink_lambda_ = lam
                rho = float(self.slot_blend_rate)
                print(
                    f"[TabLDM:enhance] holdout shrinkage: lambda={lam:.4f} "
                    f"(n_val={n_val_rows}, kappa={self.shrink_kappa}), rho={rho:.4f}, "
                    f"blend={1.0 - lam:.4f}*mix(pi_0) + {lam * (1.0 - rho):.4f}*mix(pi_hold) "
                    f"+ {lam * rho:.4f}*slot_nnls, "
                    f"pi_0=({pi_0.get('main', 0.0):.4f}, {pi_0.get('quantile', 0.0):.4f}, "
                    f"{pi_0.get('hk', 0.0):.4f}), "
                    f"pi_hold=({pi_hold.get('main', 0.0):.4f}, {pi_hold.get('quantile', 0.0):.4f}, "
                    f"{pi_hold.get('hk', 0.0):.4f})"
                )

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
        self, X: np.ndarray, output_type: str | list[str] = "mean", alphas: list[float] | None = None
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict target values for test samples.

        When ``enhance_candidates=True`` (and ``fit()`` was called with that
        flag), applies the two-level holdout shrinkage of
        ``_combine_predictions`` (group prior + holdout group masses + slot
        NNLS).  Otherwise, runs the plain single-group predict path.
        """
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        # Check if prediction is possible
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        has_training_data = (
            hasattr(self, "ensemble_generator_") and getattr(self.ensemble_generator_, "X_", None) is not None
        )
        if not has_kv_cache and not has_training_data:
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
                        f" Only {n_logical_cores} threads will be used.",
                        stacklevel=2,
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
                    batch_out = self._batch_forward_with_cache(
                        Xs_test, kv_cache, output_type=output_type, alphas=alphas
                    )
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
                    arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(
                        n_estimators, n_samples, n_quantiles
                    )
                    final_results[key] = np.mean(arr, axis=0)

            if self.n_jobs is not None:
                torch.set_num_threads(old_n_threads)
            if len(output_type) == 1:
                return final_results[output_type[0]]
            return final_results

        # ---- Enhanced path ----
        output_type = ["mean"]  # enhanced path only supports mean
        results = {key: [] for key in output_type}
        fsi = getattr(self, "feat_sample_indices_", None)

        # Main group
        use_augment = getattr(self, "svd_", None) is not None and self.n_estimators == 32
        if use_augment:
            data = self.ensemble_generator_.transform(X, mode="both")
            X_raw_test = self.ensemble_generator_.unique_filter_.transform(X)
            X_raw_train = self.ensemble_generator_.X_

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
                    for li, ol in zip(odd_idxs, odd_locals, strict=False):
                        x_both_i = Xs_both[li]
                        x_tr = self._augment_estimator(x_both_i[:n_train_rows], X_raw_train, ol, is_train=True)
                        x_te = self._augment_estimator(x_both_i[n_train_rows:], X_raw_test, ol, is_train=False)
                        Xs_odd_list.append(np.concatenate([x_tr, x_te], axis=0))
                    Xs_odd = np.stack(Xs_odd_list, axis=0)
                    fi = [fsi[g] for g in odd_global] if fsi is not None else None
                    bout = self._batch_forward(Xs_odd, ys[odd_idxs], output_type=output_type, feat_indices=fi)
                    preds_val = bout if not isinstance(bout, dict) else bout["mean"]
                    for i, li in enumerate(odd_idxs):
                        per_est_preds[li] = preds_val[i]

                for key in output_type:
                    stacked = np.stack(
                        [per_est_preds[li] for li in range(n_est_this_method)],
                        axis=0,
                    )
                    results[key].append(stacked)
        else:
            data = self.ensemble_generator_.transform(X, mode="both")
            _grp_offset = 0
            for norm_method, (Xs, ys) in data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_grp_offset + i] for i in range(n_est)] if fsi is not None else None
                bout = self._batch_forward(Xs, ys, output_type=output_type, feat_indices=fi)
                if isinstance(bout, dict):
                    for key in output_type:
                        results[key].append(bout[key])
                else:
                    results[output_type[0]].append(bout)
                _grp_offset += n_est

        # Quantile group
        q_gen = getattr(self, "quantile_ensemble_generator_", None)
        if q_gen is not None:
            q_data = q_gen.transform(X, mode="both")
            _q_offset = getattr(self, "n_main_actual_", self.n_estimators)
            for norm_method, (Xs, ys) in q_data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_q_offset + i] for i in range(n_est)] if fsi is not None else None
                bout = self._batch_forward(Xs, ys, output_type=output_type, feat_indices=fi)
                if isinstance(bout, dict):
                    for key in output_type:
                        results[key].append(bout[key])
                else:
                    results[output_type[0]].append(bout)
                _q_offset += n_est

        # Collect HK test predictions in original scale (kinds travel with rows)
        hk_preds_orig = None
        hk_names: list[str] = []
        hk_infos_full = getattr(self, "hk_generators_", [])
        if self.hk_triggered_ and hk_infos_full:
            _hk_p, _hk_kinds = self._collect_hk_test_predictions_orig(X, hk_infos_full)
            if _hk_p.ndim == 2 and _hk_p.shape[0] > 0:
                hk_preds_orig = _hk_p
                hk_names = [f"hk_{kind}[{i}]" for i, kind in enumerate(_hk_kinds)]

        # ---- v3.1 combine: two-level holdout shrinkage (§3.5-3.6) ----
        #   holdout_pred = (1 - rho) * mix(pi_hold) + rho * slot_nnls
        #   pred = (1 - lambda) * mix(pi_0) + lambda * holdout_pred
        nan_valid_mask = getattr(self, "nan_valid_mask_", None)
        if getattr(self, "group_prior_", None) is None:
            # fit() did not run the enhancement path (or was unpickled from an
            # older version) — fall back to a sane prior on the groups that
            # actually exist so predictions stay well-defined.
            self.group_prior_ = {"main": 0.60, "quantile": 0.30, "hk": 0.20}

        final_results = {}
        for key in output_type:
            arr = np.concatenate(results[key], axis=0)
            n_base = arr.shape[0]
            n_samples = arr.shape[1]

            base_names = list(getattr(self, "candidate_names_", []) or [])
            if len(base_names) != n_base:
                base_names = [self._candidate_name(i) for i in range(n_base)]
            if nan_valid_mask is not None and len(nan_valid_mask) == n_base:
                sel = np.asarray(nan_valid_mask, dtype=bool)
                row_names = [base_names[i] for i in np.where(sel)[0]]
                arr = arr[sel]
            else:
                row_names = base_names
            n_estimators = arr.shape[0]

            if self.verbose:
                main_n = getattr(self, "n_main_actual_", self.n_estimators)
                q_n = getattr(self, "n_quantile_actual_", 0)
                print(
                    f"[TabLDM:enhance] Predictions merged: main={main_n}, quantile={q_n}, "
                    f"total={n_estimators}, shape={arr.shape}"
                )

            if arr.ndim == 2:
                arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples)
                if not np.isfinite(arr).all():
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

                # Append HK predictions (already in original scale)
                if hk_preds_orig is not None and key == "mean":
                    arr = np.concatenate([arr, hk_preds_orig.astype(np.float64)], axis=0)
                    row_names = row_names + hk_names

                final_results[key] = self._combine_predictions(arr, row_names)
            else:
                n_quantiles = arr.shape[2]
                arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples, n_quantiles)
                if not np.isfinite(arr).all():
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                # Multi-quantile output: the mixture is taken over the
                # candidate axis, so the same combine math applies.
                final_results[key] = self._combine_predictions(arr, row_names)

        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)

        if len(output_type) == 1:
            return final_results[output_type[0]]
        return final_results

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
                if key not in ("global_dilation", "global_max_span") and not key.startswith("icl_moe_")
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
                raise RuntimeError("Failed to load saved MoE model weights; file may be incompatible.") from exc
            bad_missing, bad_unexpected = _moe_load_mismatch(set(self.model_.state_dict().keys()), missing, unexpected)
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    f"Failed to load saved MoE model weights; missing={bad_missing}, unexpected={bad_unexpected}"
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
