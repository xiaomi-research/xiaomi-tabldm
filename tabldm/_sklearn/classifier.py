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
"""TabLDM Classifier with inference enhancement methods.

This module provides ``TabLDMClassifier``, the public sklearn-style
estimator for TabLDM in-context tabular classification, with multi-group
candidate ensembling, NNLS weight learning, probability calibration, and
other inference-time enhancements ported from the MiTabEnhancedClassifier
reference implementation.

Inference enhancement follows the **v2 no-K-Fold rules**
(``docs/tabldm_classification_no_kfold_recommendations.md``):

* Rules decide two things only: **which groups enter the candidate pool**
  (v2 §3.4, which also sets E) and the **group prior pi** (v2 §3.6, the
  fusion shrinkage target).  Candidates are never filtered by name or slot
  index — the NNLS support drifts across seeds (v2 §2.3).
* Weight learning is kept, but only on a **single stratified holdout**.
  K-Fold is completely disabled (v2 §3.2).  Fusion shrinks the holdout NNLS
  towards the group prior (v2 §3.5)::

      p = (1 - lambda) * mix(pi_0) + lambda * p_nnls_holdout

  where ``lambda`` is tiered by ``val_per_weight = n_val / E``
  (0 / 0.25 / 0.50 / 0.75 at 5 / 10 / 20).  Per v2 §4.2 the legacy
  ``0.75*nnls + 0.25*uniform`` foundation is replaced by this group prior —
  the foundation rate is ``1 - lambda`` and is never 0 at the rule tiers.
* Routing statistics come from the training matrix actually received (v2
  §3.3), never from upstream dataset metadata (v2 §1.3).  Small-sample
  equal-weight fallbacks are gone: every fusion fallback is the group prior
  (v2 §3.5 / §4.1).

All log messages use the ``[TabLDM:...]`` prefix for consistency with the
Xiaomi-TabLDM project conventions.

Usage::

    from tabldm import TabLDMClassifier

    model = TabLDMClassifier(
        model_path="path/to/classifier_checkpoint.ckpt",
        enhance_candidates=True,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
"""

from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from scipy.optimize import (
    minimize as _scipy_minimize,
    nnls as _scipy_nnls,
)
from scipy.special import (
    expit as _expit,
)
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tabldm._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tabldm._model.embedding_dual_stream import ColEmbeddingDualStream
from tabldm._model.kv_cache import TabLDMCache

from .base import TabLDMBaseEstimator
from .preprocessing import (
    EnsembleGenerator,
    GaussianRankGenerator,
    InteractionAugmentor,
    PCADecorrelator,
    TransformToNumerical,
    UniqueFeatureFilter,
)
from .sklearn_utils import _moe_load_mismatch, _num_samples, validate_data

if TYPE_CHECKING:
    from tabldm import InferenceConfig

# ---------------------------------------------------------------------------
# Adaptive inference config routing (ported from MiTab reference).
# ---------------------------------------------------------------------------

_ADAPTIVE_DEFAULT_NORMS = ["none", "power"]
_ADAPTIVE_ENS4_NORMS = ["none", "power", "quantile", "robust"]


def _get_adaptive_inference_config(n_features, n_num, enable_augmentations=False):
    """Dataset-size routing for adaptive_plus candidate group."""
    if n_features <= 10 and n_num >= 2:
        cfg = {"use_svd": True, "svd_n_components": 10, "n_estimators": 16, "norm_methods": list(_ADAPTIVE_ENS4_NORMS)}
    elif n_features <= 10 and n_num < 2:
        cfg = {"use_svd": False, "n_estimators": 16, "norm_methods": list(_ADAPTIVE_ENS4_NORMS)}
    elif n_features <= 50:
        cfg = {"use_svd": False, "n_estimators": 32, "norm_methods": list(_ADAPTIVE_DEFAULT_NORMS)}
    else:
        cfg = {"use_svd": True, "svd_n_components": 10, "n_estimators": 16, "norm_methods": list(_ADAPTIVE_ENS4_NORMS)}

    if enable_augmentations:
        cfg["use_pca_decorr"] = n_num >= 15 and n_features >= 20
        cfg["use_interactions"] = n_features <= 5 and n_num >= 2
    return cfg


def _is_nan(value) -> bool:
    """NaN check that also works on object-dtype scalars (non-numeric -> False)."""
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _excess_kurtosis(x: np.ndarray) -> float:
    """Population excess kurtosis of the finite values of one column.

    Mirrors ``_safe_stats`` in the v2 analysis script
    (``utils/analyze_tabarena_cls_no_kfold_rules.py``) so ``share_kurt10``
    is comparable with the routing table in v2 §3.7: moment estimator on
    z-scores, ``m4/m2^2 - 3``, ``0.0`` for columns with fewer than 8 finite
    values (they count as not heavy-tailed in the share).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 0.0
    mean = x.mean()
    std = x.std()
    std = std if std > 1e-12 else 1e-12
    z = (x - mean) / std
    m2 = float(np.mean(z**2))
    m4 = float(np.mean(z**4))
    return m4 / max(m2**2, 1e-12) - 3.0


# ---------------------------------------------------------------------------
# Enhanced Classifier
# ---------------------------------------------------------------------------


class TabLDMClassifier(ClassifierMixin, TabLDMBaseEstimator):
    """TabLDM Classifier with inference enhancement.

    Supports multi-group candidate ensembling, NNLS weight learning,
    probability calibration, and other inference-time enhancements. When
    ``enhance_candidates=False`` (default), these enhancements are skipped
    and the estimator runs the plain single-group inference path.

    Parameters
    ----------
    n_estimators : int, default=32
        Number of estimators for the main ensemble group (v2 §3.2 keeps
        ``32`` so the main group splits into plain=16 + cross_svd=16).

    norm_methods : str or list[str] or None, default=None
        Normalization methods for the main group (v2 §3.2: ``none,power``).

    feat_shuffle_method : str, default='random'
        Feature permutation strategy.

    class_shuffle_method : str, default='shift'
        Class label permutation strategy (main group only; enhanced groups
        always use ``"none"``).

    max_num_features : Optional[int], default=300
        Per-candidate feature sampling width. Must be kept: the 300-dim
        sampling is itself an effective enhancement and defines ``high_dim``
        for the v2 §3.4 rules.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection.

    softmax_temperature : float, default=0.9
        Temperature for softmax.

    average_logits : bool, default=True
        Whether to average logits (True) or probabilities (False).

    support_many_classes : bool, default=True
        Enable many-class support (mixed-radix + hierarchical).

    batch_size : int, "auto", or None, default=4
        Batch size for inference. ``"auto"`` picks a value based on
        ``n_samples_in_ * n_features_in_`` to reduce CUDA memory pressure on
        large datasets (<=1M cells -> 8, <=2M -> 4, <=5M -> 2, else 1).

    kv_cache : bool or str, default=False
        KV cache mode. Not compatible with ``enhance_candidates=True``.

    model_path : Optional[str | Path], default=None
        Path to the pre-trained model checkpoint.

    allow_auto_download : bool, default=True
        Allow automatic download from Hugging Face Hub.

    checkpoint_version : str
        Checkpoint version identifier.

    device : Optional[str | torch.device], default=None
        Device for inference.

    use_amp : bool or "auto", default=True
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

    categorical_indices : array-like of int or None, default=None
        Categorical feature indices (also feeds the v2 §3.3 N/C split).

    cat_random_encode : bool, default=True
        Randomly permute categorical codes per ensemble member.

    enhance_candidates : bool, default=True
        Master switch for inference enhancement. When False, runs the plain
        single-group inference path with no ensembling enhancements.

    n_quantile_estimators : int, default=0
        Size of the optional quantile group (v2 §3.4: off by default — the
        group is a net negative, §2.7; only worth trying on ``heavy_tail``
        data as an 8-candidate experiment group). The quantile group always
        gets prior mass 0 and only enters as extra NNLS candidates (§3.6).

    use_cross_feature : bool, default=True
        Append SVD + cross features to odd-indexed main candidates. Only
        takes effect when ``n_estimators == 32`` (v2 §3.2 keeps it as-is).

    validation : bool, default=True
        Use a single stratified holdout for the NNLS weights and for the
        calibration fit (v2 §3.2). When False, the pure group prior is used.

    k_fold : bool, default=False
        **Completely disabled** (v2 §3.2). Retained only for backward
        compatibility and ignored; a DeprecationWarning is raised when True.

    n_splits : int, default=1
        Deprecated. No longer used (K-Fold is disabled).

    shrink_lambda : float or None, default=None
        Fusion weight of the holdout NNLS term (v2 §3.5):
        ``p = (1 - lambda) * mix(pi_0) + lambda * p_nnls_holdout``.
        ``None`` (default) picks the tiered rule value from
        ``val_per_weight = n_val / E`` (0/0.25/0.50/0.75 at 5/10/20, and
        always 0 for ``n_train < 1000``). An explicit float in ``[0, 1]``
        overrides the tiers (``0`` = pure group prior). Ignored when
        ``validation=False``.

    group_prior : Optional[dict[str, float]], default=None
        Explicit group mixing ratios ``pi_0``, e.g. ``{'main': 1.0}`` for a
        main-only ablation arm. Keys are ``main`` / ``ap`` / ``gr`` /
        ``svd`` / ``q``. When None, the ratios are routed from dataset
        statistics per v2 §3.6. Mass on a group that the §3.4 rules (or
        overrides) switched off is folded into ``main`` (§3.6).

    use_svd_ens : bool or None, default=None
        SVD-ensemble group (group 6) switch. ``None`` (default) follows the
        v2 §3.4 rules (on only for R3: heavy_tail with 31-500 features and
        cat_ratio < 0.6); True/False force it on/off (ablation arms).

    n_svd_ens_estimators : int or None, default=None
        Size of the SVD-ensemble group when it is on. ``None`` (default)
        uses the rule size of 8 (v2 §3.4). Ignored when the group is off
        (logged as ignored so the knob never lies, v2 §4.3).

    svd_ens_norm_methods : list[str] or None, default=None
        Normalizations for SVD ensemble group.

    svd_ens_n_components : int, default=10
        Number of SVD components.

    use_adaptive_plus_candidate : bool or None, default=None
        adaptive_plus group switch. ``None`` (default) follows the rules
        (on unless ``high_dim``); True/False force it on/off. Group size
        comes from the existing code routing (§3.4: 16 / 32 / 16+SVD by
        feature count).

    adaptive_plus_enable_augmentations : bool, default=True
        Enable PCA decorrelation and interaction augmentation in adaptive_plus.

    enable_calibration : bool, default=True
        Enable probability calibration (v2 §3.2 keeps it on: it is AUC-neutral
        but F1 +0.004, §2.9). Fitted on the final fused holdout probabilities.

    calibration_lambda : float, default=1e-2
        Regularization for calibration.

    use_gaussian_rank_ens : bool or None, default=None
        gaussian_rank group switch. ``None`` (default) follows the rules
        (on when ``(2k <= n_train <= 30k and 11 <= n_features <= 100) or
        mean_cat_card >= 8``, and never for R1 / low-cardinality R2);
        True/False force it on/off.

    n_gaussian_rank_estimators : int, default=16
        Number of Gaussian rank candidates (v2 §3.4 rule size).
    """

    def __init__(
        self,
        # -- base parameters --
        n_estimators: int = 32,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "random",
        class_shuffle_method: str = "shift",
        max_num_features: int | None = 300,
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: int | str | None = 4,
        kv_cache: bool | str = False,
        model_path: str | Path | None = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "checkpoints/clf_default.ckpt",
        device: str | torch.device | None = None,
        use_amp: bool | str = True,
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: str | None = None,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        verbose: bool = False,
        inference_config: InferenceConfig | dict | None = None,
        categorical_indices: list[int] | None = None,
        cat_random_encode: bool = True,
        # -- enhancement parameters (v2 no-K-Fold rules) --
        enhance_candidates: bool = True,
        n_quantile_estimators: int = 0,
        use_cross_feature: bool = True,
        validation: bool = True,
        k_fold: bool = False,
        n_splits: int = 1,
        shrink_lambda: float | None = None,
        group_prior: dict[str, float] | None = None,
        use_svd_ens: bool | None = None,
        n_svd_ens_estimators: int | None = None,
        svd_ens_norm_methods: list[str] | None = None,
        svd_ens_n_components: int = 10,
        use_adaptive_plus_candidate: bool | None = None,
        adaptive_plus_enable_augmentations: bool = True,
        enable_calibration: bool = True,
        calibration_lambda: float = 1e-2,
        use_gaussian_rank_ens: bool | None = None,
        n_gaussian_rank_estimators: int = 16,
    ):
        # base
        if svd_ens_norm_methods is None:
            svd_ens_norm_methods = ["none", "power", "quantile", "robust"]
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.max_num_features = max_num_features
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
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
        self.categorical_indices = categorical_indices
        self.cat_random_encode = cat_random_encode
        # enhancement
        self.enhance_candidates = enhance_candidates
        self.n_quantile_estimators = n_quantile_estimators
        self.use_cross_feature = use_cross_feature
        self.validation = validation
        self.k_fold = k_fold
        self.n_splits = n_splits
        self.shrink_lambda = shrink_lambda
        self.group_prior = group_prior
        self.use_svd_ens = use_svd_ens
        self.n_svd_ens_estimators = n_svd_ens_estimators
        self.svd_ens_norm_methods = svd_ens_norm_methods or ["none", "power", "quantile", "robust"]
        self.svd_ens_n_components = svd_ens_n_components
        self.use_adaptive_plus_candidate = use_adaptive_plus_candidate
        self.adaptive_plus_enable_augmentations = adaptive_plus_enable_augmentations
        self.enable_calibration = enable_calibration
        self.calibration_lambda = calibration_lambda
        self.use_gaussian_rank_ens = use_gaussian_rank_ens
        self.n_gaussian_rank_estimators = n_gaussian_rank_estimators

    # ==================================================================
    # Model loading (MoE architecture)
    # ==================================================================

    def _load_model(self) -> None:
        """Load a MoE model from checkpoint.

        Builds a ``TabLDMSparseMoE`` model with a ``ColEmbeddingDualStream``
        column embedder and drops the frozen dense FFN from MoE layers.
        """
        repo_id = "occams/Xiaomi-TabLDM"
        filename = self.checkpoint_version

        if self.model_path is None:
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if not self.allow_auto_download:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    ) from None
                print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
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

        self.model_.col_embedder = ColEmbeddingDualStream(
            embed_dim=config.get("embed_dim", 128),
            num_blocks=config.get("col_num_blocks", 3),
            nhead=config.get("col_nhead", 8),
            dim_feedforward=config.get("embed_dim", 128) * config.get("ff_factor", 2),
            num_inds=config.get("col_num_inds", 128),
            dropout=config.get("dropout", 0.0),
            activation=config.get("activation", "gelu"),
            norm_first=config.get("norm_first", True),
            bias_free_ln=True,
            affine=config.get("col_affine", False),
            feature_group=config.get("col_feature_group", "same"),
            feature_group_size=config.get("col_feature_group_size", 3),
            global_dilation=global_dilation,
            global_max_span=global_max_span,
            target_aware=config.get("col_target_aware", True),
            max_classes=config.get("max_classes", 0),
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
    # fit()
    # ==================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabLDMClassifier:
        """Fit the classifier to training data.

        When ``enhance_candidates=True``, fits multiple candidate generators
        and optionally learns NNLS ensemble weights via validation/OOF.
        """
        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")

        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)
        check_classification_targets(y)

        # Device + inference config
        self._resolve_device()
        self.n_samples_in_ = _num_samples(X)
        self._build_inference_config()

        # Load model
        self._load_model()
        self.model_.to(self.device_)

        # Encode labels
        self.y_encoder_ = LabelEncoder()
        y = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.y_encoder_.classes_)

        if self.n_classes_ > self.model_.max_classes:
            if self.kv_cache:
                raise ValueError(
                    f"KV caching is not supported when the number of classes ({self.n_classes_}) exceeds "
                    f"the max number of classes ({self.model_.max_classes}) natively supported by the model."
                )
            if not self.support_many_classes:
                raise ValueError(
                    f"The number of classes ({self.n_classes_}) exceeds the max number of classes "
                    f"({self.model_.max_classes}) natively supported by the model. "
                    f"Consider enabling many-class support."
                )
            if self.verbose:
                print(
                    f"[TabLDM] n_classes={self.n_classes_} > max_classes={self.model_.max_classes}; "
                    f"enabling many-class strategy."
                )

        # Transform features
        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)

        if self.enhance_candidates and self.kv_cache:
            raise ValueError("kv_cache is not supported together with enhance_candidates=True. Disable one of them.")

        # Initialize enhancement state
        self.nnls_weights_ = None
        self.nnls_weight_map_ = None
        self.nnls_valid_candidate_mask_ = None
        self.calibration_params_ = None
        self._cal_P_ = None
        self._cal_y_ = None
        self.shrink_lambda_ = 0.0

        if self.enhance_candidates:
            if self.k_fold:
                warnings.warn(
                    "k_fold is deprecated and completely disabled in TabLDMClassifier "
                    "(v2 no-K-Fold rules). The parameter is ignored; a single holdout is "
                    "used at most for NNLS shrinkage. Pass k_fold=False to silence this warning.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if self.shrink_lambda is not None and not (0.0 <= self.shrink_lambda <= 1.0):
                raise ValueError(f"shrink_lambda must be in [0, 1] or None (auto), got {self.shrink_lambda}")

            # ---- v2 §3.3 / §3.4 / §3.6 routing (before any generator is fitted) ----
            self.route_stats_ = self._compute_route_stats(X)
            self.group_config_ = self._route_candidate_pool(self.route_stats_)
            self.group_prior_ = self._resolve_group_prior(self.route_stats_, self.group_config_)
            self._log_routing()

            # Freeze adaptive_plus structure before any holdout splitting
            if self.group_config_["use_ap"]:
                self._freeze_adaptive_plus_structure(X)
            self._fit_enhanced_generators(X, y)
            self.n_candidates_ = (
                self._frozen_main_views_
                + self._frozen_q_views_
                + self._frozen_svd_ens_views_
                + self._frozen_ap_views_
                + self._frozen_gr_views_
            )

            # Feature sampling: build per-estimator column indices for ALL
            # estimator groups when n_features exceeds max_num_features.
            # Index layout mirrors the candidate order in
            # _collect_candidate_probs: [main | quantile_safe | svd_ens |
            # adaptive_plus | gaussian_rank]. Computed after
            # _fit_enhanced_generators so we use the generator's actual
            # n_features_in_ (post UniqueFeatureFilter), avoiding
            # out-of-bounds column indices.
            n_features = self.ensemble_generator_.n_features_in_
            _n_ap_est = self.adaptive_plus_structure_.get("n_estimators", 0) if self.group_config_["use_ap"] else 0
            n_total_estimators = (
                self.n_estimators
                + self.group_config_["n_quantile"]
                + self.group_config_["n_svd_ens"]
                + _n_ap_est
                + self.group_config_["n_gr"]
            )
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
                    f"[TabLDM] feature sampling triggered: "
                    f"n_features={n_features} -> max_num_features={self.max_num_features}, "
                    f"n_estimators={n_total_estimators}"
                )
            else:
                print(
                    f"[TabLDM] feature sampling off: n_features={n_features}, max_num_features={self.max_num_features}"
                )
            if (self.feat_sample_indices_ is not None) != bool(self.route_stats_["feat_sampled"]):
                print(
                    f"[TabLDM:route] WARNING: feature sampling outcome "
                    f"({self.feat_sample_indices_ is not None}) differs from the routing "
                    f"prediction ({self.route_stats_['feat_sampled']}); the §3.4 pool rules "
                    f"already ran on the predicted flag."
                )

            self._fit_nnls_weights(X, y)
            if self.enable_calibration:
                self._fit_calibration()
        else:
            # Original single-generator path
            self.feat_sample_indices_ = None
            self.ensemble_generator_ = EnsembleGenerator(
                classification=True,
                n_estimators=self.n_estimators,
                norm_methods=self.norm_methods or ["none", "power"],
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
                cat_random_encode=self.cat_random_encode,
                categorical_indices=self.categorical_indices,
            )
            self.ensemble_generator_.fit(X, y)

        # KV cache (only for non-enhanced path)
        self.model_kv_cache_ = None
        if self.kv_cache:
            if self.kv_cache is True or self.kv_cache == "kv":
                self.cache_mode_ = "kv"
            elif self.kv_cache == "repr":
                self.cache_mode_ = "repr"
            else:
                raise ValueError(f"Invalid kv_cache value '{self.kv_cache}'. Expected False, True, 'kv', or 'repr'.")
            self._build_kv_cache()

        return self

    def _build_kv_cache(self) -> None:
        """Pre-compute KV caches for training data across all ensemble batches."""
        train_data = self.ensemble_generator_.transform(X=None, mode="train")
        self.model_kv_cache_ = OrderedDict()

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
                    self.model_.forward_with_cache(
                        X_train=X_batch,
                        y_train=y_batch,
                        use_cache=False,
                        store_cache=True,
                        cache_mode=self.cache_mode_,
                        inference_config=self.inference_config_,
                    )
                caches.append(self.model_._cache)
                self.model_.clear_cache()

            self.model_kv_cache_[norm_method] = TabLDMCache.concat(caches)

    # ==================================================================
    # Adaptive-plus structure freeze (Group 7)
    # ==================================================================

    def _freeze_adaptive_plus_structure(self, X: np.ndarray) -> None:
        """Freeze adaptive_plus routing config using full X (no y, no leakage)."""
        n_features = X.shape[1]
        try:
            n_num = int(
                np.sum(
                    [
                        not np.issubdtype(X[:, c].dtype, np.integer) and len(np.unique(X[:, c])) > 10
                        for c in range(n_features)
                    ]
                )
            )
        except Exception:
            n_num = n_features

        cfg = _get_adaptive_inference_config(
            n_features,
            n_num,
            enable_augmentations=self.adaptive_plus_enable_augmentations,
        )

        use_pca = cfg.get("use_pca_decorr", False)
        pca_active = False
        if use_pca:
            probe = PCADecorrelator()
            probe.fit(X)
            pca_active = bool(probe.active_)

        use_ia = cfg.get("use_interactions", False)
        ia_active = False
        if use_ia:
            ia_probe = InteractionAugmentor()
            ia_probe.fit(X)
            ia_active = bool(ia_probe.active_)

        self.adaptive_plus_structure_ = {
            "n_estimators": int(cfg["n_estimators"]),
            "norm_methods": list(cfg["norm_methods"]),
            "use_svd": bool(cfg.get("use_svd", False)),
            "svd_n_components": cfg.get("svd_n_components", None),
            "use_pca_decorr": use_pca,
            "pca_active": pca_active,
            "use_interactions": use_ia,
            "ia_active": ia_active,
        }
        print(
            f"[TabLDM:adaptive_plus] frozen structure: "
            f"n_estimators={self.adaptive_plus_structure_['n_estimators']}, "
            f"norm_methods={self.adaptive_plus_structure_['norm_methods']}, "
            f"use_svd={self.adaptive_plus_structure_['use_svd']}, "
            f"pca_active={pca_active}, ia_active={ia_active}"
        )

    # ==================================================================
    # v2 no-K-Fold routing (candidate pool, group prior, fusion shrinkage)
    # ==================================================================

    #: Group keys of the v2 §3.6 prior table, in table order.
    _GROUP_KEYS = ("main", "ap", "gr", "svd", "q")

    @staticmethod
    def _candidate_group(name: str) -> str:
        """Map a candidate name to its v2 §3.6 group key.

        Group membership is resolved from the name prefix, never from a
        global integer slot index (v2 §2.3): which groups are present moves
        the index layout around.
        """
        prefix = name.split("[", 1)[0]
        return {
            "main": "main",
            "adaptive_plus": "ap",
            "gaussian_rank": "gr",
            "svd_ens": "svd",
            "quantile": "q",
        }.get(prefix, "main")

    def _resolve_categorical_mask(self, X: np.ndarray) -> np.ndarray:
        """Per-column categorical mask for the v2 §3.3 N/C split.

        The口径 is the structure of the training matrix this fit actually
        received — never upstream dataset metadata (v2 §1.3). Resolution
        order:

        1. ``categorical_indices`` when the caller passed one;
        2. pandas dtypes recorded by ``TransformToNumerical`` (DataFrame
           input: its ``ColumnTransformer`` emits ``[continuous | categorical]``
           blocks);
        3. ``EnsembleGenerator``'s own default — columns with fewer than 10
           distinct non-NaN values are categorical (this also matches its
           ``categorical_indices=None`` handling, so routing and
           ``cat_random_encode`` agree).
        """
        n_cols = X.shape[1]
        if self.categorical_indices is not None:
            mask = np.zeros(n_cols, dtype=bool)
            for idx in self.categorical_indices:
                if 0 <= int(idx) < n_cols:
                    mask[int(idx)] = True
            return mask

        tfm = getattr(self.X_encoder_, "tfm_", None)
        if hasattr(tfm, "transformers_"):
            n_continuous = 0
            n_categorical = 0
            for name, _sub, cols in tfm.transformers_:
                if name == "continuous":
                    n_continuous = len(cols)
                elif name == "categorical":
                    n_categorical = len(cols)
            if n_continuous + n_categorical == n_cols:
                mask = np.zeros(n_cols, dtype=bool)
                mask[n_continuous:] = True
                return mask

        mask = np.zeros(n_cols, dtype=bool)
        for c in range(n_cols):
            col = X[:, c]
            col = col[np.isfinite(col)]
            mask[c] = np.unique(col).size < 10 if col.size else True
        return mask

    def _compute_route_stats(self, X: np.ndarray) -> dict:
        """Compute the v2 §3.3 routing statistics from the training matrix.

        ``n_features`` is the post-unique-filter effective width (the rule
        windows and the feature-sampling trigger run on it); the N/C counts
        and per-column shape statistics cover every encoded column, matching
        the N_*/C_* caliber of the v2 analysis script.
        """
        X = np.asarray(X, dtype=np.float64)
        n_rows, n_cols = X.shape
        is_cat = self._resolve_categorical_mask(X)
        n_num = int((~is_cat).sum())
        n_cat = int(is_cat.sum())

        kurts: list[float] = []
        cat_cards: list[int] = []
        n_values = np.zeros(n_cols, dtype=np.int64)
        for c in range(n_cols):
            col = X[:, c]
            col = col[np.isfinite(col)]
            if is_cat[c]:
                # Drop the __MISSING__ / unknown sentinel (NaN above, -1 for
                # ordinal-encoded categoricals) before counting categories.
                col = col[col != -1]
            n_values[c] = np.unique(col).size if col.size else 0
            if is_cat[c]:
                cat_cards.append(int(n_values[c]))
            else:
                kurts.append(_excess_kurtosis(col))

        # UniqueFeatureFilter keeps columns with >1 distinct value; that is
        # the "编码后有效特征数" the rule windows use (§3.3). Reuse the real
        # filter so the count matches what EnsembleGenerator will report.
        unique_filter = UniqueFeatureFilter()
        unique_filter.fit(X)
        n_features = int(unique_filter.n_features_out_)
        feat_sampled = self.max_num_features is not None and n_features > self.max_num_features

        share_kurt10 = float(np.mean([abs(k) > 10.0 for k in kurts])) if kurts else 0.0
        mean_cat_card = float(np.mean(cat_cards)) if cat_cards else 0.0
        cat_ratio = n_cat / max(n_cols, 1)

        return dict(
            n_train=n_rows,
            n_features=n_features,
            n_features_encoded=n_cols,
            n_num=n_num,
            n_cat=n_cat,
            cat_ratio=cat_ratio,
            share_kurt10=share_kurt10,
            mean_cat_card=mean_cat_card,
            n_classes=int(self.n_classes_),
            feat_sampled=bool(feat_sampled),
            heavy_tail=bool(share_kurt10 >= 0.50),
            cat_dominant=bool(cat_ratio >= 0.80),
            high_dim=bool(n_features > 500 or feat_sampled),
        )

    def _route_candidate_pool(self, stats: dict) -> dict:
        """v2 §3.4 rule table: which groups enter the candidate pool.

        Priority order R1..R5, first match wins. ``use_*`` constructor flags
        are tri-state overrides: ``None`` follows the rules, ``True`` /
        ``False`` force the group on/off (kept for the §5 ablation arms).
        """
        n_train = stats["n_train"]
        n_features = stats["n_features"]
        mean_cat_card = stats["mean_cat_card"]

        # gaussian_rank gate shared by R2-R5 (§3.4 note), with the R1 /
        # low-cardinality-R2 override from the analysis script's _v2_route.
        gr_ok = (2000 <= n_train <= 30000 and 11 <= n_features <= 100) or mean_cat_card >= 8
        if stats["high_dim"] or (stats["cat_dominant"] and mean_cat_card < 8):
            gr_ok = False

        if stats["high_dim"]:
            rule, use_ap, use_svd = "R1-high-dim", False, False
        elif stats["cat_dominant"]:
            rule, use_ap, use_svd = "R2-cat-dominant", True, False
        elif stats["heavy_tail"] and 31 <= n_features <= 500 and stats["cat_ratio"] < 0.6:
            rule, use_ap, use_svd = "R3-heavy+svd", True, True
        elif stats["heavy_tail"]:
            rule, use_ap, use_svd = "R4-heavy", True, False
        else:
            rule, use_ap, use_svd = "R5-default", True, False

        def _override(flag: bool | None, auto: bool) -> bool:
            return auto if flag is None else bool(flag)

        use_ap = _override(self.use_adaptive_plus_candidate, use_ap)
        use_gr = _override(self.use_gaussian_rank_ens, gr_ok)
        use_svd = _override(self.use_svd_ens, use_svd)

        n_gr = int(self.n_gaussian_rank_estimators) if use_gr else 0
        # Rule size for the svd_ens group is 8 (§3.4); the constructor knob
        # overrides it. When the group is off the size is dropped entirely so
        # it can never be a silently dead knob (§4.3).
        n_svd = (int(self.n_svd_ens_estimators) if self.n_svd_ens_estimators is not None else 8) if use_svd else 0
        n_quantile = int(self.n_quantile_estimators or 0)

        return dict(
            rule=rule,
            use_ap=use_ap,
            use_gr=use_gr,
            use_svd_ens=use_svd,
            n_gr=n_gr,
            n_svd_ens=n_svd,
            n_quantile=n_quantile,
            gr_gate=gr_ok,
        )

    @staticmethod
    def _route_group_prior(stats: dict) -> dict:
        """v2 §3.6 group mixing ratios pi_0 (match order = table order)."""
        if stats["high_dim"]:
            pi = dict(main=1.00, ap=0.0, gr=0.0, svd=0.0, q=0.0)
        elif stats["cat_dominant"] and stats["mean_cat_card"] < 8:
            pi = dict(main=0.85, ap=0.15, gr=0.0, svd=0.0, q=0.0)
        elif stats["cat_dominant"]:
            pi = dict(main=0.75, ap=0.15, gr=0.10, svd=0.0, q=0.0)
        elif stats["heavy_tail"]:
            pi = dict(main=0.50, ap=0.25, gr=0.10, svd=0.15, q=0.0)
        elif stats["mean_cat_card"] >= 8:
            pi = dict(main=0.60, ap=0.25, gr=0.15, svd=0.0, q=0.0)
        else:
            pi = dict(main=0.55, ap=0.25, gr=0.20, svd=0.0, q=0.0)
        return pi

    def _resolve_group_prior(self, stats: dict, group_config: dict) -> dict:
        """pi_0 after folding off-group mass into main (v2 §3.6).

        The quantile group carries no prior mass even when it is on — it only
        exists as extra NNLS candidates (§3.6), so a routed pi always has
        ``q = 0``.
        """
        if self.group_prior is not None:
            pi = {g: float(self.group_prior.get(g, 0.0)) for g in self._GROUP_KEYS}
            unknown = set(self.group_prior) - set(self._GROUP_KEYS)
            if unknown:
                raise ValueError(
                    f"group_prior has unknown keys {sorted(unknown)}; expected a subset of {self._GROUP_KEYS}"
                )
        else:
            pi = self._route_group_prior(stats)

        on = {
            "main": True,
            "ap": bool(group_config["use_ap"]),
            "gr": bool(group_config["use_gr"]),
            "svd": bool(group_config["use_svd_ens"]),
            "q": group_config["n_quantile"] > 0,
        }
        for g in ("ap", "gr", "svd", "q"):
            if not on[g] and pi.get(g, 0.0) > 0.0:
                print(
                    f"[TabLDM:route] group '{g}' is off but carries pi={pi[g]:.4f}; "
                    f"folding its mass into 'main' (v2 §3.6)."
                )
                pi["main"] = pi.get("main", 0.0) + pi[g]
                pi[g] = 0.0

        total = float(sum(pi.values()))
        if total <= 0:
            raise ValueError(f"group_prior values must sum to a positive value, got {pi}")
        pi = {g: v / total for g, v in pi.items()}

        # §3.6: the quantile group is an NNLS-only experiment group; the
        # routed prior never pays it (an explicit group_prior may, capped by
        # the suggested <=0.05 experimental value).
        if self.group_prior is None:
            pi["q"] = 0.0
        return pi

    @staticmethod
    def _route_shrink_lambda(n_train: int, n_val: int, n_candidates: int) -> float:
        """v2 §3.5 fusion shrinkage from ``val_per_weight = n_val / E``.

        The tiers are 0.75 / 0.50 / 0.25 / 0 at val_per_weight 20 / 10 / 5.
        ``n_train < 1000`` is pinned to 0: that band must fall back to the
        group prior, never to all-candidate equal weighting (§3.5 / §4.1 —
        the measured equal-weight fallback cost LogLoss +0.0085).
        """
        if n_train < 1000:
            return 0.0
        val_per_weight = n_val / max(n_candidates, 1)
        if val_per_weight >= 20:
            return 0.75
        if val_per_weight >= 10:
            return 0.50
        if val_per_weight >= 5:
            return 0.25
        return 0.0

    def _log_routing(self) -> None:
        """One-shot log of the routing inputs and decisions (v2 §4.5)."""
        s = self.route_stats_
        cfg = self.group_config_
        pi = self.group_prior_
        print(
            f"[TabLDM:route] stats: n_train={s['n_train']}, n_features={s['n_features']} "
            f"(encoded={s['n_features_encoded']}), n_num={s['n_num']}, n_cat={s['n_cat']}, "
            f"cat_ratio={s['cat_ratio']:.4f}, share_kurt10={s['share_kurt10']:.4f}, "
            f"mean_cat_card={s['mean_cat_card']:.2f}, n_classes={s['n_classes']}, "
            f"heavy_tail={s['heavy_tail']}, cat_dominant={s['cat_dominant']}, "
            f"high_dim={s['high_dim']}, feat_sampled={s['feat_sampled']}"
        )
        print(
            f"[TabLDM:route] pool ({cfg['rule']}): main={self.n_estimators}, "
            f"adaptive_plus={'on' if cfg['use_ap'] else 'off'}, "
            f"gaussian_rank={cfg['n_gr']}, "
            f"svd_ens={cfg['n_svd_ens']}"
            + ("" if cfg["use_svd_ens"] else f" (n_svd_ens_estimators={self.n_svd_ens_estimators} ignored)")
            + f", quantile={cfg['n_quantile']}, use_cross_feature={self.use_cross_feature}"
        )
        print(
            f"[TabLDM:route] group prior pi_0: main={pi['main']:.4f}, ap={pi['ap']:.4f}, "
            f"gr={pi['gr']:.4f}, svd={pi['svd']:.4f}, q={pi['q']:.4f} "
            f"({'explicit' if self.group_prior is not None else cfg['rule']})"
        )
        print(
            f"[TabLDM:route] fusion: k_fold=disabled, shrink_lambda="
            f"{'auto(val/E tiers)' if self.shrink_lambda is None else self.shrink_lambda}, "
            f"validation={self.validation}, enable_calibration={self.enable_calibration}"
        )

    # ==================================================================
    # Enhanced X-side candidate generation
    # ==================================================================

    def _fit_enhanced_generators(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit main + optional SVD/cross + quantile_safe + svd_ens + adaptive_plus + gaussian_rank generators.

        Which groups are built (and their sizes) comes from
        ``self.group_config_`` — the v2 §3.4 pool decision — not from the raw
        constructor flags.
        """
        norm_methods = self.norm_methods or ["none", "power"]
        cfg = self.group_config_

        # ---- main group (default candidates) ----
        self.ensemble_generator_ = EnsembleGenerator(
            classification=True,
            n_estimators=self.n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=self.class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
            cat_random_encode=self.cat_random_encode,
            categorical_indices=self.categorical_indices,
        )
        self.ensemble_generator_.fit(X, y)

        # ---- SVD + feature-crossing augmentation ----
        self.svd_ = None
        if self.n_estimators == 32 and sorted(norm_methods) == ["none", "power"]:
            self._fit_svd_cross(self.ensemble_generator_.X_)

        # ---- quantile_safe group ----
        self.quantile_ensemble_generator_ = None
        if cfg["n_quantile"] > 0:
            self.quantile_ensemble_generator_ = EnsembleGenerator(
                classification=True,
                n_estimators=cfg["n_quantile"],
                norm_methods=["quantile"],
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            self.quantile_ensemble_generator_.fit(X, y)

        # ---- Group 6: SVD feature ensemble ("svd+") ----
        self.svd_ens_generator_ = None
        if cfg["n_svd_ens"] > 0:
            self.svd_ens_generator_ = EnsembleGenerator(
                classification=True,
                n_estimators=cfg["n_svd_ens"],
                norm_methods=self.svd_ens_norm_methods,
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
                use_svd=True,
                svd_n_components=self.svd_ens_n_components,
            )
            self.svd_ens_generator_.fit(X, y)

        # ---- Group 7: adaptive_plus candidate ----
        self.adaptive_plus_generator_ = None
        self.adaptive_plus_pca_decorr_ = None
        self.adaptive_plus_interaction_ = None
        if cfg["use_ap"]:
            s = self.adaptive_plus_structure_
            X_ap = X.copy()
            if s["use_pca_decorr"]:
                pca_d = PCADecorrelator()
                pca_d.fit(X_ap)
                if s["pca_active"] and pca_d.active_:
                    X_ap = pca_d.transform(X_ap)
                    self.adaptive_plus_pca_decorr_ = pca_d
                elif s["pca_active"] and not pca_d.active_:
                    self.adaptive_plus_pca_decorr_ = pca_d
            if s["use_interactions"]:
                ia = InteractionAugmentor()
                ia.fit(X_ap)
                if s["ia_active"] and ia.active_:
                    X_ap = ia.transform(X_ap)
                    self.adaptive_plus_interaction_ = ia
                elif s["ia_active"] and not ia.active_:
                    self.adaptive_plus_interaction_ = ia
            gen_kwargs = dict(
                classification=True,
                n_estimators=s["n_estimators"],
                norm_methods=s["norm_methods"],
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            if s["use_svd"]:
                gen_kwargs["use_svd"] = True
                if s["svd_n_components"] is not None:
                    gen_kwargs["svd_n_components"] = s["svd_n_components"]
            self.adaptive_plus_generator_ = EnsembleGenerator(**gen_kwargs)
            self.adaptive_plus_generator_.fit(X_ap, y)

        # ---- Group 8: Gaussian rank normalization ensemble ----
        self.gaussian_rank_generator_ = None
        if cfg["n_gr"] > 0:
            self.gaussian_rank_generator_ = GaussianRankGenerator(
                n_estimators=cfg["n_gr"],
                feat_shuffle_method="random",
                random_state=self.random_state,
            )
            self.gaussian_rank_generator_.fit(X, y)

        # ---- Freeze effective view counts ----
        def _count_views(gen):
            if gen is None:
                return 0
            return sum(len(v) for v in gen.ensemble_configs_.values())

        self._frozen_main_views_ = _count_views(self.ensemble_generator_)
        self._frozen_q_views_ = _count_views(getattr(self, "quantile_ensemble_generator_", None))
        self._frozen_svd_ens_views_ = _count_views(getattr(self, "svd_ens_generator_", None))
        self._frozen_ap_views_ = _count_views(getattr(self, "adaptive_plus_generator_", None))
        self._frozen_gr_views_ = _count_views(getattr(self, "gaussian_rank_generator_", None))

        _use_augment = getattr(self, "svd_", None) is not None and self.n_estimators == 32
        if _use_augment:
            self._frozen_plain_views_ = self._frozen_main_views_ // 2
            self._frozen_cross_svd_views_ = self._frozen_main_views_ - self._frozen_plain_views_
        else:
            self._frozen_plain_views_ = self._frozen_main_views_
            self._frozen_cross_svd_views_ = 0

        print(
            f"[TabLDM:enhance] generators fitted: "
            f"requested_n_estimators=(main={self.n_estimators}, "
            f"q={cfg['n_quantile']}, "
            f"ap={self.adaptive_plus_structure_['n_estimators'] if cfg['use_ap'] else 0}, "
            f"svd_ens={cfg['n_svd_ens']}, gr={cfg['n_gr']}), "
            f"frozen_effective_n=(main={self._frozen_main_views_}, "
            f"plain={self._frozen_plain_views_}, cross_svd={self._frozen_cross_svd_views_}, "
            f"quantile_safe={self._frozen_q_views_}, svd_ens(group6)={self._frozen_svd_ens_views_}, "
            f"adaptive_plus(group7)={self._frozen_ap_views_}, gaussian_rank(group8)={self._frozen_gr_views_}), "
            f"svd/cross={'on' if getattr(self, 'svd_', None) is not None else 'off'}, "
            f"use_cross_feature={self.use_cross_feature}"
        )

    # ==================================================================
    # SVD + cross feature fitting
    # ==================================================================

    def _fit_svd_cross(self, X_raw: np.ndarray) -> None:
        """Fit SVD pipeline and build crossing pool on unique-filtered raw training features."""
        rng = np.random.default_rng(self.random_state)
        n_train, n_orig = X_raw.shape

        is_cat = np.array(
            [np.issubdtype(X_raw[:, c].dtype, np.integer) or len(np.unique(X_raw[:, c])) <= 10 for c in range(n_orig)],
            dtype=bool,
        )
        num_idx = np.where(~is_cat)[0]
        cat_idx = np.where(is_cat)[0]

        k = max(1, int(math.sqrt(n_orig)))
        self._k_ = k

        # ---- Crossing pool (numerical cols only) ----
        if not self.use_cross_feature:
            self._cross_pool_ = []
            self._cross_scaler_ = None
            self._cross_pool_train_ = np.empty((n_train, 0), dtype=np.float32)
        elif len(num_idx) >= 2:
            all_pairs = list(itertools.combinations(num_idx.tolist(), 2))
            cross_pool_size = min(16 * k, len(all_pairs))
            pool_sel = rng.choice(len(all_pairs), size=cross_pool_size, replace=False)
            self._cross_pool_ = [all_pairs[i] for i in pool_sel]
            cross_mat = np.stack(
                [X_raw[:, i].astype(float) * X_raw[:, j].astype(float) for i, j in self._cross_pool_],
                axis=1,
            )
            self._cross_scaler_ = StandardScaler()
            self._cross_pool_train_ = self._cross_scaler_.fit_transform(cross_mat).clip(-100, 100)
        else:
            self._cross_pool_ = []
            self._cross_scaler_ = None
            self._cross_pool_train_ = np.empty((n_train, 0), dtype=np.float32)

        self._cross_selections_ = []
        for _ in range(16):
            n_sel = min(k, len(self._cross_pool_))
            if n_sel > 0:
                sel = rng.choice(len(self._cross_pool_), size=n_sel, replace=False).tolist()
            else:
                sel = []
            self._cross_selections_.append(sel)

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
            self.svd_ = None
            self._svd_pre_ = None
            self._X_train_svd_ = np.empty((n_train, 0), dtype=np.float32)
            self._svd_selections_ = [[] for _ in range(16)]
            return

        svd = TruncatedSVD(n_components=svd_pool_size, random_state=self.random_state)
        self._svd_pre_ = svd_pre
        self.svd_ = svd
        self._X_train_svd_ = svd.fit_transform(X_pre).clip(-100, 100).astype(np.float32)

        self._svd_selections_ = []
        for _ in range(16):
            n_sel = min(k, svd_pool_size)
            sel = rng.choice(svd_pool_size, size=n_sel, replace=False).tolist()
            self._svd_selections_.append(sel)

    def _augment_estimator(self, X_preprocessed, raw_X, odd_local, is_train):
        """Append selected cross and SVD features to preprocessed features."""
        parts = [X_preprocessed]

        cross_sel = self._cross_selections_[odd_local]
        if cross_sel:
            if is_train:
                cross_feats = self._cross_pool_train_[:, cross_sel].astype(np.float32)
            else:
                all_cross_mat = np.stack(
                    [raw_X[:, i].astype(float) * raw_X[:, j].astype(float) for i, j in self._cross_pool_],
                    axis=1,
                )
                all_cross_scaled = self._cross_scaler_.transform(all_cross_mat).clip(-100, 100)
                cross_feats = all_cross_scaled[:, cross_sel].astype(np.float32)
            parts.append(cross_feats)

        svd_sel = self._svd_selections_[odd_local]
        if svd_sel and getattr(self, "svd_", None) is not None:
            if is_train:
                svd_feats = self._X_train_svd_[:, svd_sel]
            else:
                svd_feats = (
                    self.svd_.transform(self._svd_pre_.transform(raw_X))[:, svd_sel].clip(-100, 100).astype(np.float32)
                )
            parts.append(svd_feats)

        return np.concatenate(parts, axis=1) if len(parts) > 1 else X_preprocessed

    # ==================================================================
    # Forward helpers
    # ==================================================================

    def _forward_candidates(
        self, Xs: np.ndarray, ys: np.ndarray, feat_indices: list[np.ndarray] | None = None
    ) -> np.ndarray:
        """Forward a batch of ensemble members and return per-estimator probabilities."""
        n_bad_xs = int(np.sum(~np.isfinite(Xs)))
        if n_bad_xs > 0:
            print(
                f"[TabLDM:forward_candidates] WARNING: transformed input Xs has "
                f"{n_bad_xs}/{Xs.size} non-finite values (shape={Xs.shape})"
            )

        if feat_indices is not None:
            Xs = np.stack([Xs[i][:, feat_indices[i]] for i in range(Xs.shape[0])], axis=0)

        batch_size = self.batch_size_ or Xs.shape[0]
        n_batches = int(np.ceil(Xs.shape[0] / batch_size))
        Xs_split = np.array_split(Xs, n_batches)
        ys_split = np.array_split(ys, n_batches)

        outputs = []
        for X_batch, y_batch in zip(Xs_split, ys_split, strict=False):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)
            with torch.no_grad():
                out = self.model_(
                    X=X_batch,
                    y_train=y_batch,
                    feature_shuffles=None,
                    return_logits=True,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            raw = out.float().cpu().numpy()
            n_bad_logits = int(np.sum(~np.isfinite(raw)))
            if n_bad_logits > 0:
                print(
                    f"[TabLDM:forward_candidates] WARNING: raw model output has "
                    f"{n_bad_logits}/{raw.size} non-finite values (batch_shape={raw.shape})"
                )
            outputs.append(raw)

        logits = np.concatenate(outputs, axis=0)
        probs = self.softmax(logits, axis=-1, temperature=self.softmax_temperature)
        n_bad_probs = int(np.sum(~np.isfinite(probs)))
        if n_bad_probs > 0:
            print(
                f"[TabLDM:forward_candidates] WARNING: final probs has "
                f"{n_bad_probs}/{probs.size} non-finite values after softmax (shape={probs.shape})"
            )
        return probs

    def _batch_forward(self, Xs: np.ndarray, ys: np.ndarray, feature_shuffles: np.ndarray | None = None) -> np.ndarray:
        """Process model forward passes in batches."""
        batch_size = self.batch_size_ or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)
        if feature_shuffles is None:
            feature_shuffles = [None] * n_batches
        else:
            feature_shuffles = np.array_split(feature_shuffles, n_batches)

        outputs = []
        for X_batch, y_batch, shuffle_batch in zip(Xs, ys, feature_shuffles, strict=False):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)
            if shuffle_batch is not None:
                shuffle_batch = shuffle_batch.tolist()
            with torch.no_grad():
                out = self.model_(
                    X=X_batch,
                    y_train=y_batch,
                    feature_shuffles=shuffle_batch,
                    return_logits=bool(self.average_logits),
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            outputs.append(out.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _batch_forward_with_cache(self, Xs: np.ndarray, kv_cache: TabLDMCache) -> np.ndarray:
        """Process model forward passes using a pre-computed KV cache."""
        n_total = Xs.shape[0]
        batch_size = self.batch_size_ or n_total
        n_batches = int(np.ceil(n_total / batch_size))
        Xs_split = np.array_split(Xs, n_batches)

        outputs = []
        offset = 0
        for X_batch in Xs_split:
            bs = X_batch.shape[0]
            cache_subset = kv_cache.slice_batch(offset, offset + bs)
            offset += bs
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            with torch.no_grad():
                out = self.model_.forward_with_cache(
                    X_test=X_batch,
                    cache=cache_subset,
                    return_logits=bool(self.average_logits),
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            outputs.append(out.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    # ==================================================================
    # Candidate probability collection
    # ==================================================================

    def _collect_candidate_probs(
        self,
        X: np.ndarray,
        main_gen: EnsembleGenerator,
        q_gen: EnsembleGenerator | None,
        svd_ens_gen: EnsembleGenerator | None = None,
        adaptive_plus_gen: EnsembleGenerator | None = None,
        adaptive_plus_pca: PCADecorrelator | None = None,
        adaptive_plus_ia: InteractionAugmentor | None = None,
        gaussian_rank_gen=None,
        tag: str = "predict",
    ) -> tuple:
        """Collect per-estimator probabilities in canonical candidate order."""
        all_probs = []
        _fold_label = getattr(self, "_current_fold_label_", "")

        def _check_and_log_finite(group_probs, mode_name, est_offset):
            n_est = group_probs.shape[0]
            valid = np.ones(n_est, dtype=bool)
            for i in range(n_est):
                p = group_probs[i]
                if np.all(np.isfinite(p)):
                    continue
                valid[i] = False
                bad_mask = ~np.isfinite(p)
                n_bad = int(bad_mask.sum())
                bad_rows, bad_cols = np.where(bad_mask)
                print(
                    f"[TabLDM:finite_check{_fold_label}] DISCARD "
                    f"mode={mode_name} estimator={est_offset + i} "
                    f"prob_shape={p.shape} n_nonfinite={n_bad}"
                )
            return valid

        fsi = getattr(self, "feat_sample_indices_", None)

        # ---- main group ----
        data = main_gen.transform(X, mode="both")
        use_augment = getattr(self, "svd_", None) is not None and self.n_estimators == 32
        if use_augment:
            X_raw_test = main_gen.unique_filter_.transform(X)
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

                per_est_probs = {}
                if even_idxs:
                    fi = [fsi[g] for g in even_global] if fsi is not None else None
                    _p = self._forward_candidates(Xs_both[even_idxs], ys[even_idxs], feat_indices=fi)
                    for i, li in enumerate(even_idxs):
                        per_est_probs[li] = _p[i]
                if odd_idxs:
                    Xs_odd_list = []
                    for li, ol in zip(odd_idxs, odd_locals, strict=False):
                        x_both_i = Xs_both[li]
                        x_tr = self._augment_estimator(x_both_i[:n_train_rows], X_raw_train, ol, is_train=True)
                        x_te = self._augment_estimator(x_both_i[n_train_rows:], X_raw_test, ol, is_train=False)
                        Xs_odd_list.append(np.concatenate([x_tr, x_te], axis=0))
                    Xs_odd = np.stack(Xs_odd_list, axis=0)
                    fi = [fsi[g] for g in odd_global] if fsi is not None else None
                    _p = self._forward_candidates(Xs_odd, ys[odd_idxs], feat_indices=fi)
                    for i, li in enumerate(odd_idxs):
                        per_est_probs[li] = _p[i]

                stacked = np.stack([per_est_probs[li] for li in range(n_est_this_method)], axis=0)
                if hasattr(main_gen, "class_shuffles_") and norm_method in main_gen.class_shuffles_:
                    shuffles = main_gen.class_shuffles_[norm_method]
                    for _si in range(n_est_this_method):
                        stacked[_si] = stacked[_si][:, shuffles[_si]]
                all_probs.append(stacked)
        else:
            _grp_offset = 0
            for norm_method, (Xs, ys) in data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_grp_offset + i] for i in range(n_est)] if fsi is not None else None
                probs_arr = self._forward_candidates(Xs, ys, feat_indices=fi)
                if hasattr(main_gen, "class_shuffles_") and norm_method in main_gen.class_shuffles_:
                    shuffles = main_gen.class_shuffles_[norm_method]
                    for _si in range(n_est):
                        probs_arr[_si] = probs_arr[_si][:, shuffles[_si]]
                all_probs.append(probs_arr)
                _grp_offset += n_est

        n_main = sum(p.shape[0] for p in all_probs)
        rep_shape = all_probs[0].shape[1:] if all_probs else (0, 0)
        print(
            f"[TabLDM:enhance:{tag}] mode=default n_estimators={n_main} "
            f"per_estimator_prob_shape=(n_test={rep_shape[0]}, n_classes={rep_shape[1]})"
        )

        # ---- quantile_safe group ----
        n_q = 0
        if q_gen is not None:
            q_probs = []
            q_data = q_gen.transform(X, mode="both")
            _q_offset = n_main
            for norm_method, (Xs, ys) in q_data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_q_offset + i] for i in range(n_est)] if fsi is not None else None
                probs_arr = self._forward_candidates(Xs, ys, feat_indices=fi)
                if hasattr(q_gen, "class_shuffles_") and norm_method in q_gen.class_shuffles_:
                    shuffles = q_gen.class_shuffles_[norm_method]
                    for _si in range(n_est):
                        probs_arr[_si] = probs_arr[_si][:, shuffles[_si]]
                q_probs.append(probs_arr)
                _q_offset += n_est
            all_probs.extend(q_probs)
            n_q = sum(p.shape[0] for p in q_probs)
            print(f"[TabLDM:enhance:{tag}] mode=quantile_safe n_estimators={n_q}")

        # ---- group 6: svd+ ensemble ----
        n_svd_ens = 0
        if svd_ens_gen is not None:
            s_probs = []
            s_data = svd_ens_gen.transform(X, mode="both")
            _s_offset = n_main + n_q
            for norm_method, (Xs, ys) in s_data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_s_offset + i] for i in range(n_est)] if fsi is not None else None
                probs_arr = self._forward_candidates(Xs, ys, feat_indices=fi)
                if hasattr(svd_ens_gen, "class_shuffles_") and norm_method in svd_ens_gen.class_shuffles_:
                    shuffles = svd_ens_gen.class_shuffles_[norm_method]
                    for _si in range(n_est):
                        probs_arr[_si] = probs_arr[_si][:, shuffles[_si]]
                s_probs.append(probs_arr)
                _s_offset += n_est
            all_probs.extend(s_probs)
            n_svd_ens = sum(p.shape[0] for p in s_probs)
            print(f"[TabLDM:enhance:{tag}] mode=svd_ens(group6) n_estimators={n_svd_ens}")

        # ---- group 7: adaptive_plus ----
        n_adaptive = 0
        if adaptive_plus_gen is not None:
            X_ap = X
            if adaptive_plus_pca is not None and getattr(adaptive_plus_pca, "active_", False):
                X_ap = adaptive_plus_pca.transform(X_ap)
            if adaptive_plus_ia is not None and getattr(adaptive_plus_ia, "active_", False):
                X_ap = adaptive_plus_ia.transform(X_ap)
            ap_probs = []
            ap_data = adaptive_plus_gen.transform(X_ap, mode="both")
            _ap_offset = n_main + n_q + n_svd_ens
            for norm_method, (Xs, ys) in ap_data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_ap_offset + i] for i in range(n_est)] if fsi is not None else None
                probs_arr = self._forward_candidates(Xs, ys, feat_indices=fi)
                if hasattr(adaptive_plus_gen, "class_shuffles_") and norm_method in adaptive_plus_gen.class_shuffles_:
                    shuffles = adaptive_plus_gen.class_shuffles_[norm_method]
                    for _si in range(n_est):
                        probs_arr[_si] = probs_arr[_si][:, shuffles[_si]]
                ap_probs.append(probs_arr)
                _ap_offset += n_est
            all_probs.extend(ap_probs)
            n_adaptive = sum(p.shape[0] for p in ap_probs)
            print(f"[TabLDM:enhance:{tag}] mode=adaptive_plus(group7) n_estimators={n_adaptive}")

        # ---- group 8: Gaussian rank ----
        n_gaussian_rank = 0
        if gaussian_rank_gen is not None:
            gr_probs = []
            gr_data = gaussian_rank_gen.transform(X, mode="both")
            _gr_offset = n_main + n_q + n_svd_ens + n_adaptive
            for norm_method, (Xs, ys) in gr_data.items():
                n_est = Xs.shape[0]
                fi = [fsi[_gr_offset + i] for i in range(n_est)] if fsi is not None else None
                gr_probs.append(self._forward_candidates(Xs, ys, feat_indices=fi))
                _gr_offset += n_est
            all_probs.extend(gr_probs)
            n_gaussian_rank = sum(p.shape[0] for p in gr_probs)
            print(f"[TabLDM:enhance:{tag}] mode=gaussian_rank(group8) n_estimators={n_gaussian_rank}")

        # ---- Per-estimator finite check ----
        all_probs_concat = np.concatenate(all_probs, axis=0)
        E_total = all_probs_concat.shape[0]

        _offsets = [0, n_main, n_main + n_q, n_main + n_q + n_svd_ens, n_main + n_q + n_svd_ens + n_adaptive]
        _group_names = ["default", "quantile_safe", "svd_ens(group6)", "adaptive_plus(group7)", "gaussian_rank(group8)"]
        _group_sizes = [n_main, n_q, n_svd_ens, n_adaptive, n_gaussian_rank]

        valid_mask = np.ones(E_total, dtype=bool)
        for _gname, _gstart, _gsize in zip(_group_names, _offsets, _group_sizes, strict=False):
            if _gsize == 0:
                continue
            _gprobs = all_probs_concat[_gstart : _gstart + _gsize]
            _gvalid = _check_and_log_finite(_gprobs, _gname, _gstart)
            valid_mask[_gstart : _gstart + _gsize] = _gvalid

        n_valid = int(valid_mask.sum())
        n_dropped = E_total - n_valid
        if n_dropped > 0:
            print(
                f"[TabLDM:finite_check{_fold_label}] dropped {n_dropped}/{E_total} "
                f"non-finite candidates; {n_valid} remaining."
            )
        if n_valid == 0:
            raise RuntimeError(
                f"[TabLDM:finite_check{_fold_label}] ALL {E_total} candidates produced "
                "non-finite probabilities; cannot continue."
            )

        # Candidate names travel with the rows (v2 §2.3): fusion weights are
        # looked up by name so dropped candidates can never shift a group.
        names = self._build_candidate_names(n_main, n_q, n_svd_ens, n_adaptive, n_gaussian_rank)

        probs = all_probs_concat[valid_mask]
        return probs, valid_mask, names

    @staticmethod
    def _build_candidate_names(n_main, n_q, n_svd_ens, n_adaptive, n_gaussian_rank) -> list[str]:
        """Canonical candidate names in ``_collect_candidate_probs`` order.

        Layout: ``[main | quantile | svd_ens | adaptive_plus | gaussian_rank]``
        — main covers both the plain and the cross_svd half under one group
        (v2 §3.6 treats them as one ``main`` mass).
        """
        names = [f"main[{i}]" for i in range(n_main)]
        names += [f"quantile[{i}]" for i in range(n_q)]
        names += [f"svd_ens[{i}]" for i in range(n_svd_ens)]
        names += [f"adaptive_plus[{i}]" for i in range(n_adaptive)]
        names += [f"gaussian_rank[{i}]" for i in range(n_gaussian_rank)]
        return names

    # ==================================================================
    # NNLS weight learning (v2: single holdout only — K-Fold disabled)
    # ==================================================================

    @staticmethod
    def _route_validation(validation: bool, k_fold: bool, n_train: int) -> str:
        """Decide which validation path fit() takes: 'default' or 'single_validation'.

        K-Fold is **completely disabled** (v2 §3.2 / §4.1). The parameter
        ``k_fold`` is accepted for backward compatibility only and never
        routes to a K-Fold path. The small-sample floor is just what
        ``train_test_split`` and the stratified NNLS need: the fusion trust
        for small data is handled by ``shrink_lambda`` (v2 §3.5 pins
        ``n_train < 1000`` to lambda=0), not by throwing the holdout away.

          | condition             | path              |
          |-----------------------|-------------------|
          | validation=False      | default           |
          | k_fold=True (ignored) | *warn*            |
          | n_train < 50          | default           |
          | n_train >= 50         | single_validation |
        """
        if k_fold:
            warnings.warn(
                "k_fold is deprecated and completely disabled in TabLDMClassifier "
                "(v2 no-K-Fold rules). The parameter is ignored; a single holdout is "
                "used at most for NNLS shrinkage. Pass k_fold=False to silence this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not validation:
            return "default"
        return "single_validation" if n_train >= 50 else "default"

    def _fit_nnls_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """v2 §3.5 fusion: single-holdout NNLS shrunk towards the group prior.

        Learns ``p = (1 - lambda) * mix(pi_0) + lambda * p_nnls_holdout``. Every
        failure path falls back to the group prior — never to all-candidate
        equal weighting (v2 §3.5 / §4.1).
        """
        self.nnls_weights_ = None
        self.nnls_weight_map_ = None
        self.nnls_valid_candidate_mask_ = None

        n_full = X.shape[0]
        n_val = math.ceil(0.2 * n_full)
        n_candidates = int(getattr(self, "n_candidates_", 0) or 0)
        if self.shrink_lambda is not None:
            lam = float(self.shrink_lambda)
        else:
            lam = self._route_shrink_lambda(n_full, n_val, n_candidates)
        lam = min(max(lam, 0.0), 1.0)
        self.shrink_lambda_ = lam

        route = self._route_validation(self.validation, self.k_fold, n_full)
        # The holdout feeds both the NNLS term (lambda > 0) and the
        # calibration fit (§2.9 keeps calibration on), so run it whenever
        # either needs it.
        need_holdout = route == "single_validation" and (lam > 0.0 or self.enable_calibration)
        print(
            f"[TabLDM:nnls] v2 fusion: k_fold=disabled, n_train={n_full}, n_val={n_val}, "
            f"E={n_candidates}, val_per_weight={n_val / max(n_candidates, 1):.2f}, "
            f"lambda={lam:.2f}, path={route}, holdout={'yes' if need_holdout else 'no'}"
        )
        if not need_holdout:
            self.shrink_lambda_ = 0.0
            print("[TabLDM:nnls] no holdout -> pure group prior pi_0")
            return

        class_counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        present = class_counts[class_counts > 0]
        min_class = int(present.min()) if present.size else 0
        if min_class < 2:
            self.shrink_lambda_ = 0.0
            print(
                f"[TabLDM:nnls] smallest class has {min_class} sample(s) (<2); cannot stratify -> pure group prior pi_0"
            )
            return

        self._current_fold_label_ = ""
        try:
            probs, y_target, valid_mask, names = self._single_val_probs(X, y)
        except Exception as e:
            self.shrink_lambda_ = 0.0
            print(f"[TabLDM:nnls] holdout collection failed ({e!r}) -> pure group prior pi_0")
            return

        self.nnls_valid_candidate_mask_ = valid_mask
        names_valid = [nm for nm, keep in zip(names, valid_mask, strict=False) if keep]

        weights = self._solve_nnls_classification(probs, y_target, names_valid)
        if weights is None:
            print("[TabLDM:nnls] NNLS could not solve stably -> pure group prior pi_0")
            self.shrink_lambda_ = 0.0
        else:
            self.nnls_weights_ = weights
            self.nnls_weight_map_ = {nm: float(w) for nm, w in zip(names_valid, weights, strict=False) if w > 0}
            print(
                f"[TabLDM:nnls] learned ensemble weights: E={weights.shape[0]} "
                f"(over {int(valid_mask.sum())}/{valid_mask.shape[0]} valid candidates), "
                f"sum={weights.sum():.6f}"
            )

        if self.enable_calibration:
            # Calibration is fitted on the *final* fused holdout probabilities
            # so it matches what predict() produces (v2 §3.5).
            self._cal_P_ = self._combine_probs(probs, names_valid)
            self._cal_y_ = np.asarray(y_target, dtype=np.int64)

    def _single_val_probs(self, X, y):
        """Single stratified holdout probability collection."""
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True,
            random_state=self.random_state,
            stratify=y,
        )
        print(f"[TabLDM:nnls] stratified validation split: n_train={X_tr.shape[0]}, n_val={X_val.shape[0]}")
        gen_info = self._make_and_fit_enhanced_generators(X_tr, y_tr)
        val_probs, valid_mask, names = self._collect_val_probs(X_val, gen_info)
        print(f"[TabLDM:nnls] validation probability matrix shape={val_probs.shape}")
        return val_probs, y_val, valid_mask, names

    def _make_and_fit_enhanced_generators(self, X_tr, y_tr):
        """Fit main + optional generators on the holdout train split.

        Group composition mirrors the full-data fit (``self.group_config_``)
        so holdout and predict candidate slots line up.
        """
        norm_methods = self.norm_methods or ["none", "power"]
        cfg = self.group_config_

        main_frozen_filter = getattr(self.ensemble_generator_, "unique_filter_", None)
        main_gen = EnsembleGenerator(
            classification=True,
            n_estimators=self.n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=self.class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
            cat_random_encode=self.cat_random_encode,
            categorical_indices=self.categorical_indices,
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

        if self.n_estimators == 32 and sorted(norm_methods) == ["none", "power"]:
            self._fit_svd_cross(main_gen.X_)
        else:
            self.svd_ = None

        q_gen = None
        if cfg["n_quantile"] > 0:
            q_frozen_filter = getattr(getattr(self, "quantile_ensemble_generator_", None), "unique_filter_", None)
            q_gen = EnsembleGenerator(
                classification=True,
                n_estimators=cfg["n_quantile"],
                norm_methods=["quantile"],
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            q_gen.fit(X_tr, y_tr, frozen_unique_filter=q_frozen_filter)

        svd_ens_gen = None
        if cfg["n_svd_ens"] > 0:
            svd_ens_frozen_filter = getattr(getattr(self, "svd_ens_generator_", None), "unique_filter_", None)
            svd_ens_gen = EnsembleGenerator(
                classification=True,
                n_estimators=cfg["n_svd_ens"],
                norm_methods=self.svd_ens_norm_methods,
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
                use_svd=True,
                svd_n_components=self.svd_ens_n_components,
            )
            svd_ens_gen.fit(X_tr, y_tr, frozen_unique_filter=svd_ens_frozen_filter)

        adaptive_plus_gen = None
        adaptive_plus_pca = None
        adaptive_plus_ia = None
        if cfg["use_ap"]:
            s = self.adaptive_plus_structure_
            ap_frozen_filter = getattr(getattr(self, "adaptive_plus_generator_", None), "unique_filter_", None)
            X_ap = X_tr.copy()
            if s["use_pca_decorr"]:
                pca_d = PCADecorrelator()
                pca_d.fit(X_ap)
                if s["pca_active"]:
                    X_ap = pca_d.transform(X_ap)
                adaptive_plus_pca = pca_d
            if s["use_interactions"]:
                ia = InteractionAugmentor()
                ia.fit(X_ap)
                if s["ia_active"]:
                    X_ap = ia.transform(X_ap)
                adaptive_plus_ia = ia
            gen_kwargs = dict(
                classification=True,
                n_estimators=s["n_estimators"],
                norm_methods=s["norm_methods"],
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            if s["use_svd"]:
                gen_kwargs["use_svd"] = True
                if s["svd_n_components"] is not None:
                    gen_kwargs["svd_n_components"] = s["svd_n_components"]
            adaptive_plus_gen = EnsembleGenerator(**gen_kwargs)
            adaptive_plus_gen.fit(X_ap, y_tr, frozen_unique_filter=ap_frozen_filter)

        gaussian_rank_gen = None
        if cfg["n_gr"] > 0:
            gr_frozen_filter = getattr(getattr(self, "gaussian_rank_generator_", None), "unique_filter_", None)
            gaussian_rank_gen = GaussianRankGenerator(
                n_estimators=cfg["n_gr"],
                feat_shuffle_method="random",
                random_state=self.random_state,
            )
            gaussian_rank_gen.fit(X_tr, y_tr, frozen_unique_filter=gr_frozen_filter)

        return {
            "main_gen": main_gen,
            "q_gen": q_gen,
            "svd_ens_gen": svd_ens_gen,
            "adaptive_plus_gen": adaptive_plus_gen,
            "adaptive_plus_pca": adaptive_plus_pca,
            "adaptive_plus_ia": adaptive_plus_ia,
            "gaussian_rank_gen": gaussian_rank_gen,
            "saved_svd": saved_svd,
        }

    def _collect_val_probs(self, X_val, gen_info):
        """Collect validation probabilities for the holdout split."""
        try:
            probs, valid_mask, names = self._collect_candidate_probs(
                X_val,
                gen_info["main_gen"],
                gen_info["q_gen"],
                gen_info.get("svd_ens_gen"),
                adaptive_plus_gen=gen_info.get("adaptive_plus_gen"),
                adaptive_plus_pca=gen_info.get("adaptive_plus_pca"),
                adaptive_plus_ia=gen_info.get("adaptive_plus_ia"),
                gaussian_rank_gen=gen_info.get("gaussian_rank_gen"),
                tag="val",
            )
        finally:
            for attr, val in gen_info["saved_svd"].items():
                if val is None and hasattr(self, attr):
                    delattr(self, attr)
                elif val is not None:
                    setattr(self, attr, val)
        return probs, valid_mask, names

    def _group_prior_prediction(
        self,
        probs: np.ndarray,
        names: list[str],
        pi: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Mix candidate probabilities with group ratios pi (v2 §3.5/§3.6).

        Within a group all candidates are averaged with equal weight; the
        group means are then mixed by pi. Groups with no surviving candidate
        row drop out and the remaining mass is renormalized.
        """
        if pi is None:
            pi = self.group_prior_
        groups = [self._candidate_group(nm) for nm in names]

        parts: list[np.ndarray] = []
        weights: list[float] = []
        for g in self._GROUP_KEYS:
            mask = np.array([gg == g for gg in groups], dtype=bool)
            w_g = float(pi.get(g, 0.0))
            if mask.any() and w_g > 0.0:
                parts.append(probs[mask].mean(axis=0))
                weights.append(w_g)

        if not parts:
            # No group matched the prior — fall back to the all-candidate mean.
            return probs.mean(axis=0)
        total = float(sum(weights))
        return sum(w * p for w, p in zip(weights, parts, strict=False)) / total

    def _nnls_prediction(self, probs: np.ndarray, names: list[str]) -> np.ndarray | None:
        """Slot-level NNLS combination, weights looked up by candidate name.

        Name lookup (not row position) keeps holdout and predict aligned when
        the finite-check drops different candidates on either side (v2 §2.3).
        """
        w_map = getattr(self, "nnls_weight_map_", None)
        if not w_map:
            return None
        w = np.array([w_map.get(nm, 0.0) for nm in names], dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0:
            return None
        return np.tensordot(w / w_sum, probs, axes=(0, 0))

    def _combine_probs(self, probs: np.ndarray, names: list[str]) -> np.ndarray:
        """v2 §3.5 fusion: ``p = (1 - lambda) * mix(pi_0) + lambda * p_nnls_holdout``.

        Per v2 §4.2 the legacy ``0.75*nnls + 0.25*uniform`` foundation is
        replaced by the group prior: ``mix(pi_0)`` is the foundation term and
        ``1 - lambda`` the foundation rate (1.0 / 0.75 / 0.50 / 0.25 at the
        §3.5 tiers — never 0). Row-stochastic output.
        """
        p_prior = self._group_prior_prediction(probs, names)
        lam = float(getattr(self, "shrink_lambda_", 0.0) or 0.0)
        if lam <= 0.0:
            out = p_prior
        else:
            p_nnls = self._nnls_prediction(probs, names)
            out = p_prior if p_nnls is None else (1.0 - lam) * p_prior + lam * p_nnls

        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return out / row_sums

    def _log_nnls_group_mass(self, weights, names, n_nonzero, raw_sum) -> None:
        """Per-group NNLS mass + per-candidate nonzero weights (v2 §4.4).

        The v2 analysis could only get group quality via ablation diffs
        because classification logs stopped at ``n_nonzero=x/y``; print the
        aggregated group masses and the named nonzero weights on every
        single-validation run.
        """
        totals = dict.fromkeys(self._GROUP_KEYS, 0.0)
        for nm, w in zip(names, weights, strict=False):
            totals[self._candidate_group(nm)] += float(w)
        print(
            f"[TabLDM:nnls] group_mass: main={totals['main']:.4f}, ap={totals['ap']:.4f}, "
            f"gr={totals['gr']:.4f}, svd_ens={totals['svd']:.4f}, quantile={totals['q']:.4f}"
        )
        nonzero_str = ", ".join(f"{nm}={w:.4f}" for nm, w in zip(names, weights, strict=False) if w > 0)
        print(
            f"[TabLDM:nnls] solved: n_nonzero={n_nonzero}/{len(weights)}, raw_sum={raw_sum:.4f}; "
            f"nonzero weights: {nonzero_str}"
        )

    def _solve_nnls_classification(self, probs, y_val, names):
        """Solve non-negative least squares for the holdout ensemble weights.

        Returns the raw normalized NNLS solution (v2 §4.2: the old
        ``0.75*nnls + 0.25*uniform`` foundation no longer lives here — the
        group-prior shrinkage is applied by ``_combine_probs``).
        """
        E, n_val, n_classes = probs.shape

        if not np.all(np.isfinite(probs)):
            print("[TabLDM:nnls] WARNING: validation probabilities have non-finite values.")
            return None
        if not np.all(probs >= 0):
            print("[TabLDM:nnls] WARNING: validation probabilities contain negative values.")
            return None

        A = probs.reshape(E, n_val * n_classes).T
        onehot = np.zeros((n_val, n_classes), dtype=np.float64)
        onehot[np.arange(n_val), y_val.astype(int)] = 1.0
        b = onehot.reshape(n_val * n_classes)

        try:
            raw_weights, _ = _scipy_nnls(A, b)
        except Exception as e:
            print(f"[TabLDM:nnls] scipy nnls solver failed: {e!r}")
            return None

        w_sum = float(raw_weights.sum())
        n_nonzero = int((raw_weights > 0).sum())
        if not np.isfinite(w_sum) or w_sum <= 0:
            print(f"[TabLDM:nnls] degenerate NNLS solution (sum={w_sum}); not usable.")
            return None

        weights = raw_weights / w_sum
        self._log_nnls_group_mass(weights, names, n_nonzero, w_sum)
        return weights

    # ==================================================================
    # Probability calibration
    # ==================================================================

    @staticmethod
    def _log_loss_safe(P, y):
        eps = 1e-15
        n = len(y)
        return -float(np.sum(np.log(np.clip(P[np.arange(n), y], eps, 1.0))) / n)

    def _fit_calibration(self) -> None:
        """Fit Platt (binary) or vector scaling (multiclass) on stored OOF probs."""
        P_cal = getattr(self, "_cal_P_", None)
        y_cal = getattr(self, "_cal_y_", None)
        if P_cal is None or y_cal is None:
            if self.verbose:
                print("[TabLDM:calibration] no OOF probabilities available; skipping.")
            return

        n_cal, n_classes = P_cal.shape
        present_classes = np.unique(y_cal)
        if len(present_classes) < n_classes:
            print(f"[TabLDM:calibration] WARNING: OOF covers {len(present_classes)}/{n_classes} classes; skipping.")
            return

        eps = 1e-15
        lam = float(self.calibration_lambda)

        if self.verbose:
            method_name = "platt_scaling" if n_classes == 2 else "vector_scaling"
            print(f"[TabLDM:calibration] method={method_name}, n_cal={n_cal}, n_classes={n_classes}, lambda={lam}")

        try:
            if n_classes == 2:
                z = np.log((P_cal[:, 1] + eps) / (P_cal[:, 0] + eps))

                def _platt_nll(params):
                    A, B = params
                    p1 = _expit(A * z + B)
                    p1 = np.clip(p1, eps, 1.0 - eps)
                    nll = -np.mean(y_cal * np.log(p1) + (1 - y_cal) * np.log(1.0 - p1))
                    reg = lam * ((A - 1.0) ** 2 + B**2)
                    return nll + reg

                result = _scipy_minimize(
                    _platt_nll,
                    x0=[1.0, 0.0],
                    method="L-BFGS-B",
                    bounds=[(0.8, 1.2), (-1.0, 1.0)],
                )
                if not result.success or not np.all(np.isfinite(result.x)):
                    reason = result.message if not result.success else "non-finite params"
                    print(f"[TabLDM:calibration] WARNING: Platt fit unusable ({reason}); skipping.")
                    return
                A, B = float(result.x[0]), float(result.x[1])
                params = {"type": "platt", "A": A, "B": B}
            else:
                log_P = np.log(np.clip(P_cal, eps, 1.0))

                def _vs_nll(params):
                    scale = params[:n_classes]
                    bias = params[n_classes:]
                    logits = log_P * scale + bias
                    logits_c = logits - logits.max(axis=1, keepdims=True)
                    exp_l = np.exp(logits_c)
                    P_new = exp_l / exp_l.sum(axis=1, keepdims=True)
                    P_new = np.clip(P_new, eps, 1.0)
                    nll = -np.mean(np.log(P_new[np.arange(n_cal), y_cal]))
                    reg = lam * (np.sum((scale - 1.0) ** 2) + np.sum(bias**2))
                    return nll + reg

                x0 = np.ones(2 * n_classes)
                x0[n_classes:] = 0.0
                bounds = [(0.8, 1.2)] * n_classes + [(-1.0, 1.0)] * n_classes
                result = _scipy_minimize(_vs_nll, x0=x0, method="L-BFGS-B", bounds=bounds)
                if not result.success or not np.all(np.isfinite(result.x)):
                    reason = result.message if not result.success else "non-finite params"
                    print(f"[TabLDM:calibration] WARNING: vector scaling fit unusable ({reason}); skipping.")
                    return
                scale = result.x[:n_classes]
                bias = result.x[n_classes:]
                params = {"type": "vector", "scale": scale.copy(), "bias": bias.copy()}

        except Exception as exc:
            print(f"[TabLDM:calibration] WARNING: calibration fitting failed ({exc!r}); skipping.")
            return

        self.calibration_params_ = params

    def _apply_calibration(self, P: np.ndarray) -> np.ndarray:
        """Apply fitted calibration to aggregated probabilities."""
        params = getattr(self, "calibration_params_", None)
        if params is None:
            return P

        eps = 1e-15
        try:
            if params["type"] == "platt":
                A, B = float(params["A"]), float(params["B"])
                z = np.log((P[:, 1] + eps) / (P[:, 0] + eps))
                p1 = _expit(A * z + B)
                P_new = np.column_stack([1.0 - p1, p1])
            else:
                scale = params["scale"]
                bias = params["bias"]
                log_P = np.log(np.clip(P, eps, 1.0))
                logits = log_P * scale + bias
                logits_c = logits - logits.max(axis=1, keepdims=True)
                exp_l = np.exp(logits_c)
                P_new = exp_l / exp_l.sum(axis=1, keepdims=True)

            if not np.all(np.isfinite(P_new)) or np.any(P_new < 0):
                print("[TabLDM:calibration] WARNING: calibrated probabilities invalid; reverting.")
                return P

            row_sums = P_new.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            return P_new / row_sums

        except Exception as exc:
            print(f"[TabLDM:calibration] WARNING: applying calibration failed ({exc!r}); using uncalibrated.")
            return P

    # ==================================================================
    # Enhanced probability prediction
    # ==================================================================

    def _predict_proba_enhanced(self, X: np.ndarray) -> np.ndarray:
        """Enhanced probability prediction over all candidate groups."""
        self._current_fold_label_ = ""

        probs_all, predict_mask, names = self._collect_candidate_probs(
            X,
            self.ensemble_generator_,
            getattr(self, "quantile_ensemble_generator_", None),
            getattr(self, "svd_ens_generator_", None),
            adaptive_plus_gen=getattr(self, "adaptive_plus_generator_", None),
            adaptive_plus_pca=getattr(self, "adaptive_plus_pca_decorr_", None),
            adaptive_plus_ia=getattr(self, "adaptive_plus_interaction_", None),
            gaussian_rank_gen=getattr(self, "gaussian_rank_generator_", None),
            tag="predict",
        )

        # Apply holdout-derived valid candidate mask
        oof_mask = getattr(self, "nnls_valid_candidate_mask_", None)
        if oof_mask is not None:
            E_total = oof_mask.shape[0]
            if predict_mask.shape[0] != E_total:
                print(
                    f"[TabLDM:enhance] WARNING: predict_mask length {predict_mask.shape[0]} != oof_mask length {E_total}; ignoring OOF mask."
                )
                effective_mask = predict_mask
            else:
                effective_mask = predict_mask & oof_mask
                n_oof_extra_dropped = int(predict_mask.sum()) - int(effective_mask.sum())
                if n_oof_extra_dropped > 0:
                    print(
                        f"[TabLDM:enhance] OOF mask drops {n_oof_extra_dropped} additional candidate(s); {int(effective_mask.sum())} remain."
                    )
            predict_valid_indices = np.where(predict_mask)[0]
            effective_local = effective_mask[predict_valid_indices]
            probs = probs_all[effective_local]
            keep_indices = predict_valid_indices[effective_local]
        else:
            effective_mask = predict_mask
            probs = probs_all
            keep_indices = np.where(predict_mask)[0]

        n_estimators = probs.shape[0]
        if n_estimators == 0:
            raise RuntimeError("[TabLDM:enhance] ALL candidates are invalid; cannot produce predictions.")

        names_valid = [names[i] for i in keep_indices]
        group_counts = dict.fromkeys(self._GROUP_KEYS, 0)
        for nm in names_valid:
            group_counts[self._candidate_group(nm)] += 1
        print(
            f"[TabLDM:enhance:predict] effective_valid_candidates={n_estimators} "
            f"(total_generated={effective_mask.shape[0]}, "
            f"main={group_counts['main']}, quantile={group_counts['q']}, "
            f"svd_ens={group_counts['svd']}, adaptive_plus={group_counts['ap']}, "
            f"gaussian_rank={group_counts['gr']})"
        )

        # v2 §3.5 fusion: shrinkage of the holdout NNLS towards the group
        # prior. Never an unweighted candidate mean (v2 §4.1).
        proba = self._combine_probs(probs, names_valid)

        nnls_map = getattr(self, "nnls_weight_map_", None)
        weight_mode = f"group_prior+nnls(lambda={self.shrink_lambda_:.2f})" if nnls_map else "group_prior"
        print(
            f"[TabLDM:enhance] weight_mode={weight_mode} valid_candidates={n_estimators} final_proba_shape={proba.shape}"
        )

        # Probability calibration
        return self._apply_calibration(proba)

    # ==================================================================
    # predict_proba() / predict()
    # ==================================================================

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for test samples."""
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        has_training_data = (
            hasattr(self, "ensemble_generator_") and getattr(self.ensemble_generator_, "X_", None) is not None
        )
        if not has_kv_cache and not has_training_data:
            raise RuntimeError(
                "Cannot predict: this estimator was saved without training data and has no KV cache. "
                "Re-fit the estimator or load from a file saved with save_training_data=True or save_kv_cache=True."
            )

        if self.n_jobs is not None:
            assert self.n_jobs != 0
            old_n_threads = torch.get_num_threads()
            n_logical_cores = mp.cpu_count()
            if self.n_jobs > 0:
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"TabLDM got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used.",
                        stacklevel=2,
                    )
                n_threads = min(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, n_logical_cores + 1 + self.n_jobs)
            torch.set_num_threads(n_threads)

        X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)

        # Detect all-NaN columns
        if hasattr(X, "columns"):
            feature_mask = X.isna().all(axis=0).to_numpy()
        else:
            arr = np.asarray(X)
            if np.issubdtype(arr.dtype, np.number):
                feature_mask = np.isnan(arr).all(axis=0)
            else:
                feature_mask = np.array([all(_is_nan(v) for v in arr[:, i]) for i in range(arr.shape[1])])

        if feature_mask is not None and not np.any(feature_mask):
            feature_mask = None

        if feature_mask is not None:
            if hasattr(X, "columns"):
                X.iloc[:, feature_mask] = 0.0
            else:
                X[:, feature_mask] = 0.0

        X = self.X_encoder_.transform(X)

        # Enhanced path
        if getattr(self, "enhance_candidates", False):
            proba = self._predict_proba_enhanced(X)
            if self.n_jobs is not None:
                torch.set_num_threads(old_n_threads)
            return proba

        # Original path
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        use_cache = has_kv_cache and feature_mask is None

        if use_cache:
            test_data = self.ensemble_generator_.transform(X, mode="test")
            outputs = []
            for norm_method, (Xs_test,) in test_data.items():
                kv_cache = self.model_kv_cache_[norm_method]
                outputs.append(self._batch_forward_with_cache(Xs_test, kv_cache))
            outputs = np.concatenate(outputs, axis=0)
        else:
            data = self.ensemble_generator_.transform(X, mode="both", feature_mask=feature_mask)
            outputs = []
            for norm_method, (Xs, ys) in data.items():
                if feature_mask is None:
                    feature_shuffles = self.ensemble_generator_.feature_shuffles_[norm_method]
                else:
                    feature_shuffles = self.ensemble_generator_.masked_feature_shuffles_[norm_method]
                outputs.append(self._batch_forward(Xs, ys, feature_shuffles))
            outputs = np.concatenate(outputs, axis=0)

        class_shuffles = []
        for shuffles in self.ensemble_generator_.class_shuffles_.values():
            class_shuffles.extend(shuffles)

        n_estimators = len(class_shuffles)
        avg = np.zeros_like(outputs[0])
        for i, shuffle in enumerate(class_shuffles):
            out = outputs[i]
            avg += out[..., shuffle]
        avg /= n_estimators

        if self.average_logits:
            avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)

        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)

        return avg / avg.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples."""
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)
        return self.y_encoder_.inverse_transform(y)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


__all__ = [
    "TabLDMClassifier",
]
