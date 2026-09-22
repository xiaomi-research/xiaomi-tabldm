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
estimator for TabLDM in-context tabular regression. When
``enhance_candidates=True``, it fits a LimiX pipeline ensemble and learns
holdout NNLS weights (see ``_fit_pipeline_enhancement``).

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

import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.optimize import nnls as _scipy_nnls
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .base import TabLDMBaseEstimator, _clear_cuda_cache
from .preprocessing import (
    EnsembleGenerator,
    TransformToNumerical,
    PipelineEnsemble,
    default_regressor_pipeline_specs,
    large_regressor_pipeline_specs,
)
from .sklearn_utils import _moe_load_mismatch, _num_samples, validate_data

from tabldm import InferenceConfig
from tabldm._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tabldm._model.embedding_dual_stream import ColEmbeddingDualStream
from tabldm._model.kv_cache import TabLDMCache


# ---------------------------------------------------------------------------
# Enhanced Regressor
# ---------------------------------------------------------------------------

class TabLDMRegressor(RegressorMixin, TabLDMBaseEstimator):
    """TabLDM Regressor with inference enhancement.

    When ``enhance_candidates=True``, fits a LimiX pipeline ensemble and
    learns holdout NNLS weights. When ``enhance_candidates=False``, runs
    the plain single-group inference path.

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

    enhance_candidates : bool, default=True
        Master switch for inference enhancement. When True, fits a LimiX
        pipeline ensemble and learns holdout NNLS weights. When False, runs
        the plain single-group inference path with no ensembling
        enhancements.

    validation : bool, default=True
        Enable NNLS weight learning via a single holdout split.

    pipeline_specs : tuple or None, default=None
        Explicit LimiX pipeline member specs. When None, the default (or
        large-dataset) regressor spec set is used.

    validation_size : float, default=0.2
        Holdout fraction used for NNLS weight learning.

    nnls_min_samples : int, default=2000
        Minimum training rows required before NNLS is attempted; smaller
        datasets fall back to equal member weights.

    pipeline_chunk_rows : int or None, default=None
        Query rows per forward chunk for each pipeline member. None keeps
        the whole query set in one chunk.
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
        validation: bool = True,
        pipeline_specs=None,
        validation_size: float = 0.2,
        nnls_min_samples: int = 2000,
        pipeline_chunk_rows: Optional[int] = None,
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
        self.validation = validation
        self.pipeline_specs = pipeline_specs
        self.validation_size = validation_size
        self.nnls_min_samples = nnls_min_samples
        self.pipeline_chunk_rows = pipeline_chunk_rows

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
    # Forward helpers
    # ==================================================================

    def _batch_forward(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        output_type: str | list[str] = "mean",
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes in batches to manage memory efficiently."""
        batch_size = self.batch_size_ or Xs.shape[0]
        n_batches = int(np.ceil(Xs.shape[0] / batch_size))
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

    def _fit_pipeline_enhancement(self, X: np.ndarray, y: np.ndarray, y_scaled: np.ndarray) -> None:
        """Fit LimiX V2 regression views and one holdout NNLS ensemble."""
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be in (0, 1)")
        encoded_categories = list(getattr(self.X_encoder_, "categorical_indices_", []))
        categorical_indices = encoded_categories
        default_specs = default_regressor_pipeline_specs()
        if self.pipeline_specs is not None:
            self.pipeline_specs_ = tuple(self.pipeline_specs)
            self.pipeline_route_ = "user"
        elif X.shape[0] > 50_000:
            self.pipeline_specs_ = large_regressor_pipeline_specs()
            self.pipeline_route_ = "large_dataset"
        else:
            self.pipeline_specs_ = default_specs
            self.pipeline_route_ = "default"
        self.pipeline_reduced_for_large_dataset_ = self.pipeline_route_ == "large_dataset"
        self.pipeline_member_names_ = [spec.name for spec in self.pipeline_specs_]
        self.pipeline_selected_member_names_ = list(self.pipeline_member_names_)
        selected_names = set(self.pipeline_member_names_)
        self.pipeline_removed_member_names_ = [
            spec.name for spec in default_specs if spec.name not in selected_names
        ]
        self.pipeline_oom_audit_ = []
        self.pipeline_validation_audit_ = []
        if self.pipeline_reduced_for_large_dataset_:
            self.pipeline_validation_audit_.append({
                "status": "large_dataset_pipeline_reduction",
                "n_train": int(X.shape[0]),
                "removed_members": list(self.pipeline_removed_member_names_),
                "remaining_members": list(self.pipeline_member_names_),
            })
        self.pipeline_failed_members_ = []
        valid_indices = list(range(len(self.pipeline_specs_)))
        weights = None

        # Avoid fitting a validation NNLS problem for small datasets; the
        # ensemble average is more stable and avoids an unnecessary peak.
        use_nnls = bool(self.validation and X.shape[0] >= self.nnls_min_samples)
        if self.validation and not use_nnls:
            self.pipeline_validation_audit_.append({
                "status": "equal_weight_small_dataset",
                "n_train": int(X.shape[0]),
                "nnls_min_samples": int(self.nnls_min_samples),
            })

        if use_nnls:
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

    def _pipeline_member_predictions(self, member, X: np.ndarray) -> np.ndarray:
        """Run one pipeline member in query chunks with a CUDA OOM retry."""
        chunk = getattr(self, "pipeline_chunk_rows", None)
        if chunk is None:
            chunk = max(1, int(getattr(self, "n_samples_in_", X.shape[0])))
        outputs = []
        for start in range(0, X.shape[0], int(chunk)):
            X_query = X[start:start + int(chunk)]
            X_both = np.concatenate([member.X_train_, member.transform(X_query)], axis=0)[None, ...]
            y_train = np.asarray(member.y_train_, dtype=np.float32)[None, ...]
            try:
                output = self._batch_forward(X_both, y_train, output_type="mean")
            except torch.cuda.OutOfMemoryError as exc:
                _clear_cuda_cache(self.device_)
                self.pipeline_oom_audit_.append({"tier": 4, "action": "forward_oom_retry", "member": member.spec.name, "error": repr(exc)})
                old_batch = self.batch_size_
                self.batch_size_ = 1
                try:
                    output = self._batch_forward(X_both, y_train, output_type="mean")
                finally:
                    self.batch_size_ = old_batch
            outputs.append(np.asarray(output)[0])
        return np.concatenate(outputs, axis=0) if outputs else np.empty(0, dtype=np.float32)


    def fit(self, X: np.ndarray, y: np.ndarray, kv_cache: bool | str = False) -> "TabLDMRegressor":
        """Fit the regressor to training data.

        When ``enhance_candidates=True``, fits a LimiX pipeline ensemble and
        learns holdout NNLS weights. When ``enhance_candidates=False``, runs
        the plain single-group inference path.
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

        When ``enhance_candidates=True`` (and ``fit()`` was called with that
        flag), applies the LimiX pipeline ensemble with NNLS weighting.
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
