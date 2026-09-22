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
estimator for TabLDM in-context tabular classification. When
``enhance_candidates=True``, it fits a conservative LimiX pipeline
ensemble and learns holdout NNLS weights (see
``_fit_pipeline_enhancement``).

All log messages use the ``[TabLDM:...]`` prefix for consistency with the
Xiaomi-TabLDM project conventions.
"""
from __future__ import annotations

import warnings
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch

from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from scipy.optimize import nnls as _scipy_nnls

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from .base import TabLDMBaseEstimator, _clear_cuda_cache
from .preprocessing import (
    TransformToNumerical,
    EnsembleGenerator,
    PipelineEnsemble,
    default_classifier_pipeline_specs,
    large_classifier_pipeline_specs,
)
from .sklearn_utils import _moe_load_mismatch, validate_data, _num_samples

from tabldm import InferenceConfig
from tabldm._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tabldm._model.embedding_dual_stream import ColEmbeddingDualStream
from tabldm._model.kv_cache import TabLDMCache


# ---------------------------------------------------------------------------
# Enhanced Classifier
# ---------------------------------------------------------------------------

class TabLDMClassifier(ClassifierMixin, TabLDMBaseEstimator):
    """TabLDM Classifier with inference enhancement.

    When ``enhance_candidates=True``, fits a conservative LimiX pipeline
    ensemble and learns holdout NNLS weights. When
    ``enhance_candidates=False``, runs the plain single-group inference
    path with no ensembling enhancements.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of estimators for the main ensemble group.

    norm_methods : str or list[str] or None, default=None
        Normalization methods for the main group.

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy.

    class_shuffle_method : str, default='shift'
        Class label permutation strategy for the plain ensemble group.

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

    categorical_indices : array-like of int or None, default=None
        Categorical feature indices.

    cat_random_encode : bool, default=False
        Randomly permute categorical codes per ensemble member.

    enhance_candidates : bool, default=True
        Master switch for inference enhancement. When True, fits a
        conservative LimiX pipeline ensemble and learns holdout NNLS
        weights. When False, runs the plain single-group inference path
        with no ensembling enhancements.

    validation : bool, default=True
        Enable NNLS weight learning via a single holdout split.

    pipeline_specs : tuple or None, default=None
        Explicit LimiX pipeline member specs. When None, the default (or
        large-dataset) classifier spec set is used and then filtered to the
        conservative safe subset.

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
        n_estimators: int = 16,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "random",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: Optional[int | str] = 4,
        kv_cache: bool | str = False,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "checkpoints/clf_default.ckpt",
        device: Optional[str | torch.device] = None,
        use_amp: bool | str = True,
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
        categorical_indices: Optional[List[int]] = None,
        cat_random_encode: bool = False,
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
        self.class_shuffle_method = class_shuffle_method
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
                    )
                print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
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
    # fit()
    # ==================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabLDMClassifier":
        """Fit the classifier to training data.

        When ``enhance_candidates=True``, fits a conservative LimiX pipeline
        ensemble and learns holdout NNLS weights. When False, fits the plain
        single-group ensemble generator.
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
            raise ValueError(
                "kv_cache is not supported together with enhance_candidates=True. "
                "Disable one of them."
            )

        # Initialize enhancement state
        self.nnls_weights_ = None

        if self.enhance_candidates:
            self._fit_pipeline_enhancement(X, y)
            self.n_features_in_ = X.shape[1]
        else:
            # Original single-generator path
            self.ensemble_generator_ = EnsembleGenerator(
                classification=True,
                n_estimators=self.n_estimators,
                norm_methods=self.norm_methods or ["none", "power"],
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
                cat_random_encode=self.cat_random_encode,
                categorical_indices=getattr(self, "_encoded_categorical_indices_", self.categorical_indices),
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

    @staticmethod
    def _safe_classifier_pipeline_spec(spec) -> bool:
        """Return whether a pipeline spec is safe for high-cardinality data.

        The conservative enhancement route keeps ordinal-only members and
        avoids dense one-hot, SVD, interaction, and original-column expansion.
        """
        return (
            isinstance(spec.categorical_encoding, str)
            and spec.categorical_encoding.startswith("ordinal")
            and not spec.discrete_flag
            and spec.svd_components is None
            and spec.max_interactions is None
            and not spec.original_flag
        )

    @classmethod
    def _select_safe_classifier_pipeline_specs(cls, specs):
        """Filter classifier pipeline specs to the conservative member set."""
        source_specs = tuple(specs)
        selected_specs = tuple(
            spec for spec in source_specs if cls._safe_classifier_pipeline_spec(spec)
        )
        if not selected_specs:
            raise ValueError(
                "No safe classifier pipeline members remain. The conservative "
                "route requires ordinal encoding without discrete, SVD, "
                "interaction, or original-feature expansion."
            )
        return source_specs, selected_specs

    def _fit_pipeline_enhancement(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit conservative LimiX member views and learn one holdout NNLS ensemble."""
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be in (0, 1)")
        default_specs = default_classifier_pipeline_specs()
        if self.pipeline_specs is not None:
            source_specs = tuple(self.pipeline_specs)
            self.pipeline_route_ = "user"
        elif X.shape[0] > 50_000:
            source_specs = tuple(large_classifier_pipeline_specs())
            self.pipeline_route_ = "large_dataset"
        else:
            source_specs = default_specs
            self.pipeline_route_ = "default"
        source_specs, selected_specs = self._select_safe_classifier_pipeline_specs(source_specs)
        self.pipeline_specs_ = selected_specs
        self.pipeline_reduced_for_large_dataset_ = self.pipeline_route_ == "large_dataset"
        self.pipeline_default_member_names_ = [spec.name for spec in default_specs]
        self.pipeline_source_member_names_ = [spec.name for spec in source_specs]
        self.pipeline_removed_member_names_ = [
            spec.name for spec in source_specs
            if not self._safe_classifier_pipeline_spec(spec)
        ]
        self.pipeline_member_names_ = [spec.name for spec in self.pipeline_specs_]
        self.pipeline_selected_member_names_ = list(self.pipeline_member_names_)
        self.pipeline_failed_members_ = []
        valid_indices = list(range(len(self.pipeline_specs_)))
        self.pipeline_oom_audit_ = []
        self.pipeline_validation_audit_ = [{
            "status": "safe_pipeline_selection",
            "route": self.pipeline_route_,
            "source_members": list(self.pipeline_source_member_names_),
            "selected_members": list(self.pipeline_member_names_),
            "removed_members": list(self.pipeline_removed_member_names_),
        }]
        if self.pipeline_reduced_for_large_dataset_:
            self.pipeline_validation_audit_.append({
                "status": "large_dataset_pipeline_reduction",
                "n_train": int(X.shape[0]),
                "removed_members": list(self.pipeline_removed_member_names_),
                "remaining_members": list(self.pipeline_member_names_),
            })
        encoded_categories = list(getattr(self.X_encoder_, "categorical_indices_", []))
        categorical_indices = encoded_categories or (self.categorical_indices or [])
        weights = None

        # Small datasets use the stable equal-weight ensemble.  NNLS is only
        # useful once the validation set is large enough to identify weights.
        use_nnls = bool(self.validation and X.shape[0] >= self.nnls_min_samples)
        if self.validation and not use_nnls:
            self.pipeline_validation_audit_.append({
                "status": "equal_weight_small_dataset",
                "n_train": int(X.shape[0]),
                "nnls_min_samples": int(self.nnls_min_samples),
            })

        if use_nnls:
            counts = np.bincount(y.astype(int), minlength=self.n_classes_)
            if counts.size == 0 or (counts[counts > 0].size and counts[counts > 0].min() < 2):
                self.pipeline_validation_audit_.append({"status": "skipped", "reason": "smallest class has fewer than two samples"})
            else:
                try:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X, y, test_size=self.validation_size, stratify=y,
                        shuffle=True, random_state=self.random_state,
                    )
                    holdout = PipelineEnsemble(
                        classification=True, specs=self.pipeline_specs_, categorical_indices=categorical_indices,
                        random_state=self.random_state,
                    ).fit(X_tr, y_tr)
                    predictions, successful = [], []
                    for local_index, member in zip(holdout.member_indices_, holdout.members_):
                        try:
                            probabilities = self._pipeline_member_probabilities(member, X_val)
                            if probabilities.shape != (X_val.shape[0], self.n_classes_):
                                raise ValueError(f"unexpected probability shape {probabilities.shape}")
                            if not np.isfinite(probabilities).all() or (probabilities < 0).any():
                                raise ValueError("probabilities are non-finite or negative")
                        except Exception as exc:
                            self.pipeline_failed_members_.append({"index": local_index, "name": member.spec.name, "stage": "validation", "reason": repr(exc)})
                            continue
                        predictions.append(probabilities)
                        successful.append(local_index)
                    self.pipeline_failed_members_.extend(holdout.failed_members_)
                    if successful:
                        valid_indices = successful
                        prediction_array = np.stack(predictions)
                        A = prediction_array.reshape(len(successful), -1).T
                        onehot = np.eye(self.n_classes_, dtype=np.float64)[y_val.astype(int)].reshape(-1)
                        try:
                            raw_weights, _ = _scipy_nnls(A, onehot)
                            if np.isfinite(raw_weights).all() and raw_weights.sum() > 0:
                                weights = raw_weights / raw_weights.sum()
                            else:
                                raise ValueError("degenerate NNLS result")
                        except Exception as exc:
                            weights = np.full(len(successful), 1 / len(successful), dtype=np.float64)
                            self.pipeline_validation_audit_.append({"status": "equal_weight_fallback", "reason": repr(exc)})
                        self.pipeline_validation_audit_.append({"status": "nnls", "n_validation": len(y_val), "valid_member_indices": successful})
                    else:
                        # A failed holdout should not turn full fitting into an
                        # empty member selection. Refit the safe candidates and
                        # use equal weights if validation produced no usable row.
                        valid_indices = list(range(len(self.pipeline_specs_)))
                        self.pipeline_validation_audit_.append({
                            "status": "validation_fallback",
                            "reason": "all holdout members failed; refitting selected members",
                            "selected_member_indices": list(valid_indices),
                        })
                except Exception as exc:
                    self.pipeline_validation_audit_.append({"status": "split_failed", "reason": repr(exc)})

        full = PipelineEnsemble(
            classification=True, specs=self.pipeline_specs_, categorical_indices=categorical_indices,
            random_state=self.random_state,
        ).fit(X, y, member_indices=valid_indices)
        self.pipeline_failed_members_.extend(full.failed_members_)
        self.pipeline_members_ = full.members_
        self.nnls_valid_member_indices_ = full.member_indices_
        if not self.pipeline_members_:
            raise RuntimeError("All LimiX pipeline members failed during full-data fitting.")
        if weights is None or len(weights) != len(self.pipeline_members_):
            weights = np.full(len(self.pipeline_members_), 1 / len(self.pipeline_members_), dtype=np.float64)
        self.nnls_weights_ = weights
        self.ensemble_generator_ = None

    def _pipeline_member_probabilities(self, member, X: np.ndarray) -> np.ndarray:
        """Run one pipeline member with bounded query chunks and CUDA retry."""
        chunk = getattr(self, "pipeline_chunk_rows", None)
        if chunk is None:
            chunk = max(1, int(getattr(self, "n_samples_in_", X.shape[0])))
        outputs = []
        for start in range(0, X.shape[0], int(chunk)):
            X_query = X[start:start + int(chunk)]
            X_view = member.transform(X_query)
            X_both = np.concatenate([member.X_train_, X_view], axis=0)[None, ...]
            y_train = np.asarray(member.y_train_, dtype=np.float32)[None, ...]
            try:
                raw = self._batch_forward(X_both, y_train, feature_shuffles=None)[0]
            except torch.cuda.OutOfMemoryError as exc:
                _clear_cuda_cache(self.device_)
                self.pipeline_oom_audit_.append({"tier": 4, "action": "forward_oom_retry", "member": member.spec.name, "error": repr(exc)})
                old_batch = self.batch_size_
                self.batch_size_ = 1
                try:
                    raw = self._batch_forward(X_both, y_train, feature_shuffles=None)[0]
                finally:
                    self.batch_size_ = old_batch
            if self.average_logits:
                raw = self.softmax(raw, axis=-1, temperature=self.softmax_temperature)
            outputs.append(member.inverse_class_probabilities(raw))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, self.n_classes_))

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
            for X_batch, y_batch in zip(Xs_split, ys_split):
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
    # Forward helpers
    # ==================================================================

    def _batch_forward(
        self, Xs: np.ndarray, ys: np.ndarray, feature_shuffles: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Process model forward passes in batches."""
        batch_size = self.batch_size_ or Xs.shape[0]
        n_batches = int(np.ceil(Xs.shape[0] / batch_size))
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)
        if feature_shuffles is None:
            feature_shuffles = [None] * n_batches
        else:
            feature_shuffles = np.array_split(feature_shuffles, n_batches)

        outputs = []
        for X_batch, y_batch, shuffle_batch in zip(Xs, ys, feature_shuffles):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)
            if shuffle_batch is not None:
                shuffle_batch = shuffle_batch.tolist()
            with torch.no_grad():
                out = self.model_(
                    X=X_batch,
                    y_train=y_batch,
                    feature_shuffles=shuffle_batch,
                    return_logits=True if self.average_logits else False,
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
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            outputs.append(out.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _predict_proba_pipeline(self, X: np.ndarray) -> np.ndarray:
        """Predict with the retained LimiX pipeline members and NNLS weights."""
        probabilities, weights = [], []
        for member, weight in zip(self.pipeline_members_, self.nnls_weights_):
            try:
                probability = self._pipeline_member_probabilities(member, X)
                if probability.shape != (X.shape[0], self.n_classes_):
                    raise ValueError(f"unexpected probability shape {probability.shape}")
                if not np.isfinite(probability).all() or (probability < 0).any():
                    raise ValueError("probabilities are non-finite or negative")
            except Exception as exc:
                warnings.warn(f"Skipping pipeline member {member.spec.name}: {exc}", UserWarning, stacklevel=2)
                continue
            probabilities.append(probability)
            weights.append(weight)
        if not probabilities:
            raise RuntimeError("All retained LimiX pipeline members failed during prediction.")
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        proba = np.einsum("e,enc->nc", weights, np.stack(probabilities))
        row_sums = proba.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0) or not np.isfinite(proba).all():
            raise RuntimeError("LimiX pipeline ensemble produced invalid probabilities.")
        return proba / row_sums

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for test samples."""
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        has_training_data = (
            hasattr(self, "ensemble_generator_") and getattr(self.ensemble_generator_, "X_", None) is not None
        )
        has_pipeline_training_data = bool(getattr(self, "pipeline_members_", []))
        if not has_kv_cache and not has_training_data and not has_pipeline_training_data:
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
                        f" Only {n_logical_cores} threads will be used."
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
                feature_mask = np.array([all(v != v for v in arr[:, i]) for i in range(arr.shape[1])])

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
            proba = self._predict_proba_pipeline(X)
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
