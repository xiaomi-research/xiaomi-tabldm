#!/usr/bin/env python3
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
"""TabLDM Enhanced Classifier inference + metrics on TALENT-cls datasets.

Mirrors the upstream TALENT evaluation harness's seed handling and result-saving
format, but uses ``TabLDMEnhancedClassifier`` with inference enhancement enabled.

Enhancement features:
  - Multi-group candidate ensembling (main + quantile + SVD + adaptive_plus + gaussian_rank)
  - NNLS ensemble weight learning via validation/OOF
  - Probability calibration (Platt/vector scaling)
  - Adaptive routing based on dataset characteristics

Uses ONLY the local ``tabldm`` package + already-installed deps
(torch / scikit-learn / numpy / pandas / scipy). No TALENT framework, no pip installs.

Seed handling (aligned with the upstream eval scripts):
  for seed in range(seed_num): set_seeds(seed); TabLDMEnhancedClassifier(random_state=seed);
  fit(train) -> predict_proba(test); save predictions_seed{seed}.npz; aggregate mean+-std.

Usage:
    # Single dataset
    python infer_talent_cls_enhanced.py --dataset Pima_Indians_Diabetes_Database

    # All datasets
    python infer_talent_cls_enhanced.py --all --seed-num 3 --limit 5

    # Disable specific enhancement groups
    python infer_talent_cls_enhanced.py --all --no-svd-ens --no-adaptive-plus --no-gaussian-rank

    # Disable NNLS (use equal-weight ensemble)
    python infer_talent_cls_enhanced.py --all --no-validation

    # Disable calibration
    python infer_talent_cls_enhanced.py --all --no-calibration

    # Disable all enhancements (equivalent to baseline)
    python infer_talent_cls_enhanced.py --all --no-enhance
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    log_loss, precision_score, roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder, label_binarize

# Import the LOCAL tabldm package
TABLDM_DIR = Path(os.environ.get("TABLDM_DIR", Path(__file__).resolve().parent.parent))
if str(TABLDM_DIR) not in sys.path:
    sys.path.insert(0, str(TABLDM_DIR))
import tabldm
from tabldm import TabLDMEnhancedClassifier

DEFAULT_CKPT = os.environ.get("TABLDM_CKPT", "checkpoints/clf_moe1.ckpt")
DEFAULT_DATA = os.environ.get("TABLDM_DATA_ROOT", "data/tabarena_cls")
DEFAULT_MODEL_TYPE = "tabldm_moe1_cls_enhanced"
METRIC_NAMES = ["Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC"]
_UNKNOWN_CATEGORY = np.iinfo(np.int64).max - 3


def set_seeds(seed: int) -> None:
    """Mirror TALENT.model.utils.set_seeds."""
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.default_generators[device_index].manual_seed(seed + 3 + device_index)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load(ds_dir: Path, split: str):
    n, c = ds_dir / f"N_{split}.npy", ds_dir / f"C_{split}.npy"
    N = np.load(n, allow_pickle=True) if n.exists() else None
    C = np.load(c, allow_pickle=True) if c.exists() else None
    y = np.load(ds_dir / f"y_{split}.npy", allow_pickle=True)
    return N, C, y


def _as_2d(array):
    array = np.asarray(array)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def _prepare_numeric(N_train, N_test):
    if N_train is None:
        return None, None
    N_train = _as_2d(N_train).astype(np.float64)
    N_test = None if N_test is None else _as_2d(N_test).astype(np.float64)
    fill_values = np.nanmean(N_train, axis=0)
    fill_values = np.nan_to_num(fill_values, nan=0.0)
    train_nan = np.isnan(N_train)
    if train_nan.any():
        train_nan_indices = np.where(train_nan)
        N_train[train_nan_indices] = np.take(fill_values, train_nan_indices[1])
    if N_test is not None:
        test_nan = np.isnan(N_test)
        if test_nan.any():
            test_nan_indices = np.where(test_nan)
            N_test[test_nan_indices] = np.take(fill_values, test_nan_indices[1])
    return N_train, N_test


def _prepare_categorical(C_train, C_test):
    if C_train is None:
        return None, None

    def normalize_categories(values):
        raw_values = _as_2d(values)
        if np.issubdtype(raw_values.dtype, np.number):
            missing = np.isnan(raw_values)
        else:
            missing = np.isin(raw_values, np.array(["nan", "NaN", "", None], dtype=object))
        values = raw_values.astype(object)
        values[missing] = "___null___"
        return values.astype(str)

    C_train = normalize_categories(C_train)
    C_test = None if C_test is None else normalize_categories(C_test)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=_UNKNOWN_CATEGORY,
        dtype=np.int64,
    ).fit(C_train)
    C_train_encoded = encoder.transform(C_train)
    if C_test is None:
        return C_train_encoded, None
    C_test_encoded = encoder.transform(C_test)
    replacement_values = [
        np.argmax(np.bincount(column[column != _UNKNOWN_CATEGORY]))
        if np.any(column == _UNKNOWN_CATEGORY)
        else column[0]
        for column in C_train_encoded.T
    ]
    for column_index, replacement_value in enumerate(replacement_values):
        unknown = C_test_encoded[:, column_index] == _UNKNOWN_CATEGORY
        C_test_encoded[unknown, column_index] = replacement_value
    return C_train_encoded, C_test_encoded


def prepare_talent_arrays(N_train, C_train, N_test, C_test):
    """Apply TALENT's external preprocessing and return numeric train/test arrays."""
    N_train, N_test = _prepare_numeric(N_train, N_test)
    C_train, C_test = _prepare_categorical(C_train, C_test)
    def concatenate(N, C):
        if N is not None and C is not None:
            return np.concatenate((N, C), axis=1)
        return N if N is not None else C
    return concatenate(N_train, C_train), concatenate(N_test, C_test)


def _check_softmax(predictions):
    if np.any((predictions < 0) | (predictions > 1)) or not np.allclose(
        predictions.sum(axis=-1), 1, atol=1e-5
    ):
        exponentials = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return predictions


def compute_metrics(y_true, y_pred, proba, n_classes):
    proba = _check_softmax(np.asarray(proba, dtype=np.float64))
    classes = np.arange(n_classes)
    m = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Avg_Recall": float(balanced_accuracy_score(y_true, y_pred)),
        "Avg_Precision": float(precision_score(y_true, y_pred, average="macro")),
        "F1": float(
            f1_score(y_true, y_pred, average="binary" if n_classes == 2 else "macro")
        ),
    }
    try:
        m["LogLoss"] = float(log_loss(y_true, proba, labels=classes))
    except Exception:
        m["LogLoss"] = float("nan")
    try:
        if n_classes == 2 and proba.shape[1] >= 2:
            m["AUC"] = float(roc_auc_score(y_true, proba[:, 1], labels=classes))
        else:
            present_classes = np.unique(y_true)
            if len(present_classes) < 2:
                m["AUC"] = float("nan")
            else:
                binarized = label_binarize(y_true, classes=classes)
                class_indices = [
                    index for index, category in enumerate(classes)
                    if category in present_classes
                ]
                m["AUC"] = float(
                    roc_auc_score(
                        binarized[:, class_indices],
                        proba[:, class_indices],
                        multi_class="ovr",
                        average="macro",
                    )
                )
    except Exception:
        m["AUC"] = float("nan")
    return m


def save_predictions(npz_path, proba, y_true):
    """Mirror the upstream eval harness predict() save block."""
    proba = proba.astype(np.float32)
    np.savez(
        npz_path,
        probas=proba,
        confidence=np.max(proba, axis=1).astype(np.float32),
        pred_label=np.argmax(proba, axis=1).astype(np.int64),
        true_label=y_true.astype(np.int64),
    )


def show_results(metric_names, loss_list, results_list, csv_prefix):
    """Mirror the upstream eval metrics.show_results."""
    arrays = {n: [] for n in metric_names}
    for res in results_list:
        for i, n in enumerate(metric_names):
            arrays[n].append(res[i])
    mean = {n: float(np.mean(arrays[n])) for n in metric_names}
    std = {n: float(np.std(arrays[n])) for n in metric_names}
    mean_loss = float(np.mean(loss_list))

    df = pd.DataFrame({"seed": range(len(loss_list)), **arrays})
    df.loc["mean"] = [np.nan] + [mean[n] for n in metric_names]
    df.loc["std"] = [np.nan] + [std[n] for n in metric_names]
    df.to_csv(f"{csv_prefix}_details.csv", index=False)

    for n in metric_names:
        print(f"{n} Results: {', '.join(f'{v:.8f}' for v in arrays[n])}")
        print(f"{n} MEAN = {mean[n]:.8f} +/- {std[n]:.8f}")
    print(f"Mean Loss: {mean_loss:.8e}")
    return mean, std, mean_loss


def summarize_all(all_summaries, save_path):
    """Mirror the upstream eval metrics.summarize_all."""
    if not all_summaries:
        return
    df = pd.DataFrame(all_summaries)
    df.to_csv(os.path.join(save_path, "all_datasets_summary.csv"), index=False)
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    overall = pd.DataFrame([df[mean_cols].mean()])
    overall.to_csv(os.path.join(save_path, "overall_averages.csv"), index=False)
    print("\n" + "=" * 64)
    print("Overall average across all datasets:")
    for c in mean_cols:
        print(f"  {c}: {df[c].mean():.6f} +/- {df[c].std():.6f}")
    print("=" * 64)
    print(f"Saved summary -> {save_path}/all_datasets_summary.csv")
    print(f"Saved overall -> {save_path}/overall_averages.csv")


def run_dataset(
    ds_dir, ckpt, n_estimators, batch_size, device, verbose, seed_num,
    save_path, model_type, offload_mode, disk_offload_dir, cat_random_encode=False,
    # enhancement parameters
    enhance_candidates=True,
    n_quantile_estimators=16,
    use_cross_feature=True,
    validation=True,
    k_fold=True,
    n_splits=5,
    use_svd_ens=True,
    n_svd_ens_estimators=16,
    svd_ens_n_components=10,
    use_adaptive_plus_candidate=True,
    adaptive_plus_enable_augmentations=True,
    enable_calibration=True,
    calibration_lambda=1e-2,
    use_gaussian_rank_ens=True,
    n_gaussian_rank_estimators=16,
):
    info = json.load(open(ds_dir / "info.json"))
    Ntr, Ctr, ytr = _load(ds_dir, "train")
    Nte, Cte, yte = _load(ds_dir, "test")
    le = LabelEncoder()
    ytr_enc = le.fit_transform(np.asarray(ytr).ravel())
    yte_enc = le.transform(np.asarray(yte).ravel())
    n_classes = len(le.classes_)
    Xtr, Xte = prepare_talent_arrays(Ntr, Ctr, Nte, Cte)
    numeric_width = 0 if Ntr is None else _as_2d(Ntr).shape[1]
    categorical_width = 0 if Ctr is None else _as_2d(Ctr).shape[1]
    categorical_indices = list(range(numeric_width, numeric_width + categorical_width))
    cpu = device == "cpu"

    save_path1 = f"{ds_dir.name}-{model_type}"
    save_path2 = "Epoch0BZ{batch_size}-Norm-none-Nan-mean-new-Cat-indices".format(batch_size=batch_size)
    ds_save = os.path.join(save_path, save_path1, save_path2)
    mkdir(ds_save)

    loss_list, results_list = [], []
    for seed in range(seed_num):
        set_seeds(seed)
        clf = TabLDMEnhancedClassifier(
            n_estimators=n_estimators,
            norm_methods=["none", "power"],
            model_path=ckpt, allow_auto_download=False,
            device=device,
            use_amp=False if cpu else True,
            use_fa3=False if cpu else "auto",
            offload_mode="cpu" if cpu else offload_mode,
            disk_offload_dir=None if cpu else disk_offload_dir,
            n_jobs=None, verbose=verbose, random_state=seed,
            cat_random_encode=cat_random_encode,
            categorical_indices=categorical_indices,
            # enhancement
            enhance_candidates=enhance_candidates,
            n_quantile_estimators=n_quantile_estimators,
            use_cross_feature=use_cross_feature,
            validation=validation,
            k_fold=k_fold,
            n_splits=n_splits,
            use_svd_ens=use_svd_ens,
            n_svd_ens_estimators=n_svd_ens_estimators,
            svd_ens_n_components=svd_ens_n_components,
            use_adaptive_plus_candidate=use_adaptive_plus_candidate,
            adaptive_plus_enable_augmentations=adaptive_plus_enable_augmentations,
            enable_calibration=enable_calibration,
            calibration_lambda=calibration_lambda,
            use_gaussian_rank_ens=use_gaussian_rank_ens,
            n_gaussian_rank_estimators=n_gaussian_rank_estimators,
        )
        t0 = time.time(); clf.fit(Xtr, ytr_enc); t_fit = time.time() - t0
        t0 = time.time(); proba = clf.predict_proba(Xte); t_pred = time.time() - t0
        pred = np.argmax(proba, axis=1)
        m = compute_metrics(yte_enc, pred, proba, n_classes)
        save_predictions(os.path.join(ds_save, f"predictions_seed{seed}.npz"), proba, yte_enc)
        loss_list.append(m["LogLoss"])
        results_list.append([m[n] for n in METRIC_NAMES])
        print(f"  [seed={seed}] fit={t_fit:.1f}s pred={t_pred:.1f}s | "
              + " ".join(f"{n}={m[n]:.4f}" for n in METRIC_NAMES))

    csv_prefix = os.path.join(ds_save, f"results_{ds_dir.name}")
    mean, std, mean_loss = show_results(METRIC_NAMES, loss_list, results_list, csv_prefix)
    return {
        "dataset": ds_dir.name, "task_type": info["task_type"], "n_classes": n_classes,
        "mean_loss": mean_loss,
        **{f"{n}_mean": mean[n] for n in METRIC_NAMES},
        **{f"{n}_std": std[n] for n in METRIC_NAMES},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offload-mode", default="auto", choices=["auto", "gpu", "cpu", "disk"])
    ap.add_argument("--disk-offload-dir", default=None)
    ap.add_argument("--dataset", default="Pima_Indians_Diabetes_Database")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed-num", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cat_random", "--cat_randomEncode", dest="cat_random_encode", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict", action="store_true")
    # Enhancement control
    ap.add_argument("--no-enhance", action="store_true", help="Disable all enhancements (same as baseline)")
    ap.add_argument("--no-validation", action="store_true", help="Disable NNLS weight learning")
    ap.add_argument("--no-kfold", action="store_true", help="Disable K-fold OOF (use single holdout)")
    ap.add_argument("--no-calibration", action="store_true", help="Disable probability calibration")
    ap.add_argument("--no-svd-ens", action="store_true", help="Disable SVD ensemble group")
    ap.add_argument("--no-adaptive-plus", action="store_true", help="Disable adaptive_plus group")
    ap.add_argument("--no-gaussian-rank", action="store_true", help="Disable Gaussian rank group")
    ap.add_argument("--no-cross-feature", action="store_true", help="Disable SVD+cross feature augmentation")
    ap.add_argument("--n-quantile-estimators", type=int, default=16)
    ap.add_argument("--n-svd-ens-estimators", type=int, default=16)
    ap.add_argument("--n-gaussian-rank-estimators", type=int, default=16)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--calibration-lambda", type=float, default=1e-2)
    args = ap.parse_args()

    if args.seed_num <= 0:
        ap.error("--seed-num must be positive")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_stem = Path(args.ckpt).stem
    save_path = args.save_path or str(Path(__file__).resolve().parent / "results" / ckpt_stem)
    mkdir(save_path)

    data_root = Path(args.data_root)
    if args.all:
        names = sorted(d.name for d in data_root.iterdir() if d.is_dir())
        if args.limit:
            names = names[:args.limit]
    else:
        names = [args.dataset]

    # Enhancement parameters
    enhance = not args.no_enhance
    print(f"tabldm {tabldm.__version__} | ckpt={ckpt_stem} | model_type={args.model_type} "
          f"| n_estimators={args.n_estimators} | batch_size={args.batch_size} "
          f"| seed_num={args.seed_num} | device={device} | enhance={enhance}")
    print(f"data_root={data_root} | datasets={len(names)} | save_path={save_path}\n")

    run_summaries = []
    failures = []
    for i, name in enumerate(names, 1):
        ds_dir = data_root / name
        print(f"[{i}/{len(names)}] {name}")
        summary = None
        if args.resume:
            # Check for existing results (simplified)
            pass
        if summary is None:
            try:
                summary = run_dataset(
                    ds_dir, args.ckpt, args.n_estimators, args.batch_size,
                    device, args.verbose, args.seed_num, save_path,
                    args.model_type, args.offload_mode, args.disk_offload_dir,
                    args.cat_random_encode,
                    enhance_candidates=enhance,
                    n_quantile_estimators=args.n_quantile_estimators,
                    use_cross_feature=not args.no_cross_feature,
                    validation=not args.no_validation,
                    k_fold=not args.no_kfold,
                    n_splits=args.n_splits,
                    use_svd_ens=not args.no_svd_ens,
                    n_svd_ens_estimators=args.n_svd_ens_estimators,
                    use_adaptive_plus_candidate=not args.no_adaptive_plus,
                    enable_calibration=not args.no_calibration,
                    calibration_lambda=args.calibration_lambda,
                    use_gaussian_rank_ens=not args.no_gaussian_rank,
                    n_gaussian_rank_estimators=args.n_gaussian_rank_estimators,
                )
            except Exception as e:
                failures.append({"dataset": name, "error": repr(e)})
                print(f"  ERROR: {e!r}")
        if summary is not None:
            run_summaries.append(summary)
        print()

    if run_summaries:
        summarize_all(run_summaries, save_path)
    if failures:
        print(f"Failed datasets: {', '.join(f['dataset'] for f in failures)}")
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
