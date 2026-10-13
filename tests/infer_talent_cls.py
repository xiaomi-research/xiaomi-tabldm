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
"""TabLDM MoE1 inference + metrics on TALENT-cls datasets (mirrors
the upstream TALENT evaluation harness's seed handling and result-saving format).

Uses ONLY the local ``tabldm`` package + already-installed deps
(torch / scikit-learn / numpy / pandas / scipy). No TALENT framework, no pip installs.

Seed handling (aligned with the upstream eval scripts):
  for seed in range(seed_num): set_seeds(seed); TabLDMClassifier(random_state=seed);
  fit(train) -> predict_proba(test); save predictions_seed{seed}.npz; aggregate mean+-std.
  (the referenced original evaluation uses seed_num=5.)

Save format (aligned with the upstream eval scripts / metrics.show_results / summarize_all):
  {save_root}/{dataset}-{model_type}/Epoch0BZ{bs}-Norm-none-Nan-mean-new-Cat-indices/
      predictions_seed{seed}.npz   # probas, confidence, pred_label, true_label
      results_{dataset}_details.csv  # per-seed metrics + mean/std rows
  {save_root}/all_datasets_summary.csv  # per-dataset mean/std (multi-dataset)
  {save_root}/overall_averages.csv      # overall means (multi-dataset)
  {save_root}/failed_datasets.csv       # datasets that failed in the current run

The feature pipeline mirrors TALENT's external preprocessing:
numeric columns are mean-imputed, categorical columns are converted to strings,
ordinal-encoded, and unknown test categories use TALENT's fallback category.

Usage:
    python infer_talent_cls.py --dataset Pima_Indians_Diabetes_Database
    python infer_talent_cls.py --dataset Pima_Indians_Diabetes_Database --seed-num 15
    python infer_talent_cls.py --all --seed-num 3 --limit 5
    python infer_talent_cls.py --all --seed-num 3 --cat_random
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

# Import the LOCAL tabldm package (this script lives in the TabLDM project root).
TABLDM_DIR = Path(os.environ.get("TABLDM_DIR", Path(__file__).resolve().parent.parent))
if str(TABLDM_DIR) not in sys.path:
    sys.path.insert(0, str(TABLDM_DIR))
import tabldm
from tabldm import TabLDMClassifier

DEFAULT_CKPT = os.environ.get("TABLDM_CKPT", "checkpoints/clf_moe1.ckpt")
DEFAULT_DATA = os.environ.get("TABLDM_DATA_ROOT", "data/tabarena_cls")
DEFAULT_MODEL_TYPE = "tabldm_moe1_cls"
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
    """Mirror the upstream eval metrics.show_results: per-seed CSV + mean/std rows + print."""
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


def run_dataset(
    ds_dir, ckpt, n_estimators, batch_size, device, verbose, seed_num,
    save_path, model_type, offload_mode, disk_offload_dir, cat_random_encode=False,
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
    categorical_indices = list(
        range(numeric_width, numeric_width + categorical_width)
    )
    cpu = device == "cpu"

    save_path1 = f"{ds_dir.name}-{model_type}"
    save_path2 = f"Epoch0BZ{batch_size}-Norm-none-Nan-mean-new-Cat-indices"
    ds_save = os.path.join(save_path, save_path1, save_path2)
    mkdir(ds_save)

    loss_list, results_list = [], []
    for seed in range(seed_num):
        set_seeds(seed)
        clf = TabLDMClassifier(
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


def summarize_all(all_summaries, save_path):
    """Mirror the upstream eval metrics.summarize_all: per-dataset + overall CSVs."""
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


def _dataset_save_path(save_path, dataset_name, model_type, batch_size):
    return os.path.join(
        save_path,
        f"{dataset_name}-{model_type}",
        f"Epoch0BZ{batch_size}-Norm-none-Nan-mean-new-Cat-indices",
    )


def _expected_summary_columns():
    return (
        ["dataset", "task_type", "n_classes", "mean_loss"]
        + [f"{name}_mean" for name in METRIC_NAMES]
        + [f"{name}_std" for name in METRIC_NAMES]
    )


def _load_completed_summary(save_path, ds_dir, model_type, batch_size, seed_num):
    details_path = os.path.join(
        _dataset_save_path(save_path, ds_dir.name, model_type, batch_size),
        f"results_{ds_dir.name}_details.csv",
    )
    if not os.path.exists(details_path):
        return None

    try:
        details = pd.read_csv(details_path)
    except Exception:
        return None

    expected_columns = {"seed", *METRIC_NAMES}
    if len(details) != seed_num + 2 or not expected_columns.issubset(details.columns):
        return None
    if not np.array_equal(details["seed"].iloc[:seed_num].to_numpy(), np.arange(seed_num)):
        return None

    info = json.load(open(ds_dir / "info.json"))
    y_train = np.load(ds_dir / "y_train.npy", allow_pickle=True)
    mean_row = details.iloc[seed_num]
    std_row = details.iloc[seed_num + 1]
    return {
        "dataset": ds_dir.name,
        "task_type": info["task_type"],
        "n_classes": int(len(np.unique(y_train))),
        "mean_loss": float(mean_row["LogLoss"]),
        **{f"{name}_mean": float(mean_row[name]) for name in METRIC_NAMES},
        **{f"{name}_std": float(std_row[name]) for name in METRIC_NAMES},
    }


def _load_existing_summaries(save_path):
    summary_path = os.path.join(save_path, "all_datasets_summary.csv")
    if not os.path.exists(summary_path):
        return []
    try:
        existing = pd.read_csv(summary_path)
    except Exception:
        return []
    if set(existing.columns) != set(_expected_summary_columns()):
        print("Ignoring existing summary because its schema differs from this evaluator.")
        return []
    return existing.to_dict("records")


def _write_failed_datasets(save_path, failures):
    pd.DataFrame(failures, columns=["dataset", "error"]).to_csv(
        os.path.join(save_path, "failed_datasets.csv"), index=False
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offload-mode", default="auto", choices=["auto", "gpu", "cpu", "disk"])
    ap.add_argument(
        "--disk-offload-dir", default=None, help="directory for disk offload (required by --offload-mode disk)"
    )
    ap.add_argument("--dataset", default="Pima_Indians_Diabetes_Database", help="dataset dir name (ignored if --all)")
    ap.add_argument("--all", action="store_true", help="run all datasets under --data-root")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed-num", type=int, default=5, help="#trials (referenced original evaluation uses 5)")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--cat_random", "--cat_randomEncode", dest="cat_random_encode",
        action="store_true",
        help="randomly permute categorical feature and class integer codes per ensemble member",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap #datasets (with --all)")
    ap.add_argument("--save-path", default=None, help="save root (default: <script_dir>/results/<ckpt_stem>)")
    ap.add_argument("--resume", action="store_true", help="skip datasets with complete matching results")
    ap.add_argument("--strict", action="store_true", help="exit nonzero if any dataset fails")
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
            names = names[: args.limit]
    else:
        names = [args.dataset]

    print(f"tabldm {tabldm.__version__} | ckpt={ckpt_stem} | model_type={args.model_type} "
          f"| n_estimators={args.n_estimators} | batch_size={args.batch_size} "
          f"| seed_num={args.seed_num} | device={device} "
          f"| cat_random={args.cat_random_encode}")
    print(f"data_root={data_root} | datasets={len(names)} | save_path={save_path} | resume={args.resume}\n")

    run_summaries = []
    failures = []
    for i, name in enumerate(names, 1):
        ds_dir = data_root / name
        print(f"[{i}/{len(names)}] {name}")
        summary = None
        if args.resume:
            summary = _load_completed_summary(
                save_path, ds_dir, args.model_type, args.batch_size, args.seed_num
            )
            if summary is not None:
                print("  SKIP: complete matching results found")
        if summary is None:
            try:
                summary = run_dataset(ds_dir, args.ckpt, args.n_estimators, args.batch_size,
                                      device, args.verbose, args.seed_num, save_path, args.model_type,
                                      args.offload_mode, args.disk_offload_dir,
                                      args.cat_random_encode)
            except Exception as e:
                failures.append({"dataset": name, "error": repr(e)})
                print(f"  ERROR: {e!r}")
        if summary is not None:
            run_summaries.append(summary)
        print()

    merge_existing = args.resume or not args.all
    existing_summaries = _load_existing_summaries(save_path) if merge_existing else []
    selected_names = set(names)
    summaries_by_dataset = {
        summary["dataset"]: summary
        for summary in existing_summaries
        if summary["dataset"] not in selected_names
    }
    summaries_by_dataset.update({summary["dataset"]: summary for summary in run_summaries})
    all_summaries = [summaries_by_dataset[name] for name in sorted(summaries_by_dataset)]

    _write_failed_datasets(save_path, failures)
    if failures:
        print(f"Failed datasets: {', '.join(failure['dataset'] for failure in failures)}")
        print(f"Saved failures -> {save_path}/failed_datasets.csv")
    summarize_all(all_summaries, save_path) if len(all_summaries) >= 1 else None
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
