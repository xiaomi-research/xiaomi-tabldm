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
"""TabLDM Enhanced Regressor inference + metrics on TALENT-reg datasets.

Uses ``TabLDMEnhancedRegressor`` with ``enhance_candidates=True`` for multi-group
candidate ensembling, NNLS weight learning, SVD+cross feature augmentation,
and high-kurtosis target transforms.

Reports RMSE / MAE / R2 in TALENT's normalized (mean/std) convention, identical
to ``infer_talent_reg.py``.

Seed handling (aligned with the upstream eval scripts):
  for seed in range(seed_num): set_seeds(seed); TabLDMEnhancedRegressor(random_state=seed);
  fit(train) -> predict(test); save predictions_seed{seed}.npz; aggregate mean+-std.

Save format (aligned with the upstream eval scripts):
  {save_root}/{dataset}-{model_type}/Epoch0BZ{bs}-Norm-none-Nan-mean-new-Cat-indices/
      predictions_seed{seed}.npz   # predictions, true_label
      results_{dataset}_details.csv  # per-seed metrics + mean/std rows
  {save_root}/all_datasets_summary.csv  # per-dataset mean/std (multi-dataset)
  {save_root}/overall_averages.csv      # overall means (multi-dataset)

Usage:
    python infer_talent_reg_enhanced.py --dataset MiamiHousing
    python infer_talent_reg_enhanced.py --dataset MiamiHousing --seed-num 15
    python infer_talent_reg_enhanced.py --all --seed-num 3 --limit 5
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)

# Import the LOCAL tabldm package (this script lives in the TabLDM project root).
TABLDM_DIR = Path(os.environ.get("TABLDM_DIR", Path(__file__).resolve().parent))
if str(TABLDM_DIR) not in sys.path:
    sys.path.insert(0, str(TABLDM_DIR))
import tabldm
from tabldm._sklearn.regressor_enhanced import TabLDMEnhancedRegressor

DEFAULT_CKPT = os.environ.get("TABLDM_REG_CKPT", "checkpoints/reg_moe1.ckpt")
DEFAULT_DATA = os.environ.get("TABLDM_REG_DATA_ROOT", "data/tabarena_reg")
DEFAULT_MODEL_TYPE = "tabldm_moe1_reg_enhanced"
METRIC_NAMES = ["rmse", "mae", "r2"]


def set_seeds(seed: int) -> None:
    """Mirror TALENT.model.utils.set_seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load(ds_dir: Path, split: str):
    n, c = ds_dir / f"N_{split}.npy", ds_dir / f"C_{split}.npy"
    N = np.load(n, allow_pickle=True) if n.exists() else None
    C = np.load(c, allow_pickle=True) if c.exists() else None
    y = np.load(ds_dir / f"y_{split}.npy", allow_pickle=True)
    return N, C, y


def to_frame(N, C) -> pd.DataFrame:
    cols = {}
    if N is not None:
        N = np.asarray(N)
        if N.ndim == 1:
            N = N.reshape(-1, 1)
        for j in range(N.shape[1]):
            cols[f"num_{j}"] = pd.to_numeric(pd.Series(N[:, j]), errors="coerce").astype("float64")
    if C is not None:
        C = np.asarray(C)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        for j in range(C.shape[1]):
            cols[f"cat_{j}"] = pd.Series(C[:, j]).astype("object")
    return pd.DataFrame(cols)


def compute_metrics(y_true, y_pred, y_std):
    """Regression metrics in TALENT's normalized (mean/std) convention."""
    s = y_std if (y_std and y_std > 0) else 1.0
    mse = float(mean_squared_error(y_true, y_pred)) / (s * s)
    m = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)) / s,
        "r2": float(r2_score(y_true, y_pred)),
    }
    return m


def save_predictions(npz_path, predictions, y_true):
    """Mirror the upstream eval harness predict() save block (regression variant)."""
    predictions = np.asarray(predictions, dtype=np.float32).ravel()
    np.savez(
        npz_path,
        predictions=predictions,
        true_label=y_true.astype(np.float32),
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


def run_dataset(ds_dir, ckpt, n_estimators, batch_size, device, verbose, seed_num,
                save_path, model_type, enhance_kwargs):
    """Run enhanced regressor on one dataset, returning per-seed summaries."""
    info = json.load(open(ds_dir / "info.json"))
    Ntr, Ctr, ytr = _load(ds_dir, "train")
    Nte, Cte, yte = _load(ds_dir, "test")
    ytr_enc = np.asarray(ytr, dtype=np.float64).ravel()
    yte_enc = np.asarray(yte, dtype=np.float64).ravel()
    y_std = float(np.std(ytr_enc))  # ddof=0, matches TALENT data_label_process
    Xtr, Xte = to_frame(Ntr, Ctr), to_frame(Nte, Cte)
    cpu = device == "cpu"

    save_path1 = f"{ds_dir.name}-{model_type}"
    save_path2 = f"Epoch0BZ{batch_size}-Norm-none-Nan-mean-new-Cat-indices"
    ds_save = os.path.join(save_path, save_path1, save_path2)
    mkdir(ds_save)

    loss_list, results_list = [], []
    for seed in range(seed_num):
        set_seeds(seed)
        reg = TabLDMEnhancedRegressor(
            n_estimators=n_estimators,
            norm_methods=["none", "power"],
            model_path=ckpt, allow_auto_download=False,
            device=device,
            use_amp=False if cpu else True,
            use_fa3=False if cpu else "auto",
            offload_mode="cpu" if cpu else "auto",
            n_jobs=1, verbose=verbose, random_state=seed,
            enhance_candidates=True,
            **enhance_kwargs,
        )
        t0 = time.time(); reg.fit(Xtr, ytr_enc); t_fit = time.time() - t0
        t0 = time.time(); pred = reg.predict(Xte); t_pred = time.time() - t0
        pred = np.asarray(pred, dtype=np.float64).ravel()
        m = compute_metrics(yte_enc, pred, y_std)
        save_predictions(os.path.join(ds_save, f"predictions_seed{seed}.npz"), pred, yte_enc)
        loss_list.append(m["mse"])
        results_list.append([m[n] for n in METRIC_NAMES])
        print(f"  [seed={seed}] fit={t_fit:.1f}s pred={t_pred:.1f}s | "
              + " ".join(f"{n}={m[n]:.4f}" for n in METRIC_NAMES))

    csv_prefix = os.path.join(ds_save, f"results_{ds_dir.name}")
    mean, std, mean_loss = show_results(METRIC_NAMES, loss_list, results_list, csv_prefix)
    return {
        "dataset": ds_dir.name, "task_type": info["task_type"], "y_std": y_std,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MiamiHousing", help="dataset dir name (ignored if --all)")
    ap.add_argument("--all", action="store_true", help="run all datasets under --data-root")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    ap.add_argument("--n-estimators", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed-num", type=int, default=1, help="#trials (upstream reference uses 15)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap #datasets (with --all)")
    ap.add_argument("--save-path", default=None, help="save root (default: <script_dir>/results/<ckpt_stem>)")
    # -- enhancement overrides --
    ap.add_argument("--quantile-est", type=int, default=16, help="n_quantile_estimators")
    ap.add_argument("--no-cross-feature", action="store_true", help="disable SVD+cross augmentation")
    ap.add_argument("--no-validation", action="store_true", help="disable NNLS weight learning")
    ap.add_argument("--no-kfold", action="store_true", help="use single validation instead of K-Fold")
    ap.add_argument("--n-splits", type=int, default=5, help="K-Fold splits")
    ap.add_argument("--foundation-rate", type=float, default=0.25, help="foundation blend ratio")
    ap.add_argument("--max-num-features", type=int, default=500, help="feature sampling threshold")
    ap.add_argument("--no-hk", action="store_true", help="disable high-kurtosis target ensemble")
    ap.add_argument("--hk-threshold", type=float, default=10.0, help="high-kurtosis threshold")
    ap.add_argument("--hk-n-est", type=int, default=8, help="HK candidate count")
    args = ap.parse_args()

    ckpt_stem = Path(args.ckpt).stem
    save_path = args.save_path or str(Path(__file__).resolve().parent / "results" / ckpt_stem)
    mkdir(save_path)

    # Build enhancement kwargs from CLI args
    enhance_kwargs = dict(
        n_quantile_estimators=args.quantile_est,
        use_cross_feature=not args.no_cross_feature,
        validation=not args.no_validation,
        k_fold=not args.no_kfold,
        n_splits=args.n_splits,
        foundation_rate=args.foundation_rate,
        max_num_features=args.max_num_features,
        enable_high_kurtosis_target_ensemble=not args.no_hk,
        high_kurtosis_threshold=args.hk_threshold,
        high_kurtosis_n_estimators=args.hk_n_est,
    )

    data_root = Path(args.data_root)
    if args.all:
        names = sorted(d.name for d in data_root.iterdir() if d.is_dir())
        if args.limit:
            names = names[: args.limit]
    else:
        names = [args.dataset]

    n_total_est = args.n_estimators + args.quantile_est + (args.hk_n_est if not args.no_hk else 0)
    print(f"tabldm {tabldm.__version__} | ckpt={ckpt_stem} | model_type={args.model_type}")
    print(f"  n_estimators={args.n_estimators} | quantile={args.quantile_est} | "
          f"hk={'off' if args.no_hk else args.hk_n_est} | total_candidates≈{n_total_est}")
    print(f"  batch_size={args.batch_size} | seed_num={args.seed_num} | device={args.device}")
    print(f"  use_cross_feature={not args.no_cross_feature} | validation={not args.no_validation} | "
          f"k_fold={not args.no_kfold} | foundation_rate={args.foundation_rate}")
    print(f"data_root={data_root} | datasets={len(names)} | save_path={save_path}\n")

    all_summaries = []
    for i, name in enumerate(names, 1):
        ds_dir = data_root / name
        print(f"[{i}/{len(names)}] {name}")
        try:
            summary = run_dataset(
                ds_dir, args.ckpt, args.n_estimators, args.batch_size,
                args.device, args.verbose, args.seed_num, save_path,
                args.model_type, enhance_kwargs,
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"  ERROR: {e!r}")
        print()

    summarize_all(all_summaries, save_path) if len(all_summaries) >= 1 else None


if __name__ == "__main__":
    main()
