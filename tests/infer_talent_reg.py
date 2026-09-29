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
"""TabLDM MoE1 inference + metrics on TALENT-reg datasets (mirrors
the upstream TALENT evaluation harness's seed handling and result-saving format).

Regression counterpart of ``infer_talent_cls.py``: uses ``TabLDMRegressor``
with ``predict`` (point predictions) instead of ``predict_proba``, and reports
RMSE / MAE / R2 in TALENT's normalized (mean/std) convention: RMSE and MAE are
divided by std(y_train), the MSE loss by std(y_train)**2; R2 is scale-invariant
and left untouched. This mirrors the upstream TALENT eval
(data_label_process + eval/metrics.normalize_regression_metrics, which use
``y_data['train'].std()`` -- numpy default ddof=0). Without this normalization
raw RMSE/MAE are dominated by high-scale targets (e.g. house prices) and are
not comparable across datasets.

Uses ONLY the local ``tabldm`` package + already-installed deps
(torch / scikit-learn / numpy / pandas / scipy). No TALENT framework, no pip installs.

Seed handling (aligned with the upstream eval scripts):
  for seed in range(seed_num): set_seeds(seed); TabLDMRegressor(random_state=seed);
  fit(train) -> predict(test); save predictions_seed{seed}.npz; aggregate mean+-std.
  (reference default seed_num=15; pass --seed-num 15 to match exactly.)

Save format (aligned with the upstream eval scripts / metrics.show_results / summarize_all):
  {save_root}/{dataset}-{model_type}/Epoch0BZ{bs}-Norm-none-Nan-mean-new-Cat-indices/
      predictions_seed{seed}.npz   # predictions, true_label
      results_{dataset}_details.csv  # per-seed metrics + mean/std rows
  {save_root}/all_datasets_summary.csv  # per-dataset mean/std (multi-dataset)
  {save_root}/overall_averages.csv      # overall means (multi-dataset)

Usage:
    python infer_talent_reg.py --dataset MiamiHousing
    python infer_talent_reg.py --dataset MiamiHousing --seed-num 15
    python infer_talent_reg.py --all --seed-num 3 --limit 5
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)

# Import the LOCAL tabldm package from the repo root (this script lives in tests/).
TABLDM_DIR = Path(os.environ.get("TABLDM_DIR", Path(__file__).resolve().parent.parent))
if str(TABLDM_DIR) not in sys.path:
    sys.path.insert(0, str(TABLDM_DIR))
import tabldm
from tabldm import TabLDMRegressor

DEFAULT_CKPT = os.environ.get("TABLDM_REG_CKPT", "checkpoints/reg_moe1.ckpt")
DEFAULT_DATA = os.environ.get("TABLDM_REG_DATA_ROOT", "data/tabarena_reg")
DEFAULT_MODEL_TYPE = "tabldm_moe1_reg"
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
    """Regression metrics in TALENT's normalized (mean/std) convention.

    RMSE and MAE are divided by ``y_std`` (std of the *training* target);
    MSE (the reported loss) by ``y_std ** 2``; R2 is scale-invariant and left
    untouched. Mirrors the upstream eval (data_label_process computes
    ``std = y_data['train'].std()`` -- numpy default ddof=0 -- then
    normalize_regression_metrics divides MAE/RMSE by it). If ``y_std`` is not
    positive the metrics fall back to the raw (unnormalized) scale.
    """
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


def run_dataset(ds_dir, ckpt, n_estimators, batch_size, device, verbose, seed_num, save_path, model_type):
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
        reg = TabLDMRegressor(
            enhance_candidates=True,
            n_estimators=n_estimators,
            norm_methods=["none", "power"],
            model_path=ckpt, allow_auto_download=False,
            device=device,
            use_amp=False if cpu else True,
            use_fa3=False if cpu else "auto",
            offload_mode="cpu" if cpu else "auto",
            n_jobs=1, verbose=verbose, random_state=seed,
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


def _parse_gpu_ids(device: str):
    """Return physical GPU ids from a device string, or [] for non-GPU devices."""
    if device is None:
        return []
    value = str(device).strip()
    if value == "" or value.startswith("cpu"):
        return []
    ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            ids.append(int(part))
        elif part == "cuda":
            ids.append(0)
        elif part.startswith("cuda:") and part[5:].isdigit():
            ids.append(int(part[5:]))
        else:
            return []
    return ids


def _worker_summary_path(save_path: str, worker_id: int) -> str:
    return os.path.join(save_path, f".worker_{worker_id}_summary.json")


def _run_datasets_subset(names, data_root, args, save_path, worker_id):
    """Run one worker's dataset slice and persist its summaries for the parent."""
    all_summaries = []
    for i, name in enumerate(names, 1):
        ds_dir = data_root / name
        if worker_id is None:
            print(f"[{i}/{len(names)}] {name}")
        else:
            print(f"[worker {worker_id}] [{i}/{len(names)}] {name}")
        try:
            summary = run_dataset(ds_dir, args.ckpt, args.n_estimators, args.batch_size,
                                  args.device, args.verbose, args.seed_num, save_path, args.model_type)
            all_summaries.append(summary)
        except Exception as e:
            print(f"  ERROR: {e!r}")
        print()

    if worker_id is not None:
        summary_path = _worker_summary_path(save_path, worker_id)
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f)
        print(f"[worker {worker_id}] saved {len(all_summaries)} summaries -> {summary_path}")
    return all_summaries


def _build_worker_cmd(args, save_path, worker_id, worker_count):
    """Build a subprocess command for one dataset-level GPU worker."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--ckpt", str(args.ckpt),
        "--data-root", str(args.data_root),
        "--model-type", str(args.model_type),
        "--n-estimators", str(args.n_estimators),
        "--batch-size", str(args.batch_size),
        "--seed-num", str(args.seed_num),
        "--device", "cuda:0",
        "--save-path", str(save_path),
        "--worker-id", str(worker_id),
        "--worker-count", str(worker_count),
    ]
    if args.all:
        cmd.append("--all")
    else:
        cmd += ["--dataset", str(args.dataset)]
    if args.verbose:
        cmd.append("--verbose")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd



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
    ap.add_argument(
        "--device",
        default="cpu",
        help="inference device: cpu, cuda, cuda:0, or comma-separated GPU ids (0,1,2,3) / cuda:0,cuda:1; with --all, each GPU runs a distinct dataset slice",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap #datasets (with --all)")
    ap.add_argument("--save-path", default=None, help="save root (default: <script_dir>/results/<ckpt_stem>)")
    ap.add_argument("--worker-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

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
          f"| seed_num={args.seed_num} | device={args.device}")
    print(f"data_root={data_root} | datasets={len(names)} | save_path={save_path}\n")

    # Worker mode: process only this worker's dataset slice, then exit before
    # the parent spawns any additional subprocesses.
    if args.worker_id >= 0:
        worker_names = names[args.worker_id::args.worker_count]
        _run_datasets_subset(worker_names, data_root, args, save_path, args.worker_id)
        return

    # Dataset-level parallelism: with --all and multiple GPU ids, one subprocess
    # per GPU runs a disjoint slice of datasets.  Single-dataset and CPU runs
    # fall through to the original sequential path below.
    gpu_ids = _parse_gpu_ids(args.device)
    if args.all and len(gpu_ids) > 1:
        worker_count = min(len(gpu_ids), len(names))
        if worker_count > 1:
            print(f"[multi-gpu] {len(names)} datasets over {worker_count} GPU worker(s): "
                  f"{gpu_ids[:worker_count]}\n")
            procs = []
            for worker_id in range(worker_count):
                stale_summary = _worker_summary_path(save_path, worker_id)
                if os.path.exists(stale_summary):
                    os.remove(stale_summary)
            for worker_id in range(worker_count):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[worker_id])
                cmd = _build_worker_cmd(args, save_path, worker_id, worker_count)
                print(f"[multi-gpu] launch worker {worker_id} on physical GPU {gpu_ids[worker_id]}: "
                      f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
                procs.append(subprocess.Popen(cmd, env=env))

            failed = False
            for proc in procs:
                proc.wait()
                if proc.returncode != 0:
                    failed = True
                    print(f"[multi-gpu] worker exited with status {proc.returncode}")

            all_summaries = []
            for worker_id in range(worker_count):
                summary_path = _worker_summary_path(save_path, worker_id)
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        all_summaries.extend(json.load(f))
            all_summaries.sort(key=lambda r: r.get("dataset", ""))
            summarize_all(all_summaries, save_path) if all_summaries else None
            if failed:
                sys.exit(1)
            return

    # Sequential single-process path (also used for single-dataset multi-GPU,
    # where TabLDMRegressor itself shards ensemble members across devices).
    all_summaries = _run_datasets_subset(names, data_root, args, save_path, None)
    summarize_all(all_summaries, save_path) if all_summaries else None

if __name__ == "__main__":
    main()
