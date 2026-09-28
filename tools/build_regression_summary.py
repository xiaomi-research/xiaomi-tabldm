#!/usr/bin/env python3
"""Reconstruct regression summaries from experiment details and a reference CSV.

The inference command writes ``all_datasets_summary.csv`` and
``overall_averages.csv`` only after all workers finish. This script can
reconstruct those files from completed per-dataset detail CSVs. By default,
datasets without completed experiment details are filled from
``regression_results.csv`` and are marked as ``reference_csv_fallback`` in the
comparison output; this is an estimate, not a replacement for the final
15-seed experiment result.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from statistics import mean, pstdev


SUMMARY_COLUMNS = [
    "dataset",
    "task_type",
    "y_std",
    "mean_loss",
    "rmse_mean",
    "mae_mean",
    "r2_mean",
    "rmse_std",
    "mae_std",
    "r2_std",
]

REFERENCE_COLUMNS = [
    "dataset",
    "R2",
    "train_std",
    "normalized_RMSE",
    "normalized_MAE",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    default_reference = repo / "regression_results.csv"
    default_results = repo / "results" / "step-6000_talent_reg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=default_reference,
        help=f"reference CSV (default: {default_reference})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help=f"experiment result directory (default: {default_results})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_results / "reconstructed_summary",
        help="directory for generated files",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=15,
        help="expected number of seeds per dataset (default: 15)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="omit datasets without completed experiment details instead of using reference fallback",
    )
    return parser.parse_args()


def as_float(value: str) -> float:
    return float(value)


def format_value(value: float | int | str | None) -> str | float | int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


def load_reference(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in REFERENCE_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        reference = {}
        for row in reader:
            dataset = row["dataset"].strip()
            if not dataset:
                continue
            reference[dataset] = {
                "r2": as_float(row["R2"]),
                "y_std": as_float(row["train_std"]),
                "rmse": as_float(row["normalized_RMSE"]),
                "mae": as_float(row["normalized_MAE"]),
            }
    return reference


def dataset_from_details(path: Path) -> str:
    filename = path.name
    prefix = "results_"
    suffix = "_details.csv"
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        raise ValueError(f"unexpected detail filename: {path}")
    return filename[len(prefix) : -len(suffix)]


def prediction_seed_count(results_dir: Path, dataset: str) -> int:
    pattern = results_dir / f"{dataset}-tabldm_moe1_reg" / "Epoch*" / "predictions_seed*.npz"
    return len(glob.glob(str(pattern)))


def read_detail(
    path: Path,
    reference: dict[str, float],
    results_dir: Path,
    expected_seeds: int,
) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    seed_rows = [row for row in rows if row.get("seed", "").strip()]
    if not seed_rows:
        raise ValueError(f"no seed rows found in {path}")

    rmse_values = [as_float(row["rmse"]) for row in seed_rows]
    mae_values = [as_float(row["mae"]) for row in seed_rows]
    r2_values = [as_float(row["r2"]) for row in seed_rows]
    dataset = dataset_from_details(path)
    return {
        "dataset": dataset,
        "task_type": "regression",
        "y_std": reference["y_std"],
        "mean_loss": mean(value * value for value in rmse_values),
        "rmse_mean": mean(rmse_values),
        "mae_mean": mean(mae_values),
        "r2_mean": mean(r2_values),
        "rmse_std": pstdev(rmse_values),
        "mae_std": pstdev(mae_values),
        "r2_std": pstdev(r2_values),
        "source": "experiment_details",
        "completed_seeds": len(seed_rows),
        "expected_seeds": expected_seeds,
        "prediction_seeds": prediction_seed_count(results_dir, dataset),
    }


def find_detail_rows(
    results_dir: Path,
    reference: dict[str, dict[str, float]],
    expected_seeds: int,
) -> dict[str, dict]:
    rows = {}
    pattern = str(results_dir / "*" / "*" / "results_*_details.csv")
    for path_string in glob.glob(pattern):
        path = Path(path_string)
        dataset = dataset_from_details(path)
        if dataset not in reference:
            continue
        rows[dataset] = read_detail(
            path,
            reference[dataset],
            results_dir,
            expected_seeds,
        )
    return rows


def make_reference_row(
    dataset: str,
    reference: dict[str, float],
    results_dir: Path,
    expected_seeds: int,
) -> dict:
    return {
        "dataset": dataset,
        "task_type": "regression",
        "y_std": reference["y_std"],
        "mean_loss": reference["rmse"] ** 2,
        "rmse_mean": reference["rmse"],
        "mae_mean": reference["mae"],
        "r2_mean": reference["r2"],
        "rmse_std": math.nan,
        "mae_std": math.nan,
        "r2_std": math.nan,
        "source": "reference_csv_fallback",
        "completed_seeds": 0,
        "expected_seeds": expected_seeds,
        "prediction_seeds": prediction_seed_count(results_dir, dataset),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fieldnames})


def build_comparison_row(row: dict, reference: dict[str, float]) -> dict:
    return {
        "dataset": row["dataset"],
        "source": row["source"],
        "completed_seeds": row["completed_seeds"],
        "prediction_seeds": row["prediction_seeds"],
        "reference_R2": reference["r2"],
        "experiment_R2": row["r2_mean"],
        "delta_R2": row["r2_mean"] - reference["r2"],
        "reference_normalized_RMSE": reference["rmse"],
        "experiment_normalized_RMSE": row["rmse_mean"],
        "delta_normalized_RMSE": row["rmse_mean"] - reference["rmse"],
        "reference_normalized_MAE": reference["mae"],
        "experiment_normalized_MAE": row["mae_mean"],
        "delta_normalized_MAE": row["mae_mean"] - reference["mae"],
    }


def main() -> None:
    args = parse_args()
    if args.expected_seeds <= 0:
        raise SystemExit("--expected-seeds must be positive")

    reference = load_reference(args.reference_csv)
    experiment_rows = find_detail_rows(
        args.results_dir,
        reference,
        args.expected_seeds,
    )
    summary_rows = []
    comparison_rows = []
    fallback_count = 0

    for dataset in sorted(reference):
        row = experiment_rows.get(dataset)
        if row is None:
            if args.strict:
                continue
            row = make_reference_row(
                dataset,
                reference[dataset],
                args.results_dir,
                args.expected_seeds,
            )
            fallback_count += 1
        summary_rows.append(row)
        comparison_rows.append(build_comparison_row(row, reference[dataset]))

    if not summary_rows:
        raise SystemExit("no summary rows found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "all_datasets_summary.csv"
    overall_path = args.output_dir / "overall_averages.csv"
    comparison_path = args.output_dir / "comparison_to_reference.csv"

    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)

    mean_columns = ["rmse_mean", "mae_mean", "r2_mean"]
    overall = {
        column: mean(float(row[column]) for row in summary_rows)
        for column in mean_columns
    }
    write_csv(overall_path, mean_columns, [overall])

    comparison_columns = list(comparison_rows[0])
    write_csv(comparison_path, comparison_columns, comparison_rows)

    print(f"wrote {summary_path} ({len(summary_rows)} datasets)")
    print(f"wrote {overall_path}")
    print(f"wrote {comparison_path}")
    print(f"completed experiment datasets: {len(experiment_rows)}")
    print(f"reference fallbacks: {fallback_count}")
    if fallback_count:
        print(
            "fallback rows are estimates from regression_results.csv, "
            "not final experiment metrics"
        )


if __name__ == "__main__":
    main()
