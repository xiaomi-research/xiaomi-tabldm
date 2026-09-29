#!/usr/bin/env python3
"""Summarize regression_results.csv into one metric-level CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median, pstdev


METRICS = ["R2", "train_std", "normalized_RMSE", "normalized_MAE"]


def parse_args() -> argparse.Namespace:
    default_input = Path(
        "/mnt/ai-car-miks/xinzhenwei/"
        "TestOpenSource/Reg/xiaomi-tabldm/regression_results.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"input CSV (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output CSV; defaults to <input_dir>/regression_results_summary.csv",
    )
    parser.add_argument(
        "--overall-output",
        type=Path,
        default=None,
        help=(
            "one-row overall averages CSV; defaults to "
            "<input_dir>/regression_overall_averages.csv"
        ),
    )
    return parser.parse_args()


def load_values(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [metric for metric in METRICS if metric not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")

        values = {metric: [] for metric in METRICS}
        for row_number, row in enumerate(reader, start=2):
            for metric in METRICS:
                try:
                    values[metric].append(float(row[metric]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid value in {path}:{row_number}, column {metric!r}: "
                        f"{row[metric]!r}"
                    ) from exc
    return values


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name("regression_results_summary.csv")
    overall_output = args.overall_output or args.input.with_name(
        "regression_overall_averages.csv"
    )
    values = load_values(args.input)
    dataset_count = len(values[METRICS[0]])

    rows = []
    for metric in METRICS:
        current = values[metric]
        if len(current) != dataset_count:
            raise ValueError(f"column {metric!r} has an inconsistent row count")
        rows.append(
            {
                "metric": metric,
                "dataset_count": dataset_count,
                "mean": mean(current),
                "std": pstdev(current),
                "median": median(current),
                "min": min(current),
                "max": max(current),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "dataset_count", "mean", "std", "median", "min", "max"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    overall_fields = ["rmse_mean", "mae_mean", "r2_mean"]
    overall_row = {
        "rmse_mean": mean(values["normalized_RMSE"]),
        "mae_mean": mean(values["normalized_MAE"]),
        "r2_mean": mean(values["R2"]),
    }
    overall_output.parent.mkdir(parents=True, exist_ok=True)
    with overall_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overall_fields)
        writer.writeheader()
        writer.writerow(overall_row)

    print(f"wrote {output}")
    print(f"wrote {overall_output}")
    print(f"datasets: {dataset_count}")


if __name__ == "__main__":
    main()
