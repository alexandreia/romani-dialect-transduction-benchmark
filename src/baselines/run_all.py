"""
Run all simple baselines and write a comparison table.

This script is useful when you want one CSV comparing every baseline on the same
split. It can also save row-level prediction files for later error analysis.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.simple import run_baseline
from src.data.load_pairs import load_pair_dataset
from src.evaluation.metrics import score_predictions, summarize_scores


BASELINES = [
    "copy_source",
    "target_form_lookup",
    "nearest_source_form",
    "suffix_rule",
    "suffix_rule_with_backoff",
    "character_rewrite_rule",
]


def split_name(split_path):
    """Use the split filename, without .json, as the run label."""
    return os.path.splitext(os.path.basename(split_path))[0]


def ensure_output_path(path, overwrite=False):
    """Protect existing files unless the user explicitly passes --overwrite."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    os.makedirs(os.path.dirname(path), exist_ok=True)


def run_baselines(args):
    """Load train/test data once, then evaluate every requested baseline."""
    train_df = load_pair_dataset(
        split_path=args.split,
        partition="train",
        exclude_check=args.exclude_check,
    )
    test_df = load_pair_dataset(
        split_path=args.split,
        partition=args.test_partition,
        exclude_check=args.exclude_check,
    )

    split_label = split_name(args.split)
    check_label = "exclude_check" if args.exclude_check else "include_check"
    summary_rows = []

    for baseline in args.baselines:
        print(f"Running {baseline}...")
        predictions = run_baseline(baseline, train_df, test_df)
        scored = score_predictions(predictions)
        summary = summarize_scores(scored)
        summary["baseline"] = baseline
        summary["split"] = split_label
        summary["partition"] = args.test_partition
        summary["check_policy"] = check_label
        summary_rows.append(summary)

        if args.save_predictions:
            prediction_path = os.path.join(
                args.prediction_folder,
                f"{baseline}_{split_label}_{args.test_partition}_{check_label}.csv",
            )
            ensure_output_path(prediction_path, overwrite=args.overwrite)
            scored.to_csv(prediction_path, index=False)
            print(f"Saved predictions to {prediction_path}")

    summary_df = pd.DataFrame(summary_rows)
    ordered_columns = [
        "baseline",
        "split",
        "partition",
        "check_policy",
        "row_count",
        "cer",
        "normalized_levenshtein_similarity",
        "chrf",
        "exact_match",
        "copy_rate",
        "degeneration_rate",
        "suffix_similarity",
        "stem_similarity",
    ]
    summary_df = summary_df[ordered_columns]

    summary_path = os.path.join(
        args.summary_folder,
        f"{split_label}_{args.test_partition}_{check_label}_baseline_summary.csv",
    )
    ensure_output_path(summary_path, overwrite=args.overwrite)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved comparison summary to {summary_path}")
    print(summary_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all simple baselines and write a comparison table."
    )
    parser.add_argument(
        "--split",
        default=os.path.join("data", "splits", "heldout_item_split.json"),
        help="Split JSON path.",
    )
    parser.add_argument(
        "--test-partition",
        choices=["dev", "test"],
        default="test",
    )
    parser.add_argument(
        "--exclude-check",
        action="store_true",
        help="Exclude CHECK rows from train/test.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=BASELINES,
        choices=BASELINES,
    )
    parser.add_argument(
        "--prediction-folder",
        default=os.path.join("reports", "predictions"),
    )
    parser.add_argument(
        "--summary-folder",
        default=os.path.join("reports", "model_comparison"),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save row-level scored predictions for each baseline.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing report files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_baselines(parse_args())
