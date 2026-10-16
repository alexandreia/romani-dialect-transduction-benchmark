"""
Scoring functions for dialect form transduction.

The primary metrics are character-level because the task often involves small
spelling, sound-correspondence, or suffix differences rather than whole-sentence
translation.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_pairs import load_pair_dataset


def character_error_rate(prediction, reference):
    """Levenshtein edit distance divided by reference length. Lower is better."""
    prediction = str(prediction)
    reference = str(reference)

    if not reference:
        return 0.0 if not prediction else 1.0

    return Levenshtein.distance(prediction, reference) / len(reference)


def normalized_levenshtein_similarity(prediction, reference):
    """Similarity from 0 to 1 based on normalized edit distance. Higher is better."""
    prediction = str(prediction)
    reference = str(reference)
    max_length = max(len(prediction), len(reference))

    if max_length == 0:
        return 1.0

    return 1 - (Levenshtein.distance(prediction, reference) / max_length)


def char_ngrams(text, n):
    """Return character n-grams with simple boundary spaces."""
    text = f" {str(text)} "
    if len(text) < n:
        return [text]
    return [text[index:index + n] for index in range(len(text) - n + 1)]


def chrf_score(prediction, reference, max_n=6, beta=2.0):
    """Compute a small local chrF-style character n-gram score."""
    prediction = str(prediction)
    reference = str(reference)

    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0

    precisions = []
    recalls = []

    for n in range(1, max_n + 1):
        pred_ngrams = char_ngrams(prediction, n)
        ref_ngrams = char_ngrams(reference, n)
        ref_counts = {}
        for ngram in ref_ngrams:
            ref_counts[ngram] = ref_counts.get(ngram, 0) + 1

        overlap = 0
        for ngram in pred_ngrams:
            if ref_counts.get(ngram, 0) > 0:
                overlap += 1
                ref_counts[ngram] -= 1

        precisions.append(overlap / len(pred_ngrams) if pred_ngrams else 0.0)
        recalls.append(overlap / len(ref_ngrams) if ref_ngrams else 0.0)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta ** 2
    return ((1 + beta_squared) * precision * recall) / ((beta_squared * precision) + recall)


def final_token(text):
    tokens = str(text).split()
    return tokens[-1] if tokens else ""


def suffix(text, length=3):
    token = final_token(text)
    return token[-length:] if token else ""


def stem_guess(text, suffix_length=3):
    token = final_token(text)
    if len(token) <= suffix_length:
        return token
    return token[:-suffix_length]


def is_degenerate_prediction(prediction):
    """Flag empty or obviously repetitive predictions."""
    prediction = str(prediction).strip()
    if not prediction:
        return True

    compact = prediction.replace(" ", "")
    if len(compact) >= 4 and len(set(compact)) == 1:
        return True

    return False


def score_predictions(df, prediction_column="prediction", reference_column="model_target_form"):
    """Add row-level metric columns to a prediction dataframe."""
    required = {prediction_column, reference_column, "model_input_form"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    scored = df.copy()
    scored["exact_match"] = scored[prediction_column] == scored[reference_column]
    scored["cer"] = scored.apply(
        lambda row: character_error_rate(row[prediction_column], row[reference_column]),
        axis=1,
    )
    scored["normalized_levenshtein_similarity"] = scored.apply(
        lambda row: normalized_levenshtein_similarity(row[prediction_column], row[reference_column]),
        axis=1,
    )
    scored["chrf"] = scored.apply(
        lambda row: chrf_score(row[prediction_column], row[reference_column]),
        axis=1,
    )
    scored["copy_prediction"] = scored[prediction_column] == scored["model_input_form"]
    scored["degenerate_prediction"] = scored[prediction_column].apply(is_degenerate_prediction)
    scored["suffix_similarity"] = scored.apply(
        lambda row: normalized_levenshtein_similarity(
            suffix(row[prediction_column]),
            suffix(row[reference_column]),
        ),
        axis=1,
    )
    scored["stem_similarity"] = scored.apply(
        lambda row: normalized_levenshtein_similarity(
            stem_guess(row[prediction_column]),
            stem_guess(row[reference_column]),
        ),
        axis=1,
    )
    return scored


def summarize_scores(scored_df):
    """Collapse row-level scores into one aggregate summary row."""
    if scored_df.empty:
        return {
            "row_count": 0,
            "cer": None,
            "normalized_levenshtein_similarity": None,
            "chrf": None,
            "exact_match": None,
            "copy_rate": None,
            "degeneration_rate": None,
            "suffix_similarity": None,
            "stem_similarity": None,
        }

    return {
        "row_count": len(scored_df),
        "cer": scored_df["cer"].mean(),
        "normalized_levenshtein_similarity": scored_df["normalized_levenshtein_similarity"].mean(),
        "chrf": scored_df["chrf"].mean(),
        "exact_match": scored_df["exact_match"].mean(),
        "copy_rate": scored_df["copy_prediction"].mean(),
        "degeneration_rate": scored_df["degenerate_prediction"].mean(),
        "suffix_similarity": scored_df["suffix_similarity"].mean(),
        "stem_similarity": scored_df["stem_similarity"].mean(),
    }


def summarize_by(scored_df, group_columns):
    """Create grouped summaries, for example by form_type or slot."""
    available_group_columns = [column for column in group_columns if column in scored_df.columns]
    if not available_group_columns:
        return pd.DataFrame([summarize_scores(scored_df)])

    rows = []
    for group_values, group_df in scored_df.groupby(available_group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = dict(zip(available_group_columns, group_values))
        row.update(summarize_scores(group_df))
        rows.append(row)

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Score dialect transduction predictions.")
    parser.add_argument("--predictions", help="CSV containing predictions to score.")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--reference-column", default="model_target_form")
    parser.add_argument("--output", help="Optional path for row-level scored CSV.")
    parser.add_argument("--summary-output", help="Optional path for summary CSV.")
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["form_type", "slot", "alignment_key_type", "quality_flag"],
        help="Columns for grouped summary.",
    )
    parser.add_argument(
        "--demo-copy-baseline",
        action="store_true",
        help="Score a copy-source baseline using the standardized pair loader.",
    )
    parser.add_argument("--split")
    parser.add_argument("--partition", choices=["train", "dev", "test"])
    parser.add_argument("--exclude-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo_copy_baseline:
        df = load_pair_dataset(
            split_path=args.split,
            partition=args.partition,
            exclude_check=args.exclude_check,
        )
        df["prediction"] = df["model_input_form"]
    elif args.predictions:
        df = pd.read_csv(args.predictions, keep_default_na=False)
    else:
        raise ValueError("Provide --predictions or use --demo-copy-baseline.")

    scored_df = score_predictions(
        df,
        prediction_column=args.prediction_column,
        reference_column=args.reference_column,
    )
    summary_df = summarize_by(scored_df, args.group_by)

    print(pd.DataFrame([summarize_scores(scored_df)]).to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        scored_df.to_csv(args.output, index=False)
        print(f"Saved scored rows to {args.output}")

    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
        summary_df.to_csv(args.summary_output, index=False)
        print(f"Saved summary to {args.summary_output}")
