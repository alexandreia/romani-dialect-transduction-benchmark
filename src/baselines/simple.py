"""
Run one simple non-neural baseline.

Baselines help answer: does a future neural model learn something, or is it only
matching easy surface patterns?
"""

import argparse
from difflib import SequenceMatcher
import os
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_pairs import load_pair_dataset
from src.evaluation.metrics import score_predictions, summarize_by, summarize_scores


def predict_copy_source(test_df):
    """
    Predict the source form unchanged.

    This is important for Romani dialects because many forms are similar or
    identical across dialect samples. More complex systems should be compared
    against this simple copying strategy.
    """
    predictions = test_df.copy()
    predictions["prediction"] = predictions["model_input_form"]
    predictions["baseline"] = "copy_source"
    return predictions


def build_target_form_lookup(train_df):
    """
    builds the lookup dictionary from the training data.
    """
    lookup = {}
    grouped = train_df.groupby(
        ["target_sample_id", "item_id", "slot"],
        dropna=False,
    )
    # this loop goes through each group and stores the most common value/form for that group
    for key, group in grouped:
        # counts how often each target form appears
        target_counts = group["model_target_form"].value_counts()
        # takes the most common form, which is the first item in the count table
        lookup[key] = target_counts.index[0]

    return lookup


def predict_target_form_lookup(train_df, test_df):
    """
    Uses the lookup dictionary to make predictions on the test data. 
    Predicts from memorized target forms when the same item/slot was seen.
    This is a diagnostic function for the dataset, not a fair form-only model.
    high target_form_lookup performance suggests possible leakage.
    """
    lookup = build_target_form_lookup(train_df)
    predictions = test_df.copy()

    def predict(row):
        key = (row["target_sample_id"], row["item_id"], row["slot"])
        return lookup.get(key, row["model_input_form"])

    predictions["prediction"] = predictions.apply(predict, axis=1)
    predictions["baseline"] = "target_form_lookup"
    return predictions


def predict_nearest_source_form(train_df, test_df):
    """Use the target form from the closest-looking training source form."""
    train_records = train_df[["model_input_form", "model_target_form"]].to_dict("records")
    records_by_target_and_slot = {}
    records_by_target = {}

    for record in train_df.to_dict("records"):
        records_by_target_and_slot.setdefault(
            (record["target_sample_id"], record["slot"]),
            [],
        ).append(record)
        records_by_target.setdefault(record["target_sample_id"], []).append(record)

    predictions = test_df.copy()

    def nearest_target(source_form, candidates):
        # Pick the training example with the smallest character edit distance.
        best_record = None
        best_distance = None
        for record in candidates:
            distance = Levenshtein.distance(source_form, record["model_input_form"])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_record = record

        if best_record is None:
            return source_form

        return best_record["model_target_form"]

    def predict(row):
        candidates = records_by_target_and_slot.get((row["target_sample_id"], row["slot"]))
        if not candidates:
            candidates = records_by_target.get(row["target_sample_id"])
        if not candidates:
            candidates = train_records
        return nearest_target(row["model_input_form"], candidates)

    predictions["prediction"] = predictions.apply(predict, axis=1)
    predictions["baseline"] = "nearest_source_form"
    return predictions


def final_token(text):
    tokens = str(text).split()
    return tokens[-1] if tokens else ""


def replace_final_token(text, replacement_token):
    tokens = str(text).split()
    if not tokens:
        return replacement_token

    tokens[-1] = replacement_token
    return " ".join(tokens)


def build_suffix_rules(train_df, max_suffix_length=4):
    """Learn common source-suffix -> target-suffix replacements."""
    rule_counts = {}

    for _, row in train_df.iterrows():
        source_token = final_token(row["model_input_form"])
        target_token = final_token(row["model_target_form"])
        if not source_token or not target_token:
            continue

        for suffix_length in range(max_suffix_length, 0, -1):
            if len(source_token) < suffix_length:
                continue

            source_suffix = source_token[-suffix_length:]
            target_suffix = target_token[-suffix_length:] if len(target_token) >= suffix_length else target_token
            rule_key = (row["target_sample_id"], row["slot"], source_suffix)
            rule_counts.setdefault(rule_key, {})
            rule_counts[rule_key][target_suffix] = rule_counts[rule_key].get(target_suffix, 0) + 1

    suffix_rules = {}
    for rule_key, target_counts in rule_counts.items():
        suffix_rules[rule_key] = max(target_counts.items(), key=lambda item: item[1])[0]

    return suffix_rules


def predict_suffix_rule(train_df, test_df, max_suffix_length=4):
    """Apply the most frequent learned suffix replacement when available."""
    suffix_rules = build_suffix_rules(train_df, max_suffix_length=max_suffix_length)
    predictions = test_df.copy()

    def predict(row):
        source_form = row["model_input_form"]
        source_token = final_token(source_form)
        if not source_token:
            return source_form

        for suffix_length in range(max_suffix_length, 0, -1):
            if len(source_token) < suffix_length:
                continue

            source_suffix = source_token[-suffix_length:]
            rule_key = (row["target_sample_id"], row["slot"], source_suffix)
            if rule_key in suffix_rules:
                target_suffix = suffix_rules[rule_key]
                if len(source_token) == suffix_length:
                    rewritten_token = target_suffix
                else:
                    rewritten_token = source_token[:-suffix_length] + target_suffix
                return replace_final_token(source_form, rewritten_token)

        return source_form

    predictions["prediction"] = predictions.apply(predict, axis=1)
    predictions["baseline"] = "suffix_rule"
    return predictions


def common_prefix_length(source_text, target_text):
    """Return how many leading characters two strings share."""
    shared = 0
    for source_char, target_char in zip(str(source_text), str(target_text)):
        if source_char != target_char:
            break
        shared += 1
    return shared


def suffix_change_from_tokens(source_token, target_token):
    """
    Convert one aligned token pair into a suffix rewrite.

    Example:
        kerav -> kerau becomes v -> u

    The rule is intentionally simple: it keeps the shared prefix and learns only
    the final changed part. This makes it a morphology-like baseline, not a full
    character transducer.
    """
    source_token = str(source_token)
    target_token = str(target_token)
    shared = common_prefix_length(source_token, target_token)
    source_suffix = source_token[shared:]
    target_suffix = target_token[shared:]

    if not source_suffix:
        return None
    if source_suffix == target_suffix:
        return None

    return source_suffix, target_suffix


def get_suffix_rule_contexts(row):
    """
    Return suffix-rule contexts from most specific to broadest.

    Prediction uses the same order. This lets the baseline first try rules
    learned for the exact source-target sample pair, then back off to variety or
    target-sample patterns when pair-specific evidence is unavailable.
    """
    source_sample = row.get("source_sample_id", "ANY_SOURCE_SAMPLE")
    target_sample = row.get("target_sample_id", "ANY_TARGET_SAMPLE")
    source_variety = row.get("source_variety", "ANY_SOURCE_VARIETY")
    target_variety = row.get("target_variety", "ANY_TARGET_VARIETY")
    form_type = row.get("form_type", "ANY_FORM_TYPE")
    slot = row.get("slot", "ANY_SLOT") or "ANY_SLOT"

    return [
        ("sample_pair_form_slot", source_sample, target_sample, form_type, slot),
        ("variety_pair_form_slot", source_variety, target_variety, form_type, slot),
        ("target_sample_form_slot", target_sample, form_type, slot),
        ("target_variety_form_slot", target_variety, form_type, slot),
        ("target_sample_slot", target_sample, slot),
        ("target_variety_slot", target_variety, slot),
        ("form_slot", form_type, slot),
        ("slot", slot),
        ("global",),
    ]


def suffix_outcomes(source_token, target_token):
    """
    B4: for every suffix window of the source token whose change is fully
    contained in that window, return (source_suffix, target_suffix).

    Unchanged tokens yield KEEP outcomes (source_suffix == target_suffix), so a
    suffix that usually stays the same is no longer rewritten just because it
    changed in a couple of training rows.

    Example: kerav -> kerau gives ("v", "u"), ("av", "au"), ..., ("kerav", "kerau").
             phenav -> phenav gives ("v", "v"), ("av", "av"), ... (keep).
    """
    outcomes = []
    for length in range(1, len(source_token) + 1):
        prefix = source_token[:-length]
        if target_token.startswith(prefix):
            outcomes.append((source_token[-length:], target_token[len(prefix):]))
    return outcomes


def build_suffix_rules_with_backoff(train_df, min_count=2):
    """
    Learn contextual suffix rewrite rules with backoff.

    The rule table maps:
        context + source_suffix -> most frequent target_suffix

    B4: keep outcomes (target_suffix == source_suffix) compete with change
    outcomes for the same key, so a rule only rewrites when changing is the
    majority outcome in its context. Previously unchanged rows were skipped
    entirely, which made every rule seen twice fire.

    `min_count` avoids applying one-off broad rules.
    """
    rule_counts = {}

    for _, row in train_df.iterrows():
        source_token = final_token(row["model_input_form"])
        target_token = final_token(row["model_target_form"])
        if not source_token or not target_token:
            continue

        contexts = get_suffix_rule_contexts(row)
        for source_suffix, target_suffix in suffix_outcomes(source_token, target_token):
            for context in contexts:
                rule_key = (context, source_suffix)
                rule_counts.setdefault(rule_key, {})
                rule_counts[rule_key][target_suffix] = rule_counts[rule_key].get(target_suffix, 0) + 1

    rules = {}
    for rule_key, target_counts in rule_counts.items():
        target_suffix, count = max(target_counts.items(), key=lambda item: item[1])
        if count >= min_count:
            rules[rule_key] = target_suffix

    return rules


def predict_suffix_rule_with_backoff(train_df, test_df, min_count=2):
    """
    Predict with contextual suffix rules and ordered backoff.

    This is a stronger morphology-like baseline than `suffix_rule`. It tests
    whether slot-, form-type-, sample-, and variety-aware suffix changes can
    explain the task before using a neural model.
    """
    suffix_rules = build_suffix_rules_with_backoff(train_df, min_count=min_count)
    predictions = test_df.copy()

    def predict(row):
        source_form = row["model_input_form"]
        source_token = final_token(source_form)
        if not source_token:
            return source_form

        suffixes = [
            source_token[start:]
            for start in range(len(source_token))
            if source_token[start:]
        ]
        suffixes = sorted(suffixes, key=len, reverse=True)

        for context in get_suffix_rule_contexts(row):
            for source_suffix in suffixes:
                rule_key = (context, source_suffix)
                if rule_key not in suffix_rules:
                    continue

                target_suffix = suffix_rules[rule_key]
                rewritten_token = source_token[:-len(source_suffix)] + target_suffix
                return replace_final_token(source_form, rewritten_token)

        return source_form

    predictions["prediction"] = predictions.apply(predict, axis=1)
    predictions["baseline"] = "suffix_rule_with_backoff"
    return predictions


def find_character_rewrite_rules(source_text, target_text):
    """
    Extract simple character rewrite rules from one aligned training pair.

    This uses SequenceMatcher to find changed spans. For example, if one form
    changes an internal vowel or consonant sequence, the changed source span and
    target span become a candidate rewrite rule.
    """
    rules = []
    matcher = SequenceMatcher(a=str(source_text), b=str(target_text), autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            continue

        source_piece = str(source_text)[source_start:source_end]
        target_piece = str(target_text)[target_start:target_end]

        # Empty-source insertions are hard to apply safely without richer
        # context, so this baseline keeps only replace/delete style rules.
        if not source_piece:
            continue

        rules.append((source_piece, target_piece))

    return rules


def is_simple_rewrite_rule(source_piece, target_piece, max_piece_length=3):
    """Keep only short word-internal rewrites for this conservative baseline."""
    if len(source_piece) > max_piece_length or len(target_piece) > max_piece_length:
        return False
    if any(character.isspace() for character in source_piece + target_piece):
        return False
    if "/" in source_piece or "/" in target_piece:
        return False
    return True


def source_substrings(text, max_piece_length=3):
    """All distinct substrings of length 1..max_piece_length (rule support counting)."""
    text = str(text)
    return {
        text[start:start + length]
        for length in range(1, max_piece_length + 1)
        for start in range(len(text) - length + 1)
    }


def build_character_rewrite_rules(train_df, min_count=3, max_piece_length=3, min_rate=0.5):
    """
    Learn frequent source substring -> target substring rewrite rules.

    Rules are stored at several specificity levels. Prediction first tries
    target-sample + slot rules, then backs off to less specific rules.

    B4: a rule is kept only if it applies in at least `min_rate` of the training
    rows whose source contains the piece (rows where the piece stayed unchanged
    count against it). Counts are per row, so count/support is a proportion.
    """
    rule_counts = {}
    support_counts = {}

    for _, row in train_df.iterrows():
        rules = set(find_character_rewrite_rules(row["model_input_form"], row["model_target_form"]))
        pieces = source_substrings(row["model_input_form"], max_piece_length=max_piece_length)
        keys = [
            (row["target_sample_id"], row["slot"]),
            (row["target_sample_id"], "ANY_SLOT"),
            ("ANY_TARGET", row["slot"]),
            ("ANY_TARGET", "ANY_SLOT"),
        ]

        for key in keys:
            rule_counts.setdefault(key, {})
            key_support = support_counts.setdefault(key, {})
            for piece in pieces:
                key_support[piece] = key_support.get(piece, 0) + 1
            for source_piece, target_piece in rules:
                if not is_simple_rewrite_rule(
                    source_piece,
                    target_piece,
                    max_piece_length=max_piece_length,
                ):
                    continue

                rule = (source_piece, target_piece)
                rule_counts[key][rule] = rule_counts[key].get(rule, 0) + 1

    learned_rules = {}
    for key, counts in rule_counts.items():
        support = support_counts.get(key, {})
        filtered = [
            (source_piece, target_piece, count)
            for (source_piece, target_piece), count in counts.items()
            if count >= min_count
            and source_piece != target_piece
            and count / max(support.get(source_piece, 0), 1) >= min_rate
        ]
        learned_rules[key] = sorted(
            filtered,
            key=lambda item: (item[2], len(item[0])),
            reverse=True,
        )

    return learned_rules


def apply_rewrite_rules(text, rules, max_rules_per_form=5):
    """Apply a small number of learned rewrite rules to one source form."""
    rewritten = str(text)
    applied = 0

    for source_piece, target_piece, _ in rules:
        if source_piece not in rewritten:
            continue

        rewritten = rewritten.replace(source_piece, target_piece, 1)
        applied += 1
        if applied >= max_rules_per_form:
            break

    return rewritten


def predict_character_rewrite_rule(train_df, test_df, min_count=3, max_rules_per_form=2):
    """
    Predict with frequent character rewrite rules.

    This baseline can model internal spelling/sound correspondences more flexibly
    than suffix_rule, while staying much simpler than a neural model.
    """
    learned_rules = build_character_rewrite_rules(train_df, min_count=min_count)
    predictions = test_df.copy()

    def predict(row):
        keys = [
            (row["target_sample_id"], row["slot"]),
            (row["target_sample_id"], "ANY_SLOT"),
            ("ANY_TARGET", row["slot"]),
            ("ANY_TARGET", "ANY_SLOT"),
        ]

        for key in keys:
            # A9: only stop at this backoff level if at least one of its rules
            # actually matches the form; previously a level with rules that did
            # not match returned the source unchanged and broader levels were
            # never tried.
            rules = [
                rule for rule in learned_rules.get(key, [])
                if rule[0] in str(row["model_input_form"])
            ]
            if rules:
                return apply_rewrite_rules(
                    row["model_input_form"],
                    rules,
                    max_rules_per_form=max_rules_per_form,
                )

        return row["model_input_form"]

    predictions["prediction"] = predictions.apply(predict, axis=1)
    predictions["baseline"] = "character_rewrite_rule"
    return predictions


def run_baseline(baseline_name, train_df, test_df):
    """Dispatch a baseline name to the corresponding prediction function."""
    if baseline_name == "copy_source":
        return predict_copy_source(test_df)
    if baseline_name == "target_form_lookup":
        return predict_target_form_lookup(train_df, test_df)
    if baseline_name == "nearest_source_form":
        return predict_nearest_source_form(train_df, test_df)
    if baseline_name == "suffix_rule":
        return predict_suffix_rule(train_df, test_df)
    if baseline_name == "suffix_rule_with_backoff":
        return predict_suffix_rule_with_backoff(train_df, test_df)
    if baseline_name == "character_rewrite_rule":
        return predict_character_rewrite_rule(train_df, test_df)

    raise ValueError(f"Unknown baseline: {baseline_name}")


def ensure_output_path(path, overwrite=False):
    """Protect existing files unless the user explicitly passes --overwrite."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    os.makedirs(os.path.dirname(path), exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simple dialect transduction baselines."
    )
    parser.add_argument(
        "--baseline",
        choices=[
            "copy_source",
            "target_form_lookup",
            "nearest_source_form",
            "suffix_rule",
            "suffix_rule_with_backoff",
            "character_rewrite_rule",
        ],
        required=True,
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
        help="Exclude CHECK rows from train and test data.",
    )
    parser.add_argument(
        "--prediction-output",
        help="Optional predictions CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional summary CSV path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing report files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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

    predictions = run_baseline(args.baseline, train_df, test_df)
    scored = score_predictions(predictions)
    overall_summary = pd.DataFrame([summarize_scores(scored)])
    grouped_summary = summarize_by(
        scored,
        ["baseline", "form_type", "slot", "alignment_key_type", "quality_flag"],
    )

    print("Overall summary")
    print(overall_summary.to_string(index=False))

    if args.prediction_output:
        ensure_output_path(args.prediction_output, overwrite=args.overwrite)
        scored.to_csv(args.prediction_output, index=False)
        print(f"Saved predictions to {args.prediction_output}")

    if args.summary_output:
        ensure_output_path(args.summary_output, overwrite=args.overwrite)
        grouped_summary.to_csv(args.summary_output, index=False)
        print(f"Saved summary to {args.summary_output}")
