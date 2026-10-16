"""Tests for A9 and B4 in src/baselines/simple.py. Run: python -m pytest tests/"""
import pandas as pd

from src.baselines.simple import (
    build_character_rewrite_rules,
    predict_character_rewrite_rule,
    predict_suffix_rule_with_backoff,
    suffix_outcomes,
)


def frame(pairs, target="T", slot="1SG"):
    return pd.DataFrame(
        {
            "model_input_form": [s for s, _ in pairs],
            "model_target_form": [t for _, t in pairs],
            "source_sample_id": "S", "target_sample_id": target,
            "source_variety": "VS", "target_variety": "VT",
            "form_type": "paradigm_form", "slot": slot,
        }
    )


def test_suffix_outcomes_include_keep():
    assert ("v", "u") in suffix_outcomes("kerav", "kerau")
    assert ("av", "av") in suffix_outcomes("phenav", "phenav")


def test_suffix_rule_does_not_fire_when_keep_is_the_majority():  # B4
    # "-av" stays "-av" in 5 rows and becomes "-au" in 2 rows.
    train = frame([("kerav", "kerav")] * 5 + [("phenav", "phenau")] * 2)
    test = frame([("dikhav", "dikhav")])
    prediction = predict_suffix_rule_with_backoff(train, test)["prediction"].iloc[0]
    assert prediction == "dikhav"          # before B4: "dikhau"


def test_suffix_rule_fires_when_change_is_the_majority():
    train = frame([("phenav", "phenau")] * 5 + [("kerav", "kerav")])
    test = frame([("dikhav", "dikhau")])
    assert predict_suffix_rule_with_backoff(train, test)["prediction"].iloc[0] == "dikhau"


def test_rewrite_rule_needs_majority_rate():  # B4
    # "e" becomes "ǝ" in 3 rows but stays "e" in 10 rows.
    train = frame([("kerel", "kǝrel")] * 3 + [("pel", "pel")] * 10)
    rules = build_character_rewrite_rules(train, min_count=3)
    assert ("e", "ǝ", 3) not in rules[("T", "1SG")]


def test_rewrite_backoff_tries_broader_levels():  # A9
    # Specific level (T, 1SG) only knows a rule for "x"; the form contains "e",
    # for which only a broader level (ANY_TARGET) has a rule.
    train = pd.concat([
        frame([("xa", "ya")] * 3, target="T", slot="1SG"),
        frame([("kerel", "kǝrǝl")] * 3, target="U", slot="2SG"),
    ], ignore_index=True)
    test = frame([("pel", "pǝl")], target="T", slot="1SG")
    prediction = predict_character_rewrite_rule(train, test)["prediction"].iloc[0]
    assert prediction != "pel"             # before A9: returned "pel" unchanged
