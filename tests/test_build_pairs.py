"""Tests for A3/A4/A5 in build_pairs.py. Run: python -m pytest tests/"""
import pandas as pd

from src.data.build_pairs import build_pair_df

COLUMNS = ["sample_id", "variety", "item_id", "meaning", "form_raw", "form_normalized",
           "form_index", "is_paradigm", "form_type", "person", "number", "slot",
           "source_file", "quality_flag", "notes"]


def row(sample, item, form, index, slot="UNK", flag="OK"):
    person, number = (slot[0], slot[1:]) if slot != "UNK" else ("UNK", "UNK")
    return [sample, "V", item, "m", form, form, index, slot != "UNK", "lexeme",
            person, number, slot, "f.csv", flag, ""]


def canonical(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_quality_flag_uses_both_sides():  # A4
    df = canonical([row("A", "1", "kerav", 1), row("B", "1", "kǝrav", 1, flag="CHECK")])
    pair_df, report, _, _ = build_pair_df(df, "A", "B")
    assert list(pair_df["quality_flag"]) == ["CHECK"]
    assert report["aligned_check_rows"] == 1


def test_unaligned_rows_are_reported_not_silently_dropped():  # A3
    df = canonical([
        row("A", "1", "me kerav", 1, slot="1SG"),   # slot on source...
        row("B", "1", "kǝrav", 1),                  # ...UNK on target -> cannot align
        row("A", "2", "pani", 1),                   # item 2 missing in B
    ])
    pair_df, report, unaligned, _ = build_pair_df(df, "A", "B")
    assert pair_df.empty
    assert report["source_lost_no_matching_key"] == 1
    assert report["source_lost_item_missing_in_target"] == 1
    assert report["target_lost_no_matching_key"] == 1
    assert set(unaligned["reason"]) == {"no_matching_key", "item_missing_on_other_side"}


def test_duplicate_keys_excluded_not_cross_multiplied():  # A5
    df = canonical([
        row("A", "1", "vo kerel", 3, slot="3SG"),
        row("A", "1", "vo kerena", 6, slot="3SG"),  # duplicate 3SG in one cell
        row("B", "1", "vo kǝrel", 3, slot="3SG"),
    ])
    pair_df, report, _, duplicates = build_pair_df(df, "A", "B")
    assert pair_df.empty                      # before: 2 cross-multiplied rows
    assert len(duplicates) == 2
    assert report["source_duplicate_key_rows_excluded"] == 2


def test_split_group_uses_real_form_index_for_fallback_rows():  # A15
    df = canonical([row("A", "7", "pani", 2), row("B", "7", "paji", 2)])
    pair_df, _, _, _ = build_pair_df(df, "A", "B")
    assert list(pair_df["split_group"]) == ["7_form_2"]   # was "7_form_UNK"
