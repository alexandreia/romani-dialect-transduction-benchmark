"""Tests for A2 (slots only for paradigm cells). Run: python -m pytest tests/"""
from src.data.build_canonical import build_canonical_rows

HEADER = "sample_id,item_id,gold_variety,gold_variety_broad,romani_phrase,meaning\n"


def test_slot_only_inferred_for_paradigm_cells(tmp_path):
    csv_path = tmp_path / "RO-TEST_items.csv"
    csv_path.write_text(
        HEADER
        + "RO-TEST,1,Test,Test,vo dikhlja le phuren,he saw the old men\n"  # sentence starting with a pronoun
        + "RO-TEST,2,Test,Test,me kerav\\tu keres,do\n",                   # paradigm cell (backslash-separated)
        encoding="utf-8",
    )
    df = build_canonical_rows([str(csv_path)])

    sentence = df[df["item_id"].astype(str) == "1"].iloc[0]
    assert sentence["slot"] == "UNK"  # was "3SG" before A2

    paradigm = df[df["item_id"].astype(str) == "2"]
    assert list(paradigm["slot"]) == ["1SG", "2SG"]
