"""Tests for A7 (corrections file) and A8 (broad variety). Run: python -m pytest tests/"""
import pytest

from src.data.build_canonical import apply_corrections, build_canonical_rows, load_corrections

HEADER = "sample_id,item_id,gold_variety,gold_variety_broad,romani_phrase,meaning\n"
CORR_HEADER = "correction_id,sample_id,item_id,old,new,error_type,reason,confidence,verified,verified_note\n"


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_separator_fix_applied_before_split(tmp_path):
    raw = write(tmp_path, "RO-T_items.csv", HEADER + "RO-T,1,Kaldaraš,Kalderash,me phandau\\vo/\\voj phanden,shut\n")
    corr = write(tmp_path, "c.csv", CORR_HEADER + "C1,RO-T,1,vo/\\voj,vo/voj,separator_typo,x,high,yes,\n")
    df = build_canonical_rows([raw], corrections=load_corrections(corr))
    assert list(df["form_raw"]) == ["me phandau", "vo/voj phanden"]   # 2 forms, not 3
    assert set(df["corrections_applied"]) == {"C1"}


def test_unverified_corrections_are_ignored_by_default(tmp_path):
    corr = write(tmp_path, "c.csv", CORR_HEADER + "C1,RO-T,1,a,b,typo,x,low,no,\n")
    assert load_corrections(corr) == {}
    assert len(load_corrections(corr, apply_unverified=True)) == 1


def test_ambiguous_or_stale_correction_stops_the_build():
    corrections = {("RO-T", "1"): [{"correction_id": "C1", "old": "von pen", "new": "x"}]}
    with pytest.raises(ValueError, match="found 2"):
        apply_corrections("von pen\\von pen", "RO-T", "1", corrections)
    with pytest.raises(ValueError, match="found 0"):
        apply_corrections("amen pas", "RO-T", "1", corrections)


def test_correction_for_missing_item_stops_the_build(tmp_path):
    raw = write(tmp_path, "RO-T_items.csv", HEADER + "RO-T,1,X,X,pani,water\n")
    corr = write(tmp_path, "c.csv", CORR_HEADER + "C9,RO-T,999,a,b,typo,x,low,yes,\n")
    with pytest.raises(ValueError, match="C9"):
        build_canonical_rows([raw], corrections=load_corrections(corr))


def test_variety_uses_broad_label(tmp_path):  # A8
    raw = write(tmp_path, "RO-T_items.csv", HEADER + "RO-T,1,Kaldaraš,Kalderash,pani,water\n")
    df = build_canonical_rows([raw])
    assert df.loc[0, "variety"] == "Kalderash"
    assert df.loc[0, "variety_label"] == "Kaldaraš"
