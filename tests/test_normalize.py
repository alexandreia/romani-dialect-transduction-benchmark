"""Tests for src/data/normalize.py (checklist items A1 and A6). Run: python -m pytest tests/"""
import pytest

from src.data.normalize import first_token, infer_person_number, normalize_for_matching


@pytest.mark.parametrize("form, expected", [
    ("(v)on kǝrǝn", ("3", "PL")),        # RO-008: was UNK before A1
    ("on bešen", ("3", "PL")),           # RO-004: was UNK before A1
    ("(v)ou/(v)oj kǝrel", ("3", "SG")),  # RO-008
    ("ou/oj bešel", ("3", "SG")),        # RO-004
    ("me kerav", ("1", "SG")),
    ("tume keren", ("2", "PL")),
    ("dešu", ("UNK", "UNK")),            # numeral, not a pronoun
])
def test_pronoun_slots(form, expected):
    assert infer_person_number(form) == expected


def test_first_token_removes_inner_parentheses():
    assert first_token("(v)on kǝrǝn") == "von"
    assert first_token("(v)ou/(v)oj kǝrel") == "vou/voj"


def test_schwa_code_points_are_unified():
    turned_e = "kǝrav"   # ǝ U+01DD, as in RO-008
    schwa = "kərav"      # ə U+0259, as in RO-025
    assert normalize_for_matching(turned_e) == normalize_for_matching(schwa) == "kərav"


def test_other_diacritics_preserved():
    assert normalize_for_matching("ŠUKAR čhavo") == "šukar čhavo"
