"""benchmarks/scoring.py: answer normalisation, field scoring, source checks and fill-plan scoring."""

import sys
from datetime import date

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
import scoring  # noqa: E402


@pytest.mark.parametrize("text", [
    "05/09/2024", "5/9/2024", "05-09-2024", "05.09.2024", "2024-09-05", "5 September 2024", "5th September, 2024",
    "September 5, 2024", "Thursday, 5 September 2024", "05 Sep 2024", "5-Sept-2024",
])
def test_parse_date_formats(text):
    assert scoring.parse_date(text) == date(2024, 9, 5)


def test_parse_date_rejects_non_dates():
    assert scoring.parse_date("UNANSWERABLE") is None
    assert scoring.parse_date("31/02/2024") is None
    assert scoring.parse_date(None) is None


@pytest.mark.parametrize("text,value", [
    ("Rs. 10,00,00,000", 100_000_000), ("100000000", 100_000_000), ("INR 100,000,000", 100_000_000),
    ("Rs. 1,234.56 lakhs", 123_456_000), ("42,318.64 lakh", 4_231_864_000), ("Rs. 5,000 crore", 50_000_000_000),
    ("8,412.35 Cr.", 84_123_500_000), ("INR 120 million", 120_000_000), ("973.16 mn", 973_160_000),
    ("30,00,00,000 (Rupees thirty crore)", 300_000_000),
])
def test_parse_amount(text, value):
    assert scoring.parse_amount(text) == value


def test_yes_no_and_identifiers():
    assert scoring.parse_yesno("Yes, annexed as MR-3") == "yes"
    assert scoring.parse_yesno("No.") == "no"
    assert scoring.parse_yesno("Not applicable") == "no"
    assert scoring.parse_yesno("UNANSWERABLE") is None
    assert scoring.parse_yesno("None") is None
    assert scoring.normalise_id(" l29120zz1994plc011872 ") == "L29120ZZ1994PLC011872"
    assert scoring.normalise_email("E-mail: CS@Example.COM.") == "cs@example.com"


def test_score_field_by_type():
    assert scoring.score_field("date", "14/05/2024", "14 May 2024")["correct"]
    assert not scoring.score_field("date", "14/05/2024", "12/05/2023")["correct"]
    assert scoring.score_field("amount", "500000000", "5,000.00 lakhs")["correct"]
    assert not scoring.score_field("amount", "500000000", "300000000")["correct"]
    assert scoring.score_field("yesno", "No", "Not applicable")["correct"]
    assert scoring.score_field("id", "L29120ZZ1994PLC011872", "CIN: L29120ZZ1994PLC011872")["correct"]
    assert scoring.score_field("email", "a@b.example", "a@b.example")["correct"]


def test_text_fields_use_token_f1():
    gold = "Velmora Pumps and Valves Limited"
    exact = scoring.score_field("text", gold, "Velmora Pumps & Valves Ltd.")
    assert exact["correct"] and exact["strict"]
    loose = scoring.score_field("text", "Plot 14, Sector 9, Examplepur Industrial Area, Examplepur, Zedland 999014",
                                "Plot 14, Sector 9, Examplepur Industrial Area, Examplepur, Zedland")
    assert loose["correct"] and not loose["strict"]
    assert not scoring.score_field("text", gold, "Velmora Holdings")["correct"]


def test_unanswerable_gold_needs_a_refusal():
    assert scoring.score_field("email", None, "UNANSWERABLE")["correct"]
    assert scoring.score_field("date", None, None)["correct"]
    assert not scoring.score_field("email", None, "rta@registrar.example")["correct"]
    refused = scoring.score_field("email", "a@b.example", "Not found in the documents")
    assert not refused["correct"] and not refused["answered"]


def test_source_correct():
    gold = [{"file": "fs.pdf", "page": 4}, {"file": "report.docx", "page": None}]
    assert scoring.source_correct({"file": "fs.pdf", "page": 4}, gold)
    assert not scoring.source_correct({"file": "fs.pdf", "page": 3}, gold)
    assert scoring.source_correct({"file": "report.docx", "page": 0}, gold)   # no pages in Word files
    assert not scoring.source_correct({"file": "tb.xlsx", "page": 1}, gold)
    assert not scoring.source_correct(None, gold)


def test_iou_and_centre_in_box():
    assert scoring.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert scoring.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert scoring.centre_in_box([100, 100, 110, 110], [90, 90, 120, 120])
    assert not scoring.centre_in_box([0, 0, 10, 10], [90, 90, 120, 120])


GOLD_FORM = [
    {"key": "Name", "kind": "text", "value": "Jordan Sample", "bbox": [200, 80, 400, 100]},
    {"key": "Gender", "kind": "checkbox", "value": "Male",
     "options": [{"value": "Male", "bbox": [200, 120, 205, 125]}, {"value": "Female", "bbox": [260, 120, 265, 125]}]},
    {"key": "Date of birth", "kind": "comb", "value": "21/11/1985", "bbox": [300, 150, 440, 170]},
]


def test_score_form_counts_correct_placements():
    plan = [{"text": "Jordan Sample", "bbox": [200, 80, 400, 100]},
            {"text": "✓", "bbox": [200, 120, 205, 125]},
            {"text": "21111985", "bbox": [300, 150, 440, 170]}]
    s = scoring.score_form(plan, GOLD_FORM)
    assert (s["n_correct"], s["n_fields"]) == (3, 3)


def test_score_form_rejects_wrong_boxes_and_double_ticks():
    plan = [{"text": "Jordan Sample", "bbox": [200, 180, 400, 200]},      # wrong row
            {"text": "✓", "bbox": [200, 120, 205, 125]},
            {"text": "X", "bbox": [260, 120, 265, 125]},                   # both options ticked
            {"text": "2", "bbox": [300, 150, 310, 170]}, {"text": "1", "bbox": [310, 150, 320, 170]}]
    s = scoring.score_form(plan, GOLD_FORM)
    assert s["n_correct"] == 0
    assert s["fields"][1]["ticked"] == ["Male", "Female"]


def test_score_form_accepts_one_character_per_comb_cell():
    plan = [{"text": ch, "bbox": [300 + 14 * i, 150, 314 + 14 * i, 170]} for i, ch in enumerate("21111985")]
    assert scoring.score_form(plan, GOLD_FORM)["fields"][2]["correct"]


def test_score_form_ignores_malformed_entries():
    plan = [{"text": "Jordan Sample"}, {"bbox": [1, 2, 3, 4]}, "junk", {"text": "Jordan Sample", "bbox": [1, 2]}]
    s = scoring.score_form(plan, GOLD_FORM)
    assert s["n_correct"] == 0 and s["n_invalid_entries"] == 4
    assert scoring.score_form(None, GOLD_FORM)["n_correct"] == 0
