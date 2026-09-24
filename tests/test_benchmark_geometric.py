"""benchmarks/geometric_filler.py: the no-LLM nearest-label form-filling baseline."""

import json
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
import geometric_filler as geo  # noqa: E402
from scoring import score_form  # noqa: E402

FORMS = REPO_ROOT / "benchmarks" / "data" / "forms"


def text(t, bbox):
    return {"type": "TEXT", "text": t, "bbox": bbox}


def box(bbox, kind="INPUT_FIELD"):
    return {"type": kind, "text": None, "bbox": bbox}


def test_label_similarity_ignores_item_numbers_and_punctuation():
    assert geo.label_similarity("Applicant name", "1. Applicant name:") == pytest.approx(1.0)
    assert geo.label_similarity("Do you smoke", "a) Do you smoke?") == pytest.approx(1.0)
    assert geo.label_similarity("E-mail", "5. E-mail:") > geo.label_similarity("E-mail", "4. Phone number:")


def test_prepare_drops_letter_artefacts_and_label_cells():
    elements = [text("Name:", [50, 100, 80, 110]),
                box([55, 101, 60, 109], "CHECKBOX"),     # a letter reported as a checkbox
                box([45, 95, 150, 115]),                  # a ruled cell holding the label
                box([160, 95, 300, 115]),                 # the real input box
                box([310, 100, 318, 108])]                # an 8 pt square: a checkbox
    texts, boxes, checks = geo.prepare(elements)
    assert [b["bbox"] for b in boxes] == [[160, 95, 300, 115]]
    assert [c["bbox"] for c in checks] == [[310, 100, 318, 108]]
    assert len(texts) == 1


def test_fill_label_left_label_above_and_checkboxes():
    elements = [
        text("1. Full name:", [50, 100, 110, 110]), box([150, 96, 350, 116]),
        text("Organisation", [50, 140, 110, 150]), box([50, 154, 350, 174]),          # label above its box
        text("2. Gender:", [50, 200, 100, 210]),
        box([150, 202, 155, 207], "CHECKBOX"), text("Male", [159, 200, 180, 210]),
        box([200, 202, 205, 207], "CHECKBOX"), text("Female", [209, 200, 240, 210]),
        text("Smoker?", [50, 240, 90, 250]),
        text("Yes", [300, 240, 316, 250]), box([320, 242, 325, 247], "CHECKBOX"),     # option text before box
        text("No", [360, 240, 373, 250]), box([377, 242, 382, 247], "CHECKBOX"),
    ]
    plan = geo.fill(elements, {"Full name": "Jordan Sample", "Organisation": "Sample Co.",
                               "Gender": "Female", "Smoker": "No"})
    placed = {p["text"]: p["bbox"] for p in plan if p["text"] != geo.TICK}
    ticks = [p["bbox"] for p in plan if p["text"] == geo.TICK]
    assert placed == {"Jordan Sample": [150, 96, 350, 116], "Sample Co.": [50, 154, 350, 174]}
    assert sorted(ticks) == [[200, 202, 205, 207], [377, 242, 382, 247]]


def test_fill_uses_each_box_once_and_skips_unknown_keys():
    elements = [text("First name:", [40, 100, 90, 110]), box([120, 96, 260, 116]),
                text("Last name:", [300, 100, 345, 110]), box([380, 96, 540, 116])]
    plan = geo.fill(elements, {"First name": "Leo", "Last name": "Specimen", "Blood group": "O+"})
    assert {p["text"]: p["bbox"][0] for p in plan} == {"Leo": 120, "Specimen": 380}


def test_grid_cells_and_comb_merging():
    h = [(100, 40, 300), (126, 40, 300), (152, 40, 300)]
    v = [(40, 100, 152), (170, 100, 152), (300, 100, 152)]
    assert geo._grid_cells(h, v) == [[40, 100, 170, 126], [170, 100, 300, 126],
                                     [40, 126, 170, 152], [170, 126, 300, 152]]
    cells = [[400, 50, 416, 70], [416, 50, 432, 70], [438, 50, 454, 70], [100, 50, 300, 70]]
    assert geo._merge_cells(cells) == [[100, 50, 300, 70], [400, 50, 454, 70]]


@pytest.mark.parametrize("form_id", ["f1_labels_left", "f4_yes_no_rows", "f5_ruled_grid", "f6_comb_kyc"])
def test_vector_perception_fills_the_benchmark_forms(form_id):
    truth = json.loads((FORMS / "ground_truth.json").read_text())
    gold = next(f for f in truth["forms"] if f["id"] == form_id)
    new_data = json.loads((FORMS / form_id / "new_data.json").read_text())
    plan = geo.fill(geo.vector_elements(FORMS / form_id / "empty.pdf"), new_data)
    s = score_form(plan, gold["fields"])
    assert s["n_correct"] == s["n_fields"], [f for f in s["fields"] if not f["correct"]]
