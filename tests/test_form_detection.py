"""Form element detection (phase1/phase1.py) on the synthetic demo form."""

import json
import subprocess
import sys

import pytest

from conftest import FIXTURES, REPO_ROOT, load_module

form = load_module("tests/fixtures/make_fixtures.py", "make_fixtures")
phase1 = load_module("phase1/phase1.py", "phase1")

EMPTY_PDF = FIXTURES / "form_empty.pdf"
FILLED_PDF = FIXTURES / "form_filled.pdf"
EMPTY_PNG = FIXTURES / "form_empty.png"


def _to_top_left(x, y, w, h):
    """reportlab (bottom-left origin) rect -> PyMuPDF (top-left origin) bbox."""
    return [x, form.PAGE_H - y - h, x + w, form.PAGE_H - y]


EXPECTED_BOXES = [_to_top_left(bx, y, bw, form.BOX_H) for _, _, y, bx, bw in form.TEXT_FIELDS]
EXPECTED_CHECKBOXES = [
    _to_top_left(x, y, form.CHECKBOX, form.CHECKBOX)
    for _, y, options in form.CHECK_GROUPS
    for _, x in options
]


def _matches(bbox, expected, tol=2.0):
    return all(abs(a - b) <= tol for a, b in zip(bbox, expected))


def _found(elements, element_type, expected, tol=2.0):
    return any(e["type"] == element_type and _matches(e["bbox"], expected, tol) for e in elements)


@pytest.fixture(scope="module")
def elements():
    return phase1.process_pdf(str(EMPTY_PDF))


def test_fixture_is_small_and_fictional():
    for path in [EMPTY_PDF, FILLED_PDF, EMPTY_PNG]:
        assert path.stat().st_size < 200_000
    import fitz

    with fitz.open(EMPTY_PDF) as doc:
        assert "Fictional" in doc[0].get_text()


def test_extracts_text_and_labels(elements):
    texts = [e["text"] for e in elements if e["type"] == "TEXT"]
    assert "1. Full Name:" in texts
    assert "4. Gender:" in texts


def test_finds_input_boxes_and_checkboxes(elements):
    for box in EXPECTED_BOXES:
        assert _found(elements, "INPUT_FIELD", box), box
    for cb in EXPECTED_CHECKBOXES:
        assert _found(elements, "CHECKBOX", cb), cb


def test_elements_carry_gestalt_fields(elements):
    # every element carries the fields the gestalt processor relies on
    for e in elements:
        assert {"type", "bbox", "center", "page_num"} <= e.keys()


def test_calculate_iou():
    assert phase1.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert phase1.calculate_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert phase1.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_filled_form_keeps_values_as_text():
    elements = phase1.process_pdf(str(FILLED_PDF))
    texts = {e.get("text") for e in elements}
    assert {"Alex Example", "14/07/1988", "DEMO-0042"} <= texts


def test_cli_writes_json(tmp_path):
    out = tmp_path / "elements.json"
    subprocess.run([sys.executable, str(REPO_ROOT / "phase1/phase1.py"), str(EMPTY_PDF), str(out)], check=True,
                   capture_output=True)
    elements = json.loads(out.read_text())
    assert elements and all(e["id"].startswith("element_") for e in elements)


def test_cli_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "form.txt"
    bad.write_text("not a form")
    out = tmp_path / "out.json"
    res = subprocess.run([sys.executable, str(REPO_ROOT / "phase1/phase1.py"), str(bad), str(out)],
                         capture_output=True, text=True)
    assert "Unsupported" in res.stdout
    assert not out.exists()


def test_visualise_elements_draws_png(tmp_path, elements):
    vis = load_module("checkbox_trials/visualise_elements.py", "visualise_elements")
    elements_json = tmp_path / "elements.json"
    elements_json.write_text(json.dumps(elements))
    out = tmp_path / "vis" / "page0.png"
    vis.draw_bboxes(str(EMPTY_PDF), str(elements_json), str(out), 0)
    assert out.exists() and out.stat().st_size > 0
