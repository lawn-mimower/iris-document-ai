"""Gestalt pre-processor (prototype1/gestalt_processor.py)."""

import json

import pytest

from conftest import load_module

GESTALT = load_module("prototype1/gestalt_processor.py", "prototype1_gestalt")


def el(id_, type_, bbox, text=None, page=0):
    x0, y0, x1, y1 = bbox
    return {"id": id_, "type": type_, "text": text, "bbox": bbox,
            "center": [(x0 + x1) / 2, (y0 + y1) / 2], "page_num": page}


SAMPLE = [
    el("t1", "TEXT", [10, 10, 40, 20], "Full"),
    el("t2", "TEXT", [45, 10, 80, 20], "Name:"),       # 5pt gap -> merged with t1
    el("t3", "TEXT", [10, 60, 50, 70], "Gender:"),     # different line -> separate
    el("f1", "INPUT_FIELD", [100, 8, 150, 22]),
    el("f2", "INPUT_FIELD", [152, 8, 200, 22]),         # 2pt gap -> merged with f1
    el("c1", "CHECKBOX", [100, 60, 105, 65]),
    el("t4", "TEXT", [10, 10, 40, 20], "Other page", page=1),
]


@pytest.fixture
def gp():
    return GESTALT


def test_merge_text_fragments(gp):
    merged = gp.merge_text_fragments(SAMPLE)
    texts = sorted(e["text"] for e in merged if e["type"] == "TEXT")
    assert texts == ["Full Name:", "Gender:", "Other page"]
    full = next(e for e in merged if e.get("text") == "Full Name:")
    assert full["bbox"] == [10, 10, 80, 20]
    assert full["id"].startswith("merged_text_")


def test_merge_input_fields(gp):
    merged = gp.merge_input_fields(SAMPLE)
    fields = [e for e in merged if e["type"] == "INPUT_FIELD"]
    assert len(fields) == 1 and fields[0]["bbox"] == [100, 8, 200, 22]
    # the checkbox is on another line and is kept as-is
    assert any(e["id"] == "c1" and e["type"] == "CHECKBOX" for e in merged)


def test_direction_and_alignment(gp):
    assert gp.calculate_direction([0, 0], [20, 0]) == "right"
    assert gp.calculate_direction([0, 0], [-20, 30]) == "below-left"
    assert gp.calculate_direction([0, 0], [2, 2]) == "same-position"
    assert gp.calculate_alignment({"center": [0, 0]}, {"center": [50, 3]}) == "horizontal"
    assert gp.calculate_alignment({"center": [0, 0]}, {"center": [3, 50]}) == "vertical"
    assert gp.calculate_distance([0, 0], [3, 4]) == 5.0


def test_neighbors_stay_on_page_and_respect_limit(gp):
    enriched = gp.calculate_relationships(SAMPLE, max_neighbors=2)
    by_id = {e["id"]: e for e in enriched}
    assert len(by_id["t1"]["neighbors"]) == 2
    assert all(n["id"] != "t4" for n in by_id["t1"]["neighbors"])
    assert by_id["t4"]["neighbors"] == []
    assert by_id["t1"]["neighbors"][0]["id"] == "t2"


def test_process_elements_end_to_end(gp, tmp_path):
    src, dst = tmp_path / "raw.json", tmp_path / "processed.json"
    src.write_text(json.dumps(SAMPLE))
    gp.process_elements(str(src), str(dst), max_neighbors=3)
    out = json.loads(dst.read_text())
    assert len(out) == 5  # 3 text groups + 1 merged field + 1 checkbox
    assert all(len(e["neighbors"]) <= 3 for e in out)


def test_invalid_json_raises_decode_error(gp, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(json.JSONDecodeError, match="Invalid JSON"):
        gp.load_elements(str(bad))


def test_non_list_json_rejected(gp, tmp_path):
    bad = tmp_path / "obj.json"
    bad.write_text('{"a": 1}')
    with pytest.raises(ValueError):
        gp.load_elements(str(bad))


def test_cli_max_neighbors(gp, tmp_path, monkeypatch):
    src, dst = tmp_path / "raw.json", tmp_path / "processed.json"
    src.write_text(json.dumps(SAMPLE))
    monkeypatch.setattr("sys.argv", ["gestalt_processor.py", str(src), str(dst), "--max-neighbors", "1"])
    assert gp.main() == 0
    assert max(len(e["neighbors"]) for e in json.loads(dst.read_text())) == 1
