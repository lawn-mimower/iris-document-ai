"""prototype1/agent.py: fill-plan generation (Gemini mocked offline, live test marked e2e)."""

import json
import sys
from types import SimpleNamespace

import pytest

from conftest import FIXTURES, gemini_key_available, load_module

agent = load_module("prototype1/agent.py", "prototype1_agent")

EMPTY = [{"id": "element_0", "type": "TEXT", "text": "1. Full Name:", "bbox": [60, 82, 125, 94],
          "center": [92.5, 88], "page_num": 0,
          "neighbors": [{"id": "element_1", "distance": 150, "direction": "right", "alignment": "horizontal"}]},
         {"id": "element_1", "type": "INPUT_FIELD", "text": None, "bbox": [200, 82, 370, 102],
          "center": [285, 92], "page_num": 0, "neighbors": []}]


def _write_inputs(tmp_path):
    paths = {}
    for name, data in {"empty": EMPTY, "golden": EMPTY, "data": {"Full Name": "Jordan Sample"}}.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(data))
        paths[name] = p
    return paths


def _run_main(monkeypatch, paths, out):
    monkeypatch.setattr(sys, "argv", [
        "agent.py",
        "--empty-form-structure", str(paths["empty"]),
        "--golden-sample-structure", str(paths["golden"]),
        "--new-data", str(paths["data"]),
        "--output-plan", str(out),
    ])
    agent.main()


def test_main_saves_valid_plan(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps([{"text": "Jordan Sample", "bbox": [200, 82, 370, 102]}])

    monkeypatch.setattr(agent, "call_llm_api", fake_llm)
    out = tmp_path / "plan.json"
    _run_main(monkeypatch, paths, out)

    assert json.loads(out.read_text()) == [{"text": "Jordan Sample", "bbox": [200, 82, 370, 102]}]
    # the three inputs are embedded in the prompt
    assert "Jordan Sample" in captured["prompt"] and "INPUT_FIELD" in captured["prompt"]


def test_main_keeps_invalid_llm_output_for_debugging(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(agent, "call_llm_api", lambda prompt: "not json")
    out = tmp_path / "plan.json"
    _run_main(monkeypatch, paths, out)
    assert not out.exists()
    assert (tmp_path / "plan_error.txt").read_text() == "not json"


def test_main_reports_missing_input(tmp_path, monkeypatch, capsys):
    paths = _write_inputs(tmp_path)
    paths["data"] = tmp_path / "missing.json"
    monkeypatch.setattr(agent, "call_llm_api", lambda prompt: pytest.fail("LLM must not be called"))
    _run_main(monkeypatch, paths, tmp_path / "plan.json")
    assert "Could not find" in capsys.readouterr().out


def test_call_llm_api_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert agent.call_llm_api("prompt") is None


def test_call_llm_api_uses_configured_model_and_json_mode(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = {}

    class FakeModel:
        def __init__(self, name):
            calls["model"] = name

        def generate_content(self, prompt, generation_config=None):
            calls["mime"] = generation_config.response_mime_type
            return SimpleNamespace(text="[]")

    monkeypatch.setattr(agent.genai, "configure", lambda api_key: calls.setdefault("key", api_key))
    monkeypatch.setattr(agent.genai, "GenerativeModel", FakeModel)
    assert agent.call_llm_api("prompt") == "[]"
    assert calls == {"key": "test-key", "model": agent.GEMINI_MODEL, "mime": "application/json"}


@pytest.mark.e2e
@pytest.mark.skipif(not gemini_key_available(), reason="GEMINI_API_KEY / GOOGLE_API_KEY not set")
def test_live_fill_plan_on_synthetic_form(tmp_path, monkeypatch, capsys):
    phase1 = load_module("phase1/phase1.py", "phase1_e2e")
    gp = load_module("prototype1/gestalt_processor.py", "prototype1_gestalt_e2e")

    structures = {}
    for name in ["empty", "filled"]:
        raw = tmp_path / f"{name}_raw.json"
        elements = phase1.process_pdf(str(FIXTURES / f"form_{name}.pdf"))
        for i, e in enumerate(elements):
            e["id"] = f"element_{i}"
        raw.write_text(json.dumps(elements))
        structures[name] = tmp_path / f"{name}_processed.json"
        gp.process_elements(str(raw), str(structures[name]))

    out = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", [
        "agent.py",
        "--empty-form-structure", str(structures["empty"]),
        "--golden-sample-structure", str(structures["filled"]),
        "--new-data", str(FIXTURES / "form_new_data.json"),
        "--output-plan", str(out),
    ])
    agent.main()
    if not out.exists() and "429" in capsys.readouterr().out:
        pytest.skip("Gemini quota exhausted (HTTP 429)")
    plan = json.loads(out.read_text())
    texts = {item["text"] for item in plan}
    assert "Jordan Sample" in texts
    name_entry = next(item for item in plan if item["text"] == "Jordan Sample")
    # the "Full Name" box sits at y = 82..102 (PDF points, top-left origin)
    assert 75 <= name_entry["bbox"][1] <= 110
