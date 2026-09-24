"""Notebooks are kept as reference material; only check that they are valid notebook JSON."""

import json

import pytest

from conftest import REPO_ROOT

NOTEBOOKS = sorted(p for p in REPO_ROOT.rglob("*.ipynb") if ".venv" not in p.parts)


def test_notebooks_found():
    assert len(NOTEBOOKS) >= 20


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[str(p.relative_to(REPO_ROOT)) for p in NOTEBOOKS])
def test_notebook_is_valid_json(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown", "raw"}
        assert isinstance(cell["source"], (list, str))
