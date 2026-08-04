"""Notebook API-drift guards (Part G).

Notebooks are not executed under pytest (no GPU), so these are static guards:
every code cell must parse, and the shipped colab notebooks must use the current
API (no by-string experiment targets, no removed TorchRL/trainer/algorithms.py
references). The REINFORCE notebook (custom-algorithm runtime is not yet wired on
the rsl-rl backend) and track_designer (user WIP) are intentionally not held to
the colab API guard — see the Part G notes.
"""

import ast
import json
import pathlib

import pytest

NB_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
ALL = sorted(NB_DIR.glob("*.ipynb"))
COLAB = [p for p in ALL if p.name.startswith("deepracer_genesis_colab")]


def _code_cells(path):
    nb = json.loads(path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _all_source(path):
    nb = json.loads(path.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def _is_shell_cell(src):
    """A cell with any shell (``!``) / magic (``%``) line — skipped by the parser.

    Stripping such lines is unreliable (multi-line ``!pip install \\`` blocks),
    and these cells carry no API usage; the pure-Python cells are what guard drift.
    """
    return any(ln.lstrip().startswith(("!", "%")) or "get_ipython" in ln
               for ln in src.splitlines())


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_notebook_code_cells_parse(path):
    for src in _code_cells(path):
        if _is_shell_cell(src):
            continue
        # allow top-level await (colab cells sometimes use it)
        compile(src, f"<{path.name}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.parametrize("path", COLAB, ids=lambda p: p.name)
def test_colab_notebooks_use_current_api(path):
    src = _all_source(path)
    assert 'rollout_video("' not in src, "by-string target: build() rejects str; pass the class"
    assert "TorchRL" not in src, "stale TorchRL reference; the backend is rsl-rl"
    assert "experiment/algorithms.py" not in src, "removed module reference"
    assert "experiment.trainer" not in src, "removed module reference"
