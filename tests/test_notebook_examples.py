import json
from pathlib import Path
from typing import Any


def test_advanced_demo_notebook_exists() -> None:
    notebook_path = Path("notebooks/advanced_demo_walkthrough.ipynb")

    assert notebook_path.exists(), f"Missing notebook: {notebook_path}"


def test_advanced_demo_notebook_mentions_research_workflow() -> None:
    notebook_path = Path("notebooks/advanced_demo_walkthrough.ipynb")
    notebook: dict[str, Any] = json.loads(notebook_path.read_text(encoding="utf-8"))

    source_text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "Advanced Regime-Aware Portfolio Research Walkthrough" in source_text
    assert "Can changing market regimes be detected" in source_text
    assert "run_advanced_research_demo_workflow" in source_text
    assert "advanced memo" in source_text.lower()


def test_examples_readme_exists() -> None:
    readme_path = Path("docs/examples/README.md")

    assert readme_path.exists(), f"Missing examples README: {readme_path}"

    readme = readme_path.read_text(encoding="utf-8")

    assert "Notebook walkthrough" in readme
    assert "run-advanced-demo" in readme
    assert "advanced_research_memo.md" in readme


def test_notebook_walkthrough_adr_exists() -> None:
    adr_path = Path("docs/adr/0066-notebook-walkthrough.md")

    assert adr_path.exists(), f"Missing ADR: {adr_path}"

    adr = adr_path.read_text(encoding="utf-8")

    assert "ADR 0066" in adr
    assert "Advanced demo notebook walkthrough" in adr
    assert "Accepted" in adr
