from pathlib import Path

README_PATH = Path("README.md")
QUALITY_CHECKLIST_PATH = Path("docs/quality-checklist.md")
CI_WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def test_readme_includes_ci_badge() -> None:
    content = README_PATH.read_text(encoding="utf-8")

    assert "actions/workflows/ci.yml/badge.svg" in content
    assert "actions/workflows/ci.yml" in content


def test_readme_links_quality_checklist() -> None:
    content = README_PATH.read_text(encoding="utf-8")

    assert "docs/quality-checklist.md" in content


def test_quality_checklist_exists() -> None:
    assert QUALITY_CHECKLIST_PATH.exists()


def test_quality_checklist_includes_required_commands() -> None:
    content = QUALITY_CHECKLIST_PATH.read_text(encoding="utf-8")

    assert "ruff format ." in content
    assert "ruff check ." in content
    assert "ruff format --check ." in content
    assert "mypy src" in content
    assert "pytest --cov=regime_risk_engine --cov-report=term-missing" in content


def test_quality_checklist_mentions_ci_expectations() -> None:
    content = QUALITY_CHECKLIST_PATH.read_text(encoding="utf-8")

    assert "GitHub Actions" in content
    assert "CI passes" in content


def test_ci_workflow_exists() -> None:
    assert CI_WORKFLOW_PATH.exists()
