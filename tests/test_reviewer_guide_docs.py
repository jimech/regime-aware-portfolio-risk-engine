from pathlib import Path


def test_reviewer_guide_exists() -> None:
    guide = Path("docs/reviewer-guide.md")

    assert guide.exists()


def test_reviewer_guide_documents_review_workflow() -> None:
    guide = Path("docs/reviewer-guide.md").read_text(encoding="utf-8")

    assert "run-advanced-demo" in guide
    assert "inspect-advanced-package" in guide
    assert "streamlit run src/regime_risk_engine/dashboard.py" in guide
    assert "pytest --cov=regime_risk_engine" in guide


def test_readme_links_reviewer_guide() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/reviewer-guide.md" in readme
    assert "Quick reviewer workflow" in readme
    assert "run-advanced-demo" in readme
    assert "inspect-advanced-package" in readme
