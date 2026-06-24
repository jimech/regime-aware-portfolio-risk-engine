from pathlib import Path


def test_readme_describes_project_demonstrations() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "What this project demonstrates" in readme
    assert "Regime detection" in readme
    assert "Rolling factor exposure diagnostics" in readme
    assert "Factor significance testing" in readme
    assert "Streamlit dashboard review interface" in readme


def test_readme_links_primary_review_artifacts() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/reviewer-guide.md" in readme
    assert "docs/thesis-analysis.md" in readme
    assert "docs/examples/advanced_research_memo_example.md" in readme
    assert "docs/examples/rolling_factor_exposure_example.md" in readme
    assert "docs/examples/factor_significance_example.md" in readme
