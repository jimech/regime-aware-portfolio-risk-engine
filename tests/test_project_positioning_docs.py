from pathlib import Path


def test_readme_describes_project_thesis() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Regime-Aware Portfolio Risk Engine" in readme
    assert "Can changing market regimes be detected" in readme
    assert "System Architecture" in readme
    assert "run-advanced-demo" in readme
    assert "mermaid" in readme
    assert "Disclaimer" in readme


def test_readme_documents_advanced_research_outputs() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    expected_outputs = [
        "advanced_research_memo.md",
        "regime_intelligence_profile.csv",
        "stress_test_summary.csv",
        "factor_exposure.csv",
        "scenario_terminal_summary.csv",
    ]

    for output in expected_outputs:
        assert output in readme


def test_thesis_analysis_document_exists() -> None:
    thesis = Path("docs/thesis-analysis.md").read_text(encoding="utf-8")

    assert "Thesis Analysis" in thesis
    assert "Research Question" in thesis
    assert "Regime Detection" in thesis
    assert "Dynamic Allocation" in thesis
    assert "Conclusion" in thesis


def test_readme_positioning_adr_exists() -> None:
    adr = Path("docs/adr/0065-readme-and-thesis-positioning.md").read_text(
        encoding="utf-8"
    )

    assert "ADR 0065" in adr
    assert "README and thesis positioning" in adr
    assert "Accepted" in adr
