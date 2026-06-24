from pathlib import Path


def test_advanced_package_summary_doc_exists() -> None:
    doc_path = Path("docs/examples/advanced_research_package_summary.md")

    assert doc_path.exists(), f"Missing doc: {doc_path}"


def test_advanced_package_summary_doc_documents_summary_fields() -> None:
    doc = Path("docs/examples/advanced_research_package_summary.md").read_text(
        encoding="utf-8"
    )

    assert "summarize_advanced_research_package" in doc
    assert "memo_title" in doc
    assert "table_count" in doc
    assert "table_names" in doc
    assert "has_factor_significance" in doc
    assert "has_rolling_factor_exposure" in doc
    assert "has_scenario_simulation" in doc
    assert "has_stress_test" in doc


def test_advanced_package_summary_docs_adr_exists() -> None:
    adr = Path("docs/adr/0088-advanced-research-package-summary-docs.md").read_text(
        encoding="utf-8"
    )

    assert "ADR 0088" in adr
    assert "Advanced research package summary documentation" in adr
    assert "Accepted" in adr
