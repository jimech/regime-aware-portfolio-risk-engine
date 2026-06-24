from pathlib import Path


def test_manifest_example_doc_exists() -> None:
    doc_path = Path("docs/examples/advanced_research_package_manifest.md")

    assert doc_path.exists(), f"Missing doc: {doc_path}"


def test_manifest_example_doc_documents_manifest_fields() -> None:
    doc = Path("docs/examples/advanced_research_package_manifest.md").read_text(
        encoding="utf-8"
    )

    assert "manifest.json" in doc
    assert '"memo"' in doc
    assert '"tables"' in doc
    assert "advanced_research_memo.md" in doc
    assert "factor_significance.csv" in doc
    assert "rolling_factor_exposures.csv" in doc


def test_manifest_docs_adr_exists() -> None:
    adr = Path("docs/adr/0082-advanced-research-package-manifest-docs.md").read_text(
        encoding="utf-8"
    )

    assert "ADR 0082" in adr
    assert "Advanced research package manifest documentation" in adr
    assert "Accepted" in adr
