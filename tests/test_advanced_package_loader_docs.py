from pathlib import Path


def test_advanced_package_loader_doc_exists() -> None:
    doc_path = Path("docs/examples/advanced_research_package_loader.md")

    assert doc_path.exists(), f"Missing doc: {doc_path}"


def test_advanced_package_loader_doc_documents_loader_usage() -> None:
    doc = Path("docs/examples/advanced_research_package_loader.md").read_text(
        encoding="utf-8"
    )

    assert "load_advanced_research_package" in doc
    assert "manifest.json" in doc
    assert "package.memo" in doc
    assert "package.table_names()" in doc
    assert "factor_significance" in doc
    assert "rolling_factor_exposures" in doc
    assert "pandas DataFrame" in doc


def test_advanced_package_loader_docs_adr_exists() -> None:
    adr = Path("docs/adr/0086-advanced-research-package-loader-docs.md").read_text(
        encoding="utf-8"
    )

    assert "ADR 0086" in adr
    assert "Advanced research package loader documentation" in adr
    assert "Accepted" in adr
