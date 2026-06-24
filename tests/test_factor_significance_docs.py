from pathlib import Path


def test_factor_significance_example_doc_exists() -> None:
    doc_path = Path("docs/examples/factor_significance_example.md")

    assert doc_path.exists(), f"Missing doc: {doc_path}"


def test_factor_significance_example_doc_documents_output_columns() -> None:
    doc = Path("docs/examples/factor_significance_example.md").read_text(
        encoding="utf-8"
    )

    assert "factor_significance.csv" in doc
    assert "beta" in doc
    assert "standard_error" in doc
    assert "t_stat" in doc
    assert "p_value" in doc
    assert "significant" in doc
    assert "statistically meaningful" in doc


def test_factor_significance_example_adr_exists() -> None:
    adr = Path("docs/adr/0077-factor-significance-example-docs.md").read_text(
        encoding="utf-8"
    )

    assert "ADR 0077" in adr
    assert "Factor significance example documentation" in adr
    assert "Accepted" in adr
