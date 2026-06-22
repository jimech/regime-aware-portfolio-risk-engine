from pathlib import Path


def test_rolling_factor_exposure_example_doc_exists() -> None:
    doc_path = Path("docs/examples/rolling_factor_exposure_example.md")

    assert doc_path.exists(), f"Missing doc: {doc_path}"


def test_rolling_factor_exposure_example_doc_documents_cli_usage() -> None:
    doc = Path("docs/examples/rolling_factor_exposure_example.md").read_text(
        encoding="utf-8"
    )

    assert "export-rolling-factor-exposure" in doc
    assert "rolling_factor_exposures.csv" in doc
    assert "rolling_factor_exposure_summary.csv" in doc
    assert "--strategy-returns" in doc
    assert "--factor-returns" in doc
    assert "equity_beta" in doc
    assert "dominant_factor" in doc


def test_rolling_factor_exposure_example_adr_exists() -> None:
    adr = Path("docs/adr/0074-rolling-factor-exposure-example-docs.md").read_text(
        encoding="utf-8"
    )

    assert "ADR 0074" in adr
    assert "Rolling factor exposure example documentation" in adr
    assert "Accepted" in adr
