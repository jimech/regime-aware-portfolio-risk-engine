from pathlib import Path

METHODOLOGY_DOC_PATH = Path("docs/methodology/rolling-factor-exposure.md")


def _read_methodology_doc() -> str:
    return METHODOLOGY_DOC_PATH.read_text(encoding="utf-8")


def test_rolling_factor_exposure_methodology_doc_exists() -> None:
    assert METHODOLOGY_DOC_PATH.exists(), (
        f"Missing methodology doc: {METHODOLOGY_DOC_PATH}"
    )


def test_rolling_factor_exposure_methodology_doc_explains_key_outputs() -> None:
    doc = _read_methodology_doc()

    required_terms = [
        "Rolling factor exposure",
        "ordinary least squares regression",
        "equity_beta",
        "alpha",
        "r_squared",
        "residual_volatility",
        "dominant_factor",
        "descriptive, not predictive",
    ]

    for term in required_terms:
        assert term in doc


def test_rolling_factor_exposure_methodology_doc_links_to_project_thesis() -> None:
    doc = _read_methodology_doc()

    assert "project thesis" in doc.lower()
    assert "change its market risk profile over time" in doc
