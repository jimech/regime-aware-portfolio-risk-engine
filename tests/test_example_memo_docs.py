from pathlib import Path


def test_example_memo_includes_factor_diagnostics_sections() -> None:
    memo = Path("docs/examples/advanced_research_memo_example.md").read_text(
        encoding="utf-8"
    )

    assert "## Rolling Factor Exposure Analysis" in memo
    assert "## Factor Significance Analysis" in memo
    assert "Regression R-squared" in memo
    assert "P-Value" in memo
