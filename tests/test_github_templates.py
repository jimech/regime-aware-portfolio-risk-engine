from pathlib import Path

BUG_REPORT_TEMPLATE = Path(".github/ISSUE_TEMPLATE/bug_report.md")
FEATURE_REQUEST_TEMPLATE = Path(".github/ISSUE_TEMPLATE/feature_request.md")
ISSUE_TEMPLATE_CONFIG = Path(".github/ISSUE_TEMPLATE/config.yml")
PULL_REQUEST_TEMPLATE = Path(".github/pull_request_template.md")


def test_bug_report_template_exists() -> None:
    assert BUG_REPORT_TEMPLATE.exists()


def test_feature_request_template_exists() -> None:
    assert FEATURE_REQUEST_TEMPLATE.exists()


def test_issue_template_config_exists() -> None:
    assert ISSUE_TEMPLATE_CONFIG.exists()


def test_pull_request_template_exists() -> None:
    assert PULL_REQUEST_TEMPLATE.exists()


def test_bug_report_template_includes_reproduction_sections() -> None:
    content = BUG_REPORT_TEMPLATE.read_text(encoding="utf-8")

    assert "Steps to reproduce" in content
    assert "Expected behavior" in content
    assert "Actual behavior" in content
    assert "Error output" in content
    assert "Environment" in content


def test_feature_request_template_includes_acceptance_criteria() -> None:
    content = FEATURE_REQUEST_TEMPLATE.read_text(encoding="utf-8")

    assert "Motivation" in content
    assert "Proposed behavior" in content
    assert "Acceptance criteria" in content


def test_pull_request_template_includes_quality_commands() -> None:
    content = PULL_REQUEST_TEMPLATE.read_text(encoding="utf-8")

    assert "ruff format ." in content
    assert "ruff check ." in content
    assert "ruff format --check ." in content
    assert "mypy src" in content
    assert "pytest --cov=regime_risk_engine --cov-report=term-missing" in content


def test_pull_request_template_mentions_documentation() -> None:
    content = PULL_REQUEST_TEMPLATE.read_text(encoding="utf-8")

    assert "README updated" in content
    assert "Methodology docs updated" in content
    assert "ADR added or updated" in content
    assert "CLI docs updated" in content
