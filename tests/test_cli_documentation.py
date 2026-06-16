from pathlib import Path

from regime_risk_engine.cli import build_parser

CLI_DOC_PATH = Path("docs/cli.md")

REQUIRED_COMMANDS = [
    "version",
    "healthcheck",
    "inspect-config",
    "create-demo-report-inputs",
    "export-report",
    "run-demo-report",
]


def test_cli_documentation_exists() -> None:
    assert CLI_DOC_PATH.exists()


def test_cli_documentation_includes_supported_commands() -> None:
    content = CLI_DOC_PATH.read_text(encoding="utf-8")

    for command in REQUIRED_COMMANDS:
        assert command in content


def test_cli_documentation_includes_module_invocation() -> None:
    content = CLI_DOC_PATH.read_text(encoding="utf-8")

    assert "python -m regime_risk_engine" in content


def test_cli_documentation_mentions_editable_install() -> None:
    content = CLI_DOC_PATH.read_text(encoding="utf-8")

    assert "pip install -e ." in content


def test_cli_parser_exposes_documented_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    for command in REQUIRED_COMMANDS:
        assert command in help_text
