from pathlib import Path

README_PATH = Path("README.md")
RELEASE_CHECKLIST_PATH = Path("docs/release-checklist.md")
VERSION_FILE_PATH = Path("src/regime_risk_engine/__init__.py")


def test_release_checklist_exists() -> None:
    assert RELEASE_CHECKLIST_PATH.exists()


def test_release_checklist_includes_quality_suite() -> None:
    content = RELEASE_CHECKLIST_PATH.read_text(encoding="utf-8")

    assert "ruff format ." in content
    assert "ruff check ." in content
    assert "mypy src" in content
    assert "pytest --cov=regime_risk_engine --cov-report=term-missing" in content


def test_release_checklist_mentions_semantic_versioning() -> None:
    content = RELEASE_CHECKLIST_PATH.read_text(encoding="utf-8")

    assert "semantic versioning" in content
    assert "MAJOR.MINOR.PATCH" in content


def test_release_checklist_mentions_annotated_tags() -> None:
    content = RELEASE_CHECKLIST_PATH.read_text(encoding="utf-8")

    assert "git tag -a" in content
    assert "git push origin v0.1.0" in content


def test_readme_links_release_checklist() -> None:
    content = README_PATH.read_text(encoding="utf-8")

    assert "docs/release-checklist.md" in content


def test_version_file_defines_package_version() -> None:
    content = VERSION_FILE_PATH.read_text(encoding="utf-8")

    assert "__version__" in content
