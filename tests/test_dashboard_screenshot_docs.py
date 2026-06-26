from pathlib import Path


def test_dashboard_screenshot_assets_exist() -> None:
    assert Path("docs/assets/dashboard-overview.png").exists()


def test_readme_links_dashboard_screenshot() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/assets/dashboard-overview.png" in readme
    assert "Dashboard preview" in readme
