from regime_risk_engine import dashboard


def test_dashboard_exposes_default_package_dir() -> None:
    assert str(dashboard.DEFAULT_PACKAGE_DIR) == "outputs/advanced_demo/package"


def test_dashboard_exposes_run_dashboard_function() -> None:
    assert callable(dashboard.run_dashboard)
