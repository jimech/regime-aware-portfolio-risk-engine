from pathlib import Path

import pandas as pd

from regime_risk_engine.research.package_manifest import (
    AdvancedPackageManifestError,
    load_advanced_research_package,
    summarize_advanced_research_package,
)

DEFAULT_PACKAGE_DIR = Path("outputs/advanced_demo/package")

_TERMINAL_CSS = """
<style>
    .stApp {
        background-color: #0b0e14;
        color: #d7dce1;
        font-family: "IBM Plex Mono", "SF Mono", Consolas, monospace;
    }
    [data-testid="stSidebar"] {
        background-color: #0f131b;
        border-right: 1px solid #1e2530;
    }
    h1, h2, h3 {
        color: #e8c468 !important;
        font-family: "IBM Plex Mono", "SF Mono", Consolas, monospace;
        letter-spacing: 0.02em;
    }
    [data-testid="stMetric"] {
        background-color: #11161f;
        border: 1px solid #1e2530;
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #8a93a3;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #4fd1a5;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8a93a3;
    }
    .stTabs [aria-selected="true"] {
        color: #e8c468 !important;
    }
    div[data-testid="stMarkdownContainer"] code {
        color: #4fd1a5;
    }
</style>
"""


def _style_numeric_table(table: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Color-code numeric columns like a trading terminal: gains green, losses red."""
    numeric_cols = table.select_dtypes(include="number").columns

    def _signed_color(value: object) -> str:
        if not isinstance(value, int | float) or pd.isna(value):
            return ""
        if value > 0:
            return "color: #4fd1a5"
        if value < 0:
            return "color: #f25f5c"
        return ""

    return table.style.map(_signed_color, subset=numeric_cols)


def run_dashboard(default_package_dir: str | Path = DEFAULT_PACKAGE_DIR) -> None:
    """Run the Streamlit dashboard for an advanced research package."""
    import streamlit as st

    st.set_page_config(
        page_title="Regime-Aware Portfolio Research",
        layout="wide",
        page_icon="📈",
    )
    st.markdown(_TERMINAL_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### REGIME-AWARE RESEARCH TERMINAL")
        package_dir = st.text_input(
            "Package directory",
            value=str(default_package_dir),
        )
        st.caption("Loads a generated `manifest.json` research package.")

    st.title("Regime-Aware Portfolio Research Dashboard")
    st.caption(
        "Live inspection of an advanced research package: memo, tables, diagnostics."
    )

    try:
        package = load_advanced_research_package(package_dir)
    except AdvancedPackageManifestError as exc:
        st.error(f"Could not load package: {exc}")
        st.stop()

    summary = summarize_advanced_research_package(package)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tables", summary.table_count)
    col2.metric(
        "Factor Significance",
        "✓" if summary.has_factor_significance else "—",
    )
    col3.metric(
        "Rolling Exposure",
        "✓" if summary.has_rolling_factor_exposure else "—",
    )
    col4.metric(
        "Scenario Simulation",
        "✓" if summary.has_scenario_simulation else "—",
    )

    st.divider()

    memo_tab, tables_tab = st.tabs(["📄 Research Memo", "📊 Tables"])

    with memo_tab:
        st.caption(f"**{summary.memo_title}**")
        st.markdown(package.memo)

    with tables_tab:
        table_name = st.selectbox(
            "Select table",
            options=summary.table_names,
        )
        table = package.tables[table_name]
        st.dataframe(_style_numeric_table(table), width="stretch")


if __name__ == "__main__":
    run_dashboard()
