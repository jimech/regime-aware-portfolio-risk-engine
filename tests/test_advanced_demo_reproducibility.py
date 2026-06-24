from pathlib import Path

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)


def test_advanced_demo_package_is_reproducible_with_same_seed(
    tmp_path: Path,
) -> None:
    first = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "first",
        analyst="Jimena Chinchilla",
        random_state=123,
    )
    second = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "second",
        analyst="Jimena Chinchilla",
        random_state=123,
    )

    first_files = _read_package_files(first.export_result.output_dir)
    second_files = _read_package_files(second.export_result.output_dir)

    assert first_files == second_files


def _read_package_files(package_dir: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(package_dir.iterdir())
        if path.is_file()
    }
