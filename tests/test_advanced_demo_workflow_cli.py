from pathlib import Path

from regime_risk_engine.cli import main


def test_cli_run_advanced_demo(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "advanced_demo"

    exit_code = main(
        [
            "run-advanced-demo",
            "--output-dir",
            str(output_dir),
            "--feature-window",
            "10",
            "--scenario-horizon",
            "5",
            "--scenario-simulations",
            "25",
            "--random-state",
            "7",
            "--analyst",
            "Jimena Chinchilla",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Advanced demo workflow completed successfully" in captured.out
    assert (output_dir / "inputs" / "prices.csv").exists()
    assert (output_dir / "package" / "advanced_research_memo.md").exists()


def test_cli_run_advanced_demo_json(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "advanced_demo"

    exit_code = main(
        [
            "run-advanced-demo",
            "--output-dir",
            str(output_dir),
            "--feature-window",
            "10",
            "--scenario-horizon",
            "5",
            "--scenario-simulations",
            "25",
            "--random-state",
            "7",
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"output_dir"' in captured.out
    assert '"memo_path"' in captured.out
    assert "advanced_research_memo.md" in captured.out


def test_cli_run_advanced_demo_respects_no_overwrite(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "advanced_demo"

    first_exit_code = main(
        [
            "run-advanced-demo",
            "--output-dir",
            str(output_dir),
            "--feature-window",
            "10",
            "--scenario-horizon",
            "5",
            "--scenario-simulations",
            "25",
        ]
    )
    second_exit_code = main(
        [
            "run-advanced-demo",
            "--output-dir",
            str(output_dir),
            "--feature-window",
            "10",
            "--scenario-horizon",
            "5",
            "--scenario-simulations",
            "25",
            "--no-overwrite",
        ]
    )

    captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 1
    assert "Advanced demo workflow failed" in captured.out
    assert "overwrite=False" in captured.out
