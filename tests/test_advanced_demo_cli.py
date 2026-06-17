from pathlib import Path

from regime_risk_engine.cli import main


def test_cli_create_advanced_demo_inputs(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "demo_inputs"

    exit_code = main(
        [
            "create-advanced-demo-inputs",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Advanced research demo inputs created successfully" in captured.out
    assert (output_dir / "prices.csv").exists()
    assert (output_dir / "static_weights.csv").exists()
    assert (output_dir / "regime_policy.csv").exists()
    assert (output_dir / "stress_periods.csv").exists()
    assert (output_dir / "factor_returns.csv").exists()


def test_cli_create_advanced_demo_inputs_respects_no_overwrite(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "demo_inputs"

    first_exit_code = main(
        [
            "create-advanced-demo-inputs",
            "--output-dir",
            str(output_dir),
        ]
    )
    second_exit_code = main(
        [
            "create-advanced-demo-inputs",
            "--output-dir",
            str(output_dir),
            "--no-overwrite",
        ]
    )

    captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 1
    assert "Advanced demo input creation failed" in captured.out
    assert "overwrite=False" in captured.out


def test_cli_create_advanced_demo_inputs_rejects_file_output_path(
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "not_a_directory.txt"
    output_path.write_text("already exists", encoding="utf-8")

    exit_code = main(
        [
            "create-advanced-demo-inputs",
            "--output-dir",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Advanced demo input creation failed" in captured.out
    assert "not a directory" in captured.out
