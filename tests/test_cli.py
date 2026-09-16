import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from strpot.cli import app

runner = CliRunner()


def test_help() -> None:
    environment = {"COLUMNS": "120"}
    result = runner.invoke(app, ["--help"], env=environment)
    run_help = runner.invoke(app, ["run", "--help"], env=environment)
    assert result.exit_code == 0
    assert "bounded" in result.stdout
    assert run_help.exit_code == 0
    assert "autotune" in run_help.stdout
    assert "token-wave" in run_help.stdout
    assert "max-proposals" in run_help.stdout


def test_production_cli_does_not_import_pytorch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import strpot.cli; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
    )

    assert result.returncode == 0


def test_compile_then_verify(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"model weights" * 100)
    image = tmp_path / "model.strpot"

    compiled = runner.invoke(
        app, ["compile", str(source), str(image), "--page-size", "64"]
    )
    verified = runner.invoke(app, ["verify", str(image)])

    assert compiled.exit_code == 0
    assert verified.exit_code == 0
    assert '"valid": true' in verified.stdout
