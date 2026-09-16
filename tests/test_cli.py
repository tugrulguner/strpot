import subprocess
import sys
from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from strpot.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    run_help = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "bounded" in result.stdout
    assert run_help.exit_code == 0
    assert "autotune" in run_help.stdout
    assert "token-wave" in run_help.stdout
    run_command = get_command(app).commands["run"]
    assert any("--max-proposals" in parameter.opts for parameter in run_command.params)


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
