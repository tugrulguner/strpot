"""StrPot command-line interface."""

from __future__ import annotations

import json
import os
import platform
import zlib
from pathlib import Path
from typing import Annotated

import typer

from strpot import __version__
from strpot.image import DEFAULT_PAGE_SIZE, compile_image, verify_image
from strpot.native_inference import run_prepared_native_model
from strpot.source import HuggingFaceSource

app = typer.Typer(
    help="Compile and inspect bounded, hardware-adaptive CPU model images.",
    no_args_is_help=True,
)
DEFAULT_STORE = Path.home() / ".strpot"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """StrPot compiles existing models for bounded CPU execution."""


@app.command("compile")
def compile_command(
    source: Annotated[Path, typer.Argument(help="Checkpoint or model file")],
    destination: Annotated[Path, typer.Argument(help="Output .strpot directory")],
    page_size: Annotated[
        int, typer.Option(min=1, help="Uncompressed bytes per immutable page")
    ] = DEFAULT_PAGE_SIZE,
) -> None:
    """Create an exact, content-addressed StrPot image."""
    try:
        manifest = compile_image(source, destination, page_size)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(manifest, indent=2))


@app.command()
def verify(
    image: Annotated[Path, typer.Argument(help="StrPot image directory")],
) -> None:
    """Verify every page and the reconstructed source digest."""
    try:
        report = verify_image(image)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zlib.error) as exc:
        typer.echo(f"verification failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2))


@app.command("run")
def run_command(
    source: Annotated[
        str, typer.Argument(help="Public Hugging Face owner/repository identifier")
    ],
    prompt: Annotated[str, typer.Option(help="Prompt to run through StrPot")],
    max_new_tokens: Annotated[
        int, typer.Option(min=1, help="Number of tokens to decode greedily")
    ] = 1,
    store: Annotated[
        Path, typer.Option(help="StrPot-owned compiled model store")
    ] = DEFAULT_STORE,
    page_size: Annotated[
        int, typer.Option(min=1, help="Bytes per immutable model page")
    ] = DEFAULT_PAGE_SIZE,
    threads: Annotated[
        int,
        typer.Option(
            min=0,
            help="Native CPU threads; 0 uses cached complete-model autotune",
        ),
    ] = 0,
    repetitions: Annotated[
        int, typer.Option(min=1, help="Runs used for median inference timing")
    ] = 1,
    token_wave: Annotated[
        bool, typer.Option("--token-wave", help="Use exact n-gram token-wave decoding")
    ] = False,
    max_proposals: Annotated[
        int, typer.Option(min=0, help="Maximum n-gram proposals per token wave")
    ] = 4,
) -> None:
    """Acquire, compile, and run a model through StrPot's native CPU engine."""
    typer.echo(f"StrPot is preparing {source}...", err=True)
    try:
        prepared = HuggingFaceSource().prepare(source, store, page_size=page_size)
        result = run_prepared_native_model(
            prepared,
            prompt,
            max_new_tokens=max_new_tokens,
            threads=threads,
            repetitions=repetitions,
            token_wave=token_wave,
            max_proposals=max_proposals,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"StrPot inference failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "prefill_seconds": result.prefill_seconds,
                "decode_seconds": result.decode_seconds,
                "decode_tokens_per_second": result.decode_tokens_per_second,
                "p50_inter_token_seconds": result.p50_inter_token_seconds,
                "p95_inter_token_seconds": result.p95_inter_token_seconds,
                "end_to_end_tokens_per_second": result.end_to_end_tokens_per_second,
                "kernel_family": result.kernel_family,
                "weight_dtype": result.weight_dtype,
                "threads": result.threads,
                "repetitions": result.repetitions,
                "token_wave": result.token_wave,
                "target_weight_traversals": result.target_weight_traversals,
                "committed_tokens_per_target_weight_traversal": (
                    result.committed_tokens_per_target_weight_traversal
                ),
                "acceptance_lengths": result.acceptance_lengths,
                "traversal_seconds": result.traversal_seconds,
                "rolled_back_tokens": result.rolled_back_tokens,
                "rollback_verified": result.rollback_verified,
                "device": "cpu",
                "engine": "strpot-native",
                "native_checkpoint": str(result.native_checkpoint),
                "revision": prepared.revision,
            },
            indent=2,
        )
    )


@app.command("test-response-atlas")
def test_response_atlas_command(
    source: Annotated[
        str, typer.Argument(help="Public Hugging Face owner/repository identifier")
    ],
    store: Annotated[
        Path, typer.Option(help="StrPot-owned compiled model store")
    ] = DEFAULT_STORE,
    sampled_tokens: Annotated[
        int, typer.Option(min=1, help="Token responses checked for exactness")
    ] = 128,
    iterations: Annotated[
        int, typer.Option(min=1, help="Single-token timing repetitions")
    ] = 30,
) -> None:
    """Test exact layer-zero response-space execution on a real model."""
    from strpot.response_atlas import benchmark_exact_token_atlas

    prepared = HuggingFaceSource().prepare(source, store)
    result = benchmark_exact_token_atlas(
        prepared, sampled_tokens=sampled_tokens, iterations=iterations
    )
    typer.echo(json.dumps(result.__dict__, indent=2))


@app.command("benchmark-pytorch")
def benchmark_pytorch_command(
    source: Annotated[
        str, typer.Argument(help="Public Hugging Face owner/repository identifier")
    ],
    prompt: Annotated[str, typer.Option(help="Prompt used by the reference backend")],
    max_new_tokens: Annotated[
        int, typer.Option(min=1, help="Number of tokens to decode greedily")
    ] = 1,
    store: Annotated[
        Path, typer.Option(help="StrPot-owned compiled model store")
    ] = DEFAULT_STORE,
    cached_pages: Annotated[
        int, typer.Option(min=1, help="Maximum decompressed pages resident at once")
    ] = 2,
    repetitions: Annotated[
        int, typer.Option(min=1, help="Runs used for median reference timing")
    ] = 1,
) -> None:
    """Run the isolated PyTorch benchmark/oracle backend, never production."""
    from strpot.inference import run_prepared_model

    prepared = HuggingFaceSource().prepare(source, store)
    result = run_prepared_model(
        prepared,
        prompt,
        max_new_tokens=max_new_tokens,
        cached_pages=cached_pages,
        repetitions=repetitions,
    )
    typer.echo(
        json.dumps(
            {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "elapsed_seconds": result.elapsed_seconds,
                "tokens_per_second": result.tokens_per_second,
                "engine": "pytorch-reference",
                "revision": prepared.revision,
            },
            indent=2,
        )
    )


@app.command()
def profile() -> None:
    """Print the CPU execution environment visible to StrPot."""
    typer.echo(
        json.dumps(
            {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "python": platform.python_version(),
            },
            indent=2,
        )
    )
