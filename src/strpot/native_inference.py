"""Production inference through StrPot's owned native CPU engine."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from strpot.native import NativeLlamaEngine
from strpot.source import PreparedModel


@dataclass(frozen=True)
class NativeInferenceResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    decode_tokens_per_second: float
    p50_inter_token_seconds: float
    p95_inter_token_seconds: float
    end_to_end_tokens_per_second: float
    kernel_family: str
    weight_dtype: str
    threads: int
    repetitions: int
    native_checkpoint: Path
    token_wave: bool
    target_weight_traversals: int
    committed_tokens_per_target_weight_traversal: float
    acceptance_lengths: tuple[int, ...]
    traversal_seconds: tuple[float, ...]
    rolled_back_tokens: int
    rollback_verified: bool


def percentile(values: list[float], percentage: float) -> float:
    """Return a linearly interpolated percentile for recorded latencies."""
    if not values:
        return 0.0
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentage / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def run_prepared_native_model(
    prepared: PreparedModel,
    prompt: str,
    *,
    max_new_tokens: int,
    threads: int,
    repetitions: int = 1,
    token_wave: bool = False,
    max_proposals: int = 4,
) -> NativeInferenceResult:
    """Generate entirely inside StrPot's dependency-free native CPU core."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    tokenizer = Tokenizer.from_file(str(prepared.tokenizer_path))
    formatted = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False).ids
    engine = NativeLlamaEngine(
        config_path=prepared.config_path,
        weights_image=prepared.weights_image,
    )

    outputs = []
    prefill_timings = []
    decode_timings = []
    inter_token_timings: list[float] = []
    reports = []
    for _ in range(repetitions):
        if token_wave:
            report = engine.generate_token_wave(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                max_proposals=max_proposals,
                threads=threads,
            )
        else:
            report = engine.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                threads=threads,
            )
        reports.append(report)
        outputs.append(report.generated_token_ids)
        prefill_timings.append(report.prefill_seconds)
        decode_timings.append(report.decode_seconds)
        inter_token_timings.extend(getattr(report, "inter_token_seconds", ()))
    if any(output != outputs[0] for output in outputs[1:]):
        raise RuntimeError("native greedy inference produced inconsistent tokens")

    generated = outputs[0]
    prefill_seconds = statistics.median(prefill_timings)
    decode_seconds = statistics.median(decode_timings)
    decoded_after_first = max(0, len(generated) - 1)
    decode_throughput = (
        decoded_after_first / decode_seconds
        if decoded_after_first and decode_seconds > 0
        else 0.0
    )
    total_seconds = prefill_seconds + decode_seconds
    end_to_end = len(generated) / total_seconds if total_seconds > 0 else 0.0
    return NativeInferenceResult(
        text=tokenizer.decode(list(generated), skip_special_tokens=True),
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=decode_throughput,
        p50_inter_token_seconds=percentile(inter_token_timings, 50),
        p95_inter_token_seconds=percentile(inter_token_timings, 95),
        end_to_end_tokens_per_second=end_to_end,
        kernel_family=reports[0].kernel_family,
        weight_dtype=reports[0].weight_dtype,
        threads=reports[0].threads,
        repetitions=repetitions,
        native_checkpoint=engine.checkpoint,
        token_wave=token_wave,
        target_weight_traversals=int(
            statistics.median(
                [getattr(report, "target_weight_traversals", 0) for report in reports]
            )
        ),
        committed_tokens_per_target_weight_traversal=statistics.median(
            [
                getattr(report, "committed_tokens_per_target_weight_traversal", 0.0)
                for report in reports
            ]
        ),
        acceptance_lengths=getattr(reports[0], "acceptance_lengths", ()),
        traversal_seconds=getattr(reports[0], "traversal_seconds", ()),
        rolled_back_tokens=getattr(reports[0], "rolled_back_tokens", 0),
        rollback_verified=getattr(reports[0], "rollback_verified", True),
    )
