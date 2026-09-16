"""Compile fixed matrix responses for input-dependent CPU lookup."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from strpot.llama import LlamaConfig
from strpot.source import PreparedModel
from strpot.tensor_store import PagedFile, SafeTensorStore


@dataclass(frozen=True)
class ExactTokenResponseAtlas:
    """Exact first-layer responses indexed by the current input token."""

    responses: dict[str, torch.Tensor]

    @classmethod
    @torch.inference_mode()
    def compile(
        cls,
        embeddings: torch.Tensor,
        norm_weight: torch.Tensor,
        projections: dict[str, torch.Tensor],
        *,
        rms_norm_eps: float,
    ) -> ExactTokenResponseAtlas:
        execution_dtype = embeddings.dtype
        values = embeddings.to(torch.float32)
        normalized = values * torch.rsqrt(
            values.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps
        )
        normalized = normalized.to(execution_dtype) * norm_weight
        responses = {
            name: functional.linear(normalized, weight).contiguous()
            for name, weight in projections.items()
        }
        return cls(responses=responses)

    def lookup(self, projection: str, token_ids: torch.Tensor) -> torch.Tensor:
        return self.responses[projection][token_ids]

    @property
    def stored_bytes(self) -> int:
        return sum(
            tensor.nelement() * tensor.element_size()
            for tensor in self.responses.values()
        )


@dataclass(frozen=True)
class AtlasBenchmark:
    compile_seconds: float
    maximum_absolute_error: float
    baseline_microseconds: float
    atlas_microseconds: float
    speedup: float
    atlas_bytes: int
    original_projection_bytes: int
    sampled_tokens: int


@torch.inference_mode()
def benchmark_exact_token_atlas(
    prepared: PreparedModel,
    *,
    sampled_tokens: int = 128,
    iterations: int = 30,
) -> AtlasBenchmark:
    """Compile and benchmark exact layer-zero Q/K/V responses on a real model."""
    config = LlamaConfig.from_path(prepared.config_path)
    tensors = SafeTensorStore(PagedFile(prepared.weights_image, max_cached_pages=2))
    embeddings = tensors.load("model.embed_tokens.weight")
    norm_weight = tensors.load("model.layers.0.input_layernorm.weight")
    projections = {
        name: tensors.load(f"model.layers.0.self_attn.{name}_proj.weight")
        for name in ("q", "k", "v")
    }

    compile_started = time.perf_counter()
    atlas = ExactTokenResponseAtlas.compile(
        embeddings,
        norm_weight,
        projections,
        rms_norm_eps=config.rms_norm_eps,
    )
    compile_seconds = time.perf_counter() - compile_started

    count = min(sampled_tokens, config.vocab_size)
    token_ids = (torch.arange(count, dtype=torch.long) * 7919 + 17) % config.vocab_size
    selected = embeddings[token_ids]
    selected_fp32 = selected.to(torch.float32)
    normalized = selected_fp32 * torch.rsqrt(
        selected_fp32.pow(2).mean(dim=-1, keepdim=True) + config.rms_norm_eps
    )
    normalized = normalized.to(selected.dtype) * norm_weight
    errors = []
    for name, projection in projections.items():
        expected = functional.linear(normalized, projection)
        actual = atlas.lookup(name, token_ids)
        errors.append(float(torch.max(torch.abs(expected - actual)).item()))

    benchmark_token = token_ids[:1]

    def baseline() -> None:
        values = embeddings[benchmark_token]
        values_fp32 = values.to(torch.float32)
        values = values_fp32 * torch.rsqrt(
            values_fp32.pow(2).mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        values = values.to(embeddings.dtype) * norm_weight
        for projection in projections.values():
            functional.linear(values, projection)

    def response_lookup() -> None:
        for name in projections:
            atlas.lookup(name, benchmark_token)

    baseline()
    response_lookup()
    baseline_timings = []
    atlas_timings = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        baseline()
        baseline_timings.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        response_lookup()
        atlas_timings.append(time.perf_counter_ns() - started)

    baseline_us = statistics.median(baseline_timings) / 1000
    atlas_us = statistics.median(atlas_timings) / 1000
    original_bytes = sum(
        tensor.nelement() * tensor.element_size() for tensor in projections.values()
    )
    return AtlasBenchmark(
        compile_seconds=compile_seconds,
        maximum_absolute_error=max(errors),
        baseline_microseconds=baseline_us,
        atlas_microseconds=atlas_us,
        speedup=baseline_us / atlas_us,
        atlas_bytes=atlas.stored_bytes,
        original_projection_bytes=original_bytes,
        sampled_tokens=count,
    )
