"""End-to-end StrPot inference orchestration."""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

from tokenizers import Tokenizer

from strpot.image import materialize_executable_pages
from strpot.llama import LlamaConfig, LlamaRuntime
from strpot.response_atlas import ExactTokenResponseAtlas
from strpot.source import PreparedModel
from strpot.tensor_store import PagedFile, ResidentTensorStore, SafeTensorStore


def physical_memory_bytes() -> int:
    """Return installed physical memory using the host's page counters."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0


def choose_weight_mode(source_size: int, *, physical_memory: int) -> str:
    """Keep native checkpoint tensors resident when they fit half of system RAM."""
    return "resident" if source_size <= physical_memory // 2 else "paged"


@dataclass(frozen=True)
class InferenceResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    page_resident_bytes: int
    response_atlas_bytes: int = 0
    response_atlas_compile_seconds: float = 0.0
    weight_mode: str = "paged"
    weight_preload_seconds: float = 0.0
    resident_weight_bytes: int = 0
    repetitions: int = 1

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return float("inf")
        return self.generated_tokens / self.elapsed_seconds


def run_prepared_model(
    prepared: PreparedModel,
    prompt: str,
    *,
    max_new_tokens: int,
    cached_pages: int = 2,
    response_atlas: bool = False,
    repetitions: int = 1,
) -> InferenceResult:
    """Run a prepared model entirely through StrPot's CPU execution graph."""
    config = LlamaConfig.from_path(prepared.config_path)
    tokenizer = Tokenizer.from_file(str(prepared.tokenizer_path))
    formatted = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False).ids

    materialize_executable_pages(prepared.weights_image)
    paged = PagedFile(prepared.weights_image, max_cached_pages=cached_pages)
    paged_tensors = SafeTensorStore(paged)
    weight_mode = choose_weight_mode(
        paged.size, physical_memory=physical_memory_bytes()
    )
    preload_started = time.perf_counter()
    if weight_mode == "resident":
        tensors = ResidentTensorStore.preload(paged_tensors)
    else:
        tensors = paged_tensors
    preload_seconds = time.perf_counter() - preload_started
    atlas = None
    atlas_compile_seconds = 0.0
    if response_atlas:
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
        atlas_compile_seconds = time.perf_counter() - compile_started
    runtime = LlamaRuntime(config, tensors, first_layer_atlas=atlas)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    timings = []
    outputs = []
    for _ in range(repetitions):
        started = time.perf_counter()
        outputs.append(runtime.generate(prompt_ids, max_new_tokens=max_new_tokens))
        timings.append(time.perf_counter() - started)
    if any(output != outputs[0] for output in outputs[1:]):
        raise RuntimeError("repeated greedy inference produced inconsistent tokens")
    generated = outputs[0]
    elapsed = statistics.median(timings)
    new_tokens = generated[len(prompt_ids) :]
    return InferenceResult(
        text=tokenizer.decode(new_tokens, skip_special_tokens=True),
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(new_tokens),
        elapsed_seconds=elapsed,
        page_resident_bytes=runtime.page_resident_bytes,
        response_atlas_bytes=atlas.stored_bytes if atlas is not None else 0,
        response_atlas_compile_seconds=atlas_compile_seconds,
        weight_mode=weight_mode,
        weight_preload_seconds=preload_seconds,
        resident_weight_bytes=(
            tensors.resident_bytes if isinstance(tensors, ResidentTensorStore) else 0
        ),
        repetitions=repetitions,
    )
