from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strpot.image import MAX_PAGE_SIZE

_SOURCE = Path(__file__).with_name("_native") / "strpot_native.cpp"
_DTYPE_SIZES = {"BF16": 2, "F32": 4}
_MAX_TENSOR_RANK = 8
_MAX_TENSOR_DIMENSION = (1 << 31) - 1
_DESCRIPTOR_CONFIG_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
    "vocab_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "bos_token_id",
    "eos_token_id",
)


@dataclass(frozen=True)
class NativeGenerationResult:
    generated_token_ids: tuple[int, ...]
    engine: str
    kernel_family: str
    weight_dtype: str
    prefill_seconds: float
    prefill_matrix_passes: int
    runtime_tensor_lookups: int
    decode_parallel_dispatches: int
    rope_table_entries: int
    attention_workspace_floats: int
    decode_seconds: float
    decode_tokens_per_second: float
    inter_token_seconds: tuple[float, ...]
    threads: int
    final_logits_hash: str
    final_kv_hash: str
    final_kv_lengths: tuple[int, ...]
    frontier_logits_hashes: tuple[str, ...]
    frontier_kv_hashes: tuple[str, ...]


@dataclass(frozen=True)
class NativeTokenWaveResult:
    generated_token_ids: tuple[int, ...]
    engine: str
    kernel_family: str
    weight_dtype: str
    prefill_seconds: float
    decode_seconds: float
    target_weight_traversals: int
    committed_decode_tokens: int
    committed_tokens_per_target_weight_traversal: float
    acceptance_lengths: tuple[int, ...]
    traversal_seconds: tuple[float, ...]
    rolled_back_tokens: int
    speculative_traversals: int
    fallback_traversals: int
    transactional_snapshot_bytes_copied: int
    rollback_verified: bool
    final_logits_hash: str
    final_kv_hash: str
    final_kv_lengths: tuple[int, ...]
    frontier_logits_hashes: tuple[str, ...]
    frontier_kv_hashes: tuple[str, ...]
    threads: int


@dataclass(frozen=True)
class NativeBatchGenerationResult:
    generated_token_ids: tuple[tuple[int, ...], ...]
    engine: str
    kernel_family: str
    weight_dtype: str
    batch_size: int
    prefill_seconds: float
    decode_seconds: float
    decode_matrix_passes: int
    decode_parallel_dispatches: int
    activation_panel_width: int
    aggregate_decode_tokens_per_second: float
    threads: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_native_engine(cache_root: Path | None = None) -> Path:
    """Compile and cache StrPot's dependency-free native CPU engine."""
    source = _SOURCE.read_bytes()
    key = hashlib.sha256(
        source + platform.machine().encode() + platform.system().encode()
    ).hexdigest()[:16]
    base = (
        cache_root.resolve()
        if cache_root is not None
        else Path.home() / ".strpot" / "native"
    )
    root = base / key
    root.mkdir(parents=True, exist_ok=True)
    executable = root / "strpot-native"
    if executable.exists():
        return executable

    temporary = root / f"strpot-native.{os.getpid()}.tmp"
    compiler = os.environ.get("CXX", "c++")
    command = [
        compiler,
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-pthread",
        str(_SOURCE),
        "-o",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to compile StrPot native engine:\n"
            + completed.stdout
            + completed.stderr
        )
    temporary.chmod(0o755)
    temporary.replace(executable)
    return executable


def _materialize_checkpoint(image: Path, destination: Path) -> dict[str, Any]:
    image = image.resolve()
    manifest = json.loads((image / "manifest.json").read_text(encoding="utf-8"))
    expected_digest = manifest.get("source_sha256")
    expected_size = manifest.get("source_size")
    pages = manifest.get("pages")
    if (
        manifest.get("format") != "strpot-image-v1"
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or not isinstance(pages, list)
    ):
        raise ValueError("native image manifest has invalid checkpoint identity")
    if (
        destination.exists()
        and destination.stat().st_size == expected_size
        and _sha256_file(destination) == expected_digest
    ):
        return manifest

    digest = hashlib.sha256()
    reconstructed_size = 0
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            for index, page in enumerate(pages):
                if not isinstance(page, dict):
                    raise ValueError(f"native image page {index} is invalid")
                relative_path = page.get("path")
                if not isinstance(relative_path, str) or not relative_path:
                    raise ValueError(f"native image page {index} has invalid path")
                page_path = (image / relative_path).resolve()
                if not page_path.is_relative_to(image):
                    raise ValueError(f"page {index} escapes the image directory")
                payload = page_path.read_bytes()
                stored_size = page.get("stored_size")
                raw_size = page.get("raw_size")
                if (
                    not isinstance(stored_size, int)
                    or isinstance(stored_size, bool)
                    or stored_size < 0
                    or len(payload) != stored_size
                    or not isinstance(raw_size, int)
                    or isinstance(raw_size, bool)
                    or raw_size < 0
                    or raw_size > MAX_PAGE_SIZE
                    or reconstructed_size + raw_size > expected_size
                ):
                    raise ValueError(f"native image page {index} has invalid sizes")
                if page.get("codec") == "raw":
                    raw = payload
                elif page.get("codec") == "zlib":
                    decompressor = zlib.decompressobj()
                    raw = decompressor.decompress(payload, raw_size + 1)
                    if (
                        len(raw) > raw_size
                        or not decompressor.eof
                        or decompressor.unconsumed_tail
                    ):
                        raise ValueError(
                            f"native image page {index} exceeds its declared size"
                        )
                else:
                    raise ValueError(f"unsupported native image codec at page {index}")
                if len(raw) != raw_size:
                    raise ValueError(f"native image page {index} has invalid raw size")
                if hashlib.sha256(raw).hexdigest() != page.get("sha256"):
                    raise ValueError(f"native image page {index} failed verification")
                digest.update(raw)
                reconstructed_size += len(raw)
                output.write(raw)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if reconstructed_size != expected_size or digest.hexdigest() != expected_digest:
        temporary.unlink(missing_ok=True)
        raise ValueError("native checkpoint reconstruction digest mismatch")
    temporary.replace(destination)
    return manifest


def _read_safetensors_header(checkpoint: Path) -> tuple[int, dict[str, Any]]:
    checkpoint_size = checkpoint.stat().st_size
    with checkpoint.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors checkpoint is missing its header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > checkpoint_size - 8:
            raise ValueError("safetensors header extends outside checkpoint")
        try:
            header = json.loads(stream.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("safetensors header is not valid JSON") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")

    data_offset = 8 + header_size
    payload_size = checkpoint_size - data_offset
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict):
                raise ValueError("safetensors metadata must be an object")
            continue
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError("safetensors tensor entries are invalid")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in _DTYPE_SIZES:
            raise ValueError(f"unsupported tensor dtype for {name}: {dtype}")
        if (
            not isinstance(shape, list)
            or not shape
            or len(shape) > _MAX_TENSOR_RANK
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                or dimension > _MAX_TENSOR_DIMENSION
                for dimension in shape
            )
        ):
            raise ValueError(f"tensor {name} has invalid or unbounded dimensions")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in offsets
            )
        ):
            raise ValueError(f"tensor {name} has invalid data offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise ValueError(f"tensor {name} range extends outside checkpoint")
        element_count = 1
        for dimension in shape:
            if element_count > payload_size // dimension:
                raise ValueError(f"tensor {name} shape overflows checkpoint bounds")
            element_count *= dimension
        byte_size = element_count * _DTYPE_SIZES[dtype]
        if end - start != byte_size:
            raise ValueError(f"tensor {name} byte range does not match its shape")
        ranges.append((start, end, name))

    previous_end = 0
    for start, end, name in sorted(ranges):
        if start < previous_end:
            raise ValueError(f"tensor {name} overlaps another tensor")
        previous_end = end
    return data_offset, header


def _validate_llama_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("model config must be an object")
    supported = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "rope_scaling": None,
        "tie_word_embeddings": False,
    }
    for key, expected in supported.items():
        if config.get(key) != expected:
            raise ValueError(f"unsupported Llama {key}: {config.get(key)!r}")

    integer_keys = _DESCRIPTOR_CONFIG_KEYS[:7]
    values: dict[str, Any] = {}
    for key in integer_keys:
        value = config.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > _MAX_TENSOR_DIMENSION
        ):
            raise ValueError(f"model config has invalid {key}")
        values[key] = value

    hidden_size = values["hidden_size"]
    attention_heads = values["num_attention_heads"]
    kv_heads = values["num_key_value_heads"]
    if (
        hidden_size % attention_heads != 0
        or attention_heads % kv_heads != 0
        or (hidden_size // attention_heads) % 2 != 0
    ):
        raise ValueError("unsupported Llama head geometry")
    head_dim = hidden_size // attention_heads
    if config.get("head_dim", head_dim) != head_dim:
        raise ValueError("unsupported Llama head_dim")

    for key, default in (("rms_norm_eps", None), ("rope_theta", 10_000.0)):
        value = config.get(key, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"model config has invalid {key}")
        values[key] = value
    for key in ("bos_token_id", "eos_token_id"):
        value = config.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < values["vocab_size"]
        ):
            raise ValueError(f"model config has invalid {key}")
        values[key] = value

    return {**supported, "head_dim": head_dim, **values}


def _expected_llama_tensors(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    expected = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (hidden, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, hidden),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return expected


def _validate_tensor_manifest(header: dict[str, Any], config: dict[str, Any]) -> None:
    tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
    expected = _expected_llama_tensors(config)
    missing = sorted(expected.keys() - tensors.keys())
    unexpected = sorted(tensors.keys() - expected.keys())
    if missing:
        raise ValueError(f"checkpoint is missing required tensors: {missing}")
    if unexpected:
        raise ValueError(f"checkpoint has unsupported tensors: {unexpected}")
    for name, shape in expected.items():
        actual = tuple(tensors[name]["shape"])
        if actual != shape:
            raise ValueError(
                f"checkpoint tensor {name} has shape {actual}, expected {shape}"
            )


def _write_descriptor(config_path: Path, checkpoint: Path, destination: Path) -> None:
    config = _validate_llama_config(json.loads(config_path.read_text(encoding="utf-8")))
    data_offset, header = _read_safetensors_header(checkpoint)
    _validate_tensor_manifest(header, config)
    lines = ["STRPOT_NATIVE_V1"]
    for key in _DESCRIPTOR_CONFIG_KEYS:
        value = config[key]
        lines.append(f"config\t{key}\t{value}")
    for name, entry in sorted(header.items()):
        if name == "__metadata__":
            continue
        start, _ = entry["data_offsets"]
        shape = ",".join(str(dimension) for dimension in entry["shape"])
        lines.append(
            f"tensor\t{name}\t{entry['dtype']}\t{data_offset + int(start)}\t{shape}"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


class NativeLlamaEngine:
    """Binding boundary for StrPot's dependency-free native Llama runtime."""

    def __init__(
        self,
        *,
        config_path: Path,
        weights_image: Path,
        cache_root: Path | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.weights_image = weights_image.resolve()
        manifest = json.loads(
            (self.weights_image / "manifest.json").read_text(encoding="utf-8")
        )
        checkpoint_sha256 = manifest.get("source_sha256")
        if (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in checkpoint_sha256
            )
        ):
            raise ValueError("native image manifest has invalid checkpoint identity")
        canonical_config = _validate_llama_config(
            json.loads(self.config_path.read_text(encoding="utf-8"))
        )
        canonical_config_bytes = json.dumps(
            canonical_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        semantic_digest = hashlib.sha256(canonical_config_bytes).hexdigest()
        self.checkpoint_sha256 = checkpoint_sha256
        root = (
            cache_root.resolve()
            if cache_root is not None
            else Path.home() / ".strpot" / "native-models"
        )
        self.model_root = root / self.checkpoint_sha256 / semantic_digest
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.model_root / "model.safetensors"
        self.descriptor = self.model_root / "model.strpot-native"
        _materialize_checkpoint(self.weights_image, self.checkpoint)
        temporary_descriptor = self.descriptor.with_suffix(".tmp")
        _write_descriptor(self.config_path, self.checkpoint, temporary_descriptor)
        temporary_descriptor.replace(self.descriptor)
        self.executable = build_native_engine(root / "engine")

    def _invoke(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        threads: int,
    ) -> dict[str, Any]:
        command = [
            str(self.executable),
            "--generate",
            str(self.descriptor),
            str(self.checkpoint),
            ",".join(str(token) for token in prompt_token_ids),
            str(max_new_tokens),
            str(threads),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "StrPot native inference failed:\n"
                + completed.stdout
                + completed.stderr
            )
        return json.loads(completed.stdout)

    def _invoke_batch(
        self,
        prompt_token_ids: list[list[int]],
        *,
        max_new_tokens: int,
        threads: int,
    ) -> dict[str, Any]:
        command = [
            str(self.executable),
            "--generate-batch",
            str(self.descriptor),
            str(self.checkpoint),
            ";".join(
                ",".join(str(token) for token in prompt) for prompt in prompt_token_ids
            ),
            str(max_new_tokens),
            str(threads),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "StrPot native batch inference failed:\n"
                + completed.stdout
                + completed.stderr
            )
        return json.loads(completed.stdout)

    def _invoke_token_wave(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        max_proposals: int,
        threads: int,
        adversarial_proposals: bool,
    ) -> dict[str, Any]:
        command = [
            str(self.executable),
            "--generate-token-wave",
            str(self.descriptor),
            str(self.checkpoint),
            ",".join(str(token) for token in prompt_token_ids),
            str(max_new_tokens),
            str(max_proposals),
            str(threads),
            "1" if adversarial_proposals else "0",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "StrPot native token-wave inference failed:\n"
                + completed.stdout
                + completed.stderr
            )
        return json.loads(completed.stdout)

    def _autotune_threads(self, prompt_token_ids: list[int]) -> int:
        tuning_path = self.model_root / "thread-tuning.json"
        build = self.executable.parent.name
        if tuning_path.exists():
            cached = json.loads(tuning_path.read_text(encoding="utf-8"))
            if cached.get("engine_build") == build:
                return int(cached["threads"])

        logical = max(1, os.cpu_count() or 1)
        candidates = set(range(1, min(logical, 4) + 1))
        candidate = 8
        while candidate < logical:
            candidates.add(candidate)
            candidate *= 2
        candidates.add(logical)
        sample_prompt = prompt_token_ids[: min(8, len(prompt_token_ids))]
        best_threads = 1
        best_elapsed = float("inf")
        expected_tokens: list[int] | None = None
        for threads in sorted(candidates):
            report = self._invoke(
                sample_prompt,
                max_new_tokens=2,
                threads=threads,
            )
            tokens = [int(token) for token in report["tokens"]]
            if expected_tokens is None:
                expected_tokens = tokens
            elif tokens != expected_tokens:
                raise RuntimeError("native thread tuning changed greedy token output")
            elapsed = float(report["prefill_seconds"]) + float(report["decode_seconds"])
            if elapsed < best_elapsed:
                best_elapsed = elapsed
                best_threads = threads
        temporary = tuning_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "engine_build": build,
                    "logical_cpus": logical,
                    "threads": best_threads,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(tuning_path)
        return best_threads

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        threads: int,
    ) -> NativeGenerationResult:
        if not prompt_token_ids:
            raise ValueError("native inference requires at least one prompt token")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if threads < 0:
            raise ValueError("threads cannot be negative")
        selected_threads = (
            self._autotune_threads(prompt_token_ids) if threads == 0 else threads
        )
        report = self._invoke(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            threads=selected_threads,
        )
        return NativeGenerationResult(
            generated_token_ids=tuple(int(token) for token in report["tokens"]),
            engine=str(report["engine"]),
            kernel_family=str(report["kernel_family"]),
            weight_dtype=str(report["weight_dtype"]),
            prefill_seconds=float(report["prefill_seconds"]),
            prefill_matrix_passes=int(report["prefill_matrix_passes"]),
            runtime_tensor_lookups=int(report["runtime_tensor_lookups"]),
            decode_parallel_dispatches=int(report["decode_parallel_dispatches"]),
            rope_table_entries=int(report["rope_table_entries"]),
            attention_workspace_floats=int(report["attention_workspace_floats"]),
            decode_seconds=float(report["decode_seconds"]),
            decode_tokens_per_second=float(report["decode_tokens_per_second"]),
            inter_token_seconds=tuple(
                float(value) for value in report["inter_token_seconds"]
            ),
            threads=int(report["threads"]),
            final_logits_hash=str(report["final_logits_hash"]),
            final_kv_hash=str(report["final_kv_hash"]),
            final_kv_lengths=tuple(int(value) for value in report["final_kv_lengths"]),
            frontier_logits_hashes=tuple(
                str(value) for value in report["frontier_logits_hashes"]
            ),
            frontier_kv_hashes=tuple(
                str(value) for value in report["frontier_kv_hashes"]
            ),
        )

    def generate_token_wave(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        max_proposals: int = 4,
        threads: int,
        adversarial_proposals: bool = False,
    ) -> NativeTokenWaveResult:
        if not prompt_token_ids:
            raise ValueError("native token-wave inference requires a prompt")
        if max_new_tokens < 1 or max_proposals < 0 or threads < 0:
            raise ValueError("token-wave limits and threads are invalid")
        selected_threads = (
            self._autotune_threads(prompt_token_ids) if threads == 0 else threads
        )
        report = self._invoke_token_wave(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            max_proposals=max_proposals,
            threads=selected_threads,
            adversarial_proposals=adversarial_proposals,
        )
        return NativeTokenWaveResult(
            generated_token_ids=tuple(int(token) for token in report["tokens"]),
            engine=str(report["engine"]),
            kernel_family=str(report["kernel_family"]),
            weight_dtype=str(report["weight_dtype"]),
            prefill_seconds=float(report["prefill_seconds"]),
            decode_seconds=float(report["decode_seconds"]),
            target_weight_traversals=int(report["target_weight_traversals"]),
            committed_decode_tokens=int(report["committed_decode_tokens"]),
            committed_tokens_per_target_weight_traversal=float(
                report["committed_tokens_per_target_weight_traversal"]
            ),
            acceptance_lengths=tuple(int(x) for x in report["acceptance_lengths"]),
            traversal_seconds=tuple(float(x) for x in report["traversal_seconds"]),
            rolled_back_tokens=int(report["rolled_back_tokens"]),
            speculative_traversals=int(report["speculative_traversals"]),
            fallback_traversals=int(report["fallback_traversals"]),
            transactional_snapshot_bytes_copied=int(
                report["transactional_snapshot_bytes_copied"]
            ),
            rollback_verified=bool(report["rollback_verified"]),
            final_logits_hash=str(report["final_logits_hash"]),
            final_kv_hash=str(report["final_kv_hash"]),
            final_kv_lengths=tuple(int(x) for x in report["final_kv_lengths"]),
            frontier_logits_hashes=tuple(
                str(value) for value in report["frontier_logits_hashes"]
            ),
            frontier_kv_hashes=tuple(
                str(value) for value in report["frontier_kv_hashes"]
            ),
            threads=int(report["threads"]),
        )

    def generate_batch(
        self,
        prompt_token_ids: list[list[int]],
        *,
        max_new_tokens: int,
        threads: int,
    ) -> NativeBatchGenerationResult:
        if not prompt_token_ids or any(not prompt for prompt in prompt_token_ids):
            raise ValueError("native batch inference requires non-empty prompts")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if threads < 1:
            raise ValueError("batch threads must be positive")
        report = self._invoke_batch(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            threads=threads,
        )
        return NativeBatchGenerationResult(
            generated_token_ids=tuple(
                tuple(int(token) for token in tokens) for tokens in report["tokens"]
            ),
            engine=str(report["engine"]),
            kernel_family=str(report["kernel_family"]),
            weight_dtype=str(report["weight_dtype"]),
            batch_size=int(report["batch_size"]),
            prefill_seconds=float(report["prefill_seconds"]),
            decode_seconds=float(report["decode_seconds"]),
            decode_matrix_passes=int(report["decode_matrix_passes"]),
            decode_parallel_dispatches=int(report["decode_parallel_dispatches"]),
            activation_panel_width=int(report["activation_panel_width"]),
            aggregate_decode_tokens_per_second=float(
                report["aggregate_decode_tokens_per_second"]
            ),
            threads=int(report["threads"]),
        )
