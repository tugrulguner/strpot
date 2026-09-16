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
_PLAN_SOURCE = Path(__file__).with_name("_native") / "strpot_plan.cpp"
_DTYPE_SIZES = {"BF16": 2, "F32": 4}
_MAX_NATIVE_PLAN_THREADS = 256
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
class NumericalSemantics:
    """Explicit arithmetic contract carried by an execution plan."""

    weight_dtype: str
    activation_dtype: str
    output_dtype: str
    accumulator_dtype: str
    accumulation_order: str
    rounding: str
    contraction_policy: str
    softmax_policy: str
    transcendental_policy: str
    gelu_formula: str
    silu_formula: str
    rope_formula: str


@dataclass(frozen=True)
class GraphValue:
    """A typed SSA value declared by an execution plan."""

    name: str
    dtype: str
    shape: tuple[int | str, ...]
    lifetime: str


@dataclass(frozen=True)
class TensorBinding:
    """Bind a graph-local tensor alias to an immutable checkpoint tensor."""

    alias: str
    checkpoint_name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Operator:
    """One ordered operation in the model-independent native graph."""

    kind: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    tensors: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    """Versioned, immutable narrow-waist artifact for native model execution."""

    version: int
    architecture_id: str
    config_identity: str
    shapes: tuple[tuple[str, int], ...]
    semantics: NumericalSemantics
    tensors: tuple[TensorBinding, ...]
    values: tuple[GraphValue, ...]
    operators: tuple[Operator, ...]
    eos_token_id: int

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported execution plan version {self.version}")
        if not self.architecture_id or not self.config_identity:
            raise ValueError("execution plan identities cannot be empty")
        supported_semantics = NumericalSemantics(
            weight_dtype=self.semantics.weight_dtype,
            activation_dtype=self.semantics.activation_dtype,
            output_dtype=self.semantics.output_dtype,
            accumulator_dtype="F32",
            accumulation_order="ordered",
            rounding="nearest_even",
            contraction_policy="disabled",
            softmax_policy="stable_max_subtraction",
            transcendental_policy="system_libm",
            gelu_formula="x_times_half_times_one_plus_erf_x_over_sqrt_two",
            silu_formula="x_over_one_plus_exp_neg_x",
            rope_formula="pair_rotation_theta_pow_2i_over_head_dim",
        )
        if self.semantics != supported_semantics:
            raise ValueError("unsupported numerical semantics")
        if self.semantics.weight_dtype not in {"F32", "BF16"}:
            raise ValueError("native plan weight dtype must be F32 or BF16")
        if self.semantics.activation_dtype not in {"F32", "BF16"}:
            raise ValueError("native plan activation dtype must be F32 or BF16")
        if self.semantics.output_dtype not in {"F32", "BF16"}:
            raise ValueError("native plan output dtype must be F32 or BF16")

        shape_names = [name for name, _ in self.shapes]
        if len(shape_names) != len(set(shape_names)):
            raise ValueError("execution plan shapes must be unique")
        shape_map = dict(self.shapes)
        if set(shape_map) != {"hidden_size", "vocab_size", "context_size"}:
            raise ValueError(
                "plan requires exactly hidden_size, vocab_size, context_size"
            )
        if any(type(value) is not int or value <= 0 for value in shape_map.values()):
            raise ValueError("execution plan shapes must be positive integers")
        if type(self.eos_token_id) is not int:
            raise ValueError("eos_token_id must be an integer")

        aliases = [binding.alias for binding in self.tensors]
        if len(aliases) != len(set(aliases)):
            raise ValueError("execution plan tensor aliases must be unique")
        for binding in self.tensors:
            if not binding.alias or not binding.checkpoint_name:
                raise ValueError("tensor identities cannot be empty")
            if binding.dtype != self.semantics.weight_dtype:
                raise ValueError(f"tensor {binding.alias} has undeclared dtype")
            if not binding.shape or any(
                type(item) is not int or item <= 0 for item in binding.shape
            ):
                raise ValueError(
                    "execution plan tensor shapes must be positive integers"
                )

        value_names = [value.name for value in self.values]
        if len(value_names) != len(set(value_names)):
            raise ValueError("execution plan value names must be unique")
        value_map = {value.name: value for value in self.values}
        for value in self.values:
            if not value.name or value.dtype not in {"F32", "BF16"}:
                raise ValueError("invalid graph value declaration")
            if value.lifetime not in {"activation", "logits", "state"}:
                raise ValueError(f"invalid graph value lifetime {value.lifetime}")
            if not value.shape:
                raise ValueError(f"graph value {value.name} has empty shape")
            for dimension in value.shape:
                if isinstance(dimension, str):
                    if dimension not in shape_map:
                        raise ValueError(f"unknown symbolic dimension {dimension}")
                elif type(dimension) is not int or dimension <= 0:
                    raise ValueError(f"invalid graph value dimension {dimension!r}")

        if not self.operators:
            raise ValueError("execution plan must contain operators")
        tensor_map = {binding.alias: binding for binding in self.tensors}
        defined: set[str] = set()
        used: set[str] = set()
        for operator in self.operators:
            attribute_names = [name for name, _ in operator.attributes]
            if len(attribute_names) != len(set(attribute_names)):
                raise ValueError(f"duplicate operator attribute on {operator.kind}")
            for name in operator.inputs:
                if name not in defined:
                    raise ValueError(f"operator dependency is unavailable: {name}")
                used.add(name)
            for alias in operator.tensors:
                if alias not in tensor_map:
                    raise ValueError(
                        f"operator {operator.kind} references unknown tensor {alias}"
                    )
            for output in operator.outputs:
                if output in defined:
                    raise ValueError(f"duplicate output definition {output}")
                if output not in value_map:
                    raise ValueError(f"operator output {output} is not declared")
                defined.add(output)
            self._validate_operator(
                operator,
                shape_map,
                value_map,
                tensor_map,
                self.semantics.activation_dtype,
                self.semantics.output_dtype,
            )
        if defined != set(value_map):
            missing = sorted(set(value_map) - defined)
            raise ValueError(f"graph values are not defined: {missing}")
        unused = defined - used - {"logits"}
        if unused:
            raise ValueError(f"graph values are unused: {sorted(unused)}")
        logits = value_map.get("logits")
        if (
            logits is None
            or logits.shape != ("vocab_size",)
            or logits.lifetime != "logits"
        ):
            raise ValueError("logits must be a vocab_size logits value")

    @staticmethod
    def _validate_operator(
        op: Operator,
        shapes: dict[str, int],
        values: dict[str, GraphValue],
        tensors: dict[str, TensorBinding],
        plan_activation_dtype: str,
        plan_output_dtype: str,
    ) -> None:
        schemas: dict[str, tuple[int, int, tuple[int, ...], frozenset[str]]] = {
            "embedding": (0, 1, (2,), frozenset()),
            "position_embedding": (1, 1, (2,), frozenset()),
            "save": (1, 1, (), frozenset()),
            "rms_norm": (1, 1, (1,), frozenset({"epsilon"})),
            "layer_norm": (1, 1, (1, 1), frozenset({"epsilon"})),
            "attention_rope": (
                1,
                1,
                (2, 2, 2, 2),
                frozenset({"cache", "heads", "kv_heads", "theta"}),
            ),
            "attention_causal": (
                1,
                1,
                (2, 2, 2, 2),
                frozenset({"cache", "heads", "kv_heads"}),
            ),
            "attention_rope_qkv_bias": (
                1,
                1,
                (2, 1, 2, 1, 2, 1, 2),
                frozenset(
                    {"cache", "heads", "kv_heads", "rope_layout", "scale", "theta"}
                ),
            ),
            "add": (2, 1, (), frozenset()),
            "swiglu": (1, 1, (2, 2, 2), frozenset()),
            "gelu_exact": (1, 1, (2, 1, 2, 1), frozenset({"formula"})),
            "linear": (1, 1, (2,), frozenset()),
        }
        if op.kind not in schemas:
            return  # Native owns unsupported-operator rejection.
        inputs, outputs, tensor_ranks, attributes = schemas[op.kind]
        if len(op.inputs) != inputs or len(op.outputs) != outputs:
            raise ValueError(f"invalid {op.kind} operator arity")
        if op.kind == "linear":
            if len(op.tensors) not in {1, 2}:
                raise ValueError("invalid linear operator arity")
        elif len(op.tensors) != len(tensor_ranks):
            raise ValueError(f"invalid {op.kind} operator arity")
        actual_attributes = frozenset(name for name, _ in op.attributes)
        if op.kind == "linear" and actual_attributes not in {
            frozenset(),
            frozenset({"result_rounding"}),
        }:
            raise ValueError("invalid attributes for linear")
        if op.kind != "linear" and actual_attributes != attributes:
            raise ValueError(f"invalid attributes for {op.kind}")
        for index, rank in enumerate(tensor_ranks[: len(op.tensors)]):
            if len(tensors[op.tensors[index]].shape) != rank:
                raise ValueError(f"tensor rank mismatch for {op.tensors[index]}")

        def resolved(value: GraphValue) -> tuple[int, ...]:
            return tuple(
                shapes[item] if isinstance(item, str) else item for item in value.shape
            )

        hidden = shapes["hidden_size"]
        vocab = shapes["vocab_size"]
        if op.kind == "embedding":
            if tensors[op.tensors[0]].shape != (vocab, hidden) or resolved(
                values[op.outputs[0]]
            ) != (hidden,):
                raise ValueError("embedding shape/dtype mismatch")
        elif op.inputs:
            for name in (*op.inputs, *op.outputs):
                if op.kind == "linear" and name == op.outputs[0]:
                    continue
                if resolved(values[name]) != (hidden,):
                    raise ValueError(f"{op.kind} value shape mismatch")
        if op.kind == "linear":
            weight = tensors[op.tensors[0]]
            output_shape = resolved(values[op.outputs[0]])
            if weight.shape[1] != resolved(values[op.inputs[0]])[0] or output_shape != (
                weight.shape[0],
            ):
                raise ValueError("linear shape/orientation mismatch")
            if len(op.tensors) == 2 and tensors[op.tensors[1]].shape != (
                weight.shape[0],
            ):
                raise ValueError("linear bias shape mismatch")
            if op.outputs[0] == "logits" and weight.shape[0] != vocab:
                raise ValueError("LM head must produce exactly vocab_size logits")
            if op.attributes != () and op.attributes != (("result_rounding", "BF16"),):
                raise ValueError("unsupported linear result rounding")
        elif op.kind == "attention_rope_qkv_bias":
            attributes_map = dict(op.attributes)
            heads = int(attributes_map["heads"])
            kv_heads = int(attributes_map["kv_heads"])
            if hidden % heads or heads % kv_heads:
                raise ValueError("invalid attention head shape")
            head_dim = hidden // heads
            expected_shapes = (
                (hidden, hidden),
                (hidden,),
                (kv_heads * head_dim, hidden),
                (kv_heads * head_dim,),
                (kv_heads * head_dim, hidden),
                (kv_heads * head_dim,),
                (hidden, hidden),
            )
            if tuple(tensors[name].shape for name in op.tensors) != expected_shapes:
                raise ValueError("attention tensor shape mismatch")
            if attributes_map["rope_layout"] != "half_split":
                raise ValueError("unsupported RoPE layout")
        for name in (*op.inputs, *op.outputs):
            expected_dtype = (
                plan_output_dtype if name == "logits" else plan_activation_dtype
            )
            if values[name].dtype != expected_dtype:
                raise ValueError(
                    f"operator {op.kind} value {name} violates declared dtype"
                )


@dataclass(frozen=True)
class NativePlanGenerationResult:
    generated_token_ids: tuple[int, ...]
    engine: str
    architecture_id: str
    config_identity: str
    weight_dtype: str
    executed_operators: int
    kv_cache_lengths: tuple[int, ...]
    frontier_logits: tuple[tuple[float, ...], ...]
    kv_cache_keys: tuple[tuple[float, ...], ...]
    kv_cache_values: tuple[tuple[float, ...], ...]
    operator_trace: tuple[str, ...]
    inter_token_seconds: tuple[float, ...]
    threads: int
    kernel_family: str
    matrix_passes: int
    parallel_dispatches: int
    threaded_matrix_dispatches: int
    runtime_tensor_lookups: int
    prefill_seconds: float
    prefill_matrix_passes: int
    prefill_physical_width: int
    frontier_logits_hashes: tuple[str, ...]
    frontier_kv_hashes: tuple[str, ...]
    final_logits_hash: str
    final_kv_hash: str


@dataclass(frozen=True)
class NativePlanTokenWaveResult:
    """Compact conformance and traversal report for generic plan token waves."""

    generated_token_ids: tuple[int, ...]
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
    matrix_passes: int
    runtime_tensor_lookups: int
    threads: int
    kernel_family: str
    native_panel_8_calls: int
    native_panel_16_calls: int


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


def build_native_plan_engine(cache_root: Path | None = None) -> Path:
    """Compile the portable execution-plan consumer."""
    source = _PLAN_SOURCE.read_bytes()
    key = hashlib.sha256(
        source + platform.machine().encode() + platform.system().encode()
    ).hexdigest()[:16]
    base = (
        cache_root.resolve()
        if cache_root is not None
        else Path.home() / ".strpot" / "native-plan"
    )
    root = base / key
    root.mkdir(parents=True, exist_ok=True)
    executable = root / "strpot-native-plan"
    if executable.exists():
        return executable
    temporary = root / f"strpot-native-plan.{os.getpid()}.tmp"
    completed = subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++20",
            "-O3",
            "-ffp-contract=off",
            "-DNDEBUG",
            "-pthread",
            str(_PLAN_SOURCE),
            "-o",
            str(temporary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to compile StrPot native plan engine:\n"
            + completed.stdout
            + completed.stderr
        )
    temporary.chmod(0o755)
    temporary.replace(executable)
    return executable


def _safe_field(value: str) -> str:
    if not value or "\t" in value or "\n" in value or "," in value:
        raise ValueError(f"invalid execution plan field {value!r}")
    return value


def _shape_field(shape: tuple[int | str, ...]) -> str:
    return ",".join(_safe_field(str(item)) for item in shape)


def _canonical_plan_lines(plan: ExecutionPlan) -> list[str]:
    lines = [
        f"identity\tarchitecture\t{_safe_field(plan.architecture_id)}",
        f"identity\tconfig\t{_safe_field(plan.config_identity)}",
        f"generation\teos_token_id\t{plan.eos_token_id}",
    ]
    for key, value in sorted(plan.shapes):
        lines.append(f"shape\t{_safe_field(key)}\t{value}")
    lines.extend(
        (
            f"semantic\tweight_dtype\t{plan.semantics.weight_dtype}",
            f"semantic\tactivation_dtype\t{plan.semantics.activation_dtype}",
            f"semantic\toutput_dtype\t{plan.semantics.output_dtype}",
            f"semantic\taccumulator_dtype\t{plan.semantics.accumulator_dtype}",
            f"semantic\taccumulation_order\t{plan.semantics.accumulation_order}",
            f"semantic\trounding\t{_safe_field(plan.semantics.rounding)}",
            f"semantic\tcontraction_policy\t{plan.semantics.contraction_policy}",
            f"semantic\tsoftmax_policy\t{plan.semantics.softmax_policy}",
            f"semantic\ttranscendental_policy\t{plan.semantics.transcendental_policy}",
            f"semantic\tgelu_formula\t{plan.semantics.gelu_formula}",
            f"semantic\tsilu_formula\t{plan.semantics.silu_formula}",
            f"semantic\trope_formula\t{plan.semantics.rope_formula}",
        )
    )
    for binding in sorted(plan.tensors, key=lambda item: item.alias):
        lines.append(
            f"binding\t{_safe_field(binding.alias)}\t"
            f"{_safe_field(binding.checkpoint_name)}\t{binding.dtype}\t"
            f"{_shape_field(binding.shape)}"
        )
    for value in sorted(plan.values, key=lambda item: item.name):
        lines.append(
            f"value\t{_safe_field(value.name)}\t{value.dtype}\t"
            f"{_shape_field(value.shape)}\t{_safe_field(value.lifetime)}"
        )
    for operator in plan.operators:
        attributes = ",".join(
            f"{_safe_field(key)}={_safe_field(value)}"
            for key, value in sorted(operator.attributes)
        )
        fields = (
            "op",
            _safe_field(operator.kind),
            ",".join(map(_safe_field, operator.inputs)) or "-",
            ",".join(map(_safe_field, operator.outputs)) or "-",
            ",".join(map(_safe_field, operator.tensors)) or "-",
            attributes or "-",
        )
        lines.append("\t".join(fields))
    return lines


def _canonical_plan_bytes(plan: ExecutionPlan) -> bytes:
    return ("\n".join(_canonical_plan_lines(plan)) + "\n").encode("utf-8")


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate checkpoint field {key}")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("checkpoint header must be an object")
    return value


def _write_execution_plan(
    plan: ExecutionPlan,
    checkpoint: Path,
    destination: Path,
    checkpoint_sha256: str,
) -> None:
    data_offset, header = _read_safetensors_header(checkpoint)
    plan_lines = _canonical_plan_lines(plan)
    data_lines: list[str] = []
    for binding in sorted(plan.tensors, key=lambda item: item.alias):
        entry = header.get(binding.checkpoint_name)
        if not isinstance(entry, dict):
            raise ValueError(f"checkpoint is missing tensor {binding.checkpoint_name}")
        if set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"malformed checkpoint tensor {binding.checkpoint_name}")
        actual_shape = tuple(entry["shape"])
        if actual_shape != binding.shape:
            raise ValueError(
                f"tensor {binding.alias} shape mismatch: plan {binding.shape}, "
                f"checkpoint {actual_shape}"
            )
        if entry["dtype"] != binding.dtype:
            raise ValueError(
                f"tensor {binding.alias} dtype {entry['dtype']} violates plan "
                f"{binding.dtype}"
            )
        start, end = entry["data_offsets"]
        elements = math.prod(binding.shape)
        expected_length = elements * _DTYPE_SIZES[binding.dtype]
        if end - start != expected_length:
            raise ValueError(f"tensor byte length mismatch for {binding.alias}")
        absolute = data_offset + start
        data_lines.append(
            f"tensor_data\t{binding.alias}\t{absolute}\t{expected_length}"
        )
    body = "\n".join((*plan_lines, *data_lines)) + "\n"
    plan_digest = hashlib.sha256(_canonical_plan_bytes(plan)).hexdigest()
    artifact_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    header_lines = [
        "STRPOT_EXECUTION_PLAN_V2",
        f"artifact\tplan_sha256\t{plan_digest}",
        f"artifact\tcheckpoint_sha256\t{checkpoint_sha256}",
        f"artifact\tbody_sha256\t{artifact_digest}",
    ]
    destination.write_text("\n".join(header_lines) + "\n" + body, encoding="utf-8")


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


class NativeExecutionPlanEngine:
    """Public Python-to-process boundary for generic native operator plans."""

    def __init__(
        self,
        *,
        plan: ExecutionPlan,
        weights_image: Path,
        cache_root: Path | None = None,
    ) -> None:
        self.plan = plan
        self.weights_image = weights_image.resolve()
        manifest = json.loads(
            (self.weights_image / "manifest.json").read_text(encoding="utf-8")
        )
        checkpoint_digest = manifest.get("source_sha256")
        if (
            not isinstance(checkpoint_digest, str)
            or len(checkpoint_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in checkpoint_digest
            )
        ):
            raise ValueError("manifest has invalid checkpoint SHA-256")
        plan_key = hashlib.sha256(_canonical_plan_bytes(plan)).hexdigest()
        base = (
            cache_root.resolve()
            if cache_root is not None
            else Path.home() / ".strpot" / "native-plan-models"
        )
        self.model_root = base / f"{checkpoint_digest}-{plan_key}"
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.model_root / "model.safetensors"
        self.artifact = self.model_root / "execution.strpot-plan"
        _materialize_checkpoint(self.weights_image, self.checkpoint)
        _write_execution_plan(plan, self.checkpoint, self.artifact, checkpoint_digest)
        self.executable = build_native_plan_engine(base / "engine")

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        threads: int,
    ) -> NativePlanGenerationResult:
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise ValueError("native inference requires at least one prompt token")
        vocab_size = dict(self.plan.shapes)["vocab_size"]
        if any(
            type(token) is not int or token < 0 or token >= vocab_size
            for token in prompt_token_ids
        ):
            raise ValueError("prompt token is outside the plan vocabulary")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if type(threads) is not int or not 1 <= threads <= _MAX_NATIVE_PLAN_THREADS:
            raise ValueError("threads must be between 1 and 256")
        completed = subprocess.run(
            [
                str(self.executable),
                str(self.artifact),
                str(self.checkpoint),
                ",".join(str(value) for value in prompt_token_ids),
                str(max_new_tokens),
                str(threads),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "StrPot native execution-plan inference failed:\n"
                + completed.stdout
                + completed.stderr
            )
        report = json.loads(completed.stdout)
        return NativePlanGenerationResult(
            generated_token_ids=tuple(int(value) for value in report["tokens"]),
            engine=str(report["engine"]),
            architecture_id=str(report["architecture_id"]),
            config_identity=str(report["config_identity"]),
            weight_dtype=str(report["weight_dtype"]),
            executed_operators=int(report["executed_operators"]),
            kv_cache_lengths=tuple(int(value) for value in report["kv_cache_lengths"]),
            frontier_logits=tuple(
                tuple(float(item) for item in values)
                for values in report["frontier_logits"]
            ),
            kv_cache_keys=tuple(
                tuple(float(item) for item in values)
                for values in report["kv_cache_keys"]
            ),
            kv_cache_values=tuple(
                tuple(float(item) for item in values)
                for values in report["kv_cache_values"]
            ),
            operator_trace=tuple(str(value) for value in report["operator_trace"]),
            inter_token_seconds=tuple(
                float(value) for value in report["inter_token_seconds"]
            ),
            threads=int(report["threads"]),
            kernel_family=str(report["kernel_family"]),
            matrix_passes=int(report["matrix_passes"]),
            parallel_dispatches=int(report["parallel_dispatches"]),
            threaded_matrix_dispatches=int(report["threaded_matrix_dispatches"]),
            runtime_tensor_lookups=int(report["runtime_tensor_lookups"]),
            prefill_seconds=float(report["prefill_seconds"]),
            prefill_matrix_passes=int(report["prefill_matrix_passes"]),
            prefill_physical_width=int(report["prefill_physical_width"]),
            frontier_logits_hashes=tuple(
                str(value) for value in report["frontier_logits_hashes"]
            ),
            frontier_kv_hashes=tuple(
                str(value) for value in report["frontier_kv_hashes"]
            ),
            final_logits_hash=str(report["final_logits_hash"]),
            final_kv_hash=str(report["final_kv_hash"]),
        )

    def generate_token_wave(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        max_proposals: int = 3,
        threads: int,
        adversarial_proposals: bool = False,
    ) -> NativePlanTokenWaveResult:
        """Generate exactly while transactionally verifying history proposals."""
        self._validate_plan_generation_inputs(
            prompt_token_ids, max_new_tokens=max_new_tokens, threads=threads
        )
        if type(max_proposals) is not int or not 0 <= max_proposals <= 3:
            raise ValueError("max_proposals must be between zero and three")
        if type(adversarial_proposals) is not bool:
            raise ValueError("adversarial_proposals must be a boolean")
        report = self._run_plan_command(
            [
                "--token-wave",
                str(self.artifact),
                str(self.checkpoint),
                ",".join(str(value) for value in prompt_token_ids),
                str(max_new_tokens),
                str(max_proposals),
                str(threads),
                "1" if adversarial_proposals else "0",
            ],
            "token-wave",
        )
        return self._plan_wave_result(report)

    def _experimental_generate_perfect_oracle(
        self,
        prompt_token_ids: list[int],
        *,
        known_future_token_ids: list[int],
        width: int,
        threads: int,
    ) -> NativePlanTokenWaveResult:
        """Run an internal perfect-proposal ceiling, not practical generation."""
        self._validate_plan_generation_inputs(
            prompt_token_ids, max_new_tokens=1, threads=threads
        )
        vocab_size = dict(self.plan.shapes)["vocab_size"]
        if not isinstance(known_future_token_ids, list) or not known_future_token_ids:
            raise ValueError(
                "native perfect-oracle inference requires known future tokens"
            )
        if any(
            type(token) is not int or token < 0 or token >= vocab_size
            for token in known_future_token_ids
        ):
            raise ValueError("perfect-oracle token is outside the plan vocabulary")
        if type(width) is not int or width not in (1, 2, 4, 8, 16):
            raise ValueError("perfect-oracle width must be 1, 2, 4, 8, or 16")
        report = self._run_plan_command(
            [
                "--experimental-perfect-oracle",
                str(self.artifact),
                str(self.checkpoint),
                ",".join(str(value) for value in prompt_token_ids),
                ",".join(str(value) for value in known_future_token_ids),
                str(width),
                str(threads),
            ],
            "experimental perfect-oracle",
        )
        if report.get("experimental") != "perfect-oracle-causal-panel-ceiling":
            raise RuntimeError("native perfect-oracle result lacks experimental marker")
        return self._plan_wave_result(
            report,
            rolled_back_tokens=0,
            speculative_traversals=int(report["target_weight_traversals"]),
            fallback_traversals=0,
        )

    def _validate_plan_generation_inputs(
        self, prompt_token_ids: list[int], *, max_new_tokens: int, threads: int
    ) -> None:
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise ValueError("native inference requires at least one prompt token")
        vocab_size = dict(self.plan.shapes)["vocab_size"]
        if any(
            type(token) is not int or token < 0 or token >= vocab_size
            for token in prompt_token_ids
        ):
            raise ValueError("prompt token is outside the plan vocabulary")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if type(threads) is not int or not 1 <= threads <= _MAX_NATIVE_PLAN_THREADS:
            raise ValueError("threads must be between 1 and 256")

    def _run_plan_command(self, arguments: list[str], operation: str) -> dict[str, Any]:
        completed = subprocess.run(
            [str(self.executable), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"StrPot native execution-plan {operation} inference failed:\n"
                + completed.stdout
                + completed.stderr
            )
        return json.loads(completed.stdout)

    @staticmethod
    def _plan_wave_result(
        report: dict[str, Any],
        *,
        rolled_back_tokens: int | None = None,
        speculative_traversals: int | None = None,
        fallback_traversals: int | None = None,
    ) -> NativePlanTokenWaveResult:
        return NativePlanTokenWaveResult(
            generated_token_ids=tuple(int(value) for value in report["tokens"]),
            target_weight_traversals=int(report["target_weight_traversals"]),
            committed_decode_tokens=int(report["committed_decode_tokens"]),
            committed_tokens_per_target_weight_traversal=float(
                report["committed_tokens_per_target_weight_traversal"]
            ),
            acceptance_lengths=tuple(
                int(value) for value in report["acceptance_lengths"]
            ),
            traversal_seconds=tuple(
                float(value) for value in report["traversal_seconds"]
            ),
            rolled_back_tokens=(
                int(report["rolled_back_tokens"])
                if rolled_back_tokens is None
                else rolled_back_tokens
            ),
            speculative_traversals=(
                int(report["speculative_traversals"])
                if speculative_traversals is None
                else speculative_traversals
            ),
            fallback_traversals=(
                int(report["fallback_traversals"])
                if fallback_traversals is None
                else fallback_traversals
            ),
            transactional_snapshot_bytes_copied=int(
                report["transactional_snapshot_bytes_copied"]
            ),
            rollback_verified=bool(report["rollback_verified"]),
            final_logits_hash=str(report["final_logits_hash"]),
            final_kv_hash=str(report["final_kv_hash"]),
            final_kv_lengths=tuple(int(value) for value in report["kv_cache_lengths"]),
            frontier_logits_hashes=tuple(
                str(value) for value in report["frontier_logits_hashes"]
            ),
            frontier_kv_hashes=tuple(
                str(value) for value in report["frontier_kv_hashes"]
            ),
            matrix_passes=int(report["matrix_passes"]),
            runtime_tensor_lookups=int(report["runtime_tensor_lookups"]),
            threads=int(report["threads"]),
            kernel_family=str(report["kernel_family"]),
            native_panel_8_calls=int(report["native_panel_8_calls"]),
            native_panel_16_calls=int(report["native_panel_16_calls"]),
        )


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
