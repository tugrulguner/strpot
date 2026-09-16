from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from decoder_oracle import DecoderOracle, OracleResult

from strpot.image import compile_image
from strpot.native import (
    ExecutionPlan,
    GraphValue,
    NativeExecutionPlanEngine,
    NativePlanGenerationResult,
    NumericalSemantics,
    Operator,
    TensorBinding,
)


def _f32_semantics() -> NumericalSemantics:
    return NumericalSemantics(
        weight_dtype="F32",
        activation_dtype="F32",
        output_dtype="F32",
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


def test_numerical_semantics_are_explicit() -> None:
    semantics = _f32_semantics()

    assert semantics.activation_dtype == "F32"
    assert semantics.output_dtype == "F32"
    assert semantics.accumulation_order == "ordered"
    assert semantics.rounding == "nearest_even"
    assert semantics.contraction_policy == "disabled"
    assert semantics.softmax_policy == "stable_max_subtraction"
    assert semantics.transcendental_policy == "system_libm"
    assert semantics.gelu_formula == "x_times_half_times_one_plus_erf_x_over_sqrt_two"
    assert semantics.silu_formula == "x_over_one_plus_exp_neg_x"
    assert semantics.rope_formula == "pair_rotation_theta_pow_2i_over_head_dim"


def test_graph_value_is_typed_and_immutable() -> None:
    value = GraphValue("hidden.0", "F32", ("hidden_size",), "activation")

    assert value.shape == ("hidden_size",)
    with pytest.raises(AttributeError):
        value.name = "changed"  # type: ignore[misc]


def test_execution_plan_rejects_duplicate_ssa_output_definitions() -> None:
    with pytest.raises(ValueError, match=r"duplicate output definition hidden\.0"):
        ExecutionPlan(
            version=1,
            architecture_id="metadata-only",
            config_identity="config-v1",
            shapes=(("hidden_size", 2), ("vocab_size", 2), ("context_size", 2)),
            semantics=_f32_semantics(),
            tensors=(),
            values=(GraphValue("hidden.0", "F32", ("hidden_size",), "activation"),),
            operators=(
                Operator("unknown", outputs=("hidden.0",)),
                Operator("unknown", outputs=("hidden.0",)),
            ),
            eos_token_id=0,
        )


def _write_f32_safetensors(
    path: Path,
    tensors: dict[str, tuple[tuple[int, ...], list[float]]],
    *,
    payload_prefix: bytes = b"",
) -> None:
    header: dict[str, object] = {}
    payload = bytearray(payload_prefix)
    for name, (shape, values) in tensors.items():
        start = len(payload)
        payload.extend(struct.pack(f"<{len(values)}f", *values))
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _write_bf16_safetensors(
    path: Path,
    tensors: dict[str, tuple[tuple[int, ...], list[float]]],
    *,
    payload_prefix: bytes = b"",
) -> None:
    header: dict[str, object] = {}
    payload = bytearray(payload_prefix)
    for name, (shape, values) in tensors.items():
        start = len(payload)
        for value in values:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            upper, lower = bits >> 16, bits & 0xFFFF
            if lower > 0x8000 or (lower == 0x8000 and upper & 1):
                upper = (upper + 1) & 0xFFFF
            payload.extend(struct.pack("<H", upper))
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _identity(rows: int, columns: int, scale: float = 1.0) -> list[float]:
    return [
        scale if row == column else 0.0
        for row in range(rows)
        for column in range(columns)
    ]


def _assert_native_matches_oracle(
    native: NativePlanGenerationResult,
    oracle: OracleResult,
    *,
    tolerance: float = 2e-6,
) -> None:
    assert native.generated_token_ids == oracle.tokens
    assert native.kv_cache_lengths == oracle.cache_lengths
    assert len(native.frontier_logits) == len(oracle.frontier_logits)
    for actual, expected in zip(
        native.frontier_logits, oracle.frontier_logits, strict=True
    ):
        assert actual == pytest.approx(expected, abs=tolerance, rel=tolerance)
    for actual_caches, expected_caches in (
        (native.kv_cache_keys, oracle.cache_keys),
        (native.kv_cache_values, oracle.cache_values),
    ):
        assert len(actual_caches) == len(expected_caches)
        for actual, expected in zip(actual_caches, expected_caches, strict=True):
            assert actual == pytest.approx(expected, abs=tolerance, rel=tolerance)


def test_native_plan_executes_rms_rope_swiglu_decoder_end_to_end(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    tensors = {
        "embed": (
            (5, 4),
            [0.0] * 4
            + [1.0, 0.5, -0.25, 0.75]
            + [0.2, -0.4, 0.8, 0.1]
            + [0.7, 0.3, 0.2, -0.6]
            + [-0.5, 0.9, 0.1, 0.4],
        ),
        "norm1": ((4,), [1.0, 0.9, 1.1, 0.8]),
        "q": ((4, 4), _identity(4, 4, 0.5)),
        "k": ((2, 4), _identity(2, 4, 0.4)),
        "v": ((2, 4), [0.3, 0.1, -0.2, 0.4, -0.1, 0.2, 0.5, 0.1]),
        "o": ((4, 4), _identity(4, 4, 0.3)),
        "norm2": ((4,), [0.8, 1.0, 0.9, 1.1]),
        "gate": ((6, 4), [0.2, -0.1, 0.3, 0.1] * 6),
        "up": ((6, 4), [0.1, 0.2, -0.1, 0.3] * 6),
        "down": ((4, 6), [0.05] * 24),
        "final_norm": ((4,), [1.0] * 4),
        "head": (
            (5, 4),
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
                0.1,
                -0.2,
                0.3,
                -0.3,
                0.9,
                0.2,
                0.1,
                0.1,
                -0.2,
                0.7,
                0.6,
                0.4,
                0.3,
                -0.5,
                0.8,
            ],
        ),
    }
    _write_f32_safetensors(source, tensors)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=61)
    bindings = tuple(
        TensorBinding(alias=name, checkpoint_name=name, dtype="F32", shape=shape)
        for name, (shape, _values) in tensors.items()
    )
    plan = ExecutionPlan(
        version=1,
        architecture_id="rms-rope-swiglu-decoder-v1",
        config_identity="fixture-a-v1",
        shapes=(("hidden_size", 4), ("vocab_size", 5), ("context_size", 12)),
        semantics=_f32_semantics(),
        tensors=bindings,
        values=tuple(
            GraphValue(
                name,
                "F32",
                ("vocab_size",) if name == "logits" else ("hidden_size",),
                "logits" if name == "logits" else "activation",
            )
            for name in (
                "hidden.0",
                "residual.0",
                "normed.0",
                "attention.0",
                "hidden.1",
                "residual.1",
                "normed.1",
                "mlp.0",
                "hidden.2",
                "hidden.3",
                "logits",
            )
        ),
        operators=(
            Operator("embedding", outputs=("hidden.0",), tensors=("embed",)),
            Operator("save", inputs=("hidden.0",), outputs=("residual.0",)),
            Operator(
                "rms_norm",
                inputs=("hidden.0",),
                outputs=("normed.0",),
                tensors=("norm1",),
                attributes=(("epsilon", "1e-5"),),
            ),
            Operator(
                "attention_rope",
                inputs=("normed.0",),
                outputs=("attention.0",),
                tensors=("q", "k", "v", "o"),
                attributes=(
                    ("cache", "layer0"),
                    ("heads", "2"),
                    ("kv_heads", "1"),
                    ("theta", "10000"),
                ),
            ),
            Operator(
                "add", inputs=("residual.0", "attention.0"), outputs=("hidden.1",)
            ),
            Operator("save", inputs=("hidden.1",), outputs=("residual.1",)),
            Operator(
                "rms_norm",
                inputs=("hidden.1",),
                outputs=("normed.1",),
                tensors=("norm2",),
                attributes=(("epsilon", "1e-5"),),
            ),
            Operator(
                "swiglu",
                inputs=("normed.1",),
                outputs=("mlp.0",),
                tensors=("gate", "up", "down"),
            ),
            Operator("add", inputs=("residual.1", "mlp.0"), outputs=("hidden.2",)),
            Operator(
                "rms_norm",
                inputs=("hidden.2",),
                outputs=("hidden.3",),
                tensors=("final_norm",),
                attributes=(("epsilon", "1e-5"),),
            ),
            Operator(
                "linear", inputs=("hidden.3",), outputs=("logits",), tensors=("head",)
            ),
        ),
        eos_token_id=0,
    )
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native"
    )

    result = engine.generate([1, 2], max_new_tokens=2, threads=1)
    oracle = DecoderOracle(plan, tensors).generate([1, 2], max_new_tokens=2)

    assert result.engine == "strpot-native-plan"
    assert result.architecture_id == "rms-rope-swiglu-decoder-v1"
    _assert_native_matches_oracle(result, oracle)
    assert result.executed_operators == len(plan.operators) * 3
    expected_trace = tuple(operator.kind for operator in plan.operators) * 3
    assert result.operator_trace == expected_trace
    assert len(result.operator_trace) == result.executed_operators
    assert len(result.frontier_logits) == 2
    assert len(result.frontier_logits[0]) == 5
    assert len(result.kv_cache_keys) == len(result.kv_cache_lengths)
    assert len(result.kv_cache_values) == len(result.kv_cache_lengths)
    artifact = engine.artifact.read_text(encoding="utf-8")
    assert artifact.startswith("STRPOT_EXECUTION_PLAN_V2\n")
    assert "identity\tarchitecture\trms-rope-swiglu-decoder-v1" in artifact
    assert "semantic\taccumulator_dtype\tF32" in artifact
    assert "op\tattention_rope\t" in artifact


def test_same_native_executor_runs_layernorm_learned_position_gelu_decoder(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    hidden = 6
    vocab = 7
    context = 9
    intermediate = 5
    tensors = {
        "token_embed": (
            (vocab, hidden),
            [((index * 7) % 19 - 9) / 10 for index in range(vocab * hidden)],
        ),
        "position_embed": (
            (context, hidden),
            [((index * 5) % 17 - 8) / 20 for index in range(context * hidden)],
        ),
        "ln1_scale": ((hidden,), [0.7, 1.1, 0.8, 1.2, 0.9, 1.3]),
        "ln1_bias": ((hidden,), [0.2, -0.1, 0.05, -0.2, 0.1, -0.05]),
        "q2": ((hidden, hidden), [((i * 3) % 13 - 6) / 15 for i in range(36)]),
        "k2": ((hidden, hidden), [((i * 5) % 11 - 5) / 16 for i in range(36)]),
        "v2": ((hidden, hidden), [((i * 7) % 17 - 8) / 18 for i in range(36)]),
        "o2": ((hidden, hidden), [((i * 2) % 9 - 4) / 14 for i in range(36)]),
        "ln2_scale": ((hidden,), [1.0, 0.8, 1.2, 0.7, 1.1, 0.9]),
        "ln2_bias": ((hidden,), [-0.1, 0.2, -0.05, 0.1, -0.2, 0.05]),
        "fc1": (
            (intermediate, hidden),
            [((i * 11) % 23 - 11) / 20 for i in range(intermediate * hidden)],
        ),
        "fc1_bias": ((intermediate,), [0.3, -0.2, 0.1, -0.4, 0.25]),
        "fc2": (
            (hidden, intermediate),
            [((i * 13) % 29 - 14) / 22 for i in range(hidden * intermediate)],
        ),
        "fc2_bias": ((hidden,), [-0.2, 0.1, 0.3, -0.1, 0.05, -0.25]),
        "final_scale": ((hidden,), [0.9, 1.1, 0.7, 1.2, 0.8, 1.0]),
        "final_bias": ((hidden,), [0.1, -0.1, 0.2, -0.2, 0.05, -0.05]),
        "head2": ((vocab, hidden), [((i * 17) % 31 - 15) / 19 for i in range(42)]),
        "head_bias": ((vocab,), [0.1, -0.2, 0.3, -0.1, 0.05, 0.2, -0.3]),
    }
    _write_f32_safetensors(source, tensors)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=79)
    plan = ExecutionPlan(
        version=1,
        architecture_id="layernorm-position-gelu-decoder-v1",
        config_identity="fixture-b-v1",
        shapes=(
            ("hidden_size", hidden),
            ("vocab_size", vocab),
            ("context_size", context),
        ),
        semantics=_f32_semantics(),
        tensors=tuple(
            TensorBinding(name, name, "F32", shape)
            for name, (shape, _values) in tensors.items()
        ),
        values=tuple(
            GraphValue(
                name,
                "F32",
                ("vocab_size",) if name == "logits" else ("hidden_size",),
                "logits" if name == "logits" else "activation",
            )
            for name in (
                "hidden.0",
                "hidden.1",
                "residual.0",
                "normed.0",
                "attention.0",
                "hidden.2",
                "residual.1",
                "normed.1",
                "mlp.0",
                "hidden.3",
                "hidden.4",
                "logits",
            )
        ),
        operators=(
            Operator("embedding", outputs=("hidden.0",), tensors=("token_embed",)),
            Operator(
                "position_embedding",
                inputs=("hidden.0",),
                outputs=("hidden.1",),
                tensors=("position_embed",),
            ),
            Operator("save", inputs=("hidden.1",), outputs=("residual.0",)),
            Operator(
                "layer_norm",
                inputs=("hidden.1",),
                outputs=("normed.0",),
                tensors=("ln1_scale", "ln1_bias"),
                attributes=(("epsilon", "1e-5"),),
            ),
            Operator(
                "attention_causal",
                inputs=("normed.0",),
                outputs=("attention.0",),
                tensors=("q2", "k2", "v2", "o2"),
                attributes=(("cache", "block0"), ("heads", "3"), ("kv_heads", "3")),
            ),
            Operator(
                "add", inputs=("residual.0", "attention.0"), outputs=("hidden.2",)
            ),
            Operator("save", inputs=("hidden.2",), outputs=("residual.1",)),
            Operator(
                "layer_norm",
                inputs=("hidden.2",),
                outputs=("normed.1",),
                tensors=("ln2_scale", "ln2_bias"),
                attributes=(("epsilon", "1e-5"),),
            ),
            Operator(
                "gelu_exact",
                inputs=("normed.1",),
                outputs=("mlp.0",),
                tensors=("fc1", "fc1_bias", "fc2", "fc2_bias"),
                attributes=(("formula", "erf"),),
            ),
            Operator("add", inputs=("residual.1", "mlp.0"), outputs=("hidden.3",)),
            Operator(
                "layer_norm",
                inputs=("hidden.3",),
                outputs=("hidden.4",),
                tensors=("final_scale", "final_bias"),
                attributes=(("epsilon", "1e-5"),),
            ),
            Operator(
                "linear",
                inputs=("hidden.4",),
                outputs=("logits",),
                tensors=("head2", "head_bias"),
            ),
        ),
        eos_token_id=0,
    )
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native"
    )

    result = engine.generate([2, 5, 1], max_new_tokens=2, threads=1)
    oracle = DecoderOracle(plan, tensors).generate([2, 5, 1], max_new_tokens=2)

    assert result.architecture_id == "layernorm-position-gelu-decoder-v1"
    _assert_native_matches_oracle(result, oracle)
    expected_trace = tuple(operator.kind for operator in plan.operators) * 4
    assert result.operator_trace == expected_trace
    assert len(result.operator_trace) == result.executed_operators
    assert engine.executable.name == "strpot-native-plan"
    artifact = engine.artifact.read_text(encoding="utf-8")
    assert "op\tposition_embedding\t" in artifact
    assert "op\tlayer_norm\t" in artifact
    assert "op\tgelu_exact\t" in artifact

    mutated_tensors = dict(tensors)
    bias_shape, bias_values = tensors["ln1_bias"]
    mutated_tensors["ln1_bias"] = (
        bias_shape,
        [bias_values[0] + 0.25, *bias_values[1:]],
    )
    mutated_source = tmp_path / "mutated-weights.safetensors"
    _write_f32_safetensors(mutated_source, mutated_tensors)
    mutated_image = tmp_path / "mutated-weights.strpot"
    compile_image(mutated_source, mutated_image, page_size=79)
    mutated_native = NativeExecutionPlanEngine(
        plan=replace(plan, config_identity="fixture-b-normalization-bias-mutation"),
        weights_image=mutated_image,
        cache_root=tmp_path / "mutated-native",
    ).generate([2, 5, 1], max_new_tokens=2, threads=1)
    mutated_oracle = DecoderOracle(plan, mutated_tensors).generate(
        [2, 5, 1], max_new_tokens=2
    )

    _assert_native_matches_oracle(mutated_native, mutated_oracle)
    assert mutated_oracle.frontier_logits != oracle.frontier_logits


def _order_fixture(tmp_path: Path) -> tuple[ExecutionPlan, Path]:
    tensors = {
        "embed": ((3, 3), [0.0, 0.0, 0.0, 1.0, 0.2, -0.5, -0.3, 0.8, 0.4]),
        "scale": ((3,), [0.5, 1.5, 0.7]),
        "transform": ((3, 3), [0.2, 0.8, -0.1, -0.5, 0.3, 0.9, 0.7, -0.4, 0.1]),
        "head": ((3, 3), [0.9, -0.2, 0.1, -0.3, 0.8, 0.4, 0.2, 0.1, 1.0]),
    }
    source = tmp_path / "order.safetensors"
    _write_f32_safetensors(source, tensors)
    image = tmp_path / "order.strpot"
    compile_image(source, image)
    common = {
        "version": 1,
        "architecture_id": "operator-order-probe-v1",
        "config_identity": "order-a",
        "shapes": (("hidden_size", 3), ("vocab_size", 3), ("context_size", 4)),
        "semantics": _f32_semantics(),
        "tensors": tuple(
            TensorBinding(name, name, "F32", shape)
            for name, (shape, _values) in tensors.items()
        ),
        "values": (
            GraphValue("hidden.0", "F32", ("hidden_size",), "activation"),
            GraphValue("hidden.1", "F32", ("hidden_size",), "activation"),
            GraphValue("hidden.2", "F32", ("hidden_size",), "activation"),
            GraphValue("logits", "F32", ("vocab_size",), "logits"),
        ),
        "eos_token_id": -1,
    }
    plan = ExecutionPlan(
        **common,
        operators=(
            Operator("embedding", outputs=("hidden.0",), tensors=("embed",)),
            Operator(
                "rms_norm",
                inputs=("hidden.0",),
                outputs=("hidden.1",),
                tensors=("scale",),
                attributes=(("epsilon", "1e-5"),),
            ),
            Operator(
                "linear",
                inputs=("hidden.1",),
                outputs=("hidden.2",),
                tensors=("transform",),
            ),
            Operator(
                "linear", inputs=("hidden.2",), outputs=("logits",), tensors=("head",)
            ),
        ),
    )
    return plan, image


def _invoke_rehashed_native_artifact(
    engine: NativeExecutionPlanEngine,
    old: str,
    new: str,
) -> subprocess.CompletedProcess[str]:
    artifact = engine.artifact.read_text(encoding="utf-8")
    assert artifact.count(old) == 1
    lines = artifact.splitlines()
    body_lines = lines[4:]
    body = "\n".join(body_lines).replace(old, new) + "\n"
    assert body != "\n".join(body_lines) + "\n"
    plan_bytes = (
        "\n".join(
            line for line in body.splitlines() if not line.startswith("tensor_data\t")
        )
        + "\n"
    )
    lines[1] = (
        f"artifact\tplan_sha256\t{hashlib.sha256(plan_bytes.encode()).hexdigest()}"
    )
    lines[3] = f"artifact\tbody_sha256\t{hashlib.sha256(body.encode()).hexdigest()}"
    engine.artifact.write_text("\n".join(lines[:4]) + "\n" + body, encoding="utf-8")
    return subprocess.run(
        [
            str(engine.executable),
            str(engine.artifact),
            str(engine.checkpoint),
            "1",
            "1",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _invoke_native_after_checkpoint_edit(
    engine: NativeExecutionPlanEngine, checkpoint: bytes
) -> subprocess.CompletedProcess[str]:
    engine.checkpoint.write_bytes(checkpoint)
    lines = engine.artifact.read_text(encoding="utf-8").splitlines()
    lines[2] = f"artifact\tcheckpoint_sha256\t{hashlib.sha256(checkpoint).hexdigest()}"
    engine.artifact.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return subprocess.run(
        [
            str(engine.executable),
            str(engine.artifact),
            str(engine.checkpoint),
            "1",
            "1",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "value\thidden.0\tF32\thidden_size\tactivation",
            "value\thidden.0\tBF16\thidden_size\tactivation",
            "violates declared dtype",
        ),
        (
            "value\thidden.0\tF32\thidden_size\tactivation",
            "value\thidden.0\tF32\thidden_size\tnonsense",
            "invalid graph value lifetime",
        ),
        (
            "value\thidden.0\tF32\thidden_size\tactivation",
            "value\thidden.0\tF32\tunknown_size\tactivation",
            "unknown symbolic dimension",
        ),
        (
            "value\thidden.0\tF32\thidden_size\tactivation",
            "value\thidden.0\tF32\t2\tactivation",
            "embedding value shape mismatch",
        ),
        (
            "value\thidden.0\tF32\thidden_size\tactivation",
            "value\thidden.0\tF32\t0\tactivation",
            "invalid graph value dimension 0",
        ),
        (
            "value\thidden.0\tF32\thidden_size\tactivation",
            "value\thidden.0\tF32\thidden_size\tstate",
            "violates declared lifetime",
        ),
        (
            "op\tlinear\thidden.1\thidden.2\ttransform\t-",
            "op\tlinear\thidden.1\thidden.0\ttransform\t-",
            "duplicate output definition",
        ),
        (
            "op\tlinear\thidden.1\thidden.2\ttransform\t-",
            "op\tlinear\thidden.1\tundeclared\ttransform\t-",
            "graph value is not declared: undeclared",
        ),
        (
            "op\trms_norm\thidden.0\thidden.1\tscale\tepsilon=1e-5\n"
            "op\tlinear\thidden.1\thidden.2\ttransform\t-",
            "op\tlinear\thidden.1\thidden.2\ttransform\t-\n"
            "op\trms_norm\thidden.0\thidden.1\tscale\tepsilon=1e-5",
            "operator dependency is unavailable: hidden.1",
        ),
        (
            "op\tlinear\thidden.2\tlogits\thead\t-",
            "op\tlinear\thidden.1\tlogits\thead\t-",
            "graph values are unused: hidden.2",
        ),
        (
            "value\thidden.2\tF32\thidden_size\tactivation\n",
            "value\thidden.2\tF32\thidden_size\tactivation\n"
            "value\tunused\tF32\thidden_size\tactivation\n",
            "graph values are not defined",
        ),
        (
            "value\tlogits\tF32\tvocab_size\tlogits",
            "value\tlogits\tF32\thidden_size\tactivation",
            "logits must be a vocab_size logits value",
        ),
    ],
)
def test_native_rejects_hash_valid_malformed_typed_ssa(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native-ssa"
    )

    completed = _invoke_rehashed_native_artifact(engine, old, new)

    assert completed.returncode != 0
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "op\trms_norm\thidden.0\thidden.1\tscale\tepsilon=1e-5",
            "op\trms_norm\thidden.0\thidden.1\tscale\tepsilon=1e-5,junk=1",
            "invalid attributes for rms_norm",
        ),
        ("epsilon=1e-5", "epsilon=nan", "invalid positive finite attribute epsilon"),
        ("epsilon=1e-5", "epsilon=1e-5junk", "malformed numeric attribute epsilon"),
    ],
)
def test_native_rejects_hash_valid_invalid_operator_attributes(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native-attributes"
    )

    completed = _invoke_rehashed_native_artifact(engine, old, new)

    assert completed.returncode != 0
    assert message in completed.stderr


def test_native_rejects_hash_valid_false_tensor_data_length(tmp_path: Path) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native-length"
    )
    artifact = engine.artifact.read_text(encoding="utf-8")
    scale_data = next(
        line
        for line in artifact.splitlines()
        if line.startswith("tensor_data\tscale\t")
    )
    fields = scale_data.split("\t")
    assert fields[3] == "12"

    completed = _invoke_rehashed_native_artifact(
        engine, scale_data, "\t".join((*fields[:3], "1"))
    )

    assert completed.returncode != 0
    assert "tensor byte length mismatch for scale" in completed.stderr


@pytest.mark.parametrize("mutation", ["checkpoint_name", "tensor_data"])
def test_native_rejects_hash_valid_inexact_checkpoint_binding(
    tmp_path: Path, mutation: str
) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / f"native-{mutation}"
    )
    artifact = engine.artifact.read_text(encoding="utf-8")
    scale_data = next(
        line
        for line in artifact.splitlines()
        if line.startswith("tensor_data\tscale\t")
    )
    fields = scale_data.split("\t")
    if mutation == "checkpoint_name":
        old = "binding\tscale\tscale\tF32\t3"
        new = "binding\tscale\tnonexistent\tF32\t3"
    else:
        old = scale_data
        new = "\t".join((fields[0], fields[1], str(int(fields[2]) - 12), fields[3]))

    completed = _invoke_rehashed_native_artifact(engine, old, new)

    assert completed.returncode != 0
    assert "checkpoint tensor binding mismatch" in completed.stderr


def test_native_rejects_hash_valid_unused_plan_tensor_binding(tmp_path: Path) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native-unused-binding"
    )

    completed = _invoke_rehashed_native_artifact(
        engine,
        "op\tlinear\thidden.1\thidden.2\ttransform\t-",
        "op\tlinear\thidden.1\thidden.2\thead\t-",
    )

    assert completed.returncode != 0
    assert "unused plan tensor binding: transform" in completed.stderr


@pytest.mark.parametrize("record", ["shape", "binding", "value"])
def test_native_rejects_hash_valid_noncanonical_plan_record_order(
    tmp_path: Path, record: str
) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / f"canonical-{record}"
    )
    artifact = engine.artifact.read_text(encoding="utf-8")
    body_lines = artifact.splitlines()[4:]
    indices = [
        index for index, line in enumerate(body_lines) if line.startswith(record + "\t")
    ]
    assert len(indices) >= 2
    first, second = indices[:2]
    old = body_lines[first] + "\n" + body_lines[second]
    new = body_lines[second] + "\n" + body_lines[first]

    completed = _invoke_rehashed_native_artifact(engine, old, new)

    assert completed.returncode != 0
    assert "noncanonical execution plan serialization" in completed.stderr


def test_native_rejects_hash_valid_tensor_shape_product_overflow(
    tmp_path: Path,
) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native-overflow"
    )

    completed = _invoke_rehashed_native_artifact(
        engine,
        "binding\tscale\tscale\tF32\t3",
        "binding\tscale\tscale\tF32\t18446744073709551615,2",
    )

    assert completed.returncode != 0
    assert "tensor shape product overflow for scale" in completed.stderr


def test_operator_order_changes_native_generation(tmp_path: Path) -> None:
    plan, image = _order_fixture(tmp_path)
    reordered = replace(
        plan,
        config_identity="order-b",
        operators=(
            plan.operators[0],
            Operator(
                "linear",
                inputs=("hidden.0",),
                outputs=("hidden.1",),
                tensors=("transform",),
            ),
            Operator(
                "rms_norm",
                inputs=("hidden.1",),
                outputs=("hidden.2",),
                tensors=("scale",),
                attributes=(("epsilon", "1e-5"),),
            ),
            plan.operators[3],
        ),
    )

    first = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "first"
    ).generate([2], max_new_tokens=1, threads=1)
    second = NativeExecutionPlanEngine(
        plan=reordered, weights_image=image, cache_root=tmp_path / "second"
    ).generate([2], max_new_tokens=1, threads=1)

    assert first.generated_token_ids != second.generated_token_ids
    assert first.operator_trace != second.operator_trace


def test_artifact_has_canonical_identity_and_native_rejects_tampering(
    tmp_path: Path,
) -> None:
    plan, image = _order_fixture(tmp_path)
    first = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "cache"
    )
    second = NativeExecutionPlanEngine(
        plan=replace(plan, architecture_id="operator-order-probe-v1"),
        weights_image=image,
        cache_root=tmp_path / "cache",
    )

    assert first.model_root == second.model_root
    artifact = first.artifact.read_text(encoding="utf-8")
    assert "artifact\tplan_sha256\t" in artifact
    assert "artifact\tcheckpoint_sha256\t" in artifact
    first.artifact.write_text(artifact.replace("order-a", "order-x"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact integrity"):
        first.generate([1], max_new_tokens=1, threads=1)


def test_native_plan_rejects_unsupported_operator(tmp_path: Path) -> None:
    plan, image = _order_fixture(tmp_path)
    unsupported = replace(
        plan,
        values=(plan.values[0], plan.values[1], plan.values[-1]),
        operators=(
            plan.operators[0],
            Operator("fourier_magic", inputs=("hidden.0",), outputs=("hidden.1",)),
            Operator(
                "linear", inputs=("hidden.1",), outputs=("logits",), tensors=("head",)
            ),
        ),
    )
    engine = NativeExecutionPlanEngine(
        plan=unsupported, weights_image=image, cache_root=tmp_path / "unsupported"
    )

    with pytest.raises(RuntimeError, match="unsupported operator fourier_magic"):
        engine.generate([1], max_new_tokens=1, threads=1)


def test_native_plan_threads_execute_equivalent_output_and_report_matrix_path(
    tmp_path: Path,
) -> None:
    plan, image = _order_fixture(tmp_path)
    single = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "single"
    ).generate([1], max_new_tokens=1, threads=1)
    threaded = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "threaded"
    ).generate([1], max_new_tokens=1, threads=2)

    assert threaded.generated_token_ids == single.generated_token_ids
    assert threaded.frontier_logits == single.frontier_logits
    assert threaded.kv_cache_keys == single.kv_cache_keys
    assert threaded.kv_cache_values == single.kv_cache_values
    assert threaded.threads == 2
    assert threaded.kernel_family == "portable-tiled"
    assert threaded.matrix_passes == single.matrix_passes > 0
    assert threaded.parallel_dispatches == threaded.matrix_passes
    assert threaded.threaded_matrix_dispatches > 0
    assert threaded.runtime_tensor_lookups == 0


@pytest.mark.parametrize("prompt_width", [1, 2, 3, 4, 5])
def test_block_prefill_preserves_sequential_semantics_and_constant_matrix_passes(
    tmp_path: Path, prompt_width: int
) -> None:
    plan, image = _order_fixture(tmp_path)
    plan = replace(
        plan,
        shapes=(("hidden_size", 3), ("vocab_size", 3), ("context_size", 6)),
    )
    prompt = [1, 2, 1, 2, 1][:prompt_width]
    native = NativeExecutionPlanEngine(
        plan=plan,
        weights_image=image,
        cache_root=tmp_path / f"block-{prompt_width}",
    ).generate(prompt, max_new_tokens=1, threads=2)
    oracle = DecoderOracle(
        plan,
        {
            "embed": ((3, 3), [0.0, 0.0, 0.0, 1.0, 0.2, -0.5, -0.3, 0.8, 0.4]),
            "scale": ((3,), [0.5, 1.5, 0.7]),
            "transform": ((3, 3), [0.2, 0.8, -0.1, -0.5, 0.3, 0.9, 0.7, -0.4, 0.1]),
            "head": ((3, 3), [0.9, -0.2, 0.1, -0.3, 0.8, 0.4, 0.2, 0.1, 1.0]),
        },
    ).generate(prompt, max_new_tokens=1)

    _assert_native_matches_oracle(native, oracle)
    assert native.prefill_matrix_passes == 2
    assert native.prefill_physical_width == prompt_width
    assert native.executed_operators == len(plan.operators) * prompt_width
    assert (
        native.operator_trace == tuple(op.kind for op in plan.operators) * prompt_width
    )


@pytest.mark.parametrize("prompt_width", [2, 3, 4, 5])
def test_block_attention_is_causal_and_matches_sequential_kv(
    tmp_path: Path, prompt_width: int
) -> None:
    # The nonzero attention fixture makes future-token leakage observable
    # in both logits and KV state.
    source = tmp_path / "weights.safetensors"
    tensors = {
        "embed": (
            (5, 4),
            [0.0] * 4
            + [1.0, 0.5, -0.25, 0.75]
            + [0.2, -0.4, 0.8, 0.1]
            + [0.7, 0.3, 0.2, -0.6]
            + [-0.5, 0.9, 0.1, 0.4],
        ),
        "norm1": ((4,), [1.0, 0.9, 1.1, 0.8]),
        "q": ((4, 4), _identity(4, 4, 0.5)),
        "k": ((2, 4), _identity(2, 4, 0.4)),
        "v": ((2, 4), [0.3, 0.1, -0.2, 0.4, -0.1, 0.2, 0.5, 0.1]),
        "o": ((4, 4), _identity(4, 4, 0.3)),
        "norm2": ((4,), [0.8, 1.0, 0.9, 1.1]),
        "gate": ((6, 4), [0.2, -0.1, 0.3, 0.1] * 6),
        "up": ((6, 4), [0.1, 0.2, -0.1, 0.3] * 6),
        "down": ((4, 6), [0.05] * 24),
        "final_norm": ((4,), [1.0] * 4),
        "head": (
            (5, 4),
            [0.0] * 4
            + [0.8, 0.1, -0.2, 0.3]
            + [-0.3, 0.9, 0.2, 0.1]
            + [0.1, -0.2, 0.7, 0.6]
            + [0.4, 0.3, -0.5, 0.8],
        ),
    }
    _write_f32_safetensors(source, tensors)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=61)
    bindings = tuple(
        TensorBinding(name, name, "F32", shape) for name, (shape, _) in tensors.items()
    )
    values = tuple(
        GraphValue(
            name,
            "F32",
            ("vocab_size",) if name == "logits" else ("hidden_size",),
            "logits" if name == "logits" else "activation",
        )
        for name in (
            "hidden.0",
            "residual.0",
            "normed.0",
            "attention.0",
            "hidden.1",
            "residual.1",
            "normed.1",
            "mlp.0",
            "hidden.2",
            "hidden.3",
            "logits",
        )
    )
    ops = (
        Operator("embedding", outputs=("hidden.0",), tensors=("embed",)),
        Operator("save", inputs=("hidden.0",), outputs=("residual.0",)),
        Operator(
            "rms_norm",
            inputs=("hidden.0",),
            outputs=("normed.0",),
            tensors=("norm1",),
            attributes=(("epsilon", "1e-5"),),
        ),
        Operator(
            "attention_rope",
            inputs=("normed.0",),
            outputs=("attention.0",),
            tensors=("q", "k", "v", "o"),
            attributes=(
                ("cache", "layer0"),
                ("heads", "2"),
                ("kv_heads", "1"),
                ("theta", "10000"),
            ),
        ),
        Operator("add", inputs=("residual.0", "attention.0"), outputs=("hidden.1",)),
        Operator("save", inputs=("hidden.1",), outputs=("residual.1",)),
        Operator(
            "rms_norm",
            inputs=("hidden.1",),
            outputs=("normed.1",),
            tensors=("norm2",),
            attributes=(("epsilon", "1e-5"),),
        ),
        Operator(
            "swiglu",
            inputs=("normed.1",),
            outputs=("mlp.0",),
            tensors=("gate", "up", "down"),
        ),
        Operator("add", inputs=("residual.1", "mlp.0"), outputs=("hidden.2",)),
        Operator(
            "rms_norm",
            inputs=("hidden.2",),
            outputs=("hidden.3",),
            tensors=("final_norm",),
            attributes=(("epsilon", "1e-5"),),
        ),
        Operator(
            "linear", inputs=("hidden.3",), outputs=("logits",), tensors=("head",)
        ),
    )
    plan = ExecutionPlan(
        version=1,
        architecture_id="causality-probe",
        config_identity="causal",
        shapes=(("hidden_size", 4), ("vocab_size", 5), ("context_size", 12)),
        semantics=_f32_semantics(),
        tensors=bindings,
        values=values,
        operators=ops,
        eos_token_id=-1,
    )
    prompt = [1, 2, 3, 4, 1][:prompt_width]
    native = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "causal-native"
    ).generate(prompt, max_new_tokens=1, threads=2)
    oracle = DecoderOracle(plan, tensors).generate(prompt, max_new_tokens=1)

    _assert_native_matches_oracle(native, oracle)
    assert native.prefill_matrix_passes == 8
    assert native.prefill_physical_width == prompt_width


def test_plan_binding_rejects_malformed_tensor_shape(tmp_path: Path) -> None:
    plan, _image = _order_fixture(tmp_path)
    with pytest.raises(ValueError, match="embedding shape"):
        replace(
            plan,
            tensors=(replace(plan.tensors[0], shape=(3, 2)), *plan.tensors[1:]),
        )


def test_native_plan_rejects_dependency_order_and_context_overflow(
    tmp_path: Path,
) -> None:
    plan, image = _order_fixture(tmp_path)
    with pytest.raises(ValueError, match="operator dependency is unavailable"):
        replace(
            plan,
            operators=(plan.operators[1], *plan.operators[0:1], *plan.operators[2:]),
        )

    bounded = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "bounded"
    )
    with pytest.raises(RuntimeError, match="context bound"):
        bounded.generate([1, 2, 1, 2, 1], max_new_tokens=1, threads=1)


def _invoke_native(
    engine: NativeExecutionPlanEngine,
    tokens: str,
    max_new: str,
    threads: str,
    *,
    timeout: float = 5,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(engine.executable),
            str(engine.artifact),
            str(engine.checkpoint),
            tokens,
            max_new,
            threads,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


@pytest.mark.parametrize(
    ("tokens", "max_new", "threads"),
    [
        ("1junk", "1", "1"),
        ("-1", "1", "1"),
        ("", "1", "1"),
        ("1,", "1", "1"),
        ("3", "1", "1"),
        ("1", "0", "1"),
        ("1", "-1", "1"),
        ("1", "18446744073709551615", "1"),
        ("1", "1junk", "1"),
        ("1", "1", "1junk"),
        ("1", "1", "-1"),
        ("1", "1", "257"),
    ],
)
def test_native_cli_rejects_noncanonical_or_unbounded_arguments(
    tmp_path: Path, tokens: str, max_new: str, threads: str
) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "strict-cli"
    )

    completed = _invoke_native(engine, tokens, max_new, threads, timeout=1)

    assert completed.returncode != 0


def test_context_frontier_allows_first_emission_at_full_context(tmp_path: Path) -> None:
    plan, image = _order_fixture(tmp_path)
    prompt = [1, 2, 1, 2]
    probe = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "context-probe"
    ).generate(prompt, max_new_tokens=1, threads=1)
    eos_plan = replace(plan, eos_token_id=probe.generated_token_ids[0])
    engine = NativeExecutionPlanEngine(
        plan=eos_plan, weights_image=image, cache_root=tmp_path / "context-eos"
    )

    result = engine.generate(prompt, max_new_tokens=(1 << 63), threads=1)

    assert result.generated_token_ids == probe.generated_token_ids
    assert result.executed_operators == len(plan.operators) * len(prompt)


def test_context_frontier_rejects_only_when_another_forward_is_required(
    tmp_path: Path,
) -> None:
    plan, image = _order_fixture(tmp_path)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "context-forward"
    )

    with pytest.raises(RuntimeError, match="context bound"):
        engine.generate([1, 2, 1, 2], max_new_tokens=2, threads=1)


def test_native_json_escapes_plan_identity_strings(tmp_path: Path) -> None:
    plan, image = _order_fixture(tmp_path)
    architecture = 'arch"\\\r\x01'
    config = 'config\\"\r\x02'
    engine = NativeExecutionPlanEngine(
        plan=replace(plan, architecture_id=architecture, config_identity=config),
        weights_image=image,
        cache_root=tmp_path / "json-identities",
    )

    result = engine.generate([1], max_new_tokens=1, threads=1)

    assert result.architecture_id == architecture
    assert result.config_identity == config


def _mixed_dtype_fixture(
    tmp_path: Path, *, output_dtype: str
) -> tuple[ExecutionPlan, Path]:
    tensors = {
        "embed": ((2, 1), [0.0, 1.00390625]),
        "head": ((2, 1), [1.00390625, 0.0]),
    }
    source = tmp_path / f"mixed-{output_dtype}.safetensors"
    _write_f32_safetensors(source, tensors)
    image = tmp_path / f"mixed-{output_dtype}.strpot"
    compile_image(source, image)
    semantics = replace(
        _f32_semantics(), activation_dtype="BF16", output_dtype=output_dtype
    )
    plan = ExecutionPlan(
        version=1,
        architecture_id="mixed-dtype-rounding-v1",
        config_identity=output_dtype,
        shapes=(("hidden_size", 1), ("vocab_size", 2), ("context_size", 1)),
        semantics=semantics,
        tensors=tuple(
            TensorBinding(name, name, "F32", shape)
            for name, (shape, _values) in tensors.items()
        ),
        values=(
            GraphValue("hidden", "BF16", ("hidden_size",), "activation"),
            GraphValue("logits", output_dtype, ("vocab_size",), "logits"),
        ),
        operators=(
            Operator("embedding", outputs=("hidden",), tensors=("embed",)),
            Operator(
                "linear", inputs=("hidden",), outputs=("logits",), tensors=("head",)
            ),
        ),
        eos_token_id=-1,
    )
    return plan, image


@pytest.mark.parametrize(
    ("output_dtype", "expected"),
    [("F32", 1.00390625), ("BF16", 1.0)],
)
def test_embedding_and_logits_obey_distinct_dtype_rounding_contract(
    tmp_path: Path, output_dtype: str, expected: float
) -> None:
    plan, image = _mixed_dtype_fixture(tmp_path, output_dtype=output_dtype)
    result = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / f"dtype-{output_dtype}"
    ).generate([1], max_new_tokens=1, threads=1)

    assert result.frontier_logits[0][0] == expected


def test_bf16_rne_preserves_nonfinite_values_and_signed_zero(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "src/strpot/_native/strpot_plan.cpp"
    probe = tmp_path / "bf16_probe.cpp"
    executable = tmp_path / "bf16_probe"
    probe.write_text(
        "#define main strpot_plan_main\n"
        f'#include "{source}"\n'
        "#undef main\n"
        "int main() {\n"
        "  const std::array<std::uint32_t, 5> bits{"
        "0x7f800000U,0xff800000U,0x00000000U,0x80000000U,0x7f800001U};\n"
        "  for (auto value : bits) std::cout << std::hex << float_to_bf16("
        "std::bit_cast<float>(value)) << '\\n';\n"
        "}\n",
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O0",
            "-fsanitize=undefined",
            str(probe),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr

    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=True
    )
    values = [int(line, 16) for line in completed.stdout.splitlines()]

    assert values[:4] == [0x7F80, 0xFF80, 0x0000, 0x8000]
    assert values[4] & 0x7F80 == 0x7F80
    assert values[4] & 0x007F != 0
    assert (
        values[4] & 0x0040 != 0
    )  # NaNs are quieted while preserving upper payload bits.


def test_unaligned_bf16_decode_offsets_are_ubsan_clean(tmp_path: Path) -> None:
    tensors = {
        "embed": ((2, 1), [0.0, 1.0]),
        "head": ((2, 1), [1.0, 0.0]),
    }
    source = tmp_path / "unaligned.safetensors"
    _write_bf16_safetensors(source, tensors, payload_prefix=b"\x00")
    image = tmp_path / "unaligned.strpot"
    compile_image(source, image)
    semantics = replace(
        _f32_semantics(),
        weight_dtype="BF16",
        activation_dtype="BF16",
        output_dtype="BF16",
    )
    plan = ExecutionPlan(
        version=1,
        architecture_id="unaligned-bf16-decode-v1",
        config_identity="unaligned-bf16",
        shapes=(("hidden_size", 1), ("vocab_size", 2), ("context_size", 2)),
        semantics=semantics,
        tensors=tuple(
            TensorBinding(name, name, "BF16", shape)
            for name, (shape, _values) in tensors.items()
        ),
        values=(
            GraphValue("hidden", "BF16", ("hidden_size",), "activation"),
            GraphValue("logits", "BF16", ("vocab_size",), "logits"),
        ),
        operators=(
            Operator("embedding", outputs=("hidden",), tensors=("embed",)),
            Operator(
                "linear",
                inputs=("hidden",),
                outputs=("logits",),
                tensors=("head",),
            ),
        ),
        eos_token_id=1,
    )
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "unaligned-native"
    )
    sanitized = tmp_path / "strpot-native-plan-sanitized"
    compiled = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O1",
            "-g",
            "-ffp-contract=off",
            "-fsanitize=undefined,address",
            "-fno-omit-frame-pointer",
            str(Path(__file__).parents[1] / "src/strpot/_native/strpot_plan.cpp"),
            "-o",
            str(sanitized),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr

    completed = subprocess.run(
        [
            str(sanitized),
            str(engine.artifact),
            str(engine.checkpoint),
            "1",
            "2",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "runtime error" not in completed.stderr
