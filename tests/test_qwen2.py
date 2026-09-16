from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest
from decoder_oracle import DecoderOracle

from strpot.image import compile_image
from strpot.native import NativeExecutionPlanEngine
from strpot.qwen2 import qwen2_execution_plan

QWEN25_3B_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "max_position_embeddings": 32768,
    "mlp_bias": False,
    "model_type": "qwen2",
    "num_attention_heads": 16,
    "num_hidden_layers": 36,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def test_qwen25_3b_adapter_emits_complete_typed_one_token_plan() -> None:
    config = dict(QWEN25_3B_CONFIG)

    plan = qwen2_execution_plan(config)

    assert plan.architecture_id == "qwen2-causal-lm-one-token-v1"
    assert dict(plan.shapes) == {
        "hidden_size": 2048,
        "vocab_size": 151936,
        "context_size": 32768,
    }
    assert plan.semantics.weight_dtype == "BF16"
    assert plan.semantics.activation_dtype == "BF16"
    assert plan.semantics.accumulator_dtype == "F32"
    assert plan.semantics.output_dtype == "F32"
    assert len(plan.tensors) == 434
    assert len(plan.operators) == 36 * 8 + 3
    assert len(plan.values) == len(plan.operators)

    embedding, *_, final_norm, head = plan.operators
    assert embedding.kind == "embedding"
    assert embedding.tensors == ("model.embed_tokens.weight",)
    assert final_norm.kind == "rms_norm"
    assert final_norm.attributes == (("epsilon", "1e-06"),)
    assert head.kind == "linear"
    assert head.tensors == ("model.embed_tokens.weight",)
    assert head.outputs == ("logits",)
    assert head.attributes == (("result_rounding", "BF16"),)

    attention = plan.operators[3]
    assert attention.kind == "attention_rope_qkv_bias"
    assert attention.tensors == (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
    )
    assert dict(attention.attributes) == {
        "cache": "layer.0",
        "heads": "16",
        "kv_heads": "2",
        "rope_layout": "half_split",
        "scale": "0.08838834764831843",
        "theta": "1000000",
    }


def test_qwen2_adapter_rejects_disabled_kv_cache() -> None:
    config = dict(QWEN25_3B_CONFIG)
    config["use_cache"] = False

    with pytest.raises(ValueError, match="use_cache"):
        qwen2_execution_plan(config)


@pytest.mark.parametrize(
    ("name", "value"),
    (("attention_bias", False), ("mlp_bias", True), ("head_dim", 64)),
)
def test_qwen2_adapter_rejects_incompatible_semantics(
    name: str, value: bool | int
) -> None:
    config = dict(QWEN25_3B_CONFIG)
    config[name] = value

    with pytest.raises(ValueError, match=name):
        qwen2_execution_plan(config)


def _bf16(value: float) -> tuple[bytes, float]:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    upper = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    encoded = struct.pack("<H", upper & 0xFFFF)
    widened = struct.unpack("<f", struct.pack("<I", (upper & 0xFFFF) << 16))[0]
    return encoded, widened


def test_qwen2_bf16_boundaries_match_torch_cpu_reference() -> None:
    """Frozen from PyTorch 2.14 CPU Qwen-style BF16 operations."""

    def f32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", value))[0]

    def bf16(value: float) -> float:
        return _bf16(f32(value))[1]

    query = [1.234375, -0.6796875, 0.333984375, -1.1171875]
    rotated = [-query[2], -query[3], query[0], query[1]]
    frequencies = [7.0, 7.0 / math.sqrt(1_000_000.0)]
    cosine = [bf16(math.cos(value)) for value in frequencies] * 2
    sine = [bf16(math.sin(value)) for value in frequencies] * 2
    rope = [
        bf16(bf16(item * cos) + bf16(rotated_item * sin))
        for item, rotated_item, cos, sin in zip(
            query, rotated, cosine, sine, strict=True
        )
    ]

    key = [0.8125, 1.1171875, -0.5546875, 0.4453125]
    score = f32(0.0)
    for left, right in zip(query, key, strict=True):
        score = f32(score + f32(left * right))
    score = bf16(bf16(score) * f32(0.5))

    gate = [0.7734375, -1.2890625]
    up = [1.3359375, -0.72265625]
    swiglu = [
        bf16(bf16(item / f32(1.0 + f32(math.exp(-item)))) * up_item)
        for item, up_item in zip(gate, up, strict=True)
    ]

    assert cosine == [0.75390625, 1.0, 0.75390625, 1.0]
    assert sine == [0.65625, 0.006988525390625, 0.65625, 0.006988525390625]
    assert rope == [0.7109375, -0.671875, 1.0625, -1.125]
    assert score == -0.2197265625
    assert swiglu == [0.703125, 0.2021484375]


def _write_bf16_safetensors(
    path: Path, tensors: dict[str, tuple[tuple[int, ...], list[float]]]
) -> dict[str, tuple[tuple[int, ...], list[float]]]:
    header: dict[str, object] = {}
    payload = bytearray()
    rounded: dict[str, tuple[tuple[int, ...], list[float]]] = {}
    for name, (shape, values) in tensors.items():
        start = len(payload)
        converted = [_bf16(value) for value in values]
        payload.extend(b"".join(item[0] for item in converted))
        rounded[name] = (shape, [item[1] for item in converted])
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return rounded


def test_qwen2_plan_executes_qkv_bias_scale_rope_and_bf16_head_rounding(
    tmp_path: Path,
) -> None:
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_bias": True,
        "attention_dropout": 0.0,
        "eos_token_id": 3,
        "hidden_act": "silu",
        "hidden_size": 4,
        "intermediate_size": 3,
        "max_position_embeddings": 8,
        "mlp_bias": False,
        "model_type": "qwen2",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 4,
    }
    plan = qwen2_execution_plan(config)
    tensors = {
        binding.alias: (binding.shape, [0.0] * __import__("math").prod(binding.shape))
        for binding in plan.tensors
    }
    tensors["model.embed_tokens.weight"] = (
        (4, 4),
        [0.0] * 4
        + [1.0, -0.5, 0.25, 0.75]
        + [-0.2, 0.4, 0.8, -0.6]
        + [0.3, 0.2, -0.7, 0.5],
    )
    tensors["model.layers.0.input_layernorm.weight"] = ((4,), [1.0] * 4)
    tensors["model.layers.0.post_attention_layernorm.weight"] = ((4,), [1.0] * 4)
    tensors["model.norm.weight"] = ((4,), [1.0, 0.9, 1.1, 0.8])
    for projection in ("q_proj", "o_proj"):
        tensors[f"model.layers.0.self_attn.{projection}.weight"] = (
            (4, 4),
            [1.0 if row == column else 0.0 for row in range(4) for column in range(4)],
        )
    for projection in ("k_proj", "v_proj"):
        tensors[f"model.layers.0.self_attn.{projection}.weight"] = (
            (2, 4),
            [0.4, -0.2, 0.1, 0.3, -0.1, 0.5, 0.2, -0.4],
        )
    tensors["model.layers.0.self_attn.q_proj.bias"] = ((4,), [0.5, -0.25, 0.75, 0.125])
    tensors["model.layers.0.self_attn.k_proj.bias"] = ((2,), [0.25, -0.5])
    tensors["model.layers.0.self_attn.v_proj.bias"] = ((2,), [-0.75, 0.5])

    source = tmp_path / "qwen2-tiny.safetensors"
    rounded = _write_bf16_safetensors(source, tensors)
    image = tmp_path / "qwen2-tiny.strpot"
    compile_image(source, image, page_size=71)
    engine = NativeExecutionPlanEngine(
        plan=plan, weights_image=image, cache_root=tmp_path / "native"
    )
    native = engine.generate([1], max_new_tokens=2, threads=1)
    oracle = DecoderOracle(plan, rounded).generate([1], max_new_tokens=2)

    assert native.generated_token_ids == oracle.tokens
    assert native.frontier_logits[0] == pytest.approx(
        oracle.frontier_logits[0], abs=2e-6, rel=2e-6
    )
    assert native.kv_cache_keys[0] == pytest.approx(oracle.cache_keys[0], abs=2e-6)
    assert native.kv_cache_values[0] == pytest.approx(oracle.cache_values[0], abs=2e-6)
    assert any(native.kv_cache_keys[0])
    assert len(native.inter_token_seconds) == 1
    for value in native.frontier_logits[0]:
        assert _bf16(value)[1] == pytest.approx(value, abs=5e-10)
