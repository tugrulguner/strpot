"""Qwen2 compatibility adapter for the generic native execution plan."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from strpot.native import (
    ExecutionPlan,
    GraphValue,
    NumericalSemantics,
    Operator,
    TensorBinding,
)


def _require(config: Mapping[str, Any], name: str, expected: Any) -> None:
    if config.get(name) != expected:
        raise ValueError(f"unsupported Qwen2 {name}: expected {expected!r}")


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Qwen2 {name} must be a positive integer")
    return value


def qwen2_execution_plan(config: Mapping[str, Any]) -> ExecutionPlan:
    """Validate supported Qwen2 semantics and emit a one-token native plan."""
    _require(config, "model_type", "qwen2")
    _require(config, "architectures", ["Qwen2ForCausalLM"])
    _require(config, "torch_dtype", "bfloat16")
    _require(config, "hidden_act", "silu")
    _require(config, "tie_word_embeddings", True)
    _require(config, "attention_bias", True)
    _require(config, "mlp_bias", False)
    _require(config, "use_cache", True)
    _require(config, "use_sliding_window", False)
    _require(config, "attention_dropout", 0.0)
    if config.get("rope_scaling") not in (None, False):
        raise ValueError("unsupported Qwen2 rope_scaling")

    hidden = _positive_int(config, "hidden_size")
    intermediate = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    kv_heads = _positive_int(config, "num_key_value_heads")
    context = _positive_int(config, "max_position_embeddings")
    vocab = _positive_int(config, "vocab_size")
    if hidden % heads or heads % kv_heads:
        raise ValueError("unsupported Qwen2 attention head shape")
    head_dim = hidden // heads
    if config.get("head_dim", head_dim) != head_dim:
        raise ValueError("unsupported Qwen2 head_dim")
    if head_dim % 2:
        raise ValueError("Qwen2 RoPE requires an even head dimension")

    epsilon = config.get("rms_norm_eps")
    theta = config.get("rope_theta")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Qwen2 rms_norm_eps must be positive and finite")
    if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("Qwen2 rope_theta must be positive and finite")
    eos = config.get("eos_token_id")
    if type(eos) is not int or not 0 <= eos < vocab:
        raise ValueError("Qwen2 eos_token_id must be inside the vocabulary")

    dtype = "BF16"
    tensors: list[TensorBinding] = [
        TensorBinding(
            "model.embed_tokens.weight",
            "model.embed_tokens.weight",
            dtype,
            (vocab, hidden),
        )
    ]
    values: list[GraphValue] = []
    operators: list[Operator] = []

    def value(name: str, *, logits: bool = False) -> str:
        values.append(
            GraphValue(
                name,
                "F32" if logits else dtype,
                ("vocab_size",) if logits else ("hidden_size",),
                "logits" if logits else "activation",
            )
        )
        return name

    current = value("hidden.0")
    operators.append(
        Operator(
            "embedding",
            outputs=(current,),
            tensors=("model.embed_tokens.weight",),
        )
    )
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        attention = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        layer_tensors = (
            (f"{prefix}.input_layernorm.weight", (hidden,)),
            (f"{attention}.q_proj.weight", (hidden, hidden)),
            (f"{attention}.q_proj.bias", (hidden,)),
            (f"{attention}.k_proj.weight", (kv_heads * head_dim, hidden)),
            (f"{attention}.k_proj.bias", (kv_heads * head_dim,)),
            (f"{attention}.v_proj.weight", (kv_heads * head_dim, hidden)),
            (f"{attention}.v_proj.bias", (kv_heads * head_dim,)),
            (f"{attention}.o_proj.weight", (hidden, hidden)),
            (f"{prefix}.post_attention_layernorm.weight", (hidden,)),
            (f"{mlp}.gate_proj.weight", (intermediate, hidden)),
            (f"{mlp}.up_proj.weight", (intermediate, hidden)),
            (f"{mlp}.down_proj.weight", (hidden, intermediate)),
        )
        tensors.extend(
            TensorBinding(name, name, dtype, shape) for name, shape in layer_tensors
        )

        residual_attention = value(f"layer.{layer}.attention_residual")
        normalized_attention = value(f"layer.{layer}.attention_norm")
        attention_output = value(f"layer.{layer}.attention")
        post_attention = value(f"layer.{layer}.post_attention")
        residual_mlp = value(f"layer.{layer}.mlp_residual")
        normalized_mlp = value(f"layer.{layer}.mlp_norm")
        mlp_output = value(f"layer.{layer}.mlp")
        post_mlp = value(f"hidden.{layer + 1}")
        operators.extend(
            (
                Operator("save", inputs=(current,), outputs=(residual_attention,)),
                Operator(
                    "rms_norm",
                    inputs=(current,),
                    outputs=(normalized_attention,),
                    tensors=(f"{prefix}.input_layernorm.weight",),
                    attributes=(("epsilon", repr(float(epsilon))),),
                ),
                Operator(
                    "attention_rope_qkv_bias",
                    inputs=(normalized_attention,),
                    outputs=(attention_output,),
                    tensors=(
                        f"{attention}.q_proj.weight",
                        f"{attention}.q_proj.bias",
                        f"{attention}.k_proj.weight",
                        f"{attention}.k_proj.bias",
                        f"{attention}.v_proj.weight",
                        f"{attention}.v_proj.bias",
                        f"{attention}.o_proj.weight",
                    ),
                    attributes=(
                        ("cache", f"layer.{layer}"),
                        ("heads", str(heads)),
                        ("kv_heads", str(kv_heads)),
                        ("rope_layout", "half_split"),
                        ("scale", repr(1.0 / math.sqrt(head_dim))),
                        ("theta", format(float(theta), ".15g")),
                    ),
                ),
                Operator(
                    "add",
                    inputs=(residual_attention, attention_output),
                    outputs=(post_attention,),
                ),
                Operator("save", inputs=(post_attention,), outputs=(residual_mlp,)),
                Operator(
                    "rms_norm",
                    inputs=(post_attention,),
                    outputs=(normalized_mlp,),
                    tensors=(f"{prefix}.post_attention_layernorm.weight",),
                    attributes=(("epsilon", repr(float(epsilon))),),
                ),
                Operator(
                    "swiglu",
                    inputs=(normalized_mlp,),
                    outputs=(mlp_output,),
                    tensors=(
                        f"{mlp}.gate_proj.weight",
                        f"{mlp}.up_proj.weight",
                        f"{mlp}.down_proj.weight",
                    ),
                ),
                Operator(
                    "add",
                    inputs=(residual_mlp, mlp_output),
                    outputs=(post_mlp,),
                ),
            )
        )
        current = post_mlp

    tensors.append(
        TensorBinding("model.norm.weight", "model.norm.weight", dtype, (hidden,))
    )
    normalized = value("model.normalized")
    logits = value("logits", logits=True)
    operators.extend(
        (
            Operator(
                "rms_norm",
                inputs=(current,),
                outputs=(normalized,),
                tensors=("model.norm.weight",),
                attributes=(("epsilon", repr(float(epsilon))),),
            ),
            Operator(
                "linear",
                inputs=(normalized,),
                outputs=(logits,),
                tensors=("model.embed_tokens.weight",),
                attributes=(("result_rounding", "BF16"),),
            ),
        )
    )

    identity_payload = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode()
    return ExecutionPlan(
        version=1,
        architecture_id="qwen2-causal-lm-one-token-v1",
        config_identity=hashlib.sha256(identity_payload).hexdigest(),
        shapes=(
            ("hidden_size", hidden),
            ("vocab_size", vocab),
            ("context_size", context),
        ),
        semantics=NumericalSemantics(
            weight_dtype=dtype,
            activation_dtype=dtype,
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
        ),
        tensors=tuple(tensors),
        values=tuple(values),
        operators=tuple(operators),
        eos_token_id=eos,
    )
