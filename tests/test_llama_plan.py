from __future__ import annotations

from copy import deepcopy

import pytest

from strpot.llama_plan import llama_execution_plan


def _representative_llama_config() -> dict[str, object]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 5632,
        "max_position_embeddings": 2048,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 22,
        "num_key_value_heads": 4,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 32000,
    }


def test_llama_adapter_emits_exact_unbiased_untied_plan() -> None:
    plan = llama_execution_plan(_representative_llama_config())

    assert plan.architecture_id == "llama-causal-lm-one-token-v1"
    assert dict(plan.shapes) == {
        "hidden_size": 2048,
        "vocab_size": 32000,
        "context_size": 2048,
    }
    assert len(plan.tensors) == 201
    assert len(plan.operators) == 22 * 8 + 3
    assert len(plan.values) == len(plan.operators)

    attention = plan.operators[3]
    assert attention.kind == "attention_rope"
    assert attention.tensors == (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    )
    assert dict(attention.attributes) == {
        "cache": "layer.0",
        "heads": "32",
        "kv_heads": "4",
        "theta": "10000",
    }
    assert not any(binding.alias.endswith(".bias") for binding in plan.tensors)
    assert plan.operators[-1].tensors == ("lm_head.weight",)
    assert plan.operators[-1].attributes == (("result_rounding", "BF16"),)
    assert plan.tensors[-1].alias == "lm_head.weight"
    assert plan.tensors[-1].shape == (32000, 2048)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "qwen2", "unsupported Llama model_type"),
        ("architectures", ["LlamaModel"], "unsupported Llama architectures"),
        ("attention_bias", True, "unsupported Llama attention_bias"),
        ("tie_word_embeddings", True, "unsupported Llama tie_word_embeddings"),
        ("pretraining_tp", 2, "unsupported Llama pretraining_tp"),
        (
            "rope_scaling",
            {"type": "linear", "factor": 2.0},
            "unsupported Llama rope_scaling",
        ),
        ("use_cache", False, "unsupported Llama use_cache"),
        ("torch_dtype", "float16", "unsupported Llama torch_dtype"),
    ],
)
def test_llama_adapter_rejects_unsupported_semantic_variants(
    field: str, value: object, message: str
) -> None:
    config = deepcopy(_representative_llama_config())
    config[field] = value

    with pytest.raises(ValueError, match=message):
        llama_execution_plan(config)


@pytest.mark.parametrize(
    "field",
    [
        "architectures",
        "attention_bias",
        "hidden_act",
        "mlp_bias",
        "model_type",
        "pretraining_tp",
        "rope_scaling",
        "tie_word_embeddings",
        "torch_dtype",
        "use_cache",
    ],
)
def test_llama_adapter_requires_every_supported_semantic_field(field: str) -> None:
    config = deepcopy(_representative_llama_config())
    del config[field]

    with pytest.raises(ValueError, match=rf"unsupported Llama {field}"):
        llama_execution_plan(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mlp_bias", True, "unsupported Llama mlp_bias"),
        ("head_dim", 128, "unsupported Llama head_dim"),
        ("bos_token_id", -1, "Llama bos_token_id must be inside the vocabulary"),
        ("eos_token_id", 32000, "Llama eos_token_id must be inside the vocabulary"),
        ("rms_norm_eps", True, "Llama rms_norm_eps must be positive and finite"),
    ],
)
def test_llama_adapter_fails_closed_on_additional_semantics(
    field: str, value: object, message: str
) -> None:
    config = deepcopy(_representative_llama_config())
    config[field] = value

    with pytest.raises(ValueError, match=message):
        llama_execution_plan(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [("attention_bias", 0), ("use_cache", 1), ("pretraining_tp", True)],
)
def test_llama_adapter_rejects_bool_integer_semantic_substitutions(
    field: str, value: object
) -> None:
    config = deepcopy(_representative_llama_config())
    config[field] = value

    with pytest.raises(ValueError, match=rf"unsupported Llama {field}"):
        llama_execution_plan(config)
