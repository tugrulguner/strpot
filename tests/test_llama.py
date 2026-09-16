from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as functional

from strpot.image import compile_image
from strpot.llama import LlamaConfig, LlamaRuntime
from strpot.response_atlas import ExactTokenResponseAtlas
from strpot.tensor_store import PagedFile, SafeTensorStore


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": {torch.bfloat16: "BF16", torch.float32: "F32"}[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_llama_runtime_executes_complete_layer_from_strpot_pages(
    tmp_path: Path,
) -> None:
    config = LlamaConfig(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        bos_token_id=1,
        eos_token_id=2,
    )
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    lm_head = torch.flip(embedding, dims=[0])
    zeros = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(2, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(8, 4),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(8, 4),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 8),
    }
    tensors = {
        "model.embed_tokens.weight": embedding,
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
        "model.norm.weight": torch.ones(4),
        "lm_head.weight": lm_head,
        **zeros,
    }
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, tensors)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=73)

    runtime = LlamaRuntime(
        config,
        SafeTensorStore(PagedFile(image, max_cached_pages=2)),
    )
    logits = runtime.forward([1, 3])

    hidden = embedding[[1, 3]]
    expected_hidden = hidden * torch.rsqrt(
        hidden.pow(2).mean(dim=-1, keepdim=True) + config.rms_norm_eps
    )
    expected = functional.linear(expected_hidden, lm_head)
    torch.testing.assert_close(logits, expected)
    atlas = ExactTokenResponseAtlas.compile(
        embedding,
        torch.ones(4),
        {
            "q": zeros["model.layers.0.self_attn.q_proj.weight"],
            "k": zeros["model.layers.0.self_attn.k_proj.weight"],
            "v": zeros["model.layers.0.self_attn.v_proj.weight"],
        },
        rms_norm_eps=config.rms_norm_eps,
    )
    atlas_runtime = LlamaRuntime(
        config,
        SafeTensorStore(PagedFile(image, max_cached_pages=2)),
        first_layer_atlas=atlas,
    )
    torch.testing.assert_close(atlas_runtime.forward([1, 3]), logits)
    generated = runtime.generate([1, 3], max_new_tokens=1)
    assert generated == [1, 3, int(expected[-1].argmax())]
    assert runtime.page_resident_bytes <= 146


def test_kv_cached_decode_matches_full_forward(tmp_path: Path) -> None:
    torch.manual_seed(7)
    config = LlamaConfig(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        bos_token_id=1,
        eos_token_id=2,
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(8, 4) * 0.1,
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4) * 0.1,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2, 4) * 0.1,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(2, 4) * 0.1,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4) * 0.1,
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4) * 0.1,
        "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4) * 0.1,
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 8) * 0.1,
        "model.norm.weight": torch.ones(4),
        "lm_head.weight": torch.randn(8, 4) * 0.1,
    }
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, tensors)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=73)
    runtime = LlamaRuntime(
        config, SafeTensorStore(PagedFile(image, max_cached_pages=2))
    )

    prefill_logits, cache = runtime.prefill([1, 3])
    decoded_logits, cache = runtime.decode_one(4, cache)

    torch.testing.assert_close(prefill_logits, runtime.forward([1, 3])[-1])
    torch.testing.assert_close(decoded_logits, runtime.forward([1, 3, 4])[-1])
    assert cache.length == 3


def test_llama_runtime_preserves_bfloat16_execution(tmp_path: Path) -> None:
    dtype = torch.bfloat16
    config = LlamaConfig(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        bos_token_id=1,
        eos_token_id=2,
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(8, 4, dtype=dtype) * 0.1,
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=dtype),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4, dtype=dtype) * 0.1,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2, 4, dtype=dtype) * 0.1,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(2, 4, dtype=dtype) * 0.1,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4, dtype=dtype) * 0.1,
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4, dtype=dtype),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, dtype=dtype) * 0.1,
        "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4, dtype=dtype) * 0.1,
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 8, dtype=dtype) * 0.1,
        "model.norm.weight": torch.ones(4, dtype=dtype),
        "lm_head.weight": torch.randn(8, 4, dtype=dtype) * 0.1,
    }
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, tensors)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=73)
    runtime = LlamaRuntime(
        config, SafeTensorStore(PagedFile(image, max_cached_pages=2))
    )

    logits = runtime.forward([1, 3])

    assert logits.dtype == torch.bfloat16
    assert runtime._weight("model.embed_tokens.weight").dtype == torch.bfloat16

    atlas = ExactTokenResponseAtlas.compile(
        tensors["model.embed_tokens.weight"],
        tensors["model.layers.0.input_layernorm.weight"],
        {
            name: tensors[f"model.layers.0.self_attn.{name}_proj.weight"]
            for name in ("q", "k", "v")
        },
        rms_norm_eps=config.rms_norm_eps,
    )
    atlas_runtime = LlamaRuntime(config, runtime.tensors, first_layer_atlas=atlas)
    atlas_logits = atlas_runtime.forward([1, 3])
    assert atlas_logits.dtype == torch.bfloat16
    torch.testing.assert_close(atlas_logits, logits)
