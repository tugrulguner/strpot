from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from strpot.image import compile_image
from strpot.native import (
    NativeLlamaEngine,
    _materialize_checkpoint,
    _write_descriptor,
    build_native_engine,
)


def test_production_dependencies_exclude_pytorch_and_numpy() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert all(not item.startswith("torch") for item in dependencies)
    assert all(not item.startswith("numpy") for item in dependencies)
    assert any(
        item.startswith("torch") for item in metadata["dependency-groups"]["dev"]
    )


def _bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounding = 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", ((bits + rounding) >> 16) & 0xFFFF)


def _write_bf16_safetensors(
    path: Path, tensors: dict[str, tuple[tuple[int, ...], list[float]]]
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (shape, values) in tensors.items():
        start = len(payload)
        for value in values:
            payload.extend(_bf16(value))
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _tiny_llama_tensors() -> dict[str, tuple[tuple[int, ...], list[float]]]:
    def zeros(size: int) -> list[float]:
        return [0.0] * size

    embeddings = zeros(32)
    embeddings[4] = 1.0
    embeddings[13] = 1.0
    lm_head = zeros(32)
    lm_head[4 * 4 + 1] = 1.0
    return {
        "model.embed_tokens.weight": ((8, 4), embeddings),
        "model.layers.0.input_layernorm.weight": ((4,), [1.0] * 4),
        "model.layers.0.self_attn.q_proj.weight": ((4, 4), zeros(16)),
        "model.layers.0.self_attn.k_proj.weight": ((2, 4), zeros(8)),
        "model.layers.0.self_attn.v_proj.weight": ((2, 4), zeros(8)),
        "model.layers.0.self_attn.o_proj.weight": ((4, 4), zeros(16)),
        "model.layers.0.post_attention_layernorm.weight": ((4,), [1.0] * 4),
        "model.layers.0.mlp.gate_proj.weight": ((8, 4), zeros(32)),
        "model.layers.0.mlp.up_proj.weight": ((8, 4), zeros(32)),
        "model.layers.0.mlp.down_proj.weight": ((4, 8), zeros(32)),
        "model.norm.weight": ((4,), [1.0] * 4),
        "lm_head.weight": ((8, 4), lm_head),
    }


def _tiny_llama_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "vocab_size": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    config.update(overrides)
    return config


def test_native_engine_rejects_manifest_digest_before_cache_path_creation(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_tiny_llama_config()), encoding="utf-8")
    image = tmp_path / "image"
    image.mkdir()
    escaped = tmp_path / "escaped"
    (image / "manifest.json").write_text(
        json.dumps(
            {
                "format": "strpot-image-v1",
                "source_size": 0,
                "source_sha256": str(escaped),
                "pages": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid checkpoint identity"):
        NativeLlamaEngine(
            config_path=config,
            weights_image=image,
            cache_root=tmp_path / "cache",
        )

    assert not escaped.exists()


def test_native_materialization_rejects_page_escape(tmp_path: Path) -> None:
    image = tmp_path / "image"
    image.mkdir()
    outside = tmp_path / "outside.page"
    payload = b"checkpoint"
    outside.write_bytes(payload)
    (image / "manifest.json").write_text(
        json.dumps(
            {
                "format": "strpot-image-v1",
                "source_size": len(payload),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "pages": [
                    {
                        "path": "../outside.page",
                        "codec": "raw",
                        "stored_size": len(payload),
                        "raw_size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the image directory"):
        _materialize_checkpoint(image, tmp_path / "model.safetensors")


def test_descriptor_rejects_tensor_range_past_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    header = {
        "tensor": {
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [0, 4],
        }
    }
    encoded = json.dumps(header).encode()
    checkpoint.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00\x00")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_tiny_llama_config()), encoding="utf-8")

    with pytest.raises(ValueError, match="outside checkpoint"):
        _write_descriptor(config, checkpoint, tmp_path / "descriptor")


def test_native_consumer_rejects_complete_tensor_range_past_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"\x00" * 16)
    descriptor = tmp_path / "descriptor"
    descriptor.write_text(
        "\n".join(
            [
                "STRPOT_NATIVE_V1",
                "config\thidden_size\t4",
                "config\tintermediate_size\t8",
                "config\tnum_attention_heads\t2",
                "config\tnum_key_value_heads\t1",
                "config\tnum_hidden_layers\t1",
                "config\tvocab_size\t8",
                "config\tmax_position_embeddings\t32",
                "config\trms_norm_eps\t1e-5",
                "config\trope_theta\t10000",
                "config\tbos_token_id\t1",
                "config\teos_token_id\t2",
                "tensor\tbad\tBF16\t15\t2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(build_native_engine(tmp_path / "native")),
            "--generate",
            str(descriptor),
            str(checkpoint),
            "1",
            "1",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "tensor range outside checkpoint" in result.stderr


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model_type": "mistral"}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, "architectures"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"attention_bias": True}, "attention_bias"),
        ({"mlp_bias": True}, "mlp_bias"),
        ({"rope_scaling": {"type": "linear", "factor": 2.0}}, "rope_scaling"),
        ({"head_dim": 3}, "head_dim"),
        ({"tie_word_embeddings": True}, "tie_word_embeddings"),
        ({"hidden_size": 5}, "head geometry"),
        ({"num_attention_heads": 3}, "head geometry"),
        ({"num_key_value_heads": 3}, "head geometry"),
    ],
)
def test_descriptor_rejects_unsupported_llama_semantics(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_bf16_safetensors(checkpoint, _tiny_llama_tensors())
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(_tiny_llama_config(**override)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _write_descriptor(config, checkpoint, tmp_path / "descriptor")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing required tensors"),
        ("unexpected", "unsupported tensors"),
        ("shape", "has shape"),
    ],
)
def test_descriptor_rejects_unsupported_tensor_manifest(
    tmp_path: Path, mutation: str, message: str
) -> None:
    tensors = _tiny_llama_tensors()
    if mutation == "missing":
        del tensors["lm_head.weight"]
    elif mutation == "unexpected":
        tensors["model.layers.0.self_attn.q_proj.bias"] = ((4,), [0.0] * 4)
    else:
        tensors["model.layers.0.self_attn.k_proj.weight"] = ((4, 2), [0.0] * 8)
    checkpoint = tmp_path / "model.safetensors"
    _write_bf16_safetensors(checkpoint, tensors)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_tiny_llama_config()), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _write_descriptor(config, checkpoint, tmp_path / "descriptor")


def _create_tiny_native_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "model.safetensors"
    _write_bf16_safetensors(source, _tiny_llama_tensors())
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=73)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_tiny_llama_config()), encoding="utf-8")
    return image, config


def test_native_checkpoint_cache_repairs_same_size_digest_mismatch(
    tmp_path: Path,
) -> None:
    image, config = _create_tiny_native_inputs(tmp_path)
    cache_root = tmp_path / "native"
    first = NativeLlamaEngine(
        config_path=config, weights_image=image, cache_root=cache_root
    )
    expected_digest = json.loads((image / "manifest.json").read_text(encoding="utf-8"))[
        "source_sha256"
    ]
    first.checkpoint.write_bytes(b"x" * first.checkpoint.stat().st_size)

    second = NativeLlamaEngine(
        config_path=config, weights_image=image, cache_root=cache_root
    )

    assert hashlib.sha256(second.checkpoint.read_bytes()).hexdigest() == expected_digest


def test_native_descriptor_cache_includes_canonical_semantic_config(
    tmp_path: Path,
) -> None:
    image, config = _create_tiny_native_inputs(tmp_path)
    cache_root = tmp_path / "native"
    first = NativeLlamaEngine(
        config_path=config, weights_image=image, cache_root=cache_root
    )
    alternate_config = tmp_path / "alternate-config.json"
    alternate_config.write_text(
        json.dumps(_tiny_llama_config(eos_token_id=3)), encoding="utf-8"
    )

    second = NativeLlamaEngine(
        config_path=alternate_config,
        weights_image=image,
        cache_root=cache_root,
    )

    assert first.model_root != second.model_root
    assert "config\teos_token_id\t3\n" in second.descriptor.read_text(encoding="utf-8")


def test_native_module_does_not_import_pytorch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import strpot.native; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
    )

    assert result.returncode == 0


def test_native_engine_executes_its_own_bfloat16_matvec(tmp_path: Path) -> None:
    executable = build_native_engine(tmp_path)

    assert executable.parent.parent == tmp_path.resolve()
    assert len(executable.parent.name) == 16
    result = subprocess.run(
        [str(executable), "--self-test"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["engine"] == "strpot-native"
    assert report["dtype"] == "bfloat16"
    assert report["kernel_family"] == "portable-tiled"
    assert report["matvec"] == [3.5, 7.5]


def test_native_engine_executes_complete_llama_generation(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    _write_bf16_safetensors(source, _tiny_llama_tensors())
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=73)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_tiny_llama_config()), encoding="utf-8")
    engine = NativeLlamaEngine(
        config_path=config,
        weights_image=image,
        cache_root=tmp_path / "native",
    )

    result = engine.generate([1, 3], max_new_tokens=2, threads=0)

    assert result.generated_token_ids == (4, 0)
    assert result.engine == "strpot-native"
    assert result.weight_dtype == "BF16"
    assert result.prefill_seconds >= 0
    assert result.decode_seconds >= 0
    assert result.prefill_matrix_passes == 8
    assert result.runtime_tensor_lookups == 0
    assert result.decode_parallel_dispatches == 5
    assert result.rope_table_entries == 32
    assert result.attention_workspace_floats == 32
    assert result.threads >= 1
    assert (engine.model_root / "thread-tuning.json").exists()
    assert len(result.inter_token_seconds) == 1
    assert result.inter_token_seconds[0] >= 0


def test_native_batch_reuses_weight_passes_and_matches_independent_requests(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.safetensors"
    _write_bf16_safetensors(source, _tiny_llama_tensors())
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=73)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_tiny_llama_config()), encoding="utf-8")
    engine = NativeLlamaEngine(
        config_path=config,
        weights_image=image,
        cache_root=tmp_path / "native",
    )
    prompts = [[1, 3], [3, 1]]
    independent = tuple(
        engine.generate(prompt, max_new_tokens=2, threads=2).generated_token_ids
        for prompt in prompts
    )

    batch = engine.generate_batch(prompts, max_new_tokens=2, threads=2)

    assert batch.generated_token_ids == independent
    assert batch.batch_size == 2
    assert batch.decode_matrix_passes == 8
    assert batch.decode_parallel_dispatches == 5
    assert batch.activation_panel_width == 16
    assert batch.aggregate_decode_tokens_per_second >= 0


def test_token_wave_is_exact_commits_repetitions_and_rolls_back_rejections(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.safetensors"
    tensors = _tiny_llama_tensors()
    # Exercise non-zero attention and MLP paths while keeping generation deterministic.
    tensors["model.layers.0.self_attn.q_proj.weight"] = (
        (4, 4),
        [0.125 if row == column else 0.0 for row in range(4) for column in range(4)],
    )
    tensors["model.layers.0.self_attn.k_proj.weight"] = (
        (2, 4),
        [0.0625, 0.0, 0.0, 0.0, 0.0, 0.0625, 0.0, 0.0],
    )
    tensors["model.layers.0.self_attn.v_proj.weight"] = (
        (2, 4),
        [0.03125, 0.0, 0.0, 0.0, 0.0, 0.03125, 0.0, 0.0],
    )
    tensors["model.layers.0.self_attn.o_proj.weight"] = (
        (4, 4),
        [0.015625 if row == column else 0.0 for row in range(4) for column in range(4)],
    )
    tensors["model.layers.0.mlp.gate_proj.weight"] = (
        (8, 4),
        [
            0.0078125 if row % 4 == column else 0.0
            for row in range(8)
            for column in range(4)
        ],
    )
    tensors["model.layers.0.mlp.up_proj.weight"] = (
        (8, 4),
        [
            0.0078125 if row % 4 == column else 0.0
            for row in range(8)
            for column in range(4)
        ],
    )
    tensors["model.layers.0.mlp.down_proj.weight"] = (
        (4, 8),
        [
            0.0078125 if column % 4 == row else 0.0
            for row in range(4)
            for column in range(8)
        ],
    )
    _write_bf16_safetensors(source, tensors)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=73)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(_tiny_llama_config(max_position_embeddings=64)),
        encoding="utf-8",
    )
    engine = NativeLlamaEngine(
        config_path=config, weights_image=image, cache_root=tmp_path / "native"
    )
    prompt = [1, 3, 1, 3]
    ordinary = engine.generate(prompt, max_new_tokens=8, threads=2)

    wave = engine.generate_token_wave(
        prompt, max_new_tokens=8, max_proposals=4, threads=2
    )

    assert wave.generated_token_ids == ordinary.generated_token_ids
    assert wave.frontier_logits_hashes == ordinary.frontier_logits_hashes
    assert wave.frontier_kv_hashes == ordinary.frontier_kv_hashes
    assert wave.final_logits_hash == ordinary.final_logits_hash
    assert wave.final_kv_hash == ordinary.final_kv_hash
    assert wave.final_kv_lengths == ordinary.final_kv_lengths
    assert max(wave.acceptance_lengths) > 0
    assert wave.committed_tokens_per_target_weight_traversal > 1.0
    assert wave.speculative_traversals > 0
    assert wave.transactional_snapshot_bytes_copied == 0

    rejected = engine.generate_token_wave(
        prompt,
        max_new_tokens=8,
        max_proposals=4,
        threads=2,
        adversarial_proposals=True,
    )

    assert rejected.generated_token_ids == ordinary.generated_token_ids
    assert rejected.final_logits_hash == ordinary.final_logits_hash
    assert rejected.final_kv_hash == ordinary.final_kv_hash
    assert rejected.final_kv_lengths == ordinary.final_kv_lengths
    assert rejected.rolled_back_tokens > 0
    assert rejected.rollback_verified
    assert rejected.fallback_traversals > 0
    assert 0 in rejected.acceptance_lengths
