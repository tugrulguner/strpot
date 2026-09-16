"""StrPot-owned CPU execution for Llama-family transformer graphs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as functional

from strpot.tensor_store import SafeTensorStore


class TokenResponseAtlas(Protocol):
    def lookup(self, projection: str, token_ids: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class LlamaCache:
    layers: tuple[LayerKV, ...]
    length: int


@dataclass(frozen=True)
class LlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    bos_token_id: int
    eos_token_id: int

    @classmethod
    def from_path(cls, path: Path) -> LlamaConfig:
        values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if values.get("model_type") != "llama":
            raise ValueError("StrPot currently supports model_type=llama")
        return cls(
            hidden_size=int(values["hidden_size"]),
            intermediate_size=int(values["intermediate_size"]),
            num_attention_heads=int(values["num_attention_heads"]),
            num_key_value_heads=int(values["num_key_value_heads"]),
            num_hidden_layers=int(values["num_hidden_layers"]),
            vocab_size=int(values["vocab_size"]),
            max_position_embeddings=int(values["max_position_embeddings"]),
            rms_norm_eps=float(values["rms_norm_eps"]),
            rope_theta=float(values.get("rope_theta", 10_000.0)),
            bos_token_id=int(values["bos_token_id"]),
            eos_token_id=int(values["eos_token_id"]),
        )


class LlamaRuntime:
    """Execute a Llama graph using tensors read from bounded StrPot pages."""

    def __init__(
        self,
        config: LlamaConfig,
        tensors: SafeTensorStore,
        *,
        first_layer_atlas: TokenResponseAtlas | None = None,
    ) -> None:
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        self.config = config
        self.tensors = tensors
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.first_layer_atlas = first_layer_atlas

    @property
    def page_resident_bytes(self) -> int:
        return self.tensors.source.resident_bytes

    def _weight(self, name: str) -> torch.Tensor:
        return self.tensors.load(name)

    def _linear(self, values: torch.Tensor, name: str) -> torch.Tensor:
        return functional.linear(values, self._weight(name))

    def _rms_norm(self, values: torch.Tensor, name: str) -> torch.Tensor:
        input_dtype = values.dtype
        values_fp32 = values.to(torch.float32)
        variance = values_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = values_fp32 * torch.rsqrt(variance + self.config.rms_norm_eps)
        normalized = normalized.to(input_dtype)
        return normalized * self._weight(name)

    @staticmethod
    def _rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor, start_position: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        frequencies = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(
            start_position,
            start_position + sequence_length,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, frequencies)
        embedding = torch.cat((angles, angles), dim=-1).unsqueeze(0)
        cosine = embedding.cos().to(query.dtype)
        sine = embedding.sin().to(query.dtype)
        return (
            query * cosine + self._rotate_half(query) * sine,
            key * cosine + self._rotate_half(key) * sine,
        )

    def _attention(
        self,
        hidden: torch.Tensor,
        layer: int,
        token_ids: torch.Tensor | None = None,
        cached: LayerKV | None = None,
        start_position: int = 0,
        capture_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerKV | None]:
        sequence_length = hidden.shape[0]
        prefix = f"model.layers.{layer}.self_attn"
        if layer == 0 and self.first_layer_atlas is not None:
            if token_ids is None:
                raise ValueError(
                    "token IDs are required for layer-zero response lookup"
                )
            query = self.first_layer_atlas.lookup("q", token_ids)
            key = self.first_layer_atlas.lookup("k", token_ids)
            value = self.first_layer_atlas.lookup("v", token_ids)
        else:
            query = self._linear(hidden, f"{prefix}.q_proj.weight")
            key = self._linear(hidden, f"{prefix}.k_proj.weight")
            value = self._linear(hidden, f"{prefix}.v_proj.weight")

        query = query.reshape(
            sequence_length, self.config.num_attention_heads, self.head_dim
        ).transpose(0, 1)
        key = key.reshape(
            sequence_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 1)
        value = value.reshape(
            sequence_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 1)
        query, key = self._apply_rope(query, key, start_position)

        if cached is not None:
            key = torch.cat((cached.key, key), dim=-2)
            value = torch.cat((cached.value, value), dim=-2)
        next_cache = LayerKV(key=key, value=value) if capture_cache else None

        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        key = key.repeat_interleave(repeats, dim=0)
        value = value.repeat_interleave(repeats, dim=0)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = torch.arange(
            start_position, start_position + sequence_length
        ).unsqueeze(1)
        key_positions = torch.arange(key.shape[-2]).unsqueeze(0)
        causal_mask = key_positions > query_positions
        scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(0, 1).reshape(
            sequence_length, self.config.hidden_size
        )
        return self._linear(attended, f"{prefix}.o_proj.weight"), next_cache

    def _mlp(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        prefix = f"model.layers.{layer}.mlp"
        gate = functional.silu(self._linear(hidden, f"{prefix}.gate_proj.weight"))
        up = self._linear(hidden, f"{prefix}.up_proj.weight")
        return self._linear(gate * up, f"{prefix}.down_proj.weight")

    def _execute(
        self,
        token_ids: list[int],
        *,
        cache: LlamaCache | None,
        capture_cache: bool,
    ) -> tuple[torch.Tensor, LlamaCache | None]:
        if not token_ids:
            raise ValueError("at least one token is required")
        start_position = cache.length if cache is not None else 0
        if start_position + len(token_ids) > self.config.max_position_embeddings:
            raise ValueError("token sequence exceeds the configured context length")
        if cache is not None and len(cache.layers) != self.config.num_hidden_layers:
            raise ValueError("KV cache layer count does not match model configuration")

        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        embeddings = self._weight("model.embed_tokens.weight")
        hidden = embeddings[token_tensor]
        del embeddings

        next_layers: list[LayerKV] = []
        for layer in range(self.config.num_hidden_layers):
            residual = hidden
            normalized = self._rms_norm(
                hidden, f"model.layers.{layer}.input_layernorm.weight"
            )
            attention, layer_cache = self._attention(
                normalized,
                layer,
                token_tensor if layer == 0 else None,
                cache.layers[layer] if cache is not None else None,
                start_position,
                capture_cache,
            )
            hidden = residual + attention
            if layer_cache is not None:
                next_layers.append(layer_cache)

            residual = hidden
            normalized = self._rms_norm(
                hidden, f"model.layers.{layer}.post_attention_layernorm.weight"
            )
            hidden = residual + self._mlp(normalized, layer)

        hidden = self._rms_norm(hidden, "model.norm.weight")
        logits = self._linear(hidden, "lm_head.weight")
        next_cache = None
        if capture_cache:
            next_cache = LlamaCache(
                layers=tuple(next_layers),
                length=start_position + len(token_ids),
            )
        return logits, next_cache

    @torch.inference_mode()
    def forward(self, token_ids: list[int]) -> torch.Tensor:
        logits, _ = self._execute(token_ids, cache=None, capture_cache=False)
        return logits

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> tuple[torch.Tensor, LlamaCache]:
        logits, cache = self._execute(token_ids, cache=None, capture_cache=True)
        if cache is None:
            raise RuntimeError("prefill did not produce a KV cache")
        return logits[-1], cache

    @torch.inference_mode()
    def decode_one(
        self, token_id: int, cache: LlamaCache
    ) -> tuple[torch.Tensor, LlamaCache]:
        logits, next_cache = self._execute([token_id], cache=cache, capture_cache=True)
        if next_cache is None:
            raise RuntimeError("decode did not produce a KV cache")
        return logits[-1], next_cache

    @torch.inference_mode()
    def generate(self, token_ids: list[int], *, max_new_tokens: int) -> list[int]:
        """Greedily decode tokens through StrPot's own forward path."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        generated = list(token_ids)
        if max_new_tokens == 0:
            return generated
        logits, cache = self.prefill(generated)
        for index in range(max_new_tokens):
            next_token = int(torch.argmax(logits).item())
            generated.append(next_token)
            if next_token == self.config.eos_token_id:
                break
            if index + 1 < max_new_tokens:
                logits, cache = self.decode_one(next_token, cache)
        return generated
