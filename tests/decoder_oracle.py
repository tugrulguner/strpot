"""Pure-stdlib numerical oracle for generic decoder-plan tests only."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _div(left: float, right: float) -> float:
    return _f32(_f32(left) / _f32(right))


def _bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", _f32(value)))[0]
    if bits & 0x7F800000 == 0x7F800000:
        upper = bits >> 16
        if bits & 0x007FFFFF:
            upper |= 0x0040
    else:
        upper = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return struct.unpack("<f", struct.pack("<I", (upper & 0xFFFF) << 16))[0]


@dataclass
class OracleResult:
    tokens: tuple[int, ...]
    frontier_logits: tuple[tuple[float, ...], ...]
    cache_keys: tuple[tuple[float, ...], ...]
    cache_values: tuple[tuple[float, ...], ...]
    cache_lengths: tuple[int, ...]


class DecoderOracle:
    """Independent scalar interpretation of the exercised plan semantics."""

    def __init__(
        self, plan: Any, tensors: dict[str, tuple[tuple[int, ...], list[float]]]
    ):
        self.plan = plan
        self.tensors = {
            name: (shape, [_f32(value) for value in values])
            for name, (shape, values) in tensors.items()
        }
        self.caches: dict[str, dict[str, Any]] = {}
        self.cache_order: list[str] = []

    def _tensor(self, alias: str) -> tuple[tuple[int, ...], list[float]]:
        return self.tensors[alias]

    def _quantize(self, value: float, *, output: bool = False) -> float:
        dtype = (
            self.plan.semantics.output_dtype
            if output
            else self.plan.semantics.activation_dtype
        )
        if dtype == "F32":
            return _f32(value)
        return _bf16(value)

    def _linear(
        self,
        values: list[float],
        weight_name: str,
        bias_name: str | None = None,
        *,
        logits: bool = False,
    ) -> list[float]:
        shape, weight = self._tensor(weight_name)
        bias = self._tensor(bias_name)[1] if bias_name is not None else None
        output = []
        for row in range(shape[0]):
            total = bias[row] if bias is not None else _f32(0.0)
            for column, item in enumerate(values):
                total = _add(total, _mul(weight[row * shape[1] + column], item))
            output.append(self._quantize(total, output=logits))
        return output

    def _normalize(
        self,
        values: list[float],
        scale_name: str,
        bias_name: str | None,
        epsilon: float,
    ) -> list[float]:
        scale = self._tensor(scale_name)[1]
        bias = self._tensor(bias_name)[1] if bias_name is not None else None
        mean = _f32(0.0)
        if bias is not None:
            for item in values:
                mean = _add(mean, item)
            mean = _div(mean, _f32(len(values)))
        variance = _f32(0.0)
        for item in values:
            centered = _add(item, -mean)
            variance = _add(variance, _mul(centered, centered))
        mean_variance = _div(variance, _f32(len(values)))
        root = _f32(math.sqrt(_add(mean_variance, _f32(epsilon))))
        inverse = _div(_f32(1.0), root)
        output = []
        for index, item in enumerate(values):
            normalized = self._quantize(_mul(_add(item, -mean), inverse))
            normalized = _mul(normalized, scale[index])
            if bias is not None:
                normalized = _add(normalized, bias[index])
            output.append(self._quantize(normalized))
        return output

    def _rope(
        self,
        values: list[float],
        heads: int,
        head_dim: int,
        position: int,
        theta: float,
    ) -> None:
        half = head_dim // 2
        for head in range(heads):
            for index in range(half):
                exponent = _div(_f32(2 * index), _f32(head_dim))
                denominator = _f32(math.pow(_f32(theta), exponent))
                angle = _div(_f32(position), denominator)
                cosine = self._quantize(_f32(math.cos(angle)))
                sine = self._quantize(_f32(math.sin(angle)))
                first_index = head * head_dim + index
                second_index = first_index + half
                first, second = values[first_index], values[second_index]
                first_cosine = self._quantize(_mul(first, cosine))
                second_sine = self._quantize(_mul(second, sine))
                second_cosine = self._quantize(_mul(second, cosine))
                first_sine = self._quantize(_mul(first, sine))
                values[first_index] = self._quantize(_add(first_cosine, -second_sine))
                values[second_index] = self._quantize(_add(second_cosine, first_sine))

    def _attention(self, op: Any, values: list[float], position: int) -> list[float]:
        attributes = dict(op.attributes)
        heads = int(attributes["heads"])
        kv_heads = int(attributes["kv_heads"])
        hidden = dict(self.plan.shapes)["hidden_size"]
        head_dim = hidden // heads
        biased = op.kind == "attention_rope_qkv_bias"
        query = self._linear(values, op.tensors[0], op.tensors[1] if biased else None)
        key = self._linear(
            values, op.tensors[2 if biased else 1], op.tensors[3] if biased else None
        )
        val = self._linear(
            values, op.tensors[4 if biased else 2], op.tensors[5] if biased else None
        )
        if op.kind in {"attention_rope", "attention_rope_qkv_bias"}:
            theta = _f32(float(attributes["theta"]))
            self._rope(query, heads, head_dim, position, theta)
            self._rope(key, kv_heads, head_dim, position, theta)
        cache_name = attributes["cache"]
        if cache_name not in self.caches:
            self.caches[cache_name] = {
                "key": [],
                "value": [],
                "length": 0,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
            }
            self.cache_order.append(cache_name)
        cache = self.caches[cache_name]
        cache["key"].extend(key)
        cache["value"].extend(val)
        cache["length"] += 1
        attended = [_f32(0.0)] * hidden
        repeats = heads // kv_heads
        scale = (
            _f32(float(attributes["scale"]))
            if biased
            else _div(_f32(1.0), _f32(math.sqrt(_f32(head_dim))))
        )
        for head in range(heads):
            kv_head = head // repeats
            scores = []
            for token in range(cache["length"]):
                score = _f32(0.0)
                for dimension in range(head_dim):
                    cache_index = (token * kv_heads + kv_head) * head_dim + dimension
                    score = _add(
                        score,
                        _mul(
                            query[head * head_dim + dimension],
                            cache["key"][cache_index],
                        ),
                    )
                scores.append(self._quantize(_mul(self._quantize(score), scale)))
            maximum = max(scores)
            probabilities = [_f32(math.exp(_add(score, -maximum))) for score in scores]
            denominator = _f32(0.0)
            for probability in probabilities:
                denominator = _add(denominator, probability)
            for dimension in range(head_dim):
                total = _f32(0.0)
                for token, probability in enumerate(probabilities):
                    cache_index = (token * kv_heads + kv_head) * head_dim + dimension
                    total = _add(
                        total,
                        _mul(
                            self._quantize(_div(probability, denominator)),
                            cache["value"][cache_index],
                        ),
                    )
                attended[head * head_dim + dimension] = self._quantize(total)
        return self._linear(attended, op.tensors[6 if biased else 3])

    def _forward(self, token: int, position: int) -> list[float]:
        values: dict[str, list[float]] = {}
        for op in self.plan.operators:
            attributes = dict(op.attributes)
            if op.kind == "embedding":
                shape, weight = self._tensor(op.tensors[0])
                values[op.outputs[0]] = [
                    self._quantize(item)
                    for item in weight[token * shape[1] : (token + 1) * shape[1]]
                ]
            elif op.kind == "position_embedding":
                source = values[op.inputs[0]]
                shape, weight = self._tensor(op.tensors[0])
                values[op.outputs[0]] = [
                    self._quantize(_add(item, weight[position * shape[1] + index]))
                    for index, item in enumerate(source)
                ]
            elif op.kind == "save":
                values[op.outputs[0]] = list(values[op.inputs[0]])
            elif op.kind in {"rms_norm", "layer_norm"}:
                values[op.outputs[0]] = self._normalize(
                    values[op.inputs[0]],
                    op.tensors[0],
                    op.tensors[1] if op.kind == "layer_norm" else None,
                    _f32(float(attributes["epsilon"])),
                )
            elif op.kind in {
                "attention_rope",
                "attention_rope_qkv_bias",
                "attention_causal",
            }:
                values[op.outputs[0]] = self._attention(
                    op, values[op.inputs[0]], position
                )
            elif op.kind == "add":
                values[op.outputs[0]] = [
                    self._quantize(_add(left, right))
                    for left, right in zip(
                        values[op.inputs[0]], values[op.inputs[1]], strict=True
                    )
                ]
            elif op.kind == "swiglu":
                gate = self._linear(values[op.inputs[0]], op.tensors[0])
                up = self._linear(values[op.inputs[0]], op.tensors[1])
                activated = []
                for gate_item, up_item in zip(gate, up, strict=True):
                    sigmoid_denominator = _add(_f32(1.0), _f32(math.exp(-gate_item)))
                    silu = self._quantize(_div(gate_item, sigmoid_denominator))
                    activated.append(self._quantize(_mul(silu, up_item)))
                values[op.outputs[0]] = self._linear(activated, op.tensors[2])
            elif op.kind == "gelu_exact":
                hidden = self._linear(
                    values[op.inputs[0]], op.tensors[0], op.tensors[1]
                )
                activated = []
                root_two = _f32(math.sqrt(_f32(2.0)))
                for item in hidden:
                    erf_term = _f32(math.erf(_div(item, root_two)))
                    activated.append(
                        self._quantize(
                            _mul(_mul(_f32(0.5), item), _add(_f32(1.0), erf_term))
                        )
                    )
                values[op.outputs[0]] = self._linear(
                    activated, op.tensors[2], op.tensors[3]
                )
            elif op.kind == "linear":
                values[op.outputs[0]] = self._linear(
                    values[op.inputs[0]],
                    op.tensors[0],
                    op.tensors[1] if len(op.tensors) == 2 else None,
                    logits=op.outputs[0] == "logits",
                )
                if dict(op.attributes).get("result_rounding") == "BF16":
                    values[op.outputs[0]] = [
                        _bf16(item) for item in values[op.outputs[0]]
                    ]
            else:
                raise AssertionError(f"oracle does not implement {op.kind}")
        return values["logits"]

    def generate(self, prompt: list[int], max_new_tokens: int) -> OracleResult:
        logits: list[float] = []
        position = 0
        for token in prompt:
            logits = self._forward(token, position)
            position += 1
        tokens: list[int] = []
        frontiers: list[tuple[float, ...]] = []
        for index in range(max_new_tokens):
            frontiers.append(tuple(logits))
            token = max(range(len(logits)), key=logits.__getitem__)
            tokens.append(token)
            if token == self.plan.eos_token_id or index + 1 == max_new_tokens:
                break
            logits = self._forward(token, position)
            position += 1
        return OracleResult(
            tokens=tuple(tokens),
            frontier_logits=tuple(frontiers),
            cache_keys=tuple(
                tuple(self.caches[name]["key"]) for name in self.cache_order
            ),
            cache_values=tuple(
                tuple(self.caches[name]["value"]) for name in self.cache_order
            ),
            cache_lengths=tuple(
                self.caches[name]["length"] for name in self.cache_order
            ),
        )
