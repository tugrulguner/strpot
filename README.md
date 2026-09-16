# StrPot

**Compile an existing neural model into a bounded, hardware-adaptive CPU execution image.**

StrPot is an experimental CPU-only inference framework. It treats fixed model weights as compiler input, divides the resulting model image into independently verifiable pages, and will execute those pages across the CPU, RAM, storage, and CPU computers that are available. A model does not need to fit in any one machine.

StrPot does not silently replace, quantize, prune, or retrain the model. Exact images preserve the source checkpoint byte for byte, and resident execution preserves each tensor's declared dtype. Approximate execution will be a separate, explicit mode.

> **Status:** `strpot run` now executes TinyLlama 1.1B through StrPot's own C++20 CPU engine: typed memory-mapped tensors, block prefill, matrix kernels, RMSNorm, RoPE, GQA attention, SwiGLU, KV caching, and greedy generation. Its architecture-neutral portable kernel reached a five-run median of 18.66 decode tokens/second on an Apple M4 with 10 runtime-autotuned threads while matching the complete 16-token PyTorch-oracle sequence. PyTorch is an optional benchmark dependency and is not imported or installed by the production path.

## Working foundation

Package any checkpoint or model file into bounded pages:

```bash
uv run strpot compile model.bin model.strpot --page-size 4194304
```

StrPot stores a page compressed only when compression makes it smaller. Every page records its raw SHA-256 digest, and identical pages are stored once.

Verify every page and the reconstructed source identity:

```bash
uv run strpot verify model.strpot
```

Inspect the local CPU execution environment:

```bash
uv run strpot profile
```

Run the native CPU-only Llama path (a C++20 compiler is required for the first cached build):

```bash
strpot run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Reply with one word: CPU" \
  --max-new-tokens 1
```

StrPot benchmarks complete native prefill and decode candidates once and caches the fastest portable thread count for that model and engine build. Pass `--threads N` to override autotuning explicitly.

The native data plane also accepts immutable, canonically hashed execution plans. Plans declare typed SSA values, exact checkpoint tensor bindings, operator order, and numerical semantics; the native consumer independently validates those declarations before execution. The shared operator registry is exercised by materially different RMSNorm/RoPE/SwiGLU and LayerNorm/learned-position/GELU decoder graphs.

An experimental Qwen2-family adapter emits this same plan contract, including Q/K/V projection bias, grouped-query attention, RoPE, SwiGLU, tied output heads, and declared BF16 boundaries. Deterministic nonzero fixtures validate the adapter and native executor; support outside the explicitly validated configuration fails closed.

Test exact response-space execution for layer-0 Q/K/V:

```bash
uv run --extra benchmark strpot test-response-atlas \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The measured TinyLlama experiments and their limitations are recorded in:

- [`spikes/001-exact-token-response-atlas`](spikes/001-exact-token-response-atlas/README.md)
- [`spikes/002-native-dtype-reference`](spikes/002-native-dtype-reference/README.md)
- [`spikes/003-strpot-native-engine`](spikes/003-strpot-native-engine/README.md)

## Intended inference path

```text
existing checkpoint
        ↓
portable operator graph
        ↓
fixed-matrix compiler
        ↓
runtime kernel dispatch
   ↙             ↘
portable fallback  optional ISA-specific kernels
        ↓
bounded content-addressed pages
        ↓
local or distributed CPU execution
        ↓
verified logits and generated tokens
```

## Non-negotiable contracts

- The inference path does not require a GPU.
- Exact mode preserves and verifies the complete source model.
- Checkpoint dtype is model input, not a StrPot architecture choice.
- Memory use is bounded independently of total model size.
- Compression is reversible and opportunistic, not a claim of free computation.
- Approximation is never enabled silently.
- Performance claims require end-to-end measurements against strong CPU baselines.
- Hardware-specific kernels are optional runtime backends; the model graph, numerical contract, and portable fallback remain architecture-neutral.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
make check
uv build
```

## License

MIT
