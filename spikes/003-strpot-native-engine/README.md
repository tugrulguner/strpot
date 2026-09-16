# 003: StrPot native engine

## Question

Can StrPot execute an unchanged TinyLlama 1.1B BF16 checkpoint through an owned CPU engine, without PyTorch, Transformers, llama.cpp, NumPy, or another inference runtime in the production request path?

## Native boundary

The production `strpot run` path now performs acquisition and tokenization in Python and transfers prompt token IDs once to a C++20 executable. The native process owns:

- memory-mapped typed BF16 and FP32 tensor views;
- matrix-vector multiplication and threading;
- RMSNorm, RoPE, grouped-query attention, softmax, and SwiGLU;
- per-layer KV-cache allocation and updates;
- the complete Llama layer/model loop;
- argmax and autoregressive greedy generation.

PyTorch remains available only through the explicit `benchmark-pytorch` command and benchmark/development dependency groups. Fresh-process tests fail if importing the production CLI or native binding imports PyTorch.

## Portability contract

The optimized kernel family is `portable-tiled`. Decode processes four output rows together so one activation load feeds four independent dot products. Block prefill additionally tiles four prompt positions, allowing one decoded weight value to feed several independent position accumulators. It contains no Apple, ARM, x86, or vendor-library calls. The checkpoint remains BF16 and each row's accumulation order remains unchanged.

The kernel set is selected behind an engine-level interface. Future NEON, AVX, or other ISA implementations must be optional runtime-selected backends. They must preserve the same graph and numerical contract, and the portable implementation remains the fallback. StrPot does not define its architecture around the development machine.

## Frozen model

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Checkpoint dtype: BF16
- Reconstructed checkpoint SHA-256: `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`
- Resident/mapped weight bytes: `2,200,096,768`
- Batch size: 1
- Prompt tokens: 27
- Generated tokens: 16
- Threads: 10, selected by StrPot's portable runtime autotuner
- Host measured: Apple M4

Prompt:

```text
Write a detailed sentence explaining why CPUs are useful.
```

## Complete-model result

Five repeated warm runs through the public command reported:

- prefill / time to first token, median: `0.700561250 s`;
- post-first-token decode time, median: `0.803828875 s`;
- decode throughput: `18.660688 tok/s`;
- p50 inter-token latency: `54.481792 ms`;
- p95 inter-token latency: `56.692358 ms`;
- end-to-end generated-token throughput including prefill: `10.635539 tok/s`.

The previous accepted portable result reached `9.388698 tok/s` with four explicitly selected threads and `2.718955 s` prefill on the same 27-token/16-token workload. Block prefill, compiled tensor binding, fused/coarsened projection scheduling, invariant RoPE tables, reusable attention-score storage, and portable thread autotuning raised the current result to `18.660688 tok/s` and reduced prefill to `0.700561 s`. Relative to that earlier configuration, decode is `1.9876x` faster and prefill is `3.8811x` faster; the thread count differs, so these ratios describe the complete runtime configuration rather than an isolated kernel change.

The native counters report:

- `155` prompt matrix passes, independent of the 27-token prompt length;
- `111` parallel decode phases per post-first-token step, down from `155`;
- zero runtime tensor-name lookups after plan construction;
- one fixed `2,048`-float attention-score workspace reused across layers and heads;
- precomputed RoPE sine/cosine tables reused across all layers.

A thread sweep produced:

- 1 thread: `2.689869 tok/s`;
- 2 threads: `5.159913 tok/s`;
- 3 threads: `7.186585 tok/s`;
- 4 threads: `9.342906 tok/s`.

The earlier one-through-four-thread sweep and the current autotuning candidates produced the same token sequence. Autotuning checks this invariant before caching a thread count.

## Oracle conformance

`verify.py` compares native generation with the existing dtype-preserving PyTorch reference graph. PyTorch is invoked only after the independent native request completes.

```bash
uv run --extra benchmark python spikes/003-strpot-native-engine/verify.py \
  ~/.strpot/TinyLlama/TinyLlama-1.1B-Chat-v1.0/fe8a4ea1ffedaf415f4da2f062534de366a451e6
```

The complete 16-token sequences matched. Both token-sequence digests were:

```text
9223886c5b69c80514609408ccec4fa492fc9487bb159129995bf79ecfef4db8
```

Decoded text:

```text
CPUs (Central Processing Units) are essential components of modern computers and
```

This establishes greedy-token identity for this frozen prompt. It does **not** establish native full-logit equality; native prefill and cached-decode logit comparison remains required.

## Current limitations

- The native binding currently reconstructs one exact contiguous safetensors file and memory-maps it. Direct execution from bounded StrPot pages is not implemented yet.
- The first run requires a local C++20 compiler. Prebuilt platform artifacts are not shipped yet.
- Only the exercised Llama/TinyLlama operator contract is supported.
- Sampling is not implemented; generation is greedy.
- No architecture-specific backend has been accepted yet.
- Decode still reads every dense target matrix for each emitted token. Request-local n-gram drafting with exact transactional block verification is available as an opt-in research path, but it is not accepted as an acceleration because representative acceptance-rate and latency gains have not been demonstrated.
- Greedy sequence identity is verified for the frozen prompt, but complete native prefill/decode logit vectors are not yet compared with the external oracle.

## Verdict: VALIDATED

StrPot now owns a complete native TinyLlama request path. Architecture-neutral block prefill and portable decode scheduling materially improved complete-model latency while preserving the frozen greedy sequence. The remaining fundamental limit is one dense target-weight traversal per emitted decode token; speculative block verification or exact fixed-matrix compilation must pass complete-model acceptance before either becomes a production claim.
