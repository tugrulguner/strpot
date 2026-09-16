# 001: Exact Token Response Atlas on TinyLlama 1.1B

## Question

Can StrPot replace conventional matrix multiplication with input-dependent response lookup without caching generated outputs or changing model behavior?

## Model and hardware

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Host: Apple M4 Mac mini, 24 GB RAM
- Runtime: StrPot CPU graph
- Operation: exact layer-0 Q/K/V projections

Layer 0 receives the normalized embedding associated with each token before any contextual mixing. StrPot compiled the exact Q/K/V response for every vocabulary token, then selected responses using the current input token IDs.

This is an input-dependent mathematical lookup. It does not cache prompts, logits, generated text, or answers.

## Component result

Command:

```bash
strpot test-response-atlas TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --sampled-tokens 128 \
  --iterations 30
```

Measured result:

- Atlas compilation: 0.678 seconds
- Maximum absolute error across 128 sampled tokens: 0.0
- Conventional layer-0 Q/K/V median: 336.312 microseconds
- Exact response lookup median: 8.2085 microseconds
- Component speedup: 40.971x
- Original Q/K/V projection storage in FP32: 20,971,520 bytes
- Response atlas storage in FP32: 327,680,000 bytes
- Storage multiplier: 15.625x

## Full-model result

The exact response atlas was inserted into the complete StrPot TinyLlama forward graph. Both paths generated the same first token, `CP`, from the same 23-token prompt.

Three baseline timings:

```text
14.1636s
13.3757s
13.3231s
```

Three atlas timings:

```text
14.4092s
13.6508s
13.8569s
```

- Baseline median: 13.3757 seconds
- Atlas median: 13.8569 seconds
- End-to-end change: 3.598% slower

A complete greedy generation run requested up to eight tokens. Both paths stopped naturally on EOS after producing the same three-token answer, `CPU`:

```text
Baseline: 41.3251 seconds, 0.072595 tokens/second
Atlas:    41.8171 seconds, 0.071741 tokens/second
```

Excluding one-time atlas compilation, the atlas path was 1.191% slower in this complete generation. Including its 0.5226-second compilation, total atlas time was 42.3397 seconds and throughput was 0.070856 tokens/second. Runtime page residency was 8 MiB; the in-memory FP32 response table added 312.5 MiB.

The full model remains dominated by rereading and reconstructing every other matrix from an 8 MiB page cache. Replacing only three projections in one of 22 layers cannot overcome that cost, and the additional atlas memory does not yet replace original projection storage.

## Historical FP32-expanded runtime result

Profiling showed that the first full-model comparison measured archival zlib decompression rather than matrix execution: zlib consumed 11.172 seconds of a 14.436-second forward pass. StrPot was corrected to materialize raw executable pages once, retain FP32 execution weights when the model fits host memory, and use prefill plus per-layer KV-cached decoding.

With the corrected runtime, the same short prompt improved from 0.072595 to 7.781755 tokens/second, a 107.194x increase.

A longer generation produced 16 tokens and used three in-process repetitions:

```text
Resident + KV baseline: 1.386109 seconds, 11.543108 tokens/second
Resident + KV atlas:    1.362596 seconds, 11.742291 tokens/second
```

The atlas path was 1.726% faster in this run, but the difference is too small to establish a durable end-to-end gain. It still adds 312.5 MiB and accelerates only layer-0 Q/K/V. Both paths generated the identical text:

```text
CPUs (Central Processing Units) are essential components of modern computers and
```

This historical baseline uses 4.098 GiB of resident FP32 weights. Its one-time preload took 2.055 seconds; including preload, first-request throughput was 4.650 tokens/second. Repeated requests reuse the resident weights.

StrPot no longer forces checkpoint tensors to FP32. The current native-dtype result and independent logit verification are recorded in [`../002-native-dtype-reference`](../002-native-dtype-reference/README.md).

## Verdict: PARTIAL

### What worked

- Completely removed three real 1.1B-model matrix multiplications from the layer-0 online path.
- Preserved exact outputs for all sampled token responses.
- Preserved the complete model's observed generated token.
- Achieved a measured 40.971x component speedup.
- Demonstrated that response lookup remains input-dependent.

### What did not work

- Has not established a durable complete-model latency improvement; the corrected three-run result was only 1.726% faster.
- Increased layer-0 Q/K/V execution storage by 15.625x in the FP32 prototype.
- Applies exactly only before contextual mixing; later-layer activations need a compact manifold representation and residual correction.
- The initial runtime reconstructed the model on every token and had no KV cache; the corrected runtime removes both confounders.

### Recommendation

Do not scale the full-token lookup table to every layer. Preserve it as an exact specialization for discrete or otherwise finite activation sites. The next response-space experiment must target contextual activations using compact local bases or residual vector quantization, and it must replace enough projection traffic to affect end-to-end latency without a prohibitive storage multiplier.
