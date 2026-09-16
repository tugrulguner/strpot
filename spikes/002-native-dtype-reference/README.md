# 002: Native-dtype reference verification

## Question

Can StrPot execute the original TinyLlama checkpoint in its declared BF16 dtype and reproduce an independent Hugging Face eager reference without converting resident weights to FP32?

## Contract

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Checkpoint dtype: BF16
- Device: CPU
- StrPot owns acquisition, image reconstruction, graph execution, KV cache, and generation.
- Transformers is used only as an offline correctness oracle.
- No quantization, pruning, retraining, or parameter replacement is permitted.

## Run

```bash
MODEL_ROOT="$HOME/.strpot/TinyLlama/TinyLlama-1.1B-Chat-v1.0/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
PYTHONPATH=src uv run \
  --with 'transformers>=5.0' \
  --with safetensors \
  python spikes/002-native-dtype-reference/verify.py "$MODEL_ROOT"
```

The verifier reconstructs `model.safetensors` from the StrPot image into a temporary directory, executes the prompt independently through StrPot and Hugging Face eager CPU paths, compares the complete final-token logit vector, and deletes the temporary checkpoint.

## Measured result

Prompt:

```text
Write a detailed sentence explaining why CPUs are useful.
```

```text
Prompt tokens:                  27
StrPot checkpoint dtype:        torch.bfloat16
StrPot argmax token:            6271
Reference argmax token:         6271
Maximum absolute logit error:   0.0
Mean absolute logit error:      0.0
Top-20 overlap:                 20 / 20
StrPot logits SHA-256:          8ba18a7dc497b8ae73f064662de797fad9e36ac8624a4c3472421b9362a4ce5a
Reference logits SHA-256:       8ba18a7dc497b8ae73f064662de797fad9e36ac8624a4c3472421b9362a4ce5a
```

Native-BF16 complete-model generation on the Apple M4 produced the same 16-token text as the former FP32-expanded path:

```text
CPUs (Central Processing Units) are essential components of modern computers and
```

Three-run median:

```text
Inference:          1.517680 seconds
Decode throughput:  10.542407 tokens/second
Resident weights:   2,200,096,768 bytes
Weight preload:     3.012105 seconds
```

The previous FP32-expanded result was 11.543108 tokens/second with 4,400,193,536 resident weight bytes. Native BF16 therefore halves resident weight memory but is currently slightly slower through PyTorch's CPU substrate. This is a measured kernel limitation, not justification for changing the checkpoint dtype.

## Verdict: VALIDATED

### What worked

- Resident loading preserves each tensor's source dtype by default.
- The Llama runtime keeps BF16 activations through linear, RoPE, attention, MLP, and output projection operations while using FP32 where the reference numerical contract requires it.
- The complete final-token logit vector was byte-identical to the independent eager reference for the tested prompt.
- Optional layer-zero Response Atlas execution now preserves BF16 as well.

### What remains

- Expand oracle coverage across multiple prompts, sequence lengths, cached decode steps, and sampling processors.
- Replace generic PyTorch linear dispatch with a dtype-native StrPot fixed-operator baseline.
- Measure layer-level hardware counters and bandwidth before evaluating alternative exact matrix programs.
