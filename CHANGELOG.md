# Changelog

All notable changes to StrPot will be documented here.

## Unreleased

- Created the initial StrPot package and lossless paged model-image foundation.
- Added StrPot-owned C++20 TinyLlama inference with block prefill, cached decode, runtime thread autotuning, and exact greedy generation.
- Added opt-in exact token-wave verification with transactional KV rollback.
- Added canonically hashed, typed execution plans with independent native validation and a shared portable operator registry.
- Added an experimental Qwen2-family plan adapter with deterministic end-to-end native conformance coverage.
- Added generic exact token-wave execution, P8/P16 AArch64 position-panel kernels with a portable fallback, perfect-oracle ceiling measurement, and a strict Llama-family plan adapter.
