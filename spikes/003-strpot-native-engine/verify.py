from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from strpot.image import materialize_executable_pages
from strpot.llama import LlamaConfig, LlamaRuntime
from strpot.native import NativeLlamaEngine
from strpot.tensor_store import PagedFile, ResidentTensorStore, SafeTensorStore


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument(
        "--prompt",
        default="Write a detailed sentence explaining why CPUs are useful.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()

    root = arguments.model_root.resolve()
    formatted = f"<|user|>\n{arguments.prompt}</s>\n<|assistant|>\n"
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False).ids

    native = NativeLlamaEngine(
        config_path=root / "config.json",
        weights_image=root / "weights.strpot",
    ).generate(
        prompt_ids,
        max_new_tokens=arguments.max_new_tokens,
        threads=arguments.threads,
    )

    materialize_executable_pages(root / "weights.strpot")
    paged = PagedFile(root / "weights.strpot", max_cached_pages=2)
    tensors = ResidentTensorStore.preload(SafeTensorStore(paged))
    oracle = LlamaRuntime(LlamaConfig.from_path(root / "config.json"), tensors)
    oracle_tokens = oracle.generate(
        prompt_ids,
        max_new_tokens=arguments.max_new_tokens,
    )[len(prompt_ids) :]
    native_tokens = list(native.generated_token_ids)

    report = {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(native_tokens),
        "checkpoint_dtype": native.weight_dtype,
        "native_engine": native.engine,
        "native_kernel_family": native.kernel_family,
        "oracle_engine": "pytorch-reference",
        "token_sequence_equal": native_tokens == oracle_tokens,
        "native_tokens_sha256": token_digest(native_tokens),
        "oracle_tokens_sha256": token_digest(oracle_tokens),
        "native_tokens": native_tokens,
        "oracle_tokens": oracle_tokens,
        "text": tokenizer.decode(native_tokens, skip_special_tokens=True),
    }
    print(json.dumps(report, indent=2))
    if native_tokens != oracle_tokens:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
