from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from strpot.llama import LlamaConfig, LlamaRuntime
from strpot.tensor_store import PagedFile, ResidentTensorStore, SafeTensorStore


def reconstruct_weights(root: Path, destination: Path) -> None:
    image = root / "weights.strpot"
    manifest = json.loads((image / "manifest.json").read_text())
    with destination.open("wb") as output:
        for page in manifest["pages"]:
            if page["codec"] != "raw":
                raise RuntimeError(
                    "materialize StrPot executable pages before verification"
                )
            with (image / page["path"]).open("rb") as source:
                shutil.copyfileobj(source, output)


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument(
        "--prompt",
        default="Write a detailed sentence explaining why CPUs are useful.",
    )
    arguments = parser.parse_args()
    root = arguments.model_root.resolve()
    formatted = f"<|user|>\n{arguments.prompt}</s>\n<|assistant|>\n"
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False).ids

    paged = PagedFile(root / "weights.strpot", max_cached_pages=2)
    tensors = ResidentTensorStore.preload(SafeTensorStore(paged))
    runtime = LlamaRuntime(LlamaConfig.from_path(root / "config.json"), tensors)
    strpot_logits = runtime.forward(prompt_ids)[-1].float()

    with tempfile.TemporaryDirectory(prefix="strpot-reference-") as temporary:
        reference_root = Path(temporary)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(root / name, reference_root / name)
        reconstruct_weights(root, reference_root / "model.safetensors")
        reference = AutoModelForCausalLM.from_pretrained(
            reference_root,
            local_files_only=True,
            dtype="auto",
            attn_implementation="eager",
        ).eval()
        with torch.inference_mode():
            reference_logits = (
                reference(
                    input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                    use_cache=False,
                )
                .logits[0, -1]
                .float()
            )

    difference = (strpot_logits - reference_logits).abs()
    topk = 20
    strpot_top = set(torch.topk(strpot_logits, topk).indices.tolist())
    reference_top = set(torch.topk(reference_logits, topk).indices.tolist())
    print(
        json.dumps(
            {
                "prompt_tokens": len(prompt_ids),
                "checkpoint_dtype": str(
                    tensors.load("model.embed_tokens.weight").dtype
                ),
                "strpot_argmax": int(strpot_logits.argmax()),
                "reference_argmax": int(reference_logits.argmax()),
                "maximum_absolute_logit_error": float(difference.max()),
                "mean_absolute_logit_error": float(difference.mean()),
                "top_20_overlap": len(strpot_top & reference_top),
                "strpot_logits_sha256": tensor_sha256(strpot_logits),
                "reference_logits_sha256": tensor_sha256(reference_logits),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
