from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from strpot.image import compile_image
from strpot.tensor_store import PagedFile, ResidentTensorStore, SafeTensorStore


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        start = len(payload)
        payload.extend(raw)
        dtype = {torch.bfloat16: "BF16", torch.float32: "F32"}[tensor.dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_paged_file_reads_across_page_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(bytes(range(100)))
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=16)

    paged = PagedFile(image, max_cached_pages=2)

    assert paged.read(13, 20) == bytes(range(13, 33))
    assert paged.resident_bytes <= 32


def test_paged_file_rejects_malformed_page_layout(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(bytes(range(32)))
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=16)
    manifest_path = image / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"][0]["raw_size"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="page layout"):
        PagedFile(image)


def test_paged_file_rejects_same_size_page_corruption(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(bytes(range(100)))
    image = tmp_path / "weights.strpot"
    manifest = compile_image(source, image, page_size=100)
    page = manifest["pages"][0]
    assert page["codec"] == "raw"
    (image / page["path"]).write_bytes(b"X" * page["stored_size"])

    with pytest.raises(ValueError, match="digest mismatch"):
        PagedFile(image).read(0, 1)


@pytest.mark.parametrize(
    "entries, message",
    [
        (
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            },
            "overlap",
        ),
        (
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [-4, 0]},
            },
            "outside checkpoint",
        ),
    ],
)
def test_safetensor_store_rejects_invalid_tensor_ranges(
    tmp_path: Path, entries: dict[str, object], message: str
) -> None:
    source = tmp_path / "model.safetensors"
    encoded = json.dumps(entries, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=17)

    with pytest.raises(ValueError, match=message):
        SafeTensorStore(PagedFile(image))


def test_safetensor_store_loads_bfloat16_from_strpot_pages(tmp_path: Path) -> None:
    expected = torch.tensor([[1.5, -2.0], [3.25, 4.5]], dtype=torch.bfloat16)
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, {"model.weight": expected})
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=17)

    store = SafeTensorStore(PagedFile(image, max_cached_pages=2))
    actual = store.load("model.weight")

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    assert store.names == ("model.weight",)


def test_resident_store_decodes_each_tensor_once(tmp_path: Path) -> None:
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, {"model.weight": expected})
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=17)

    resident = ResidentTensorStore.preload(
        SafeTensorStore(PagedFile(image, max_cached_pages=1)),
        dtype=torch.float32,
    )
    first = resident.load("model.weight", dtype=torch.float32)
    second = resident.load("model.weight", dtype=torch.float32)

    assert first.data_ptr() == second.data_ptr()
    assert resident.resident_bytes == expected.nelement() * expected.element_size()


def test_resident_store_preserves_checkpoint_dtypes_by_default(tmp_path: Path) -> None:
    bf16 = torch.tensor([[1.5, -2.0]], dtype=torch.bfloat16)
    fp32 = torch.tensor([3.25], dtype=torch.float32)
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, {"bf16.weight": bf16, "fp32.weight": fp32})
    image = tmp_path / "weights.strpot"
    compile_image(source, image, page_size=17)

    resident = ResidentTensorStore.preload(
        SafeTensorStore(PagedFile(image, max_cached_pages=1))
    )

    assert resident.load("bf16.weight").dtype == torch.bfloat16
    assert resident.load("fp32.weight").dtype == torch.float32
    assert torch.equal(resident.load("bf16.weight"), bf16)
    assert torch.equal(resident.load("fp32.weight"), fp32)
    assert resident.resident_bytes == 8
