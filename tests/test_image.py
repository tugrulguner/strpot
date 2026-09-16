from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import pytest

from strpot.image import compile_image, materialize_executable_pages, verify_image


def test_compile_and_verify_lossless_image(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    raw = (b"transformer-weights" * 100) + bytes(range(255))
    source.write_bytes(raw)

    image = tmp_path / "model.strpot"
    manifest = compile_image(source, image, page_size=128)
    report = verify_image(image)

    assert report["valid"] is True
    assert report["source_size"] == len(raw)
    assert report["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(manifest["pages"]) > 1
    assert {page["codec"] for page in manifest["pages"]} == {"raw", "zlib"}


def test_compile_deduplicates_identical_pages(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"A" * 64 + b"A" * 64)

    image = tmp_path / "model.strpot"
    manifest = compile_image(source, image, page_size=64)

    assert len(manifest["pages"]) == 2
    assert len(list((image / "pages").iterdir())) == 1
    verify_image(image)


def test_compile_repairs_corrupt_existing_page(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(bytes(range(100)))
    image = tmp_path / "model.strpot"
    manifest = compile_image(source, image, page_size=100)
    page_path = image / manifest["pages"][0]["path"]
    page_path.write_bytes(b"X" * manifest["pages"][0]["stored_size"])

    compile_image(source, image, page_size=100)

    assert verify_image(image)["valid"] is True


def test_materialize_executable_pages_removes_runtime_decompression(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"compressible fixed weights" * 100)
    image = tmp_path / "model.strpot"
    manifest = compile_image(source, image, page_size=64)
    assert any(page["codec"] == "zlib" for page in manifest["pages"])

    converted = materialize_executable_pages(image)

    assert {page["codec"] for page in converted["pages"]} == {"raw"}
    assert verify_image(image)["valid"] is True


def test_materialize_repairs_corrupt_existing_raw_page(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"compressible fixed weights" * 100)
    image = tmp_path / "model.strpot"
    manifest = compile_image(source, image, page_size=64)
    page = manifest["pages"][0]
    assert page["codec"] == "zlib"
    raw_path = image / "pages" / f"{page['sha256']}.raw"
    raw_path.write_bytes(b"X" * page["raw_size"])

    materialize_executable_pages(image)

    assert verify_image(image)["valid"] is True


def test_materialize_executable_pages_verifies_existing_raw_pages(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(bytes(range(100)))
    image = tmp_path / "model.strpot"
    manifest = compile_image(source, image, page_size=100)
    page = manifest["pages"][0]
    assert page["codec"] == "raw"
    (image / page["path"]).write_bytes(b"X" * page["stored_size"])

    with pytest.raises(ValueError, match="digest mismatch"):
        materialize_executable_pages(image)


def test_verify_detects_corruption(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"compressible model weights" * 20)
    image = tmp_path / "model.strpot"
    manifest = compile_image(source, image, page_size=64)

    page_path = image / manifest["pages"][0]["path"]
    page_path.write_bytes(zlib.compress(b"different contents"))

    with pytest.raises(ValueError, match=r"stored size mismatch|raw size mismatch"):
        verify_image(image)


def test_verify_rejects_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    image = tmp_path / "model.strpot"
    compile_image(source, image)

    manifest_path = image / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pages"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="escapes"):
        verify_image(image)
