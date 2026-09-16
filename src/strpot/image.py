"""Lossless, content-addressed StrPot model images."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

FORMAT = "strpot-image-v1"
DEFAULT_PAGE_SIZE = 4 * 1024 * 1024
MAX_PAGE_SIZE = 64 * 1024 * 1024


def _publish_page(path: Path, payload: bytes) -> None:
    """Publish verified page bytes atomically, repairing stale content."""
    if (
        path.exists()
        and not path.is_symlink()
        and path.is_file()
        and path.read_bytes() == payload
    ):
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(payload)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_verified_page(image: Path, page: dict[str, Any], index: int) -> bytes:
    """Read one manifest page with bounded decompression and integrity checks."""
    image = image.resolve()
    relative_path = page.get("path")
    stored_size = page.get("stored_size")
    raw_size = page.get("raw_size")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"invalid path for page {index}")
    if (
        not isinstance(stored_size, int)
        or isinstance(stored_size, bool)
        or stored_size < 0
        or not isinstance(raw_size, int)
        or isinstance(raw_size, bool)
        or raw_size < 0
        or raw_size > MAX_PAGE_SIZE
    ):
        raise ValueError(f"invalid size for page {index}")
    page_path = (image / relative_path).resolve()
    if not page_path.is_relative_to(image):
        raise ValueError(f"page {index} escapes the image directory")
    payload = page_path.read_bytes()
    if len(payload) != stored_size:
        raise ValueError(f"stored size mismatch for page {index}")

    codec = page.get("codec")
    if codec == "raw":
        raw = payload
    elif codec == "zlib":
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(payload, raw_size + 1)
        if (
            len(raw) > raw_size
            or not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
        ):
            raise ValueError(f"page {index} exceeds its declared size")
    else:
        raise ValueError(f"unsupported codec for page {index}: {codec}")
    if len(raw) != raw_size:
        raise ValueError(f"raw size mismatch for page {index}")
    if hashlib.sha256(raw).hexdigest() != page.get("sha256"):
        raise ValueError(f"digest mismatch for page {index}")
    return raw


def _chunks(source: Path, page_size: int) -> Iterator[bytes]:
    with source.open("rb") as stream:
        while chunk := stream.read(page_size):
            yield chunk


def compile_image(
    source: Path, destination: Path, page_size: int = DEFAULT_PAGE_SIZE
) -> dict[str, Any]:
    """Package a file into independently verified, optionally compressed pages."""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise ValueError(f"source is not a file: {source}")
    if page_size <= 0 or page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page size must be between 1 and {MAX_PAGE_SIZE}")

    pages_dir = destination / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    pages: list[dict[str, Any]] = []

    for raw in _chunks(source, page_size):
        source_hash.update(raw)
        digest = hashlib.sha256(raw).hexdigest()
        compressed = zlib.compress(raw, level=1)
        if len(compressed) < len(raw):
            payload, codec = compressed, "zlib"
        else:
            payload, codec = raw, "raw"

        relative_path = Path("pages") / f"{digest}.{codec}"
        page_path = destination / relative_path
        _publish_page(page_path, payload)

        pages.append(
            {
                "sha256": digest,
                "raw_size": len(raw),
                "stored_size": len(payload),
                "codec": codec,
                "path": relative_path.as_posix(),
            }
        )

    manifest: dict[str, Any] = {
        "format": FORMAT,
        "source_name": source.name,
        "source_size": source.stat().st_size,
        "source_sha256": source_hash.hexdigest(),
        "page_size": page_size,
        "pages": pages,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def materialize_executable_pages(image: Path) -> dict[str, Any]:
    """Convert archival pages to raw pages once, outside the inference loop."""
    image = image.resolve()
    manifest_path = image / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported StrPot image format")

    obsolete_paths: set[Path] = set()
    for index, page in enumerate(manifest["pages"]):
        raw = _read_verified_page(image, page, index)
        if page["codec"] == "raw":
            continue
        source_path = (image / page["path"]).resolve()

        relative_path = Path("pages") / f"{page['sha256']}.raw"
        raw_path = image / relative_path
        _publish_page(raw_path, raw)
        obsolete_paths.add(source_path)
        page.update(
            {
                "codec": "raw",
                "path": relative_path.as_posix(),
                "stored_size": len(raw),
            }
        )

    temporary_manifest = image / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    referenced = {(image / page["path"]).resolve() for page in manifest["pages"]}
    for path in obsolete_paths - referenced:
        path.unlink(missing_ok=True)
    return manifest


def verify_image(image: Path) -> dict[str, Any]:
    """Verify page integrity and the reconstructed source digest."""
    image = image.resolve()
    manifest_path = image / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported StrPot image format")

    source_hash = hashlib.sha256()
    reconstructed_size = 0
    for index, page in enumerate(manifest["pages"]):
        raw = _read_verified_page(image, page, index)
        source_hash.update(raw)
        reconstructed_size += len(raw)

    if reconstructed_size != manifest["source_size"]:
        raise ValueError("reconstructed source size does not match manifest")
    if source_hash.hexdigest() != manifest["source_sha256"]:
        raise ValueError("reconstructed source digest does not match manifest")

    stored_size = sum(page["stored_size"] for page in manifest["pages"])
    return {
        "valid": True,
        "pages": len(manifest["pages"]),
        "source_size": reconstructed_size,
        "stored_size": stored_size,
        "source_sha256": source_hash.hexdigest(),
    }
