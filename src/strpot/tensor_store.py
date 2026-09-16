"""Read tensors directly from bounded StrPot pages."""

from __future__ import annotations

import json
import struct
from collections import OrderedDict
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from strpot.image import FORMAT, MAX_PAGE_SIZE, _read_verified_page


class PagedFile:
    """Expose a StrPot image as a bounded random-access byte source."""

    def __init__(self, image: Path, max_cached_pages: int = 2) -> None:
        if max_cached_pages < 1:
            raise ValueError("max_cached_pages must be positive")
        self.image = image.resolve()
        self.manifest: dict[str, Any] = json.loads(
            (self.image / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("format") != FORMAT:
            raise ValueError("unsupported StrPot image format")
        pages = self.manifest.get("pages")
        source_size = self.manifest.get("source_size")
        page_size = self.manifest.get("page_size")
        if (
            not isinstance(pages, list)
            or not isinstance(source_size, int)
            or isinstance(source_size, bool)
            or source_size < 0
            or not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or not 0 < page_size <= MAX_PAGE_SIZE
        ):
            raise ValueError("invalid StrPot image dimensions")
        expected_page_count = (source_size + page_size - 1) // page_size
        if len(pages) != expected_page_count:
            raise ValueError("invalid StrPot page layout")
        for index, page in enumerate(pages):
            expected_raw_size = min(page_size, source_size - index * page_size)
            if (
                not isinstance(page, dict)
                or page.get("raw_size") != expected_raw_size
                or isinstance(page.get("raw_size"), bool)
            ):
                raise ValueError("invalid StrPot page layout")
        self.pages = pages
        self.size = source_size
        self.page_size = page_size
        self.max_cached_pages = max_cached_pages
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    @property
    def resident_bytes(self) -> int:
        return sum(len(page) for page in self._cache.values())

    def _load_page(self, index: int) -> bytes:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        raw = _read_verified_page(self.image, self.pages[index], index)

        self._cache[index] = raw
        self._cache.move_to_end(index)
        while len(self._cache) > self.max_cached_pages:
            self._cache.popitem(last=False)
        return raw

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError("read is outside the paged file")
        if size == 0:
            return b""

        output = bytearray()
        cursor = offset
        remaining = size
        while remaining:
            page_index = cursor // self.page_size
            page_offset = cursor % self.page_size
            page = self._load_page(page_index)
            count = min(remaining, len(page) - page_offset)
            output.extend(page[page_offset : page_offset + count])
            cursor += count
            remaining -= count
        return bytes(output)


_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
}


class SafeTensorStore:
    """Load individual safetensors tensors through a bounded PagedFile."""

    def __init__(self, source: PagedFile) -> None:
        self.source = source
        if source.size < 8:
            raise ValueError("safetensors checkpoint is missing its header length")
        header_size = struct.unpack("<Q", source.read(0, 8))[0]
        if header_size <= 0 or header_size > source.size - 8:
            raise ValueError("invalid safetensors header length")
        header = json.loads(source.read(8, header_size).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be an object")
        self._data_offset = 8 + header_size
        payload_size = source.size - self._data_offset
        entries: dict[str, dict[str, Any]] = {}
        ranges: list[tuple[int, int, str]] = []
        for name, entry in header.items():
            if name == "__metadata__":
                if not isinstance(entry, dict):
                    raise ValueError("safetensors metadata must be an object")
                continue
            if not isinstance(name, str) or not name or not isinstance(entry, dict):
                raise ValueError("invalid safetensors tensor entry")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if dtype not in _DTYPES:
                raise ValueError(f"unsupported tensor dtype: {dtype}")
            if (
                not isinstance(shape, list)
                or len(shape) > 8
                or any(
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                    or dimension > payload_size
                    for dimension in shape
                )
            ):
                raise ValueError(f"invalid tensor shape: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in offsets
                )
            ):
                raise ValueError(f"invalid tensor offsets: {name}")
            start, end = offsets
            if start < 0 or end < start or end > payload_size:
                raise ValueError(f"tensor {name} range is outside checkpoint")
            _, item_size = _DTYPES[dtype]
            expected_size = item_size
            for dimension in shape:
                if expected_size > payload_size // dimension:
                    raise ValueError(f"tensor {name} shape exceeds checkpoint")
                expected_size *= dimension
            if end - start != expected_size:
                raise ValueError(f"invalid byte size for tensor: {name}")
            entries[name] = entry
            ranges.append((start, end, name))

        ranges.sort()
        for previous, current in pairwise(ranges):
            if current[0] < previous[1]:
                raise ValueError(
                    f"tensor ranges overlap: {previous[2]} and {current[2]}"
                )
        self._entries = entries
        self.names = tuple(sorted(self._entries))

    def load(self, name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        entry = self._entries[name]
        stored_dtype = entry["dtype"]
        try:
            torch_dtype, item_size = _DTYPES[stored_dtype]
        except KeyError as exc:
            raise ValueError(f"unsupported tensor dtype: {stored_dtype}") from exc
        shape = tuple(int(dimension) for dimension in entry["shape"])
        start, end = (int(value) for value in entry["data_offsets"])
        expected_size = item_size
        for dimension in shape:
            expected_size *= dimension
        if end - start != expected_size:
            raise ValueError(f"invalid byte size for tensor: {name}")

        raw = bytearray(self.source.read(self._data_offset + start, end - start))
        tensor = torch.frombuffer(raw, dtype=torch_dtype).reshape(shape).clone()
        return tensor.to(dtype=dtype) if dtype is not None else tensor


class ResidentTensorStore:
    """Decode and convert every tensor once when the model fits available RAM."""

    def __init__(
        self,
        source: SafeTensorStore,
        tensors: dict[str, torch.Tensor],
        dtype: torch.dtype | None,
    ) -> None:
        self.source = source.source
        self.names = source.names
        self._tensors = tensors
        self._dtype = dtype

    @classmethod
    def preload(
        cls, source: SafeTensorStore, *, dtype: torch.dtype | None = None
    ) -> ResidentTensorStore:
        tensors = {name: source.load(name, dtype=dtype) for name in source.names}
        return cls(source, tensors, dtype)

    def load(self, name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = self._tensors[name]
        if dtype is None or dtype == tensor.dtype:
            return tensor
        return tensor.to(dtype=dtype)

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.nelement() * tensor.element_size()
            for tensor in self._tensors.values()
        )
