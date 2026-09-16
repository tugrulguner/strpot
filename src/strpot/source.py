"""Model sources owned by StrPot rather than external runtime caches."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from strpot.image import DEFAULT_PAGE_SIZE, compile_image

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_METADATA_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


@dataclass(frozen=True)
class PreparedModel:
    repo_id: str
    revision: str
    root: Path
    weights_image: Path
    config_path: Path
    tokenizer_path: Path
    tokenizer_config_path: Path


class HuggingFaceSource:
    """Acquire public Hugging Face weights directly into a StrPot image."""

    def __init__(
        self,
        *,
        base_url: str = "https://huggingface.co",
        api_url: str = "https://huggingface.co/api/models",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_url = api_url.rstrip("/")

    def _read_url(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "strpot/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    def _download(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": "strpot/0.1"})
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            destination.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream, length=1024 * 1024)

    def prepare(
        self,
        repo_id: str,
        store: Path,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> PreparedModel:
        if not _REPO_PATTERN.fullmatch(repo_id):
            raise ValueError("model source must be an owner/repository identifier")
        owner, repository = repo_id.split("/")
        if owner in {".", ".."} or repository in {".", ".."}:
            raise ValueError("model source must be an owner/repository identifier")

        store_root = store.resolve()
        repository_root = (store_root / owner / repository).resolve()
        if not repository_root.is_relative_to(store_root):
            raise ValueError("model source resolves outside model store")

        encoded_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
        metadata = json.loads(self._read_url(f"{self.api_url}/{encoded_repo}"))
        revision = str(metadata["sha"])
        if not _REVISION_PATTERN.fullmatch(revision):
            raise ValueError("model repository returned an invalid revision")
        available = {entry["rfilename"] for entry in metadata["siblings"]}
        required = {*_METADATA_FILES, "model.safetensors"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"model repository is missing required files: {missing}")

        root = (repository_root / revision).resolve()
        if not root.is_relative_to(repository_root) or not root.is_relative_to(
            store_root
        ):
            raise ValueError("model revision resolves outside model store")
        marker = root / "prepared.json"
        if marker.exists():
            return self._prepared(repo_id, revision, root)

        root.mkdir(parents=True, exist_ok=True)
        encoded_revision = quote(revision, safe="")
        resolve_base = f"{self.base_url}/{encoded_repo}/resolve/{encoded_revision}"
        for filename in _METADATA_FILES:
            (root / filename).write_bytes(self._read_url(f"{resolve_base}/{filename}"))

        with tempfile.NamedTemporaryFile(
            prefix="strpot-weights-", suffix=".safetensors", delete=False, dir=root
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            self._download(
                f"{resolve_base}/model.safetensors",
                temporary_path,
            )
            compile_image(temporary_path, root / "weights.strpot", page_size=page_size)
        finally:
            temporary_path.unlink(missing_ok=True)

        marker.write_text(
            json.dumps({"repo_id": repo_id, "revision": revision}, indent=2) + "\n",
            encoding="utf-8",
        )
        return self._prepared(repo_id, revision, root)

    @staticmethod
    def _prepared(repo_id: str, revision: str, root: Path) -> PreparedModel:
        return PreparedModel(
            repo_id=repo_id,
            revision=revision,
            root=root,
            weights_image=root / "weights.strpot",
            config_path=root / "config.json",
            tokenizer_path=root / "tokenizer.json",
            tokenizer_config_path=root / "tokenizer_config.json",
        )
