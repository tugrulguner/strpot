from __future__ import annotations

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from strpot.source import HuggingFaceSource

_REVISION = "a" * 40


def _metadata(revision: str) -> bytes:
    return json.dumps(
        {
            "sha": revision,
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "tokenizer_config.json"},
                {"rfilename": "model.safetensors"},
            ],
        }
    ).encode()


def test_huggingface_source_rejects_repository_path_traversal(tmp_path: Path) -> None:
    source = HuggingFaceSource()

    for repo_id in ("../model", "owner/..", "./model", "owner/."):
        with pytest.raises(ValueError, match="owner/repository"):
            source.prepare(repo_id, tmp_path / "store")


def test_huggingface_source_rejects_symlink_escape(tmp_path: Path) -> None:
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    store.mkdir()
    outside.mkdir()
    (store / "owner").symlink_to(outside, target_is_directory=True)

    source = HuggingFaceSource()
    with pytest.raises(ValueError, match="outside model store"):
        source.prepare("owner/model", store)


@pytest.mark.parametrize("revision", ["../escape", "abc123", "A" * 40, "a" * 39])
def test_huggingface_source_rejects_invalid_remote_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    source = HuggingFaceSource()
    monkeypatch.setattr(source, "_read_url", lambda _url: _metadata(revision))

    with pytest.raises(ValueError, match="revision"):
        source.prepare("owner/model", tmp_path / "store")


def test_huggingface_source_rejects_revision_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    repository = store / "owner" / "model"
    outside = tmp_path / "outside"
    repository.mkdir(parents=True)
    outside.mkdir()
    (repository / _REVISION).symlink_to(outside, target_is_directory=True)
    source = HuggingFaceSource()
    monkeypatch.setattr(source, "_read_url", lambda _url: _metadata(_REVISION))

    with pytest.raises(ValueError, match="outside model store"):
        source.prepare("owner/model", store)


def test_huggingface_source_builds_strpot_image_without_external_cache(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    repo_files = remote / "owner" / "model" / "resolve" / _REVISION
    repo_files.mkdir(parents=True)
    (repo_files / "config.json").write_text('{"model_type":"llama"}')
    (repo_files / "tokenizer.json").write_text("{}")
    (repo_files / "tokenizer_config.json").write_text("{}")
    weights = b"fixed-neural-weights" * 50
    (repo_files / "model.safetensors").write_bytes(weights)

    api = remote / "api" / "models" / "owner"
    api.mkdir(parents=True)
    (api / "model").write_text(
        json.dumps(
            {
                "sha": _REVISION,
                "siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "tokenizer.json"},
                    {"rfilename": "tokenizer_config.json"},
                    {"rfilename": "model.safetensors"},
                ],
            }
        )
    )

    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=remote, **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        source = HuggingFaceSource(base_url=base, api_url=f"{base}/api/models")
        prepared = source.prepare("owner/model", tmp_path / "strpot-store")
    finally:
        server.shutdown()
        thread.join()

    assert prepared.revision == _REVISION
    assert prepared.config_path.read_text() == '{"model_type":"llama"}'
    assert prepared.weights_image.joinpath("manifest.json").exists()
    assert not prepared.root.joinpath("model.safetensors").exists()
    assert (
        prepared.weights_image.joinpath("manifest.json")
        .read_text()
        .count("source_sha256")
        == 1
    )
