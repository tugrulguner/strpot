from __future__ import annotations

import torch
import torch.nn.functional as functional

from strpot.response_atlas import ExactTokenResponseAtlas


def test_exact_token_atlas_matches_projection_without_caching_outputs() -> None:
    embeddings = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [-3.0, 0.5]], dtype=torch.float32
    )
    norm_weight = torch.tensor([0.75, 1.25], dtype=torch.float32)
    projection = torch.tensor(
        [[1.0, 2.0], [-0.5, 3.0], [4.0, -1.0]], dtype=torch.float32
    )
    atlas = ExactTokenResponseAtlas.compile(
        embeddings, norm_weight, {"q": projection}, rms_norm_eps=1e-5
    )

    token_ids = torch.tensor([0, 2])
    selected = embeddings[token_ids]
    normalized = selected * torch.rsqrt(
        selected.pow(2).mean(dim=-1, keepdim=True) + 1e-5
    )
    normalized = normalized * norm_weight
    expected = functional.linear(normalized, projection)

    actual = atlas.lookup("q", token_ids)

    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[0], actual[1])
