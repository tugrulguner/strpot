from __future__ import annotations

import strpot.native_inference as native_inference


def test_percentile_interpolates_inter_token_latency() -> None:
    assert native_inference.percentile([0.1, 0.2, 0.3], 50) == 0.2
    assert native_inference.percentile([0.1, 0.2, 0.3], 95) == 0.29
