from strpot.inference import choose_weight_mode


def test_weight_mode_adapts_to_machine_memory() -> None:
    assert (
        choose_weight_mode(2_200_000_000, physical_memory=24_000_000_000) == "resident"
    )
    assert choose_weight_mode(20_000_000_000, physical_memory=24_000_000_000) == "paged"


def test_weight_mode_budgets_native_checkpoint_bytes() -> None:
    assert (
        choose_weight_mode(8_000_000_000, physical_memory=16_000_000_000) == "resident"
    )
