import numpy as np
import pandas as pd
import pytest

from f1pred.evaluation.uncertainty import paired_block_bootstrap


def test_pairing_precedes_blocks_and_loss_direction_is_correct():
    chronology = pd.DataFrame(
        {
            "race_id": [f"r{i}" for i in range(16)],
            "season": [2022] * 8 + [2023] * 8,
            "event_index": np.arange(16),
        }
    )
    a = chronology.assign(winner_logloss=1.0)
    b = chronology.assign(winner_logloss=0.8).sample(frac=1, random_state=2)
    comparison = paired_block_bootstrap(a, b, "winner_logloss", chronology, n_boot=500)
    assert comparison["delta"] == pytest.approx(-0.2)
    assert comparison["ci_low"] == pytest.approx(-0.2)
    assert comparison["ci_high"] == pytest.approx(-0.2)
    assert comparison["p_better"] == 1
    with pytest.raises(ValueError, match="chronology"):
        paired_block_bootstrap(a, b, "winner_logloss", chronology.iloc[1:])


def test_correlated_race_errors_widen_the_interval():
    chronology = pd.DataFrame(
        {
            "race_id": [f"r{i}" for i in range(80)],
            "season": np.repeat([2022, 2023, 2024, 2025], 20),
            "event_index": np.arange(80),
        }
    )
    a = chronology.assign(top1=0.0)
    b = chronology.assign(top1=np.tile(np.repeat([-1.0, 1.0], 10), 4))
    independent = paired_block_bootstrap(a, b, "top1", chronology, block_size=1)
    blocks = paired_block_bootstrap(a, b, "top1", chronology, block_size=4)
    assert blocks["ci_high"] - blocks["ci_low"] > independent["ci_high"] - independent["ci_low"]
