"""PBO on rigged strategy sets where we KNOW the answer's direction."""

import math

import numpy as np
import pandas as pd
import pytest

from overfitting_detector.pbo import compute_pbo


def _matrix(n_rows, n_strats, seed, signal_col=None, signal=0.0):
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.02, (n_rows, n_strats))
    if signal_col is not None:
        M[:, signal_col] += signal
    idx = pd.date_range("2016-01-08", periods=n_rows, freq="W-FRI")
    return pd.DataFrame(M, index=idx, columns=[f"s{i}" for i in range(n_strats)])


def test_genuine_edge_gives_low_pbo():
    # one strategy has a real, large, persistent edge -> IS winner keeps
    # winning OOS -> PBO near 0
    M = _matrix(320, 10, seed=0, signal_col=3, signal=0.01)
    out = compute_pbo(M, num_partitions=8)
    assert out["pbo"] < 0.10
    assert out["num_combinations"] == math.comb(8, 4)
    assert len(out["lambdas"]) == len(out["omegas"]) == out["num_combinations"]


def test_pure_noise_gives_pbo_near_half():
    M = _matrix(320, 10, seed=1)
    out = compute_pbo(M, num_partitions=8)
    assert 0.3 < out["pbo"] < 0.7


def test_lambda_is_logit_of_omega():
    out = compute_pbo(_matrix(160, 6, seed=2), num_partitions=4)
    for lam, om in zip(out["lambdas"], out["omegas"]):
        assert lam == pytest.approx(math.log(om / (1 - om)))
        assert 0 < om < 1


def test_odd_partitions_rejected():
    with pytest.raises(ValueError):
        compute_pbo(_matrix(100, 5, seed=3), num_partitions=5)


def test_few_strategies_warns():
    out = compute_pbo(_matrix(160, 3, seed=4), num_partitions=4)
    assert any("strategies" in w for w in out["warnings"])
