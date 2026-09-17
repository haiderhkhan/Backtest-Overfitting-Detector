"""Data layer tests. ZERO network: everything runs on the JSON fixture."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import cache as cache_mod
from data import psx_client
from data.returns import weekly_log_returns
from data.universe import load_universe

FIX = json.loads((Path(__file__).parent / "fixtures" / "historical_2023_2024.json").read_text())


def test_parse_historical_orders_oldest_first_and_types():
    df = psx_client.parse_historical(FIX["OGDC"])
    assert list(df.columns) == psx_client.PRICE_COLUMNS
    assert df["date"].is_monotonic_increasing
    assert df["close"].dtype == float
    assert len(df) == FIX["OGDC"]["meta"]["count"]


def test_missing_api_key_is_one_clear_error(monkeypatch):
    monkeypatch.delenv(psx_client.ENV_VAR, raising=False)
    monkeypatch.setattr(psx_client, "load_dotenv", lambda: None)
    with pytest.raises(psx_client.MissingApiKey, match=psx_client.ENV_VAR):
        psx_client.load_api_key()


class _FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
    def json(self):
        return self._payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _FakeSession:
    """Returns 429 twice, then 200 — exercises retry-with-backoff."""
    def __init__(self):
        self.calls = 0
    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        assert headers["X-API-Key"] == "k"
        return _FakeResponse(429 if self.calls < 3 else 200, FIX["LUCK"])


def test_client_retries_on_429(monkeypatch):
    monkeypatch.setattr(psx_client.time, "sleep", lambda s: None)
    sess = _FakeSession()
    c = psx_client.PsxClient(api_key="k", session=sess)
    df = c.get_historical("LUCK", "2023-01-01", "2024-12-31")
    assert sess.calls == 3
    assert len(df) == FIX["LUCK"]["meta"]["count"]


def test_cache_roundtrip_and_offline_guard(tmp_path, monkeypatch):
    cache = cache_mod.PriceCache(tmp_path)
    df = psx_client.parse_historical(FIX["OGDC"])
    cache.put("OGDC", "a", "b", df)
    back = cache.get("OGDC", "a", "b")
    pd.testing.assert_frame_equal(back, df)
    assert cache.get("NOPE", "a", "b") is None
    monkeypatch.setenv("PSX_OFFLINE", "1")
    got = cache_mod.fetch_universe(["OGDC"], "a", "b", cache=cache)
    assert list(got) == ["OGDC"]
    with pytest.raises(cache_mod.CacheMiss):
        cache_mod.fetch_universe(["OGDC", "NOPE"], "a", "b", cache=cache)


def test_fetch_universe_only_calls_api_for_misses(tmp_path):
    cache = cache_mod.PriceCache(tmp_path)
    cache.put("OGDC", "a", "b", psx_client.parse_historical(FIX["OGDC"]))
    class FakeClient:
        asked = []
        def get_historical(self, s, a, b):
            self.asked.append(s)
            return psx_client.parse_historical(FIX["LUCK"])
    got = cache_mod.fetch_universe(["OGDC", "LUCK"], "a", "b", cache=cache, client=FakeClient())
    assert FakeClient.asked == ["LUCK"]
    assert cache.has("LUCK", "a", "b")
    assert set(got) == {"OGDC", "LUCK"}


def test_universe_config_loads_and_validates(tmp_path):
    u = load_universe()
    assert len(u["symbols"]) >= 30 and len(set(u["symbols"])) == len(u["symbols"])
    bad = tmp_path / "u.yaml"
    bad.write_text("name: x\nstart: 2020-01-01\nend: 2021-01-01\nsymbols: [A, A]\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_universe(bad)


# ---------------------------------------------------------------- returns.py
def _daily(start="2020-01-01", n=520, seed=0, price0=100.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = price0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    return pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close, "volume": 1000})


def test_weekly_returns_clean_case_has_no_quality_warnings():
    prices = {"A": _daily(seed=1), "B": _daily(seed=2)}
    out = weekly_log_returns(prices)
    r = out["returns"]
    assert list(r.columns) == ["A", "B"]
    assert isinstance(r.index, pd.DatetimeIndex)
    assert (r.index.dayofweek == 4).all()  # W-FRI
    assert not any("missing" in w or "stale" in w or "starts" in w for w in out["warnings"])
    wc = out["close"]["A"]
    np.testing.assert_allclose(r["A"].iloc[5], np.log(wc.iloc[6] / wc.iloc[5]))


def test_pathologies_warn_but_do_not_crash():
    a = _daily(seed=1)
    gaps = a.drop(a.index[50:120])                       # missing days
    stale = _daily(seed=2); stale.loc[200:230, "close"] = stale.loc[200, "close"]  # stale run
    late = _daily(seed=3).iloc[260:]                     # lists mid-history
    dead = _daily(seed=4).iloc[:300]                     # delisted early
    out = weekly_log_returns({"GAP": gaps, "STALE": stale, "LATE": late, "DEAD": dead},
                             start="2020-01-01", end="2021-12-30")
    w = "\n".join(out["warnings"])
    assert "GAP:" in w and "missing" in w
    assert "STALE:" in w and "stale" in w
    assert "LATE:" in w and "starts" in w
    assert "DEAD:" in w and "ends" in w
    r = out["returns"]
    assert r["LATE"].first_valid_index() > r["GAP"].first_valid_index()
    assert r["DEAD"].last_valid_index() < r["GAP"].last_valid_index()
    assert np.isfinite(r.fillna(0).to_numpy()).all()


def test_too_short_history_warns():
    out = weekly_log_returns({"A": _daily(n=100)})
    assert any("too thin" in w for w in out["warnings"])
