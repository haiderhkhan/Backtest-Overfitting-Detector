"""Local parquet cache of raw daily prices, so a second run needs no network.

One file per (symbol, start, end) under data/cache/. `fetch_universe()` is
the only entry point most code should use: it checks the cache first and
only calls the API for misses. Set PSX_OFFLINE=1 to forbid network use
entirely (a miss then raises instead of fetching).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"


class CacheMiss(LookupError):
    pass


class PriceCache:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, start: str, end: str) -> Path:
        return self.dir / f"{symbol}_{start}_{end}.parquet"

    def get(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        p = self._path(symbol, start, end)
        return pd.read_parquet(p) if p.exists() else None

    def put(self, symbol: str, start: str, end: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(symbol, start, end), index=False)

    def has(self, symbol: str, start: str, end: str) -> bool:
        return self._path(symbol, start, end).exists()


def is_offline() -> bool:
    return os.environ.get("PSX_OFFLINE", "").strip() in ("1", "true", "yes")


def fetch_universe(
    symbols: list[str],
    start: str,
    end: str,
    cache: PriceCache | None = None,
    client=None,
) -> dict[str, pd.DataFrame]:
    """Cache-first fetch of daily prices for every symbol.

    `client` is only constructed (and the API key only required) if at
    least one symbol is missing from the cache.
    Returns {symbol: DataFrame(date, open, high, low, close, volume)}.
    """
    cache = cache or PriceCache()
    out: dict[str, pd.DataFrame] = {}
    misses = []
    for s in symbols:
        df = cache.get(s, start, end)
        if df is None:
            misses.append(s)
        else:
            out[s] = df
    if misses:
        if is_offline():
            raise CacheMiss(f"PSX_OFFLINE=1 but not cached: {misses}")
        if client is None:
            from data.psx_client import PsxClient
            client = PsxClient()
        for s in misses:
            df = client.get_historical(s, start, end)
            cache.put(s, start, end, df)
            out[s] = df
    return {s: out[s] for s in symbols}
