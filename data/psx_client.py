"""Thin client for the psxdata REST API (https://psxdata.mintlify.app).

Base URL and endpoint shapes come from the published OpenAPI spec.
Historical rows look like:
    {"date": "2024-01-15", "open": 128.6, "high": 133.8, "low": 128.51,
     "close": 130.29, "volume": 18594025, "is_anomaly": false}
The API returns newest-first; we flip to oldest-first.

The key is read from the PSXDATA_API_KEY environment variable (a .env
file is loaded if present). It is sent as an X-API-Key header. Missing
key -> one clear error, no retries, no guessing.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

BASE_URL = "https://psxdata-api.fastapicloud.dev"
ENV_VAR = "PSXDATA_API_KEY"
PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

MIN_SECONDS_BETWEEN_CALLS = 0.5   # be polite
MAX_ATTEMPTS = 4                  # 1 try + 3 retries
BACKOFF_SECONDS = (1, 2, 4)       # after attempt 1, 2, 3
RETRY_STATUSES = {429, 500, 502, 503, 504}


class MissingApiKey(RuntimeError):
    pass


def load_api_key() -> str:
    """Read PSXDATA_API_KEY from the environment (.env is loaded first).

    utf-8-sig: Windows editors often save .env with a BOM, which would
    otherwise turn the variable name into "﻿PSXDATA_API_KEY". The
    second lookup covers a shell that already exported it under that name.
    """
    load_dotenv(find_dotenv(usecwd=True), encoding="utf-8-sig")
    key = (os.environ.get(ENV_VAR) or os.environ.get("﻿" + ENV_VAR) or "").strip()
    if not key:
        raise MissingApiKey(
            f"{ENV_VAR} is not set. Put it in a .env file (see .env.example) "
            "or export it in your shell."
        )
    return key


def parse_historical(payload: dict) -> pd.DataFrame:
    """API JSON -> tidy DataFrame, oldest first, one row per trading day."""
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    df = pd.DataFrame(rows)[PRICE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


class PsxClient:
    """Rate-limited, retrying HTTP client. One instance per run."""

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or load_api_key()
        self.session = session or requests.Session()
        self._last_call = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = BASE_URL + path
        headers = {"X-API-Key": self.api_key, "Accept": "application/json"}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = self.session.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS[attempt - 1])
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")

    def list_stocks(self, index: str = "KSE100") -> list[str]:
        return list(self._get("/stocks", {"index": index})["data"])

    def get_historical(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Daily OHLCV for one symbol between ISO dates (inclusive)."""
        return parse_historical(
            self._get(f"/stocks/{symbol}/historical", {"start": start, "end": end})
        )
