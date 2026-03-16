"""
Deribit API client for fetching options chain data.

Uses the public (unauthenticated) Deribit v2 REST API to fetch:
- Available instruments (options, futures)
- Order book summaries for all options on an underlying
- Index/underlying price
- Mark prices and IVs

Data is cached locally as timestamped Parquet snapshots.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class DeribitClient:
    """Async client for Deribit public API."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or get_config()["data"]
        base_url = cfg.get("testnet_url") if cfg.get("use_testnet") else cfg.get("mainnet_url")
        self.base_url = base_url or "https://www.deribit.com/api/v2"
        self.max_retries = cfg.get("max_retries", 3)
        self.retry_delay = cfg.get("retry_delay_seconds", 1.0)
        self.timeout = cfg.get("request_timeout_seconds", 30)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> DeribitClient:
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Make API request with retries."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

        url = f"{self.base_url}/public/{method}"
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.get(url, params=params or {})
                resp.raise_for_status()
                data = resp.json()
                if "result" in data:
                    return data["result"]
                if "error" in data:
                    raise RuntimeError(f"Deribit API error: {data['error']}")
                return data
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2**attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise

    async def get_index_price(self, currency: str) -> dict[str, Any]:
        """Get current index price for a currency."""
        result = await self._request("get_index_price", {"index_name": f"{currency.lower()}_usd"})
        return result

    async def get_instruments(self, currency: str, kind: str = "option") -> list[dict[str, Any]]:
        """Get all active instruments of a given kind."""
        result = await self._request(
            "get_instruments",
            {"currency": currency.upper(), "kind": kind, "expired": "false"},
        )
        return result

    async def get_book_summary_by_currency(
        self, currency: str, kind: str = "option"
    ) -> list[dict[str, Any]]:
        """Get order book summaries for all instruments of a currency."""
        result = await self._request(
            "get_book_summary_by_currency",
            {"currency": currency.upper(), "kind": kind},
        )
        return result

    async def get_order_book(self, instrument_name: str, depth: int = 5) -> dict[str, Any]:
        """Get order book for a specific instrument."""
        result = await self._request(
            "get_order_book",
            {"instrument_name": instrument_name, "depth": depth},
        )
        return result

    async def get_ticker(self, instrument_name: str) -> dict[str, Any]:
        """Get ticker for a specific instrument."""
        result = await self._request("ticker", {"instrument_name": instrument_name})
        return result


def _snapshot_path(data_dir: Path, asset: str, kind: str, timestamp: datetime) -> Path:
    """Generate path for a snapshot file."""
    ts_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return data_dir / "raw" / asset.upper() / f"{kind}_{ts_str}.parquet"


async def fetch_and_save_options(
    asset: str,
    data_dir: str | Path = "data",
) -> Path:
    """
    Fetch full options chain for an asset and save as Parquet.

    Returns the path to the saved snapshot.
    """
    data_dir = Path(data_dir)
    now = datetime.now(timezone.utc)

    async with DeribitClient() as client:
        # Fetch index price
        index_data = await client.get_index_price(asset)
        index_price = index_data.get("index_price", 0)
        logger.info(f"{asset} index price: {index_price}")

        # Fetch all option book summaries
        summaries = await client.get_book_summary_by_currency(asset, "option")
        logger.info(f"Fetched {len(summaries)} option summaries for {asset}")

        # Also fetch futures for forward extraction
        futures = await client.get_book_summary_by_currency(asset, "future")
        logger.info(f"Fetched {len(futures)} futures summaries for {asset}")

    if not summaries:
        raise ValueError(f"No option data returned for {asset}")

    # Convert to DataFrame
    df_options = pd.DataFrame(summaries)
    df_options["fetch_timestamp"] = now.isoformat()
    df_options["index_price"] = index_price

    df_futures = pd.DataFrame(futures)
    df_futures["fetch_timestamp"] = now.isoformat()
    df_futures["index_price"] = index_price

    # Save options snapshot
    opt_path = _snapshot_path(data_dir, asset, "options", now)
    opt_path.parent.mkdir(parents=True, exist_ok=True)
    df_options.to_parquet(opt_path, engine="pyarrow", index=False)
    logger.info(f"Saved options snapshot: {opt_path}")

    # Save futures snapshot
    fut_path = _snapshot_path(data_dir, asset, "futures", now)
    df_futures.to_parquet(fut_path, engine="pyarrow", index=False)
    logger.info(f"Saved futures snapshot: {fut_path}")

    return opt_path


async def fetch_and_save_all(
    assets: list[str] | None = None,
    data_dir: str | Path = "data",
) -> list[Path]:
    """Fetch options data for multiple assets."""
    assets = assets or ["BTC", "ETH"]
    paths = []
    for asset in assets:
        path = await fetch_and_save_options(asset, data_dir)
        paths.append(path)
    return paths


def get_latest_snapshot(
    asset: str,
    kind: str = "options",
    data_dir: str | Path = "data",
) -> Path | None:
    """Find the most recent snapshot file for an asset."""
    data_dir = Path(data_dir)
    snapshot_dir = data_dir / "raw" / asset.upper()
    if not snapshot_dir.exists():
        return None
    files = sorted(snapshot_dir.glob(f"{kind}_*.parquet"), reverse=True)
    return files[0] if files else None


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load a Parquet snapshot."""
    return pd.read_parquet(path)


def run_fetch(asset: str, data_dir: str = "data") -> Path:
    """Synchronous wrapper for fetch_and_save_options."""
    return asyncio.run(fetch_and_save_options(asset, data_dir))
