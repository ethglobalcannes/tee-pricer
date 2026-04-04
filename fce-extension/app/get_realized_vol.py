"""
Compute 30-day realized volatility for XRP/USD from FTSO historical data.
Samples one price per day over the past 30 days using parallel RPC calls.

Falls back to SIGMA_OVERRIDE env var if RPC sampling fails.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from web3 import Web3

COSTON2_RPC = "https://coston2-api.flare.network/ext/bc/C/rpc"

CONTRACT_REGISTRY_ADDRESS = "0xaD67FE66660Fb8dFE9d6b1b4240d8650e30F6019"
XRP_USD_FEED_ID = bytes.fromhex("015852502f55534400000000000000000000000000")

# ~1.8s per block; 48_000 blocks ≈ 24 hours
_BLOCKS_PER_DAY = 48_000
_SAMPLE_DAYS = 30          # number of daily price points
_ANNUALISATION = math.sqrt(365)  # crypto trades 24/7/365

# Cache: recompute RV30 at most once per hour. Per-quote calls are instant.
_cache: dict = {"sigma": None, "expires_at": 0.0}
_CACHE_TTL_SECONDS = 3600  # 1 hour

_CONTRACT_REGISTRY_ABI = [
    {
        "inputs": [{"name": "_name", "type": "string"}],
        "name": "getContractAddressByName",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    }
]

_FTSO_V2_ABI = [
    {
        "inputs": [{"name": "_feedId", "type": "bytes21"}],
        "name": "getFeedByIdInWei",
        "outputs": [
            {"name": "_value", "type": "uint256"},
            {"name": "_timestamp", "type": "uint64"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]


def _fetch_price_at_block(ftso, block: int) -> tuple[int, float]:
    """Fetch XRP/USD price at a specific block. Returns (block, price)."""
    value, _ = ftso.functions.getFeedByIdInWei(XRP_USD_FEED_ID).call(
        block_identifier=block
    )
    return block, value / 1e18


def _rv30_from_prices(prices: list[float]) -> float:
    """
    Compute annualised realized volatility from a list of daily prices.

    Uses close-to-close log-return standard deviation, annualised by sqrt(365).
    """
    log_returns = [
        math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))
    ]
    n = len(log_returns)
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    return math.sqrt(variance) * _ANNUALISATION


def get_realized_vol(rpc_url: str = COSTON2_RPC) -> float:
    """
    Return 30-day realized volatility (RV30) for XRP/USD sourced entirely from FTSO.

    Result is cached for 1 hour — per-quote calls after the first are instant.
    First call fires all 31 RPC requests in parallel (ThreadPoolExecutor).
    Annualises with sqrt(365) — correct for 24/7 crypto markets.

    Falls back to SIGMA_OVERRIDE env var if sampling fails (e.g. RPC down).

    Parameters
    ----------
    rpc_url : Coston2 RPC endpoint

    Returns
    -------
    sigma : float — annualised realized vol in decimal form (e.g. 0.82)

    Raises
    ------
    RuntimeError — if RPC fails and SIGMA_OVERRIDE is not set
    """
    import time

    now = time.monotonic()
    if _cache["sigma"] is not None and now < _cache["expires_at"]:
        return _cache["sigma"]

    sigma_override = os.environ.get("SIGMA_OVERRIDE")

    w3 = Web3(Web3.HTTPProvider(rpc_url))

    try:
        registry = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACT_REGISTRY_ADDRESS),
            abi=_CONTRACT_REGISTRY_ABI,
        )
        ftso_address = registry.functions.getContractAddressByName("FtsoV2").call()
        ftso = w3.eth.contract(
            address=Web3.to_checksum_address(ftso_address),
            abi=_FTSO_V2_ABI,
        )

        latest = w3.eth.block_number
        # One block per day for the past SAMPLE_DAYS days (inclusive of today)
        sample_blocks = [
            latest - day * _BLOCKS_PER_DAY for day in range(_SAMPLE_DAYS, -1, -1)
        ]  # oldest → newest, length = SAMPLE_DAYS + 1

        prices_by_block: dict[int, float] = {}
        with ThreadPoolExecutor(max_workers=len(sample_blocks)) as pool:
            futures = {
                pool.submit(_fetch_price_at_block, ftso, b): b
                for b in sample_blocks
            }
            for future in as_completed(futures):
                block, price = future.result()
                prices_by_block[block] = price

        # Restore chronological order
        prices = [prices_by_block[b] for b in sample_blocks]
        sigma = _rv30_from_prices(prices)
        _cache["sigma"] = sigma
        _cache["expires_at"] = now + _CACHE_TTL_SECONDS
        return sigma

    except Exception as exc:
        if sigma_override:
            return float(sigma_override)
        raise RuntimeError(
            f"FTSO RV30 sampling failed and SIGMA_OVERRIDE is not set: {exc}"
        ) from exc
