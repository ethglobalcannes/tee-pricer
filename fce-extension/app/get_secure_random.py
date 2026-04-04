"""
Fetch on-chain SecureRandom seed from Flare Relay contract (Coston2).

The Relay contract produces a fresh random number each FTSO voting epoch (~90s)
via commit-reveal across all ~100 data providers. The value is verifiably
unpredictable at RFQ submission time and is stored on-chain alongside each
quote, letting anyone reproduce and audit the exact MC draw sequence.

Usage:
    from get_secure_random import get_mc_seed_for_rfq
    seed = get_mc_seed_for_rfq(rfq_id)   # rfq_id: bytes32 from decode_rfq()
    price, stderr = mc_pricer(S0, sigma, r, T, K, seed)
"""

import hashlib

from web3 import Web3

COSTON2_RPC = "https://coston2-api.flare.network/ext/bc/C/rpc"

# ContractRegistry is stable per network — address never changes
CONTRACT_REGISTRY_ADDRESS = "0xaD67FE66660Fb8dFE9d6b1b4240d8650e30F6019"

_CONTRACT_REGISTRY_ABI = [
    {
        "inputs": [{"name": "_name", "type": "string"}],
        "name": "getContractAddressByName",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    }
]

# RandomNumberV2Interface — implemented by the Relay contract
_RELAY_RANDOM_ABI = [
    {
        "inputs": [],
        "name": "getRandomNumber",
        "outputs": [
            {"name": "_randomNumber",    "type": "uint256"},
            {"name": "_isSecureRandom",  "type": "bool"},
            {"name": "_randomTimestamp", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]


def get_secure_random(rpc_url: str = COSTON2_RPC) -> tuple[int, bool, int]:
    """
    Fetch the current SecureRandom value from Flare's Relay contract.

    The random number is produced each FTSO voting epoch (~90s) via
    commit-reveal across all data providers. It is verifiably unpredictable
    at RFQ time and stored on-chain alongside the quote for audit.

    Resolves the Relay address at runtime via ContractRegistry (never hardcoded).

    Parameters
    ----------
    rpc_url : Coston2 (or Flare mainnet) RPC endpoint

    Returns
    -------
    random_number    : uint256 from Relay.getRandomNumber()
    is_secure        : True if produced via full commit-reveal (not fallback)
    random_timestamp : block timestamp when this epoch's randomness was finalised
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    registry = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_REGISTRY_ADDRESS),
        abi=_CONTRACT_REGISTRY_ABI,
    )
    relay_address = registry.functions.getContractAddressByName("Relay").call()

    relay = w3.eth.contract(
        address=Web3.to_checksum_address(relay_address),
        abi=_RELAY_RANDOM_ABI,
    )
    random_number, is_secure, random_timestamp = relay.functions.getRandomNumber().call()
    return random_number, is_secure, random_timestamp


def get_mc_seed_for_rfq(rfq_id: bytes, rpc_url: str = COSTON2_RPC) -> int:
    """
    Return a per-RFQ uint32 MC seed: sha256(epoch_random || rfq_id) % 2**32.

    rfq_id must be the contract-assigned bytes32 keccak256:
        keccak256(abi.encode(address(this), chainid, nonce, asset, strike, expiry, isPut, quantity))

    Mixing the epoch random with the contract-assigned rfq_id gives each RFQ
    a unique, unpredictable draw sequence while keeping full on-chain auditability:
    anyone can fetch Relay.getRandomNumber() at the quote timestamp and recompute
    sha256(epoch_random || rfq_id) to reproduce the exact MC paths.

    SHA-256 prevents a taker from engineering rfq_id = target XOR epoch_random
    to steer the seed (preimage resistance). Safe because rfq_id includes a
    contract-incremented nonce — cannot be predicted or replayed by the taker.

    Raises RuntimeError if the current epoch's random is not secure
    (commit-reveal incomplete — rare, auto-recovers next epoch in ~90s).

    Parameters
    ----------
    rfq_id : bytes32 from decode_rfq() — the contract-assigned RFQ identifier

    Returns
    -------
    seed : int — safe for numpy.random.default_rng(seed)
    """
    random_number, is_secure, _ = get_secure_random(rpc_url)
    if not is_secure:
        raise RuntimeError(
            "SecureRandom: latest random number is not secure "
            "(commit-reveal incomplete — retry after next FTSO epoch, ~90s)"
        )
    combined = random_number.to_bytes(32, "big") + rfq_id  # 64 bytes total
    digest = int.from_bytes(hashlib.sha256(combined).digest(), "big")
    return digest % (2**32)


# ---------------------------------------------------------------------------
# Quick sanity check — run directly: python get_secure_random.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rand, is_secure, ts = get_secure_random()
    print(f"random_number   : {rand}")
    print(f"is_secure       : {is_secure}")
    print(f"random_timestamp: {ts}")

    # Simulate a contract-assigned rfq_id (bytes32)
    dummy_rfq_id = bytes.fromhex("abcd" * 16)
    seed = get_mc_seed_for_rfq(dummy_rfq_id)
    print(f"mc_seed (rfq)   : {seed}")
