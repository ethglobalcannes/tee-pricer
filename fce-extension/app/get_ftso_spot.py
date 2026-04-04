"""
Fetch XRP/USD spot price from Flare FTSO v2 (Coston2).
Returns the spot as a float (USD per XRP).
"""

from web3 import Web3

COSTON2_RPC = "https://coston2-api.flare.network/ext/bc/C/rpc"

# ContractRegistry is stable per network — address never changes
CONTRACT_REGISTRY_ADDRESS = "0xaD67FE66660Fb8dFE9d6b1b4240d8650e30F6019"

# XRP/USD feed ID (bytes21): 0x01 + "XRP/USD" + zero-padding
XRP_USD_FEED_ID = bytes.fromhex("015852502f55534400000000000000000000000000")

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


def get_ftso_spot(rpc_url: str = COSTON2_RPC) -> float:
    """
    Fetch XRP/USD spot price from FTSO v2 on Coston2.

    Resolves FtsoV2 address at runtime via ContractRegistry (never hardcoded).
    Calls getFeedByIdInWei — free view call, no fee required, ~1.8s freshness.

    Parameters
    ----------
    rpc_url : Coston2 (or Flare mainnet) RPC endpoint

    Returns
    -------
    spot : float — USD per XRP (18-decimal wei value divided by 1e18)
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    registry = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_REGISTRY_ADDRESS),
        abi=_CONTRACT_REGISTRY_ABI,
    )
    ftso_address = registry.functions.getContractAddressByName("FtsoV2").call()

    ftso = w3.eth.contract(
        address=Web3.to_checksum_address(ftso_address),
        abi=_FTSO_V2_ABI,
    )
    value, _timestamp = ftso.functions.getFeedByIdInWei(XRP_USD_FEED_ID).call()
    return value / 1e18
