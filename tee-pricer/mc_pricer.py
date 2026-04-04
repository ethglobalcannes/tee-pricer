"""
Monte Carlo options pricer — GBM, European call/put.
Runs inside the FCE Python extension (TEE enclave).
"""

import numpy as np


def mc_pricer(
    S0: float,
    sigma: float,
    r: float,
    T: float,
    K: float,
    seed: int,
    N: int = 10_000,
    is_put: bool = False,
) -> tuple[float, float]:
    """
    Price a European option via GBM Monte Carlo.

    Parameters
    ----------
    S0    : spot price (e.g. FTSO XRP/USD)
    sigma : annualised volatility (decimal, e.g. 0.80 = 80%)  — RV30 from FTSO
    r     : risk-free rate (decimal, e.g. 0.05)
    T     : time to expiry in years
    K     : strike price
    seed  : on-chain SecureRandom seed (int) — makes draw sequence reproducible
    N     : number of simulation paths (default 10 000)
    is_put: True for put, False for call

    Returns
    -------
    price  : discounted expected payoff
    stderr : Monte Carlo standard error
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if is_put:
        payoffs = np.maximum(K - ST, 0.0)
    else:
        payoffs = np.maximum(ST - K, 0.0)

    price = np.exp(-r * T) * payoffs.mean()
    stderr = payoffs.std() / np.sqrt(N)
    return price, stderr


# ---------------------------------------------------------------------------
# Quick sanity check — run directly: python mc_pricer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    S0, K, T, sigma, r = 1.0, 1.0, 0.25, 0.80, 0.05
    seed = 42

    call_mc, call_se = mc_pricer(S0, sigma, r, T, K, seed, is_put=False)
    put_mc, put_se   = mc_pricer(S0, sigma, r, T, K, seed, is_put=True)

    print(f"ATM call  MC: {call_mc:.6f}  ±{call_se:.6f}")
    print(f"ATM put   MC: {put_mc:.6f}  ±{put_se:.6f}")

    # Put-call parity check: C - P ≈ S0 - K*e^(-rT)
    parity = S0 - K * np.exp(-r * T)
    parity_mc = call_mc - put_mc
    print(f"Put-call parity  theory={parity:.6f}  MC={parity_mc:.6f}  diff={abs(parity - parity_mc):.6f}")
