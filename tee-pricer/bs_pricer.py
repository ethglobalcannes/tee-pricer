"""
Black-Scholes closed-form pricer — European call/put.
Used as a convergence check against mc_pricer.py output.
"""

import numpy as np
from scipy.stats import norm


def bs_pricer(
    S0: float,
    sigma: float,
    r: float,
    T: float,
    K: float,
    is_put: bool = False,
) -> float:
    """
    Black-Scholes price for a European option.

    Parameters
    ----------
    S0    : spot price
    sigma : annualised volatility (decimal)  — RV30 from FTSO
    r     : risk-free rate (decimal)
    T     : time to expiry in years
    K     : strike price
    is_put: True for put, False for call

    Returns
    -------
    price : BS fair value
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_put:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def greeks(
    S0: float,
    sigma: float,
    r: float,
    T: float,
    K: float,
    is_put: bool = False,
) -> tuple[float, float]:
    """
    First-order Greeks: delta and vega.

    Parameters
    ----------
    Same as bs_pricer.

    Returns
    -------
    delta : dV/dS  (call: N(d1), put: N(d1) - 1)
    vega  : dV/d(sigma) per 1% move in vol
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    delta = norm.cdf(d1) if not is_put else norm.cdf(d1) - 1.0
    vega  = S0 * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol move
    return delta, vega


# ---------------------------------------------------------------------------
# Quick sanity check — run directly: python bs_pricer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from mc_pricer import mc_pricer

    S0, K, T, sigma, r = 1.0, 1.0, 0.25, 0.80, 0.05
    seed = 42
    N = 100_000  # larger N for tighter convergence check

    call_bs = bs_pricer(S0, sigma, r, T, K, is_put=False)
    put_bs  = bs_pricer(S0, sigma, r, T, K, is_put=True)

    call_mc, call_se = mc_pricer(S0, sigma, r, T, K, seed, N=N, is_put=False)
    put_mc,  put_se  = mc_pricer(S0, sigma, r, T, K, seed, N=N, is_put=True)

    call_delta, call_vega = greeks(S0, sigma, r, T, K, is_put=False)
    put_delta,  put_vega  = greeks(S0, sigma, r, T, K, is_put=True)

    print("=== Call ===")
    print(f"  BS:    {call_bs:.6f}")
    print(f"  MC:    {call_mc:.6f}  ±{call_se:.6f}")
    print(f"  diff:  {abs(call_bs - call_mc):.6f}  ({'OK' if abs(call_bs - call_mc) < 3 * call_se else 'WARN'})")
    print(f"  delta: {call_delta:.4f}  vega: {call_vega:.6f}")

    print("=== Put ===")
    print(f"  BS:    {put_bs:.6f}")
    print(f"  MC:    {put_mc:.6f}  ±{put_se:.6f}")
    print(f"  diff:  {abs(put_bs - put_mc):.6f}  ({'OK' if abs(put_bs - put_mc) < 3 * put_se else 'WARN'})")
    print(f"  delta: {put_delta:.4f}  vega: {put_vega:.6f}")

    # Put-call parity
    parity_theory = S0 - K * np.exp(-r * T)
    parity_bs     = call_bs - put_bs
    parity_mc     = call_mc - put_mc
    print(f"\n=== Put-call parity ===")
    print(f"  theory: {parity_theory:.6f}")
    print(f"  BS:     {parity_bs:.6f}  diff={abs(parity_theory - parity_bs):.8f}")
    print(f"  MC:     {parity_mc:.6f}  diff={abs(parity_theory - parity_mc):.6f}")
