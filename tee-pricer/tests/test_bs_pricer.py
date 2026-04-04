"""
Tests for bs_pricer.py (Black-Scholes + greeks)

Run with:  pytest tee-pricer/tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from bs_pricer import bs_pricer, greeks


BASE = dict(S0=1.0, K=1.0, T=0.25, sigma=0.80, r=0.05)


# ---------------------------------------------------------------------------
# Good tests — should pass
# ---------------------------------------------------------------------------

def test_atm_call_positive():
    assert bs_pricer(**BASE, is_put=False) > 0


def test_atm_put_positive():
    assert bs_pricer(**BASE, is_put=True) > 0


def test_put_call_parity_exact():
    """BS satisfies put-call parity to machine precision."""
    call  = bs_pricer(**BASE, is_put=False)
    put   = bs_pricer(**BASE, is_put=True)
    theory = BASE["S0"] - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
    assert abs((call - put) - theory) < 1e-12, (
        f"parity error: {abs((call - put) - theory):.2e}"
    )


def test_call_delta_atm_near_half():
    """ATM call delta should be close to 0.5 (slightly above for positive drift)."""
    delta, _ = greeks(**BASE, is_put=False)
    assert 0.5 < delta < 0.7, f"ATM call delta={delta:.4f} out of expected range"


def test_put_delta_atm_near_neg_half():
    """ATM put delta should be close to -0.5."""
    delta, _ = greeks(**BASE, is_put=True)
    assert -0.7 < delta < -0.3, f"ATM put delta={delta:.4f} out of expected range"


def test_call_put_delta_sum_equals_one():
    """call_delta - put_delta == 1 (put-call delta parity)."""
    call_delta, _ = greeks(**BASE, is_put=False)
    put_delta,  _ = greeks(**BASE, is_put=True)
    assert abs((call_delta - put_delta) - 1.0) < 1e-12


def test_vega_positive():
    """Vega must always be positive for both calls and puts."""
    _, call_vega = greeks(**BASE, is_put=False)
    _, put_vega  = greeks(**BASE, is_put=True)
    assert call_vega > 0
    assert put_vega  > 0


def test_call_put_vega_equal():
    """Call and put vega are identical (same underlying sensitivity to vol)."""
    _, call_vega = greeks(**BASE, is_put=False)
    _, put_vega  = greeks(**BASE, is_put=True)
    assert abs(call_vega - put_vega) < 1e-12


def test_deep_itm_call_near_intrinsic():
    """Deep ITM call (S0 >> K) should be close to intrinsic value S0 - K*e^(-rT)."""
    call  = bs_pricer(S0=10.0, K=1.0, T=0.25, sigma=0.80, r=0.05, is_put=False)
    intrinsic = 10.0 - 1.0 * np.exp(-0.05 * 0.25)
    assert abs(call - intrinsic) < 0.01, f"deep ITM call={call:.6f}  intrinsic={intrinsic:.6f}"


# ---------------------------------------------------------------------------
# Deliberate failure — xfail: proves delta is bounded
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="ATM call delta cannot exceed 1.0 — expected failure")
def test_call_delta_exceeds_one_impossible():
    """
    Delta of a call is bounded in (0, 1).
    This test asserts delta > 1.0 — it MUST fail.
    Proves the greeks() function is correctly bounded.
    """
    delta, _ = greeks(**BASE, is_put=False)
    assert delta > 1.0, f"delta={delta:.4f} — correctly bounded below 1.0"
