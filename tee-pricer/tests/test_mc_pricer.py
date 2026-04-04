"""
Tests for mc_pricer.py

Run with:  pytest tee-pricer/tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from mc_pricer import mc_pricer
from bs_pricer import bs_pricer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = dict(S0=1.0, K=1.0, T=0.25, sigma=0.80, r=0.05, seed=42)

def convergence_ok(mc_price, bs_price, stderr, z=3.0) -> bool:
    """True when |MC - BS| < z * stderr (within z standard errors)."""
    return abs(mc_price - bs_price) < z * stderr


# ---------------------------------------------------------------------------
# Good tests — should pass
# ---------------------------------------------------------------------------

def test_call_converges_to_bs():
    """N=100k paths — MC call must converge to BS within 3 sigma."""
    mc, se = mc_pricer(**BASE, N=100_000, is_put=False)
    bs     = bs_pricer(**{k: BASE[k] for k in ("S0", "sigma", "r", "T", "K")}, is_put=False)
    assert convergence_ok(mc, bs, se), f"call MC={mc:.6f} BS={bs:.6f} se={se:.6f}"


def test_put_converges_to_bs():
    """N=100k paths — MC put must converge to BS within 3 sigma."""
    mc, se = mc_pricer(**BASE, N=100_000, is_put=True)
    bs     = bs_pricer(**{k: BASE[k] for k in ("S0", "sigma", "r", "T", "K")}, is_put=True)
    assert convergence_ok(mc, bs, se), f"put MC={mc:.6f} BS={bs:.6f} se={se:.6f}"


def test_put_call_parity():
    """C - P must equal S0 - K*e^(-rT) within 1% of spot."""
    call, _ = mc_pricer(**BASE, N=100_000, is_put=False)
    put,  _ = mc_pricer(**BASE, N=100_000, is_put=True)
    theory  = BASE["S0"] - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
    assert abs((call - put) - theory) < 0.01, (
        f"parity fail: C-P={call - put:.6f}  theory={theory:.6f}"
    )


def test_seed_reproducibility():
    """Same seed must produce identical price every time."""
    price_a, _ = mc_pricer(**BASE, N=10_000, is_put=False)
    price_b, _ = mc_pricer(**BASE, N=10_000, is_put=False)
    assert price_a == price_b, "same seed produced different prices — RNG not reproducible"


def test_different_seeds_differ():
    """Different seeds should produce different prices."""
    price_a, _ = mc_pricer(**{**BASE, "seed": 1}, N=10_000, is_put=False)
    price_b, _ = mc_pricer(**{**BASE, "seed": 2}, N=10_000, is_put=False)
    assert price_a != price_b, "different seeds produced identical prices — suspicious"


def test_call_price_positive():
    price, _ = mc_pricer(**BASE, N=10_000, is_put=False)
    assert price > 0


def test_put_price_positive():
    price, _ = mc_pricer(**BASE, N=10_000, is_put=True)
    assert price > 0


# ---------------------------------------------------------------------------
# Deliberate failure — xfail: proves the convergence check catches bad inputs
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="N=5 paths cannot price accurately — expected failure")
def test_tiny_n_fails_convergence():
    """
    With only N=5 paths the MC estimate is unreliable.
    This test asserts a tight absolute tolerance (0.001) that N=5 cannot meet.

    Why not use the 3-sigma check here?
    When N is tiny, stderr is huge, so |MC - BS| < 3*stderr trivially holds —
    the relative check has no discriminating power at tiny N.
    The absolute check reveals what the relative check masks: N=5 is simply
    too few paths to price accurately.

    This test MUST fail. If it ever passes, either the RNG got astronomically
    lucky or the check is broken.
    """
    mc, _  = mc_pricer(**BASE, N=5, is_put=False)
    bs     = bs_pricer(**{k: BASE[k] for k in ("S0", "sigma", "r", "T", "K")}, is_put=False)
    assert abs(mc - bs) < 0.001, (
        f"N=5 MC={mc:.6f} BS={bs:.6f} abs_err={abs(mc - bs):.6f} — correctly flagged as inaccurate"
    )
