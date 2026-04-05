# tee-pricer — Attested Options Pricing Engine

Pricing primitives for a TEE-powered options market maker on Flare (Coston2).
These modules run **inside a Flare Compute Extension (FCE)** — Intel TDX hardware
isolation with on-chain attestation. Every quote is cryptographically bound to
the exact code that produced it.

---

## The Core Idea

RFQ-based options markets ask you to trust that the market maker priced fairly.
With a TEE, you don't have to. The enclave runs the pricing code, signs the
quote with a hardware-attested key, and publishes the attestation token on-chain
alongside every quote. Anyone can verify:

- the exact extension hash (Docker image) that ran
- the inputs used (FTSO spot, realized vol, on-chain seed)
- that the quote was signed by the registered TEE key

The quote is either honest — or the proof doesn't verify.

---

## Repository Layout

```
tee-pricer/
  mc_pricer.py         ← GBM Monte Carlo — 10k paths, vectorised NumPy
  bs_pricer.py         ← Black-Scholes closed form + delta/vega greeks
  tests/
    test_mc_pricer.py
    test_bs_pricer.py

fce-extension/app/
  get_ftso_spot.py     ← XRP/USD spot from FTSO v2 (live on-chain, ~1.8s freshness)
  get_realized_vol.py  ← 30-day realized vol from 31 FTSO archive blocks
  get_secure_random.py ← per-RFQ MC seed via sha256(epoch_random ‖ rfq_id)
```

The deployed FCE extension lives in [`ethglobalcannes/fce-sign`](https://github.com/ethglobalcannes/fce-sign).
This repo contains the pricing bricks that run inside it.

---

## Pricing Pipeline (per RFQ, executed inside the TEE)

```
RFQ received (asset, strike K, expiry T, isPut, quantity, rfq_id)
        │
        ├─ 1. get_ftso_spot()        → S0     (XRP/USD, live FTSO feed)
        ├─ 2. get_realized_vol()     → sigma  (RV30 from FTSO archive blocks)
        ├─ 3. get_mc_seed_for_rfq()  → seed   (sha256(epoch_random ‖ rfq_id) % 2³²)
        │
        ├─ 4. mc_pricer(S0, sigma, r, T, K, seed, N=10_000, is_put)
        │       → price_mc, stderr
        │
        ├─ 5. bs_pricer(S0, sigma, r, T, K, is_put)
        │       → price_bs  (convergence check: |mc − bs| < 15%)
        │
        └─ 6. greeks(S0, sigma, r, T, K, is_put)
                → delta = N(d1), vega = S0·N'(d1)·√T / 100
```

Result `(price_mc, price_bs, delta, vega, seed)` is ABI-encoded, signed with
the TEE-held EIP-712 key, and returned as the FCE action result.

---

## Module Details

### `mc_pricer.py` — GBM Monte Carlo

Models each price path as Geometric Brownian Motion:

$$S_T = S_0 \cdot \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\, Z\right], \quad Z \sim \mathcal{N}(0,1)$$

$$\text{price} = e^{-rT} \cdot \mathbb{E}\!\left[\max(S_T - K,\, 0)\right]$$

Key properties:
- **Seeded**: `np.random.default_rng(seed)` — the seed comes from on-chain
  SecureRandom, making the entire draw sequence reproducible and auditable
- **Vectorised**: all N=10,000 paths computed in a single NumPy operation, < 100ms
- **Standard error returned**: caller can verify Monte Carlo convergence per quote

### `bs_pricer.py` — Black-Scholes + Greeks

Closed-form European option price as a convergence sanity check on MC output.

$$d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S_0\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2)$$

Greeks:
- **Delta** $= \Phi(d_1)$ — rate of change of price with respect to spot
- **Vega** $= S_0\,\phi(d_1)\sqrt{T} / 100$ — sensitivity to a 1% move in vol

When `|price_mc − price_bs| < 15%` the MC result is published. A larger gap
would indicate a sampling anomaly worth flagging to the frontend fairness panel.

### `get_ftso_spot.py` — Live FTSO XRP/USD

Fetches `XRP/USD` spot at the moment of pricing from Flare's FTSO v2.

- Resolves `FtsoV2` address via `ContractRegistry` at runtime — no hardcoded feed address
- Calls `getFeedByIdInWei(feedId)` — free view call, no fee, ~1.8s update cadence
- Feed ID: `0x015852502f555344...` (bytes21 encoding of `"XRP/USD"`)

Because this call happens **inside the TEE**, the spot price used is bound to
the attestation token. The input is not taken on faith — it is part of the proof.

### `get_realized_vol.py` — 30-Day Realized Vol from FTSO

Historical volatility computed entirely from on-chain FTSO data — no off-chain
data source required for the vol input.

- Samples 31 daily FTSO prices via `getFeedByIdInWei(..., block_identifier=N)`
- Block spacing: ~48,000 blocks per day (1.8s per block on Flare C-chain)
- Parallel RPC: all 31 calls fired simultaneously via `ThreadPoolExecutor`
- Computes close-to-close log-return standard deviation, annualised by $\sqrt{365}$
- **1-hour cache** — first call per hour costs ~31 archive RPC lookups; all subsequent per-quote calls are instant

```
sigma = std(log(P[i] / P[i-1]) for i in 1..30) × sqrt(365)
```

Fallback to `SIGMA_OVERRIDE` env var if archive RPC is unavailable.

### `get_secure_random.py` — Verifiable Per-RFQ MC Seed

The Monte Carlo draw sequence must be reproducible for any external auditor to
verify a quoted price. The seed comes from two sources:

1. **`Relay.getRandomNumber()`** — Flare's on-chain SecureRandom, produced
   each FTSO voting epoch (~90s) via commit-reveal across ~100 data providers.
   Verifiably unpredictable at RFQ submission time.
2. **`rfq_id`** — unique identifier of this specific RFQ (bytes32).

Combined as:

$$\text{seed} = \text{sha256}(\text{epoch\_random} \mathbin{\|} \text{rfq\_id}) \bmod 2^{32}$$

This means:
- The taker **cannot steer** the seed (epoch random is fixed before the RFQ arrives)
- The seed is **stored on-chain** with the quote — anyone can reproduce the exact draw
- Each RFQ gets a **unique seed** even within the same block

---

## Running the Tests

Unit tests cover pricing correctness, put-call parity, and convergence.
No network access required.

```bash
cd tee-pricer
python -m unittest discover -s tests -p 'test_*.py'
# 16 passing, 2 xfail
```

Key test cases:
- ATM call/put prices converge to BS within 3σ at N=10,000
- Put-call parity: `|C_mc − P_mc − (S0 − K·e^{−rT})| < 0.01`
- Delta bounds: call delta ∈ (0, 1), put delta ∈ (−1, 0)
- Deep ITM/OTM edge cases
- Seed determinism: same seed → identical price every run

---

## Why This Matters for TEE Attestation

Each of these modules is stateless and pure (given its inputs). That is not
a coincidence — it is a design requirement for attestable computation.

| Property | Why it matters for attestation |
|---|---|
| **Deterministic given seed** | Any auditor can reproduce the exact MC draw sequence |
| **Inputs from on-chain sources** | FTSO spot + SecureRandom are verifiable on Coston2 |
| **No external HTTP calls at price time** | RV30 is cached; spot + seed are view calls — no off-chain oracle trust |
| **Output is a pure function** | Same (S0, sigma, seed, K, T) always produces the same price |

The TEE proves the computation was honest. The on-chain inputs prove the data
was honest. Together they close the loop.

---

*EthGlobal Cannes 2026 — [ethglobalcannes/fce-sign](https://github.com/ethglobalcannes/fce-sign) for the deployed FCE extension.*
