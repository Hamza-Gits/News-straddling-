# Pilot Results — MNQ News Straddle (56 events, Jan 2 – Mar 13 2026)

**Data**: MNQ 03-26 tick data (90.2M ticks, bid/ask included), 56 USD
high+medium news events, verified UTC alignment, measured per-second spreads.

## Verdict: the stop-entry straddle does not survive contact with reality

5,130 parameter configs (stop distance × TP × SL × arm/cancel/hold timing ×
OCO) were each tested at three cost levels:

| Cost assumption | Profitable configs (in-sample) | Profitable configs (OOS) | Walk-forward P&L |
|---|---|---|---|
| 1.5× stress (recommended) | **0 / 5,130** | 0 / 5,130 | −$913 |
| 1.0× (measured costs, face value) | **0 / 5,130** | 0 / 5,130 | −$805 |
| **0.0× — ZERO slippage, half-spread + commission only** | **0 / 5,130** | **0 / 5,130** | −$448 |

The zero-slippage row is the kill shot: even with physically impossible
perfect stop fills, every configuration loses money. This is not an execution
problem. It is structural.

## Why it fails

Measured from the data (medians across 56 events):

| Quantity | Value |
|---|---|
| Best one-directional excursion within 60s | 28 pts |
| Net directional move at 60s | **14.4 pts** |
| Spread during first 5s of news | 8 pts (p90: 33 pts) |
| Avg entry slippage (modeled @1.5×, vol-scaled) | 11.3 pts |
| Round-trip friction @1.5× | ~23 pts |

1. **The median event doesn't move enough.** 14 pts of net drift cannot pay
   for brackets + friction that cost roughly the same.
2. **A stop entry mechanically buys the top of the initial burst** — the
   moment of maximum spread and thinnest book. Win rate at zero slippage was
   still only ~23%: price V-reverses through the SL far more often than it
   runs to the TP.
3. **Costs scale with exactly the events that move.** The big movers (top
   decile: 60+ pts) come with 30+ pt spreads and Velocity Logic halts —
   e.g. NFP 2026-03-06: 4-second trading halt, reopen 65 pts lower, 29-pt
   spread on the reopen bar.

## Caveats

- 56 events from a single quarter/regime. A clean 0-for-5,130 across three
  cost tiers is unlikely to flip with more data, but 6-year confirmation is
  cheap via `download_databento.py`.
- Only the stop-entry-straddle family was tested. The failure mode points
  directly at testable variants (below), which reuse this entire pipeline.

## Variant sweep (exhaustive): 23,040 additional configs

Four alternative entry mechanisms — post-spike **continuation**, spike
**fade**, **pretrend** (pre-news drift direction), delayed **breakout**
(stops at the post-burst range) — crossed with entry delay (5–60s),
spike-scaled brackets (TP/SL as multiples of the initial move), max-hold,
and four filters: minimum spike size, high-impact-only, forecast-surprise
z-score (from 6y history), and a measured-spread cap at entry.

| Stress | Gate survivors | Expected from luck |
|---|---|---|
| 1.0× | 130 | — |
| 1.5× | **22** | **~23** |
| 2.0× | 6 | — |

Survivor count at the recommended stress tier sits exactly at the noise
floor. The 6 configs passing all three tiers collapse to **2 correlated
families**, and neither is statistically distinguishable from zero:

| Candidate | n | OOS-ish net | t-stat | Red flag |
|---|---|---|---|---|
| breakout/30s/high-impact/spread≤8 | 18 | +$159 | 1.20 | 44% of profit from one trade |
| pretrend/60s/spike≥20/spread≤8 | 17 | +$163 | 0.72 | **72% of profit from one trade** |

### The marginal effects are the real finding

Averaged over all 23k configs (mean OOS net P&L):

- surprise z filter: none −406 → z≥1 −142 → z≥2 **−49** (monotone)
- spread cap 8 ticks: −291 → −107
- entry delay: 5s −281 → 60s −107 (monotone)
- min spike 20pts: −293 → −115
- variant: breakout −86, pretrend −231, fade −224, cont −254

Every sensible filter cuts losses by 2–8×. **None flips the sign.** The
disciplined version of news trading on MNQ loses slowly instead of quickly.

### Endpoint

28,170 total configurations tested across 3 cost tiers. On 56 events the
honest search space is exhausted — additional dimensions on this sample
would manufacture false positives, not find edges. The next informative
step is **more events** (6-year windows via `download_databento.py`), which
would either resurrect the two candidate families with real sample size or
(more likely) bury them.

## Where the data says to look next

1. **Post-spike continuation** — enter 10–30s AFTER the release in the
   direction of the initial move, once spread normalizes. Avoids paying the
   burst.
2. **Spike fade** — the low straddle win rate implies reversals are common;
   the mirror-image trade deserves a test.
3. **Surprise filter** — the news CSV has actual vs forecast; condition
   trading on surprise magnitude instead of trading every release.
4. **Volatility-scaled brackets** — event move distributions vary 10×;
   fixed-tick brackets can't fit both CPI and Chicago PMI.
