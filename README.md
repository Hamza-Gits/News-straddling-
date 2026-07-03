# MNQ News-Straddle Backtest

Backtest + parameter optimization for a news-straddling strategy on **MNQ**
(Micro E-mini Nasdaq-100): place a buy stop and sell stop bracketing price
just before a scheduled news release, ride the spike with a TP/SL bracket.

## Why this backtest is different

News straddling lives or dies on **execution costs at the worst possible
moment**. A naive backtest assumes perfect fills and looks amazing; live, the
strategy pays:

| Cost | How it's modeled here |
|---|---|
| Spread | 1 tick base, **blows out 4x for ~10s** after the release, decaying back |
| Entry slippage | Stop→market fill: base + volatility-scaled (local 1s range) × stress multiplier, always **against** you |
| SL slippage | Same model, always against you |
| TP fill | Limit order: **never** slips in your favor; must trade *through* the level by ≥1 tick (a touch is not a fill) |
| Intrabar ambiguity | If TP and SL both possible within a 1s bar → **SL first** |
| Whipsaw | Non-OCO mode models the double-trigger loss explicitly |
| Commission | $1.98 round turn per contract |

The `stress_multiplier` knob (default 1.5x) exists to run the whole search at
1.0x / 1.5x / 2.0x costs. A strategy that's only profitable at 1.0x is not a
strategy.

## Anti-overfitting design

The optimizer does **not** pick the config with the best backtest PnL. That
number is curve-fit by construction. Instead:

1. **Walk-forward folds** — chronological expanding-window splits; every
   config is scored on data it was never tuned on.
2. **Plateau scoring** — configs are rewarded when their grid *neighbors* are
   also profitable out-of-sample. Isolated parameter peaks are noise.
3. **Minimum sample gate** — configs with <30 OOS trades are ineligible.
4. Champion = rank-combination of OOS expectancy, OOS profit factor, plateau
   stability, and OOS max drawdown.

The full leaderboard (in-sample *and* out-of-sample columns) is saved for
transparency.

## Data

- **News**: `ff_scraper/forexfactory_news.csv` — 6 years of ForexFactory
  high+medium impact events (USD/JPY/CNY), scraped with `ff_scraper/ff_scraper.py`.
  `datetime_utc` comes from the event's unix timestamp → timezone-unambiguous.
  Backtest uses **USD only** (JPY/CNY don't move MNQ); scheduled speeches are
  excluded (no exact-second data drop to straddle).
- **Price**: MNQ tick or 1-second data → `data/ticks/` (CSV/CSV.GZ/Parquet).
  Loader auto-detects raw ticks, 1s bars, or bid/ask quotes and aggregates to
  a strict 1-second grid. **Not committed** (size + licensing).

## Usage

```bash
pip install -r requirements.txt

# baseline single-config backtest (params from config/config.yaml)
python run_backtest.py

# full grid search + walk-forward optimization
python run_optimize.py                 # default 1.5x cost stress
python run_optimize.py --stress 1.0    # optimistic costs
python run_optimize.py --stress 2.0    # pessimistic costs
```

Outputs land in `results/`: `baseline_trades.csv`, `leaderboard.csv`,
`champion.json`.

## Repo layout

```
config/config.yaml     all knobs: data paths, cost model, strategy, grid
src/costs.py           spread/slippage/commission model
src/straddle.py        one-event simulator (fills, OCO, whipsaw, pessimistic rules)
src/news_loader.py     ForexFactory CSV → filtered UTC event list
src/tick_loader.py     tick/1s/bid-ask → strict 1s OHLC bars
src/backtest.py        run all events
src/metrics.py         PF, expectancy, drawdown, whipsaw rate, ...
src/optimize.py        grid search + walk-forward + plateau → champion
ff_scraper/            ForexFactory calendar scraper + scraped news CSV
```

## Interpreting results

Judge only the `oos_*` columns. Expect the OOS numbers to be well below the
in-sample ones — that gap is the overfitting you avoided. If no champion
passes the gate, that is a **result**, not a failure: the edge doesn't survive
realistic execution.
