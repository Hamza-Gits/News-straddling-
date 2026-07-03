"""Backtest driver: run the straddle simulator across all news events."""

import pandas as pd

from .costs import CostModel
from .straddle import StraddleParams, simulate_event, TradeResult
from .tick_loader import event_window


def run_backtest(bars: pd.DataFrame, news: pd.DataFrame,
                 params: StraddleParams, cm: CostModel,
                 pad_secs: float = 60.0) -> list[TradeResult]:
    """bars: 1s UTC-indexed OHLC. news: from news_loader.load_news."""
    results = []
    before = params.arm_secs + pad_secs
    after = max(params.cancel_secs, params.max_hold_secs) + pad_secs
    for row in news.itertuples(index=False):
        win = event_window(bars, row.ts_utc, before, after)
        if win is None:
            continue  # no data coverage for this event
        ts, o, h, l, c = win
        res = simulate_event(ts, o, h, l, c, row.ts_utc.timestamp(),
                             row.event, params, cm)
        results.append(res)
    return results


def precompute_windows(bars: pd.DataFrame, news: pd.DataFrame,
                       before_secs: float, after_secs: float) -> list[dict]:
    """Slice every event's 1s window ONCE (pandas .loc is the hot cost when a
    grid search re-visits the same events thousands of times).

    Window must cover the LARGEST arm/cancel/hold in the grid; extra bars at
    the edges are harmless because simulate_event is time-driven."""
    out = []
    for row in news.itertuples(index=False):
        win = event_window(bars, row.ts_utc, before_secs, after_secs)
        if win is None:
            continue
        ts, o, h, l, c = win
        out.append({'ts': ts, 'o': o, 'h': h, 'l': l, 'c': c,
                    'event_ts': row.ts_utc.timestamp(), 'name': row.event})
    return out


def run_cached(windows: list[dict], params: StraddleParams,
               cm: CostModel) -> list[TradeResult]:
    """Run one config across precomputed event windows."""
    return [simulate_event(w['ts'], w['o'], w['h'], w['l'], w['c'],
                           w['event_ts'], w['name'], params, cm)
            for w in windows]
