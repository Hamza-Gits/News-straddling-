"""Directional news-trade variants - everything that is NOT the stop straddle.

Variants (direction source at entry time T+delay):
  cont      trade WITH the initial spike (sign of move release -> T+delay)
  fade      trade AGAINST the initial spike
  pretrend  trade in the direction of the pre-news drift (last ~3 min)
  breakout  stop orders at the high/low of the [release, T+delay] range
            (the straddle's smarter cousin: levels set AFTER the burst)

Brackets are SPIKE-SCALED: TP = tp_k x |spike|, SL = sl_k x |spike|
(floored at MIN_BRACKET_PTS) - they self-adapt to how much the event moved,
answering the fixed-vs-vol-scaled bracket question inside the same grid.

Filters (all evaluated with information available AT entry time):
  min_spike_pts   skip if |initial move| below this
  require_high    high-impact events only
  min_surprise_z  |actual-forecast| z-score vs that event's 6y history
  spread_cap      skip if measured spread at entry second exceeds (ticks)

Execution realism: timed entries are MARKET orders - they pay half the
MEASURED spread at that second + vol-scaled slippage. Breakout entries are
stops - full stop slippage + gap-aware fills. Exits keep the pessimistic
rules (SL before TP intrabar, TP needs trade-through, stops gap from open).
"""

from dataclasses import dataclass
import numpy as np

from .costs import CostModel, MNQ_TICK
from .straddle import TradeResult, _local_range_ticks, _sp_at, _finish

MIN_BRACKET_PTS = 4.0


@dataclass
class VariantParams:
    variant: str = 'cont'            # cont | fade | pretrend | breakout
    delay_secs: float = 10.0         # decision/entry time after release
    tp_k: float = 1.0                # TP = tp_k x |spike|
    sl_k: float = 0.5                # SL = sl_k x |spike|
    max_hold_secs: float = 300.0
    cancel_secs: float = 60.0        # breakout only: stop-order lifetime
    min_spike_pts: float = 0.0
    require_high: bool = False
    min_surprise_z: float = 0.0
    spread_cap_ticks: float | None = None
    contracts: int = 1

    def key(self) -> tuple:
        return (self.variant, self.delay_secs, self.tp_k, self.sl_k,
                self.max_hold_secs, self.min_spike_pts, self.require_high,
                self.min_surprise_z, self.spread_cap_ticks)


def simulate_variant(ts, o, h, l, c, sp, event_ts, event_name,
                     feat: dict, p: VariantParams, cm: CostModel) -> TradeResult:
    """feat: {'impact_high': bool, 'surprise_z': float (nan ok)}"""
    no_trade = TradeResult(event_ts, event_name, 0, None, None, 'no_fill',
                           0.0, 0.0, False, 0.0, 0.0)

    # ---- event-level filters ----
    if p.require_high and not feat.get('impact_high', False):
        return no_trade
    if p.min_surprise_z > 0:
        z = feat.get('surprise_z', float('nan'))
        if not (z == z) or abs(z) < p.min_surprise_z:
            return no_trade

    i_rel = int(np.searchsorted(ts, event_ts))
    if i_rel <= 0 or i_rel >= len(ts):
        return no_trade
    p0 = c[i_rel - 1]                       # last pre-release price

    entry_time = event_ts + p.delay_secs
    idx = int(np.searchsorted(ts, entry_time, side='right')) - 1
    if idx <= i_rel - 1 or idx >= len(ts) - 1:
        return no_trade

    spike = c[idx] - p0
    if abs(spike) < p.min_spike_pts:
        return no_trade
    if p.spread_cap_ticks is not None and sp is not None:
        if sp[idx] == sp[idx] and sp[idx] > p.spread_cap_ticks:
            return no_trade

    bracket = max(abs(spike), MIN_BRACKET_PTS)
    tp_pts = p.tp_k * bracket
    sl_pts = p.sl_k * bracket
    secs_since = ts[idx] - event_ts
    lr = _local_range_ticks(h, l, idx)

    if p.variant == 'breakout':
        return _breakout(ts, o, h, l, c, sp, event_ts, event_name,
                         i_rel, idx, tp_pts, sl_pts, p, cm, no_trade)

    # ---- direction ----
    if p.variant == 'cont':
        side = 1 if spike > 0 else -1
    elif p.variant == 'fade':
        side = -1 if spike > 0 else 1
    elif p.variant == 'pretrend':
        drift = p0 - c[0]
        if drift == 0:
            return no_trade
        side = 1 if drift > 0 else -1
    else:
        raise ValueError(p.variant)

    # ---- market entry at close of decision bar ----
    slip = cm.stop_fill_slippage_ticks(lr, secs_since, _sp_at(sp, idx))
    entry_px = c[idx] + side * slip * MNQ_TICK
    tp_px = entry_px + side * tp_pts
    sl_px = entry_px - side * sl_pts
    hard_exit_ts = event_ts + p.max_hold_secs

    return _manage(ts, o, h, l, c, sp, event_ts, event_name, idx + 1, side,
                   entry_px, tp_px, sl_px, hard_exit_ts, slip, p, cm)


def _manage(ts, o, h, l, c, sp, event_ts, event_name, start, side,
            entry_px, tp_px, sl_px, hard_exit_ts, entry_slip,
            p: VariantParams, cm: CostModel) -> TradeResult:
    through = cm.tp_requires_through_ticks * MNQ_TICK
    n = len(ts)
    for i in range(start, n):
        secs_since = ts[i] - event_ts
        lr = _local_range_ticks(h, l, i)
        hit_sl = (l[i] <= sl_px) if side == 1 else (h[i] >= sl_px)
        hit_tp = (h[i] >= tp_px + through) if side == 1 else (l[i] <= tp_px - through)
        if hit_sl:                       # pessimistic: SL before TP intrabar
            slip = cm.stop_fill_slippage_ticks(lr, secs_since, _sp_at(sp, i))
            ref = min(sl_px, o[i]) if side == 1 else max(sl_px, o[i])
            exit_px = ref - side * slip * MNQ_TICK
            return _finish(event_ts, event_name, side, entry_px, exit_px,
                           'sl', entry_slip, slip, False, p, cm)
        if hit_tp:
            return _finish(event_ts, event_name, side, entry_px, tp_px,
                           'tp', entry_slip, 0.0, False, p, cm)
        if ts[i] >= hard_exit_ts:
            slip = cm.stop_fill_slippage_ticks(lr, secs_since, _sp_at(sp, i))
            exit_px = c[i] - side * slip * MNQ_TICK
            return _finish(event_ts, event_name, side, entry_px, exit_px,
                           'time', entry_slip, slip, False, p, cm)
    slip = cm.stop_fill_slippage_ticks(_local_range_ticks(h, l, n - 1),
                                       ts[-1] - event_ts, _sp_at(sp, n - 1))
    exit_px = c[-1] - side * slip * MNQ_TICK
    return _finish(event_ts, event_name, side, entry_px, exit_px, 'time',
                   entry_slip, slip, False, p, cm)


def _breakout(ts, o, h, l, c, sp, event_ts, event_name, i_rel, idx,
              tp_pts, sl_pts, p: VariantParams, cm: CostModel, no_trade):
    """Stop orders one tick beyond the post-release range [i_rel..idx]."""
    hi = float(np.max(h[i_rel:idx + 1])) + MNQ_TICK
    lo = float(np.min(l[i_rel:idx + 1])) - MNQ_TICK
    cancel_ts = event_ts + p.delay_secs + p.cancel_secs
    hard_exit_ts = event_ts + p.max_hold_secs
    for i in range(idx + 1, len(ts)):
        if ts[i] > cancel_ts:
            return no_trade
        secs_since = ts[i] - event_ts
        hit_buy, hit_sell = h[i] >= hi, l[i] <= lo
        if not (hit_buy or hit_sell):
            continue
        take_buy = (c[i] >= o[i]) if (hit_buy and hit_sell) else hit_buy
        lr = _local_range_ticks(h, l, i)
        slip = cm.stop_fill_slippage_ticks(lr, secs_since, _sp_at(sp, i))
        if take_buy:
            side, base = 1, max(hi, o[i])
        else:
            side, base = -1, min(lo, o[i])
        entry_px = base + side * slip * MNQ_TICK
        tp_px = entry_px + side * tp_pts
        sl_px = entry_px - side * sl_pts
        # entry bar: close-only exit check (bar extremes predate the trigger)
        if (c[i] <= sl_px if side == 1 else c[i] >= sl_px):
            slip2 = cm.stop_fill_slippage_ticks(lr, secs_since, _sp_at(sp, i))
            exit_px = sl_px - side * slip2 * MNQ_TICK
            return _finish(event_ts, event_name, side, entry_px, exit_px,
                           'sl', slip, slip2, False, p, cm)
        return _manage(ts, o, h, l, c, sp, event_ts, event_name, i + 1, side,
                       entry_px, tp_px, sl_px, hard_exit_ts, slip, p, cm)
    return no_trade
