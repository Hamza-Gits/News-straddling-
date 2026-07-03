"""News-straddle event simulator.

Simulates ONE news event on a 1-second (or tick-aggregated-to-1s) price series:

  1. At T-arm_secs before release: record reference price, place
     buy stop  @ ref + stop_dist_ticks
     sell stop @ ref - stop_dist_ticks
  2. On fill: apply adverse slippage (stop -> market in fast tape),
     attach TP (limit) and SL (stop) from the ACTUAL fill price.
  3. OCO handling: optionally cancel the opposite entry stop on first fill.
     If OCO is off and the second stop triggers, the position is closed at
     the second stop (net flat via opposing fill) - the classic whipsaw loss.
  4. TP is a limit: fills only if price trades THROUGH it (conservative).
     SL is a stop: fills WITH adverse slippage.
  5. Unfilled entries cancelled at T+cancel_secs. Open positions force-closed
     at T+max_hold_secs at market (with slippage).

Intrabar (intra-second) ambiguity rule - pessimistic: if both TP and SL could
have been hit within the same 1s bar, assume SL hit first.
"""

from dataclasses import dataclass
import numpy as np

from .costs import CostModel, MNQ_TICK


@dataclass
class StraddleParams:
    stop_dist_ticks: float = 40.0     # 10 pts from ref price to each entry stop
    tp_ticks: float = 80.0            # 20 pts target
    sl_ticks: float = 40.0            # 10 pts stop loss
    arm_secs: float = 30.0            # place brackets this many secs before release
    cancel_secs: float = 60.0         # cancel unfilled entries this long after release
    max_hold_secs: float = 300.0      # time-based exit for open positions
    oco: bool = True                  # cancel opposite entry when one side fills
    contracts: int = 1

    def key(self) -> tuple:
        return (self.stop_dist_ticks, self.tp_ticks, self.sl_ticks,
                self.arm_secs, self.cancel_secs, self.max_hold_secs, self.oco)


@dataclass
class TradeResult:
    event_ts: float               # epoch seconds of the news release
    event_name: str
    side: int                     # +1 long, -1 short, 0 = no fill
    entry_price: float | None
    exit_price: float | None
    exit_reason: str              # 'tp' | 'sl' | 'whipsaw' | 'time' | 'no_fill'
    pnl_points: float             # AFTER slippage/spread, BEFORE commission
    pnl_dollars: float            # net of everything
    whipsaw: bool                 # did both entry stops trigger?
    entry_slip_ticks: float
    exit_slip_ticks: float


def _local_range_ticks(high: np.ndarray, low: np.ndarray, i: int, window: int = 3) -> float:
    """Range of the surrounding few 1s bars, in ticks - proxy for tape speed."""
    lo = max(0, i - window)
    hi = min(len(high), i + 1)
    return (np.max(high[lo:hi]) - np.min(low[lo:hi])) / MNQ_TICK


def simulate_event(ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                   c: np.ndarray, event_ts: float, event_name: str,
                   p: StraddleParams, cm: CostModel) -> TradeResult:
    """Run one event. Arrays are the 1s series covering [event - arm, event + hold]."""
    no_trade = TradeResult(event_ts, event_name, 0, None, None, 'no_fill',
                           0.0, 0.0, False, 0.0, 0.0)

    arm_ts = event_ts - p.arm_secs
    i0 = int(np.searchsorted(ts, arm_ts))
    if i0 >= len(ts):
        return no_trade
    ref = c[i0]

    buy_stop = ref + p.stop_dist_ticks * MNQ_TICK
    sell_stop = ref - p.stop_dist_ticks * MNQ_TICK
    cancel_ts = event_ts + p.cancel_secs
    hard_exit_ts = event_ts + p.max_hold_secs

    side = 0
    entry_px = tp_px = sl_px = None
    entry_slip = 0.0
    buy_live, sell_live = True, True
    whipsaw = False
    entry_bar = -1

    n = len(ts)
    i = i0
    while i < n:
        t = ts[i]
        secs_since = t - event_ts if t >= event_ts else None

        if side == 0:
            if t > cancel_ts:
                return no_trade
            hit_buy = buy_live and h[i] >= buy_stop
            hit_sell = sell_live and l[i] <= sell_stop
            if hit_buy or hit_sell:
                # both in same 1s bar: pessimistic - take the one whose SL
                # region the bar then swept; approximate by entering the side
                # of the bar's open direction, then whipsaw logic handles it
                if hit_buy and hit_sell:
                    take_buy = c[i] >= o[i]
                else:
                    take_buy = hit_buy
                lr = _local_range_ticks(h, l, i)
                slip = cm.stop_fill_slippage_ticks(lr, secs_since)
                if take_buy:
                    side = 1
                    entry_px = buy_stop + slip * MNQ_TICK
                    buy_live = False
                    if p.oco:
                        sell_live = False
                else:
                    side = -1
                    entry_px = sell_stop - slip * MNQ_TICK
                    sell_live = False
                    if p.oco:
                        buy_live = False
                entry_slip = slip
                entry_bar = i
                tp_px = entry_px + side * p.tp_ticks * MNQ_TICK
                sl_px = entry_px - side * p.sl_ticks * MNQ_TICK
        if side != 0:
            lr = _local_range_ticks(h, l, i)
            through = cm.tp_requires_through_ticks * MNQ_TICK
            if i == entry_bar:
                # Entry bar: the bar's high/low may predate the trigger moment,
                # so only the CLOSE (post-trigger by construction) counts.
                # No TP on the entry bar (pessimistic).
                hit_sl = (c[i] <= sl_px) if side == 1 else (c[i] >= sl_px)
                hit_tp = False
                hit_opp = ((sell_live and c[i] <= sell_stop) if side == 1
                           else (buy_live and c[i] >= buy_stop))
            else:
                hit_sl = (l[i] <= sl_px) if side == 1 else (h[i] >= sl_px)
                hit_tp = (h[i] >= tp_px + through) if side == 1 else (l[i] <= tp_px - through)
                # opposite entry stop still live (non-OCO) -> whipsaw close
                hit_opp = ((sell_live and l[i] <= sell_stop) if side == 1
                           else (buy_live and h[i] >= buy_stop))

            # pessimistic ordering: SL/whipsaw before TP within the same bar
            if hit_sl or hit_opp:
                slip = cm.stop_fill_slippage_ticks(lr, secs_since)
                stop_ref = sl_px if hit_sl else (sell_stop if side == 1 else buy_stop)
                exit_px = stop_ref - side * slip * MNQ_TICK
                reason = 'sl' if hit_sl else 'whipsaw'
                return _finish(event_ts, event_name, side, entry_px, exit_px,
                               reason, entry_slip, slip, hit_opp or whipsaw, p, cm)
            if hit_tp:
                return _finish(event_ts, event_name, side, entry_px, tp_px,
                               'tp', entry_slip, 0.0, whipsaw, p, cm)
            if t >= hard_exit_ts:
                slip = cm.stop_fill_slippage_ticks(lr, secs_since)
                exit_px = c[i] - side * (slip * MNQ_TICK)
                return _finish(event_ts, event_name, side, entry_px, exit_px,
                               'time', entry_slip, slip, whipsaw, p, cm)
        i += 1

    # ran out of data with open position: close at last price with slippage
    if side != 0:
        lr = _local_range_ticks(h, l, n - 1)
        slip = cm.stop_fill_slippage_ticks(lr, ts[-1] - event_ts)
        exit_px = c[-1] - side * slip * MNQ_TICK
        return _finish(event_ts, event_name, side, entry_px, exit_px,
                       'time', entry_slip, slip, whipsaw, p, cm)
    return no_trade


def _finish(event_ts, event_name, side, entry_px, exit_px, reason,
            entry_slip, exit_slip, whipsaw, p: StraddleParams,
            cm: CostModel) -> TradeResult:
    pnl_pts = side * (exit_px - entry_px)
    from .costs import points_to_dollars
    dollars = points_to_dollars(pnl_pts, p.contracts) - cm.round_trip_commission() * p.contracts
    return TradeResult(event_ts, event_name, side, entry_px, exit_px, reason,
                       pnl_pts, dollars, whipsaw, entry_slip, exit_slip)
