"""Execution cost model: spread, slippage, commissions.

Philosophy (backtest-expert): punish the strategy. Every assumption here is
deliberately pessimistic. The asymmetry of news execution is modeled explicitly:

  - Stop entries  -> market orders in a fast tape: fill AT OR WORSE than stop price.
  - Stop losses   -> same: slip AGAINST you, scaled by local volatility.
  - Take profits  -> limit orders: NEVER slip in your favor. Conservative fill
                     rule requires price to trade THROUGH the limit by >= 1 tick
                     (touch alone is not a fill).

Slippage = base_ticks + vol_coeff * (local 1s range in ticks), then multiplied
by stress_multiplier. During the first seconds of a news release the local
range explodes, so slippage scales with it automatically.
"""

from dataclasses import dataclass

MNQ_TICK = 0.25          # points per tick
MNQ_TICK_VALUE = 0.50    # dollars per tick per contract
MNQ_POINT_VALUE = 2.00   # dollars per point per contract


@dataclass
class CostModel:
    # Spread (in ticks). Applied as half-spread cost on each marketable fill.
    base_spread_ticks: float = 1.0        # normal MNQ spread ~1 tick
    news_spread_mult: float = 4.0         # spread blowout multiplier inside news window
    news_spread_secs: float = 10.0        # how long after release the blowout lasts

    # Slippage for stop->market fills (entries and stop losses)
    base_slip_ticks: float = 1.0          # slippage even in calm tape
    vol_slip_coeff: float = 0.25          # extra ticks per tick of local 1s range
    stress_multiplier: float = 1.5        # global pessimism knob (test at 1.0/1.5/2.0)
    max_slip_ticks: float = 80.0          # cap (20 pts) to keep pathological bars sane

    # Take-profit limit fill rule
    tp_requires_through_ticks: float = 1.0  # must trade through limit by this much

    # Commission (per side, per contract, dollars) - round turn ~ $1.24 typical retail
    commission_per_side: float = 0.62
    exchange_fees_per_side: float = 0.37

    def spread_ticks(self, secs_since_news: float | None) -> float:
        """Effective spread in ticks at a moment in time."""
        if secs_since_news is not None and 0 <= secs_since_news <= self.news_spread_secs:
            # linearly decay the blowout back to base
            frac = 1.0 - (secs_since_news / self.news_spread_secs)
            return self.base_spread_ticks * (1.0 + (self.news_spread_mult - 1.0) * frac)
        return self.base_spread_ticks

    def stop_fill_slippage_ticks(self, local_range_ticks: float,
                                 secs_since_news: float | None,
                                 measured_spread_ticks: float | None = None) -> float:
        """Adverse slippage in ticks for a stop order that just triggered.
        If a MEASURED spread is available (from bid/ask tick data) it replaces
        the modeled spread - reality beats assumptions."""
        slip = self.base_slip_ticks + self.vol_slip_coeff * max(0.0, local_range_ticks)
        slip *= self.stress_multiplier
        # half the effective spread is also paid crossing the book
        if measured_spread_ticks is not None and measured_spread_ticks == measured_spread_ticks:
            slip += 0.5 * max(measured_spread_ticks, self.base_spread_ticks)
        else:
            slip += 0.5 * self.spread_ticks(secs_since_news)
        return min(slip, self.max_slip_ticks)

    def round_trip_commission(self) -> float:
        return 2.0 * (self.commission_per_side + self.exchange_fees_per_side)


def ticks_to_points(ticks: float) -> float:
    return ticks * MNQ_TICK


def points_to_dollars(points: float, contracts: int = 1) -> float:
    return points * MNQ_POINT_VALUE * contracts
