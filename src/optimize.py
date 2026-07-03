"""Parameter optimization with walk-forward validation and plateau scoring.

Champion selection is NOT 'best in-sample net PnL' - that number is curve-fit
by construction. Instead:

  1. Every (config, event) pair is simulated exactly ONCE (events are
     independent, so fold slicing afterwards is free array masking).
  2. Chronological folds: fold k trains on events [0 .. b_k) and tests on
     [b_k .. b_{k+1}). A config's OOS stats aggregate its test-slice trades -
     performance on events its selection never saw.
  3. TRUE walk-forward selection curve: per fold, pick the best config on the
     train slice, record ITS test-slice trades. This simulates the real
     process of re-optimizing as time passes - the honest equity curve.
  4. Plateau score: mean OOS expectancy of grid NEIGHBORS (one step away in
     stop/tp/sl). Isolated peaks are noise, plateaus are edge.
  5. Champion = rank-combo of OOS expectancy, OOS PF, plateau, OOS drawdown,
     gated by a minimum OOS trade count.

Parallelized across configs with ProcessPoolExecutor.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
import os

import numpy as np
import pandas as pd

from .costs import CostModel
from .straddle import StraddleParams
from .backtest import precompute_windows, run_cached


@dataclass
class Grid:
    stop_dist_ticks: tuple = (20.0, 30.0, 40.0, 60.0, 80.0)
    tp_ticks: tuple = (40.0, 60.0, 80.0, 120.0, 160.0)
    sl_ticks: tuple = (20.0, 30.0, 40.0, 60.0)
    arm_secs: tuple = (10.0, 30.0, 60.0)
    cancel_secs: tuple = (30.0, 60.0, 120.0)
    max_hold_secs: tuple = (120.0, 300.0, 600.0)
    oco: tuple = (True, False)

    def configs(self) -> list[StraddleParams]:
        out = []
        for sd, tp, sl, arm, canc, hold, oco in product(
                self.stop_dist_ticks, self.tp_ticks, self.sl_ticks,
                self.arm_secs, self.cancel_secs, self.max_hold_secs, self.oco):
            if tp < sl:          # never risk more than the target
                continue
            out.append(StraddleParams(stop_dist_ticks=sd, tp_ticks=tp,
                                      sl_ticks=sl, arm_secs=arm,
                                      cancel_secs=canc, max_hold_secs=hold,
                                      oco=oco))
        return out

    def max_before(self) -> float:
        return max(self.arm_secs)

    def max_after(self) -> float:
        return max(max(self.cancel_secs), max(self.max_hold_secs))


# ---------------- worker (module-level for Windows spawn pickling) ----------
_G = {}


def _init_worker(windows, costs_kwargs):
    _G['windows'] = windows
    _G['cm'] = CostModel(**costs_kwargs)


def _sim_config(args):
    idx, params_kwargs = args
    p = StraddleParams(**params_kwargs)
    trades = run_cached(_G['windows'], p, _G['cm'])
    pnl = np.array([t.pnl_dollars if t.side != 0 else np.nan for t in trades])
    whip = np.array([t.whipsaw for t in trades], dtype=bool)
    return idx, pnl, whip


# ---------------- stats over pnl arrays -------------------------------------
def _stats(pnl: np.ndarray, whip: np.ndarray, prefix: str = '') -> dict:
    filled = ~np.isnan(pnl)
    x = pnl[filled]
    n = int(filled.sum())
    if n == 0:
        return {f'{prefix}net_pnl': 0.0, f'{prefix}pf': 0.0,
                f'{prefix}expectancy': 0.0, f'{prefix}trades': 0,
                f'{prefix}win_rate': 0.0, f'{prefix}max_dd': 0.0,
                f'{prefix}whipsaw_rate': 0.0}
    eq = np.cumsum(x)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    gw = float(x[x > 0].sum())
    gl = float(-x[x <= 0].sum())
    return {
        f'{prefix}net_pnl': float(x.sum()),
        f'{prefix}pf': float(gw / gl) if gl > 0 else float('inf'),
        f'{prefix}expectancy': float(x.mean()),
        f'{prefix}trades': n,
        f'{prefix}win_rate': float((x > 0).mean()),
        f'{prefix}max_dd': dd,
        f'{prefix}whipsaw_rate': float(whip[filled].mean()),
    }


def _fold_bounds(n_events: int, n_folds: int) -> list[tuple[int, int]]:
    """Chronological expanding-window folds: [(train_end, test_end), ...]."""
    chunk = n_events // (n_folds + 1)
    bounds = []
    for k in range(1, n_folds + 1):
        tr_end, te_end = chunk * k, min(chunk * (k + 1), n_events)
        if te_end > tr_end:
            bounds.append((tr_end, te_end))
    return bounds


def grid_search(bars, news, grid: Grid, cm: CostModel, n_folds: int = 4,
                min_oos_trades: int = 30, progress=None,
                n_workers: int | None = None) -> dict:
    news = news.sort_values('ts_utc').reset_index(drop=True)
    configs = grid.configs()
    pad = 60.0
    windows = precompute_windows(bars, news, grid.max_before() + pad,
                                 grid.max_after() + pad)
    n_ev = len(windows)
    bounds = _fold_bounds(n_ev, n_folds)
    oos_mask = np.zeros(n_ev, dtype=bool)
    for tr_end, te_end in bounds:
        oos_mask[tr_end:te_end] = True

    # ---- simulate every (config, event) once, in parallel ----
    pnl_mat = np.full((len(configs), n_ev), np.nan)
    whip_mat = np.zeros((len(configs), n_ev), dtype=bool)
    jobs = [(i, p.__dict__) for i, p in enumerate(configs)]
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 1)
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(windows, cm.__dict__)) as ex:
        for idx, pnl, whip in ex.map(_sim_config, jobs, chunksize=8):
            pnl_mat[idx], whip_mat[idx] = pnl, whip
            done += 1
            if progress and done % 100 == 0:
                progress(done, len(configs))

    # ---- per-config stats: full-sample (in-sample ref) + OOS union ----
    rows = []
    for i, p in enumerate(configs):
        row = {f'p_{k}': v for k, v in zip(
            ['stop_dist', 'tp', 'sl', 'arm', 'cancel', 'hold', 'oco'], p.key())}
        row.update(_stats(pnl_mat[i], whip_mat[i], 'is_'))
        row.update(_stats(pnl_mat[i, oos_mask], whip_mat[i, oos_mask], 'oos_'))
        rows.append(row)
    df = pd.DataFrame(rows)
    df['plateau'] = _plateau_score(df)

    # ---- TRUE walk-forward selection curve ----
    wf_trades = []
    wf_picks = []
    for tr_end, te_end in bounds:
        tr = pnl_mat[:, :tr_end]
        exp = np.nanmean(np.where(np.isnan(tr), np.nan, tr), axis=1)
        n_tr = np.sum(~np.isnan(tr), axis=1)
        exp[n_tr < max(10, min_oos_trades // 2)] = -np.inf
        best = int(np.nanargmax(exp))
        seg = pnl_mat[best, tr_end:te_end]
        wf_trades.append(seg[~np.isnan(seg)])
        wf_picks.append({'train_events': tr_end, 'picked': configs[best].key()})
    wf_pnl = np.concatenate(wf_trades) if wf_trades else np.array([])
    wf_stats = _stats(wf_pnl, np.zeros(len(wf_pnl), dtype=bool), 'wf_')

    # ---- champion ----
    eligible = df[(df['oos_trades'] >= min_oos_trades) &
                  (df['oos_pf'] > 1.0) & np.isfinite(df['oos_pf'])].copy()
    if eligible.empty:
        champion = None
    else:
        for col, asc in [('oos_expectancy', False), ('oos_pf', False),
                         ('plateau', False), ('oos_max_dd', True)]:
            eligible[f'rk_{col}'] = eligible[col].rank(ascending=asc)
        eligible['score'] = (eligible['rk_oos_expectancy'] + eligible['rk_oos_pf']
                             + eligible['rk_plateau'] + 0.5 * eligible['rk_oos_max_dd'])
        champion = eligible.sort_values('score').iloc[0].to_dict()

    return {'leaderboard': df, 'champion': champion, 'n_configs': len(configs),
            'n_folds': len(bounds), 'n_events': n_ev,
            'wf_stats': wf_stats, 'wf_picks': wf_picks}


def _plateau_score(df: pd.DataFrame) -> np.ndarray:
    """Mean OOS expectancy of configs one grid step away in stop/tp/sl
    (same timing/oco). Isolated peaks score low, plateaus score high."""
    key_cols = ['p_arm', 'p_cancel', 'p_hold', 'p_oco']
    dim_cols = ['p_stop_dist', 'p_tp', 'p_sl']
    score = np.zeros(len(df))
    for _, g in df.groupby(key_cols):
        vals = {tuple(r[c] for c in dim_cols): r['oos_expectancy']
                for _, r in g.iterrows()}
        axes = {c: sorted(g[c].unique()) for c in dim_cols}
        for i, r in g.iterrows():
            here = tuple(r[c] for c in dim_cols)
            neigh = []
            for d, c in enumerate(dim_cols):
                ax = axes[c]
                j = ax.index(here[d])
                for jj in (j - 1, j + 1):
                    if 0 <= jj < len(ax):
                        k = list(here)
                        k[d] = ax[jj]
                        v = vals.get(tuple(k))
                        if v is not None:
                            neigh.append(v)
            score[df.index.get_loc(i)] = np.mean(neigh) if neigh else r['oos_expectancy']
    return score
