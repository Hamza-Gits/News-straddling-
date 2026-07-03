"""Exhaustive variant sweep: cont / fade / pretrend / breakout x timing x
spike-scaled brackets x filters (spike size, impact, surprise-z, spread cap).

Usage: python run_variants.py [--stress 1.5] [--workers N]

Multiple-testing warning is printed with results: ~23k configs on 56 events
WILL produce lucky survivors; anything that passes here is a candidate for
6-year confirmation, not a tradeable edge.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.costs import CostModel
from src.variants import VariantParams, simulate_variant
from src.features import build_features
from src.news_loader import load_news
from src.backtest import load_cached_windows
from src.optimize import _stats, _fold_bounds

GRID = {
    'variant': ['cont', 'fade', 'pretrend', 'breakout'],
    'delay_secs': [5.0, 10.0, 20.0, 30.0, 60.0],
    'tp_k': [0.5, 1.0, 1.5, 2.0],
    'sl_k': [0.5, 1.0],
    'max_hold_secs': [120.0, 300.0, 600.0],
    'min_spike_pts': [0.0, 5.0, 10.0, 20.0],
    'require_high': [False, True],
    'min_surprise_z': [0.0, 1.0, 2.0],
    'spread_cap_ticks': [None, 8.0],
}

_G = {}


def _init(windows, feats, costs_kwargs):
    _G['w'] = windows
    _G['f'] = feats
    _G['cm'] = CostModel(**costs_kwargs)


def _sim(args):
    idx, kw = args
    p = VariantParams(**kw)
    cm = _G['cm']
    pnl = np.full(len(_G['w']), np.nan)
    for j, w in enumerate(_G['w']):
        t = simulate_variant(w['ts'], w['o'], w['h'], w['l'], w['c'],
                             w.get('sp'), w['event_ts'], w['name'],
                             _G['f'].get(w['event_ts'], {}), p, cm)
        if t.side != 0:
            pnl[j] = t.pnl_dollars
    return idx, pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--stress', type=float, default=1.5)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--min-oos-trades', type=int, default=12)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    costs = dict(cfg['costs'])
    costs['stress_multiplier'] = args.stress

    windows = load_cached_windows('data/event_windows.parquet')
    news = load_news(sorted(Path(cfg['data']['news_dir']).glob('*.csv')),
                     currencies=tuple(cfg['data']['currencies']),
                     impacts=tuple(cfg['data']['impacts']),
                     include_speeches=False)
    feats = build_features(
        sorted(Path(cfg['data']['news_dir']).glob('*.csv'))[0], news)
    n_ev = len(windows)
    print(f'{n_ev} events, features for {len(feats)}')

    keys = list(GRID)
    configs = [dict(zip(keys, vals)) for vals in product(*GRID.values())]
    print(f'{len(configs)} variant configs x {n_ev} events '
          f'(stress={args.stress}x)')

    bounds = _fold_bounds(n_ev, cfg['optimize']['n_folds'])
    oos_mask = np.zeros(n_ev, dtype=bool)
    for tr_end, te_end in bounds:
        oos_mask[tr_end:te_end] = True

    pnl_mat = np.full((len(configs), n_ev), np.nan)
    t0 = time.time()
    jobs = [(i, kw) for i, kw in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(windows, feats, costs)) as ex:
        done = 0
        for idx, pnl in ex.map(_sim, jobs, chunksize=32):
            pnl_mat[idx] = pnl
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f'  {done}/{len(configs)}  {el:,.0f}s '
                      f'eta {el/done*(len(configs)-done):,.0f}s', flush=True)

    zeros = np.zeros(n_ev, dtype=bool)
    rows = []
    for i, kw in enumerate(configs):
        row = {f'p_{k}': (v if v is not None else 'none') for k, v in kw.items()}
        row.update(_stats(pnl_mat[i], zeros, 'is_'))
        row.update(_stats(pnl_mat[i, oos_mask], zeros[oos_mask], 'oos_'))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('results/variants_leaderboard.csv', index=False)

    print(f'\nDone in {time.time()-t0:,.0f}s. '
          f'{(df.is_net_pnl > 0).sum()} IS-positive, '
          f'{(df.oos_net_pnl > 0).sum()} OOS-positive (any n).')

    gate = ((df.oos_trades >= args.min_oos_trades) & (df.oos_pf > 1.0)
            & np.isfinite(df.oos_pf) & (df.is_net_pnl > 0))
    surv = df[gate].sort_values('oos_expectancy', ascending=False)
    print(f'{len(surv)} configs pass the gate '
          f'(>= {args.min_oos_trades} OOS trades, OOS PF>1, IS>0).')
    print('\n*** {} of {} configs tested: expect ~{} false survivors from '
          'luck alone. Survivors are candidates for 6-year confirmation, '
          'NOT edges. ***'.format(len(surv), len(configs),
                                  max(1, int(len(configs) * 0.001))))
    if len(surv):
        cols = [c for c in surv.columns if c.startswith('p_')] + [
            'is_net_pnl', 'is_pf', 'oos_net_pnl', 'oos_pf',
            'oos_expectancy', 'oos_trades', 'oos_win_rate', 'oos_max_dd']
        print(surv.head(25)[cols].round(2).to_string(index=False))
        surv.head(100).to_json('results/variant_survivors.json',
                               orient='records', indent=1)

    print('\n=== marginal mean OOS net pnl by dimension ===')
    for k in GRID:
        m = df.groupby(f'p_{k}')['oos_net_pnl'].mean().round(1)
        print(f'  {k:>16}: ' + '  '.join(f'{iv}={v}' for iv, v in m.items()))


if __name__ == '__main__':
    main()
