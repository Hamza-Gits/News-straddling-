"""Parameter optimization: grid search + walk-forward + plateau scoring.

Usage:  python run_optimize.py [--config config/config.yaml] [--stress 1.5]

Outputs:
  results/leaderboard.csv   - every config, in-sample AND out-of-sample stats
  results/champion.json     - the selected champion config + its OOS stats
"""

import argparse
import json
import time
from pathlib import Path

import yaml

from src.costs import CostModel
from src.optimize import Grid, grid_search
from src.backtest import load_cached_windows, precompute_windows
from run_backtest import load_all

CACHE = Path('data/event_windows.parquet')


def get_windows(cfg, before, after):
    if CACHE.exists():
        print(f'Loading cached event windows: {CACHE}')
        return load_cached_windows(CACHE)
    print('No cache found - loading raw data (run preprocess_ticks.py to '
          'build the cache for large tick files).')
    bars, news = load_all(cfg)
    return precompute_windows(bars, news.sort_values('ts_utc'), before, after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--stress', type=float, default=None,
                    help='override cost stress_multiplier (try 1.0 / 1.5 / 2.0)')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    costs = dict(cfg['costs'])
    if args.stress is not None:
        costs['stress_multiplier'] = args.stress
    cm = CostModel(**costs)

    g = cfg['optimize']['grid']
    grid = Grid(stop_dist_ticks=tuple(g['stop_dist_ticks']),
                tp_ticks=tuple(g['tp_ticks']),
                sl_ticks=tuple(g['sl_ticks']),
                arm_secs=tuple(g['arm_secs']),
                cancel_secs=tuple(g['cancel_secs']),
                max_hold_secs=tuple(g['max_hold_secs']),
                oco=tuple(g['oco']))

    windows = get_windows(cfg, grid.max_before() + 60, grid.max_after() + 60)
    print(f'{len(windows)} event windows loaded.')

    t0 = time.time()

    def progress(i, n):
        el = time.time() - t0
        eta = el / max(i, 1) * (n - i)
        print(f'  config {i}/{n}  elapsed {el:,.0f}s  eta {eta:,.0f}s', flush=True)

    res = grid_search(windows, grid, cm,
                      n_folds=cfg['optimize']['n_folds'],
                      min_oos_trades=cfg['optimize']['min_oos_trades'],
                      progress=progress)

    out = Path('results')
    out.mkdir(exist_ok=True)
    lb = res['leaderboard'].sort_values('oos_net_pnl', ascending=False)
    lb.to_csv(out / 'leaderboard.csv', index=False)

    print(f"\nSearched {res['n_configs']} configs x {res['n_events']} events "
          f"over {res['n_folds']} walk-forward folds in {time.time()-t0:,.0f}s "
          f"(stress={costs['stress_multiplier']}x)")

    print('\n=== TRUE walk-forward selection curve ===')
    print('(re-pick best config on each expanding train window, trade it on'
          ' the next unseen chunk - the honest simulation of the process)')
    for k, v in res['wf_stats'].items():
        print(f'  {k:>18}: {v:,.4f}' if isinstance(v, float) else f'  {k:>18}: {v}')
    for pick in res['wf_picks']:
        print(f"  after {pick['train_events']:>4} events -> picked {pick['picked']}")

    if res['champion'] is None:
        print('\n*** NO CHAMPION: no config met the OOS profitability + '
              'min-trades gate. The edge does not survive realistic costs. ***')
        (out / 'champion.json').write_text(json.dumps(
            {'champion': None, 'stress_multiplier': costs['stress_multiplier']},
            indent=2))
    else:
        ch = {k: (v.item() if hasattr(v, 'item') else v)
              for k, v in res['champion'].items()}
        ch['stress_multiplier'] = costs['stress_multiplier']
        (out / 'champion.json').write_text(json.dumps(ch, indent=2, default=str))
        print('\n=== CHAMPION (selected on OUT-OF-SAMPLE + plateau stability) ===')
        for k in ['p_stop_dist', 'p_tp', 'p_sl', 'p_arm', 'p_cancel', 'p_hold',
                  'p_oco', 'oos_net_pnl', 'oos_pf', 'oos_expectancy',
                  'oos_trades', 'oos_win_rate', 'oos_max_dd',
                  'oos_whipsaw_rate', 'plateau', 'is_net_pnl', 'is_pf']:
            print(f'  {k:>18}: {ch.get(k)}')
        print('\nNote: is_* columns are IN-SAMPLE (optimistic by construction).'
              '\nJudge the strategy on oos_* numbers only.')
    print('Saved: results/leaderboard.csv, results/champion.json')


if __name__ == '__main__':
    main()
