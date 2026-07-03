"""Run the baseline news-straddle backtest with the config.yaml parameters.

Usage:  python run_backtest.py [--config config/config.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.costs import CostModel
from src.straddle import StraddleParams
from src.news_loader import load_news
from src.tick_loader import load_ticks
from src.backtest import run_backtest
from src.metrics import summarize, trades_to_frame

DATA_EXTS = ('.csv', '.gz', '.zip', '.parquet', '.pq')


def collect_files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob('*') if p.suffix.lower() in DATA_EXTS)


def load_all(cfg: dict):
    news_files = collect_files(Path(cfg['data']['news_dir']))
    tick_files = collect_files(Path(cfg['data']['ticks_dir']))
    if not news_files:
        sys.exit(f"No news files in {cfg['data']['news_dir']} - drop the FF CSVs there.")
    if not tick_files:
        sys.exit(f"No tick files in {cfg['data']['ticks_dir']} - drop the MNQ data there.")
    print(f"Loading {len(news_files)} news file(s), {len(tick_files)} tick file(s)...")
    news = load_news(news_files, news_tz=cfg['data']['news_tz'],
                     currencies=tuple(cfg['data']['currencies']),
                     impacts=tuple(cfg['data']['impacts']),
                     include_speeches=cfg['data'].get('include_speeches', False))
    bars = load_ticks(tick_files, data_tz=cfg['data']['data_tz'])
    print(f"  {len(news)} filtered news events "
          f"({news['ts_utc'].min()} .. {news['ts_utc'].max()})")
    print(f"  {len(bars):,} one-second bars "
          f"({bars.index[0]} .. {bars.index[-1]})")
    return bars, news


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    bars, news = load_all(cfg)
    cm = CostModel(**cfg['costs'])
    p = StraddleParams(**cfg['strategy'])

    trades = run_backtest(bars, news, p, cm)
    stats = summarize(trades)

    out = Path('results')
    out.mkdir(exist_ok=True)
    trades_to_frame(trades).to_csv(out / 'baseline_trades.csv', index=False)
    (out / 'baseline_summary.json').write_text(json.dumps(stats, indent=2))

    print('\n=== Baseline results (slippage/spread/commission INCLUDED) ===')
    for k, v in stats.items():
        print(f'  {k:>18}: {v:,.4f}' if isinstance(v, float) else f'  {k:>18}: {v}')
    print(f"\nSaved: results/baseline_trades.csv, results/baseline_summary.json")


if __name__ == '__main__':
    main()
