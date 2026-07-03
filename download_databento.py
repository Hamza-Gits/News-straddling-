"""Download ONLY the news-event windows for MNQ from Databento - no bulk files.

Reads the ForexFactory news CSV, builds the exact UTC time slices the backtest
needs (BEFORE_SECS before each release .. AFTER_SECS after), pulls 1-second
OHLCV bars for the continuous front-month MNQ contract, and writes straight
into the same parquet cache format preprocess_ticks.py produces.

Setup:
  pip install databento
  set DATABENTO_API_KEY=db-XXXX...     (or pass --api-key)

Usage:
  python download_databento.py --start 2020-06-29 --end 2025-12-31
  python download_databento.py --dry-run          # count windows + estimate size

Cost note: ohlcv-1s for ~25min windows x ~2000 events is a few hundred MB of
raw feed -> typically a few dollars at most. Run --dry-run first; Databento's
metadata API returns an exact cost estimate before you commit.

Timestamps from Databento are true UTC nanoseconds - no timezone ambiguity.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.news_loader import load_news
from preprocess_ticks import BEFORE_SECS, AFTER_SECS

DATASET = 'GLBX.MDP3'
SYMBOL = 'MNQ.c.0'          # continuous front month, volume-rolled
SCHEMA = 'ohlcv-1s'


def build_windows(cfg, start, end) -> pd.DataFrame:
    news_files = sorted(Path(cfg['data']['news_dir']).glob('*.csv'))
    news = load_news(news_files, news_tz=cfg['data']['news_tz'],
                     currencies=tuple(cfg['data']['currencies']),
                     impacts=tuple(cfg['data']['impacts']),
                     include_speeches=cfg['data'].get('include_speeches', False))
    news = news[(news['ts_utc'] >= start) & (news['ts_utc'] <= end)]
    return news.sort_values('ts_utc').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--start', default='2020-06-29')
    ap.add_argument('--end', default='2026-07-01')
    ap.add_argument('--api-key', default=None)
    ap.add_argument('--out', default='data/event_windows.parquet')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--merge', action='store_true',
                    help='merge with an existing cache instead of overwriting')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1)
    news = build_windows(cfg, start, end)
    total_secs = len(news) * (BEFORE_SECS + AFTER_SECS)
    print(f'{len(news)} events -> {total_secs/3600:.1f} hours of 1s bars total')

    if args.dry_run:
        print('Dry run only. Rough raw-size estimate: '
              f'~{len(news) * 0.05:.0f} MB of ohlcv-1s. '
              'Databento will quote exact cost per request.')
        return

    import databento as db
    key = args.api_key or os.environ.get('DATABENTO_API_KEY')
    if not key:
        sys.exit('Set DATABENTO_API_KEY or pass --api-key.')
    client = db.Historical(key)

    parts = []
    for i, row in news.iterrows():
        t0 = row.ts_utc - pd.Timedelta(seconds=BEFORE_SECS)
        t1 = row.ts_utc + pd.Timedelta(seconds=AFTER_SECS)
        try:
            data = client.timeseries.get_range(
                dataset=DATASET, symbols=[SYMBOL], stype_in='continuous',
                schema=SCHEMA, start=t0.isoformat(), end=t1.isoformat())
            df = data.to_df()
        except Exception as e:  # noqa: BLE001 - log and continue, retry later
            print(f'  [{i+1}/{len(news)}] {row.event[:40]}: FAILED {e}')
            time.sleep(1.0)
            continue
        if df.empty:
            continue
        # databento ohlcv prices are in 1e-9 units in raw; to_df() gives floats
        out = pd.DataFrame({
            'open': df['open'], 'high': df['high'],
            'low': df['low'], 'close': df['close'],
            'volume': df['volume'],
            'spread_ticks': np.nan,
            'ev_idx': i, 'contract': 'MNQ.c.0',
            'event_ts': row.ts_utc.timestamp(),
            'event_name': row.event,
            'bar_epoch': df.index.as_unit('ns').asi8 / 1e9,
        })
        parts.append(out)
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(news)}] {len(parts)} windows downloaded...')
        time.sleep(0.1)   # be polite

    cache = pd.concat(parts, ignore_index=True)
    out_path = Path(args.out)
    if args.merge and out_path.exists():
        old = pd.read_parquet(out_path)
        keep = ~old['event_ts'].isin(set(cache['event_ts']))
        cache = pd.concat([old[keep], cache], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(out_path, index=False)
    print(f'Saved {out_path}: {cache["event_ts"].nunique()} events, '
          f'{len(cache):,} bars.')


if __name__ == '__main__':
    main()
