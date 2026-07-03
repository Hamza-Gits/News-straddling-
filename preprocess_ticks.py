"""One-time preprocessing: stream giant NinjaTrader tick exports, keep only
1-second bars around news events, cache to parquet.

NinjaTrader 'Last' export format (semicolon, no header):
    YYYYMMDD HHMMSS fffffff;last;bid;ask;volume

Usage:
  python preprocess_ticks.py --detect-tz          # find export timezone first
  python preprocess_ticks.py                      # build data/event_windows.parquet
  python preprocess_ticks.py --tz-offset 2.0      # override detected offset (hours)

Timezone detection: news releases hit at exact known UTC seconds and produce
violent 1s-volatility bursts. We scan candidate UTC offsets and pick the one
where post-release volatility (0..+45s) most exceeds pre-release (-300..-60s)
across sampled high-impact events. Winter and summer are checked separately to
catch DST-observing export timezones.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.news_loader import load_news

CHUNK = 4_000_000
BEFORE_SECS = 180.0     # window kept before each event (covers max arm + pad)
AFTER_SECS = 780.0      # after each event (covers max cancel/hold + pad)
NT_COLS = ['ts_raw', 'last', 'bid', 'ask', 'vol']


def tick_files(cfg) -> list[Path]:
    d = Path(cfg['data']['ticks_dir'])
    files = sorted(d.glob('*.txt')) + sorted(d.glob('*.csv')) + sorted(d.glob('*.gz'))
    if not files:
        raise SystemExit(f'No tick files found in "{d}"')
    return files


def stream(path: Path):
    """Yield chunks with naive datetime64[s] 'ts' (fractional secs dropped)."""
    for chunk in pd.read_csv(path, sep=';', header=None, names=NT_COLS,
                             chunksize=CHUNK, dtype={'ts_raw': str}):
        # 'YYYYMMDD HHMMSS fffffff' -> second resolution
        ts = pd.to_datetime(chunk['ts_raw'].str.slice(0, 15),
                            format='%Y%m%d %H%M%S')
        chunk = chunk.drop(columns='ts_raw')
        chunk.index = ts
        yield chunk


def load_news_cfg(cfg) -> pd.DataFrame:
    news_files = sorted(Path(cfg['data']['news_dir']).glob('*.csv'))
    return load_news(news_files, news_tz=cfg['data']['news_tz'],
                     currencies=tuple(cfg['data']['currencies']),
                     impacts=tuple(cfg['data']['impacts']),
                     include_speeches=cfg['data'].get('include_speeches', False))


# --------------------------- timezone detection -----------------------------

def detect_tz(cfg, sample_file: Path, offsets=None) -> dict:
    """Return {'winter': best_offset_h, 'summer': best_offset_h, 'table': df}.
    offset means: naive_file_time = utc_time + offset (hours)."""
    if offsets is None:
        offsets = [h / 2.0 for h in range(-24, 29)]  # -12h .. +14h in 30min steps
    news = load_news_cfg(cfg)
    high = news.copy()

    # limit to events plausibly inside this file's date range: probe cheaply
    first = next(iter(stream(sample_file)))
    t0, t1 = first.index.min(), None
    # read last ~2MB for end date
    with open(sample_file, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 2_000_000))
        tail = f.read().decode(errors='ignore').strip().splitlines()
    t1 = pd.to_datetime(tail[-1].split(';')[0][:15], format='%Y%m%d %H%M%S')
    print(f'{sample_file.name}: naive range {t0} .. {t1}')

    # candidate windows in naive file-time for every (event, offset)
    span_lo = high['ts_utc'].dt.tz_localize(None)
    mask = (span_lo >= t0 - pd.Timedelta(hours=14)) & (span_lo <= t1 + pd.Timedelta(hours=14))
    ev = high[mask]
    if len(ev) > 60:
        ev = ev.sample(60, random_state=0).sort_values('ts_utc')
    if len(ev) < 8:
        raise SystemExit('Too few events overlap this file for tz detection.')
    print(f'Using {len(ev)} events for detection.')

    ev_utc = ev['ts_utc'].dt.tz_localize(None).values  # naive UTC

    # accumulate 1s close series around every candidate window
    want = []  # (event_i, offset_i, start_naive, end_naive)
    for i, e in enumerate(ev_utc):
        for j, off in enumerate(offsets):
            s = e + np.timedelta64(int(off * 3600), 's')
            want.append((i, j, s - np.timedelta64(300, 's'),
                         s + np.timedelta64(60, 's')))
    starts = np.array([w[2] for w in want])
    ends = np.array([w[3] for w in want])
    series = [None] * len(want)

    for chunk in stream(sample_file):
        cts = chunk.index.values
        lo, hi = cts.min(), cts.max()
        sel = np.where((starts <= hi) & (ends >= lo))[0]
        for k in sel:
            sl = chunk.loc[want[k][2]:want[k][3], 'last']
            if len(sl):
                series[k] = sl if series[k] is None else pd.concat([series[k], sl])

    rows = []
    for j, off in enumerate(offsets):
        ratios = []
        for i in range(len(ev_utc)):
            s = series[i * len(offsets) + j]
            if s is None or len(s) < 60:
                continue
            sec = s.groupby(s.index.floor('s')).last()
            sec = sec.reindex(pd.date_range(sec.index[0], sec.index[-1],
                                            freq='1s')).ffill()
            r = sec.diff().abs()
            anchor = want[i * len(offsets) + j][2] + np.timedelta64(300, 's')
            pre = r[(r.index >= anchor - pd.Timedelta(seconds=240)) &
                    (r.index < anchor - pd.Timedelta(seconds=60))]
            post = r[(r.index >= anchor) & (r.index < anchor + pd.Timedelta(seconds=45))]
            if len(pre) > 30 and len(post) > 10 and pre.mean() > 0:
                ratios.append(post.mean() / pre.mean())
        if ratios:
            rows.append({'offset_h': off, 'n': len(ratios),
                         'median_ratio': float(np.median(ratios)),
                         'mean_ratio': float(np.mean(ratios))})
    table = pd.DataFrame(rows).sort_values('median_ratio', ascending=False)
    print('\nTop candidate offsets (naive = UTC + offset):')
    print(table.head(8).to_string(index=False))
    best = float(table.iloc[0]['offset_h'])
    print(f'\nBEST OFFSET: {best:+.1f}h  '
          f'(median post/pre vol ratio {table.iloc[0]["median_ratio"]:.2f})')
    return {'best_offset_h': best, 'table': table}


# --------------------------- window extraction ------------------------------

def build_cache(cfg, offset_h: float, out_path: Path):
    news = load_news_cfg(cfg)
    ev_naive = (news['ts_utc'].dt.tz_localize(None)
                + pd.Timedelta(hours=offset_h)).values
    # .dt.as_unit('ns') pins the epoch unit (pandas 3.x default is us)
    ev_utc_epoch = news['ts_utc'].dt.as_unit('ns').astype('int64').values / 1e9
    names = news['event'].values

    starts = ev_naive - np.timedelta64(int(BEFORE_SECS), 's')
    ends = ev_naive + np.timedelta64(int(AFTER_SECS), 's')

    all_parts = []
    for path in tick_files(cfg):
        contract = path.stem
        print(f'--- {contract} ({path.stat().st_size/1e9:.2f} GB)')
        kept = []
        rows_seen = 0
        for chunk in stream(path):
            rows_seen += len(chunk)
            cts = chunk.index.values
            lo, hi = cts.min(), cts.max()
            sel = np.where((starts <= hi) & (ends >= lo))[0]
            for k in sel:
                sl = chunk.loc[starts[k]:ends[k]]
                if len(sl):
                    sl = sl.copy()
                    sl['ev'] = k
                    kept.append(sl)
        if not kept:
            print(f'    {rows_seen:,} rows, no event overlap')
            continue
        raw = pd.concat(kept)
        parts = []
        for k, g in raw.groupby('ev'):
            # keep ALL ticks within each second (stable sort preserves the
            # file's intra-second trade order for correct open/close)
            g = g.sort_index(kind='stable')
            sec = pd.DataFrame({
                'open': g['last'].resample('1s').first(),
                'high': g['last'].resample('1s').max(),
                'low': g['last'].resample('1s').min(),
                'close': g['last'].resample('1s').last(),
                'volume': g['vol'].resample('1s').sum(),
                'spread_ticks': ((g['ask'] - g['bid']).clip(lower=0)
                                 .resample('1s').max() / 0.25),
            }).dropna(subset=['close'])
            sec['ev_idx'] = k
            sec['contract'] = contract
            parts.append(sec.reset_index(names='bar_naive'))
        dfc = pd.concat(parts, ignore_index=True)
        print(f'    {rows_seen:,} ticks -> {len(dfc):,} event-window bars '
              f'({dfc["ev_idx"].nunique()} events)')
        all_parts.append(dfc)

    cache = pd.concat(all_parts, ignore_index=True)
    # overlapping contracts: keep, per event, the contract with max volume
    vol = (cache.groupby(['ev_idx', 'contract'])['volume'].sum()
           .rename('cvol').reset_index())
    best = vol.sort_values('cvol').drop_duplicates('ev_idx', keep='last')
    cache = cache.merge(best[['ev_idx', 'contract']], on=['ev_idx', 'contract'])

    # attach event metadata + convert bar time to UTC epoch seconds
    cache['event_ts'] = ev_utc_epoch[cache['ev_idx'].values]
    cache['event_name'] = names[cache['ev_idx'].values]
    bar_naive = pd.DatetimeIndex(cache['bar_naive']).as_unit('ns')
    cache['bar_epoch'] = (bar_naive.asi8 / 1e9) - offset_h * 3600.0
    cache = cache.drop(columns=['bar_naive'])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(out_path, index=False)
    n_ev = cache['ev_idx'].nunique()
    print(f'\nSaved {out_path}: {len(cache):,} bars, {n_ev}/{len(news)} events covered.')

    # measured spread report -> calibrate the cost model
    at_news = cache[(cache['bar_epoch'] - cache['event_ts']).between(0, 10)]
    base = cache[(cache['bar_epoch'] - cache['event_ts']).between(-120, -30)]
    if len(at_news) and len(base):
        print('\nMeasured MNQ spread (ticks):')
        print(f'  normal (pre-news): median {base["spread_ticks"].median():.1f}, '
              f'p90 {base["spread_ticks"].quantile(.9):.1f}')
        print(f'  news 0-10s:        median {at_news["spread_ticks"].median():.1f}, '
              f'p90 {at_news["spread_ticks"].quantile(.9):.1f}, '
              f'p99 {at_news["spread_ticks"].quantile(.99):.1f}')
        print('  -> set costs.base_spread_ticks / news_spread_mult in config '
              'to match or exceed these.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--detect-tz', action='store_true')
    ap.add_argument('--tz-file', default=None,
                    help='file to use for tz detection (default: largest)')
    ap.add_argument('--tz-offset', type=float, default=None,
                    help='naive_file_time = UTC + this many hours')
    ap.add_argument('--out', default='data/event_windows.parquet')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    files = tick_files(cfg)
    if args.detect_tz:
        f = Path(args.tz_file) if args.tz_file else max(files, key=lambda p: p.stat().st_size)
        res = detect_tz(cfg, f)
        Path('data').mkdir(exist_ok=True)
        Path('data/tz_offset.json').write_text(
            json.dumps({'offset_h': res['best_offset_h']}))
        print('Saved data/tz_offset.json - rerun without --detect-tz to build cache.')
        return

    if args.tz_offset is not None:
        offset = args.tz_offset
    elif Path('data/tz_offset.json').exists():
        offset = json.loads(Path('data/tz_offset.json').read_text())['offset_h']
        print(f'Using detected tz offset {offset:+.1f}h from data/tz_offset.json')
    else:
        raise SystemExit('Run --detect-tz first (or pass --tz-offset).')

    build_cache(cfg, offset, Path(args.out))


if __name__ == '__main__':
    main()
