"""MNQ tick / 1-second data loader.

Accepts:
  - Raw tick data (timestamp, price[, volume])  -> aggregated to 1s OHLC
  - 1-second bars (timestamp, open, high, low, close[, volume])
  - Bid/ask tick data (timestamp, bid, ask)     -> mid aggregated, spread measured

Auto-detects which one it is from the columns. Timestamps must be parseable;
timezone set via data_tz config ('UTC' if already UTC / epoch).

Files can be CSV (optionally .gz/.zip) or Parquet. Multiple files are
concatenated and sorted.
"""

from pathlib import Path
import numpy as np
import pandas as pd

TS_ALIASES = ['timestamp', 'ts', 'time', 'datetime', 'date_time', 'date']
PRICE_ALIASES = ['price', 'last', 'trade', 'px', 'close']


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find(cols, names):
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def load_ticks(paths: list[str | Path], data_tz: str = 'UTC') -> pd.DataFrame:
    """Return 1-second bars: DataFrame indexed by tz-aware UTC DatetimeIndex,
    columns [open, high, low, close, volume, spread_ticks(optional)].
    """
    frames = [_read_any(Path(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    cols = list(df.columns)

    ts_col = _find(cols, TS_ALIASES)
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {cols}")

    ts = df[ts_col]
    if np.issubdtype(ts.dtype, np.number):
        # epoch: guess unit by magnitude
        mx = float(ts.iloc[:1000].max())
        unit = 'ns' if mx > 1e17 else 'us' if mx > 1e14 else 'ms' if mx > 1e11 else 's'
        idx = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(ts, errors='coerce', format='mixed')
        if idx.dt.tz is None:
            idx = idx.dt.tz_localize(data_tz, ambiguous='NaT', nonexistent='NaT')
        idx = idx.dt.tz_convert('UTC')
    df = df.set_index(pd.DatetimeIndex(idx)).sort_index()
    df = df[df.index.notna()]

    lower = {c.lower().strip() for c in df.columns}

    if {'open', 'high', 'low', 'close'} <= lower:
        # already bars
        ren = {c: c.lower().strip() for c in df.columns}
        bars = df.rename(columns=ren)[['open', 'high', 'low', 'close']]
        if 'volume' in lower:
            bars['volume'] = df.rename(columns=ren)['volume']
        else:
            bars['volume'] = np.nan
        # resample to strict 1s grid in case source is finer/irregular
        bars = bars.resample('1s').agg({'open': 'first', 'high': 'max',
                                        'low': 'min', 'close': 'last',
                                        'volume': 'sum'}).dropna(subset=['close'])
        return bars

    if {'bid', 'ask'} <= lower:
        ren = {c: c.lower().strip() for c in df.columns}
        d = df.rename(columns=ren)
        mid = (d['bid'] + d['ask']) / 2.0
        spread = (d['ask'] - d['bid']) / 0.25  # ticks
        bars = mid.resample('1s').ohlc()
        bars['volume'] = np.nan
        bars['spread_ticks'] = spread.resample('1s').max()  # worst spread each second
        return bars.dropna(subset=['close'])

    price_col = _find(list(df.columns), PRICE_ALIASES)
    if price_col is None:
        raise ValueError(f"Can't identify price columns in {cols}")
    px = pd.to_numeric(df[price_col], errors='coerce').dropna()
    bars = px.resample('1s').ohlc()
    vol_col = _find(list(df.columns), ['volume', 'size', 'qty', 'quantity'])
    bars['volume'] = (pd.to_numeric(df[vol_col], errors='coerce')
                      .resample('1s').sum() if vol_col else np.nan)
    return bars.dropna(subset=['close'])


def event_window(bars: pd.DataFrame, event_utc: pd.Timestamp,
                 before_secs: float, after_secs: float):
    """Slice bars around one event; returns (ts_epoch, o, h, l, c) numpy arrays
    or None if there's no data coverage (holiday, missing session, etc.)."""
    w = bars.loc[event_utc - pd.Timedelta(seconds=before_secs):
                 event_utc + pd.Timedelta(seconds=after_secs)]
    if len(w) < 10:
        return None
    # require data actually spanning the release moment
    if w.index[0] > event_utc or w.index[-1] < event_utc:
        return None
    # as_unit('ns') pins the epoch unit explicitly (pandas 3.x defaults to us)
    ts = w.index.as_unit('ns').asi8 / 1e9
    return (ts.astype(np.float64), w['open'].to_numpy(np.float64),
            w['high'].to_numpy(np.float64), w['low'].to_numpy(np.float64),
            w['close'].to_numpy(np.float64))
