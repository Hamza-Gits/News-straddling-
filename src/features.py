"""Per-event features: impact level + forecast-surprise z-score.

Surprise z = (actual - forecast) / std(historical surprises for that event
name over the full CSV). Uses the event's OWN release values (known at
release time). The std uses full-sample history - a mild scale lookahead,
acceptable for exploratory filtering; a live system would use expanding std.

Collapsed simultaneous releases (NFP + AHE + ...) take the max |z| across
components and 'high' if any component is high.
"""

import re
import numpy as np
import pandas as pd

_NUM = re.compile(r'(-?\d+\.?\d*)\s*([KMBT%]?)', re.I)
_SCALE = {'': 1.0, '%': 1.0, 'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}


def _parse(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    m = _NUM.search(str(v).replace(',', ''))
    if not m:
        return np.nan
    return float(m.group(1)) * _SCALE[m.group(2).upper()]


def build_features(ff_csv_path, news: pd.DataFrame) -> dict[float, dict]:
    """news: collapsed events from load_news (ts_utc, impact, event).
    Returns {event_epoch: {'impact_high': bool, 'surprise_z': float}}."""
    raw = pd.read_csv(ff_csv_path)
    raw['ts_utc'] = pd.to_datetime(raw['datetime_utc'], utc=True)
    raw['a'] = raw['actual'].map(_parse)
    raw['f'] = raw['forecast'].map(_parse)
    raw['surprise'] = raw['a'] - raw['f']

    # per-event-name surprise std over the full 6-year history
    stds = (raw.dropna(subset=['surprise'])
            .groupby('event')['surprise'].std().rename('sstd'))
    raw = raw.merge(stds, on='event', how='left')
    raw['z'] = np.where(raw['sstd'] > 0, raw['surprise'] / raw['sstd'], np.nan)

    by_ts = raw.groupby('ts_utc').agg(
        z=('z', lambda s: np.nan if s.abs().max() != s.abs().max()
           else float(s.loc[s.abs().idxmax()])),
        high=('impact', lambda s: bool((s.str.lower() == 'high').any())),
    )

    feats = {}
    for row in news.itertuples(index=False):
        epoch = row.ts_utc.timestamp()
        if row.ts_utc in by_ts.index:
            r = by_ts.loc[row.ts_utc]
            feats[epoch] = {'impact_high': bool(r['high']),
                            'surprise_z': float(r['z']) if r['z'] == r['z'] else np.nan}
        else:
            feats[epoch] = {'impact_high': row.impact == 'high',
                            'surprise_z': np.nan}
    return feats
