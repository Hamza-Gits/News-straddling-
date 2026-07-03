"""Forex Factory news calendar loader.

Auto-detects common FF export formats. Expected columns (any casing/order):
date, time, currency, impact, event/title. Extra columns ignored.

IMPORTANT: FF exports carry times in the timezone your FF profile was set to
when you exported. Set news_tz in config to match, or everything is garbage.
"""

from pathlib import Path
import pandas as pd

IMPACT_MAP = {
    'high': 'high', 'red': 'high', 'high impact expected': 'high',
    'medium': 'medium', 'orange': 'medium', 'org': 'medium',
    'medium impact expected': 'medium',
    'low': 'low', 'yellow': 'low', 'yel': 'low', 'low impact expected': 'low',
}

COL_ALIASES = {
    'date': ['date', 'day'],
    'time': ['time', 'time_et', 'timestamp'],
    'currency': ['currency', 'country', 'cur', 'ccy'],
    'impact': ['impact', 'importance', 'imp'],
    'event': ['event', 'title', 'name', 'description'],
}


def _find_col(cols: list[str], aliases: list[str]) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for a in aliases:
        if a in low:
            return low[a]
    return None


# Scheduled speeches/testimony have no exact-second data drop - straddling
# them is noise. Excluded by default (config: include_speeches).
SPEECH_PAT = r'Speaks|Speech|Testif|Press Conference'


def load_news(paths: list[str | Path], news_tz: str = 'US/Eastern',
              currencies: tuple = ('USD',),
              impacts: tuple = ('high', 'medium'),
              include_speeches: bool = False) -> pd.DataFrame:
    """Load one or more FF CSVs -> DataFrame with columns:
    [ts_utc, currency, impact, event]. ts_utc is tz-aware UTC Timestamp.

    Fast path: if the CSV has a `datetime_utc` column (ff_scraper output),
    it's parsed directly as UTC - no timezone guessing.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        cols = list(df.columns)

        if 'datetime_utc' in {c.lower().strip() for c in cols}:
            col = next(c for c in cols if c.lower().strip() == 'datetime_utc')
            out = pd.DataFrame({
                'ts_utc': pd.to_datetime(df[col], utc=True, errors='coerce'),
                'currency': df['currency'].astype(str).str.upper().str.strip(),
                'impact': (df['impact'].astype(str).str.lower().str.strip()
                           .map(IMPACT_MAP).fillna('unknown')),
                'event': df['event'].astype(str) if 'event' in df else '',
            }).dropna(subset=['ts_utc'])
            frames.append(out)
            continue
        mapping = {}
        for key, aliases in COL_ALIASES.items():
            col = _find_col(cols, aliases)
            if col is not None:
                mapping[key] = col

        missing = {'currency', 'impact'} - set(mapping)
        if missing:
            raise ValueError(f"{path}: can't find columns {missing}; has {cols}")

        # timestamp: either a single datetime column, or date + time
        if 'date' in mapping and 'time' in mapping:
            raw = df[mapping['date']].astype(str) + ' ' + df[mapping['time']].astype(str)
        elif 'date' in mapping:
            raw = df[mapping['date']].astype(str)
        else:
            raise ValueError(f"{path}: no date/time columns found; has {cols}")

        ts = pd.to_datetime(raw, errors='coerce', format='mixed')
        out = pd.DataFrame({
            'ts_local': ts,
            'currency': df[mapping['currency']].astype(str).str.upper().str.strip(),
            'impact': (df[mapping['impact']].astype(str).str.lower().str.strip()
                       .map(IMPACT_MAP).fillna('unknown')),
            'event': df[mapping['event']].astype(str) if 'event' in mapping else '',
        })
        out = out.dropna(subset=['ts_local'])
        # drop all-day / tentative rows that parsed to midnight with no time col
        frames.append(out)

    news = pd.concat(frames, ignore_index=True)
    if 'ts_utc' not in news.columns or news['ts_utc'].isna().all():
        news['ts_utc'] = (news['ts_local'].dt.tz_localize(news_tz, ambiguous='NaT',
                                                          nonexistent='NaT')
                          .dt.tz_convert('UTC'))
    news = news.dropna(subset=['ts_utc'])
    news = news[news['currency'].isin(currencies) & news['impact'].isin(impacts)]
    if not include_speeches:
        news = news[~news['event'].str.contains(SPEECH_PAT, case=False, na=False)]

    # collapse simultaneous releases (e.g. CPI + jobless claims same second)
    # into one event so we don't double-trade the same spike
    news = (news.sort_values('ts_utc')
                .groupby('ts_utc', as_index=False)
                .agg(currency=('currency', 'first'),
                     impact=('impact', lambda s: 'high' if 'high' in set(s) else s.iloc[0]),
                     event=('event', ' + '.join)))
    return news.reset_index(drop=True)
