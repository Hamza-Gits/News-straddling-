# ForexFactory Calendar Scraper

Scrapes the [ForexFactory](https://www.forexfactory.com/calendar) economic
calendar week-by-week and keeps only the events you want. Defaults to
**HIGH + MEDIUM** impact for **USD, JPY, CNY** over the **past 6 years**.

## How it works

Each calendar week page embeds all its events as JSON inside a
`window.calendarComponentStates[...]` script block. The scraper fetches one
week at a time, extracts that block, parses each event, and filters by currency
and impact. Cloudflare is handled by `cloudscraper`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Defaults (last 6 years, USD/JPY/CNY, high+medium):

```bash
python ff_scraper.py
```

The full 6-year run is ~310 weekly requests. With the default ~3s spacing that
is roughly 15–25 minutes. Be patient and don't lower `--delay` too far, or
Cloudflare may start throttling.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--years N` | `6` | Years back from today |
| `--start YYYY-MM-DD` | — | Explicit start (overrides `--years`) |
| `--end YYYY-MM-DD` | today | Explicit end |
| `--currencies` | `USD,JPY,CNY` | Currencies to keep |
| `--impacts` | `high,medium` | Impact levels (`high,medium,low,holiday`) |
| `--delay S` | `2.0` | Base seconds between weeks (actual wait is random 1–2×) |
| `--out PATH` | `forexfactory_news.csv` | Output CSV |

Example:

```bash
python ff_scraper.py --start 2020-01-01 --end 2026-07-01 --delay 3
```

## Output

CSV with one row per event:

| column | notes |
|--------|-------|
| `event_id` | ForexFactory event id (used for dedup) |
| `datetime_utc` | Event time in **UTC** (from the unix `dateline`) |
| `date_ff`, `time_ff` | Date/time labels as shown on the site |
| `currency` | USD / JPY / CNY |
| `impact` | high / medium |
| `event` | Event name |
| `actual`, `forecast`, `previous`, `revision` | Released values (blank if not yet released) |
| `country` | FF country code (China = `CH`) |
| `week` | Week token that produced the row |
| `detail_url` | Link to the event on ForexFactory |

## Resuming

Progress is stored in `<out>.progress.json` (completed weeks + seen event ids).
If the run is interrupted (Ctrl-C, network drop, crash), just run the same
command again — it skips finished weeks and never writes an event twice. Delete
the `.progress.json` file to force a clean re-scrape.

## Notes / caveats

- **Personal / research use.** Scrape politely (keep the delay). Heavy hammering
  can get your IP rate-limited by Cloudflare.
- `datetime_utc` is timezone-unambiguous because it comes from the event's unix
  timestamp; `time_ff` reflects ForexFactory's own display timezone.
- Events far in the past may lack `forecast`; upcoming events lack `actual`.
