"""
ForexFactory economic-calendar scraper.

Scrapes the ForexFactory calendar week-by-week and keeps only the events you
care about (by default: HIGH and MEDIUM impact for USD, JPY and CNY over the
past 6 years).

Data source: the calendar page embeds every event as JSON inside the
`window.calendarComponentStates[...]` script block. We fetch each week, pull
that block out, and parse the individual event objects.

Usage (defaults = last 6 years, USD/JPY/CNY, high+medium):
    python ff_scraper.py

Common overrides:
    python ff_scraper.py --years 6
    python ff_scraper.py --start 2020-01-01 --end 2026-07-01
    python ff_scraper.py --currencies USD,JPY,CNY --impacts high,medium
    python ff_scraper.py --delay 2.5 --out ff_news.csv

The run is resumable: completed weeks are recorded in <out>.progress.json, so
re-running continues where it left off and never double-writes an event.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    import cloudscraper
except ImportError:
    sys.exit("Missing dependency. Run:  pip install cloudscraper requests")

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

CSV_FIELDS = [
    "event_id", "datetime_utc", "date_ff", "time_ff", "currency",
    "impact", "event", "actual", "forecast", "previous", "revision",
    "country", "week", "detail_url",
]

STATE_RE = re.compile(r"window\.calendarComponentStates\[\d+\]\s*=\s*")


def week_param(d):
    """ForexFactory week token, e.g. 2020-01-06 -> 'jan6.2020'."""
    return f"{MONTHS[d.month - 1]}{d.day}.{d.year}"


def iter_week_starts(start, end):
    """Yield Monday of each week from start to end (inclusive-ish)."""
    d = start - timedelta(days=start.weekday())  # back up to Monday
    while d <= end:
        yield d
        d += timedelta(days=7)


def extract_state_blob(html):
    """Return the substring of the first calendarComponentStates object."""
    m = STATE_RE.search(html)
    if not m:
        return None
    i = html.find("{", m.end())
    if i == -1:
        return None
    depth, in_str, esc = 0, False, False
    for j in range(i, len(html)):
        c = html[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return html[i:j + 1]
    return None


def iter_event_objects(blob):
    """Yield each event dict. Event objects start with `{"id":` and have
    fully quoted keys, so each one is valid JSON on its own."""
    for m in re.finditer(r'\{"id":', blob):
        start = m.start()
        depth, in_str, esc = 0, False, False
        for j in range(start, len(blob)):
            c = blob[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads(blob[start:j + 1])
                        except json.JSONDecodeError:
                            pass
                        break


def normalize_impact(ev):
    """Return 'high' | 'medium' | 'low' | 'holiday' | '' from an event."""
    name = (ev.get("impactName") or "").lower()
    if name in ("high", "medium", "low", "holiday"):
        return name
    cls = ev.get("impactClass") or ""
    if "red" in cls:
        return "high"
    if "ora" in cls:
        return "medium"
    if "yel" in cls:
        return "low"
    return name


def fetch_week(scraper, token, retries=4):
    url = f"https://www.forexfactory.com/calendar?week={token}"
    for attempt in range(1, retries + 1):
        try:
            r = scraper.get(url, timeout=40)
            if r.status_code == 200 and "calendarComponentStates" in r.text:
                return r.text
            print(f"    [{token}] status {r.status_code}, retry {attempt}/{retries}")
        except Exception as e:
            print(f"    [{token}] error {e!r}, retry {attempt}/{retries}")
        time.sleep(min(60, 5 * attempt) + random.uniform(0, 3))
    return None


def load_progress(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("done_weeks", [])), set(data.get("seen_ids", []))
    return set(), set()


def save_progress(path, done_weeks, seen_ids):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done_weeks": sorted(done_weeks),
                   "seen_ids": sorted(seen_ids)}, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Scrape ForexFactory calendar.")
    ap.add_argument("--years", type=int, default=6,
                    help="How many years back from today (default 6).")
    ap.add_argument("--start", help="Start date YYYY-MM-DD (overrides --years).")
    ap.add_argument("--end", help="End date YYYY-MM-DD (default: today).")
    ap.add_argument("--currencies", default="USD,JPY,CNY",
                    help="Comma-separated currencies to keep.")
    ap.add_argument("--impacts", default="high,medium",
                    help="Comma-separated impact levels to keep.")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Base seconds to wait between weeks (default 2.0).")
    ap.add_argument("--out", default="forexfactory_news.csv",
                    help="Output CSV path.")
    args = ap.parse_args()

    end = (datetime.strptime(args.end, "%Y-%m-%d").date()
           if args.end else datetime.now(timezone.utc).date())
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start = end.replace(year=end.year - args.years)

    currencies = {c.strip().upper() for c in args.currencies.split(",") if c.strip()}
    impacts = {i.strip().lower() for i in args.impacts.split(",") if i.strip()}

    progress_path = args.out + ".progress.json"
    done_weeks, seen_ids = load_progress(progress_path)

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )

    weeks = list(iter_week_starts(start, end))
    print(f"Range {start} -> {end}  ({len(weeks)} weeks)")
    print(f"Currencies: {sorted(currencies)}   Impacts: {sorted(impacts)}")
    print(f"Output: {args.out}\n")

    file_exists = os.path.exists(args.out)
    out_f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()

    kept_total = 0
    try:
        for n, monday in enumerate(weeks, 1):
            token = week_param(monday)
            if token in done_weeks:
                continue
            print(f"[{n}/{len(weeks)}] {token} ...", end=" ", flush=True)

            html = fetch_week(scraper, token)
            if html is None:
                print("FAILED (skipped, will retry on next run)")
                continue

            blob = extract_state_blob(html)
            kept_here = 0
            if blob:
                for ev in iter_event_objects(blob):
                    cur = (ev.get("currency") or "").upper()
                    imp = normalize_impact(ev)
                    if cur not in currencies or imp not in impacts:
                        continue
                    eid = ev.get("id")
                    if eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    dl = ev.get("dateline")
                    dt_utc = (datetime.fromtimestamp(dl, tz=timezone.utc)
                              .strftime("%Y-%m-%d %H:%M:%S") if dl else "")
                    writer.writerow({
                        "event_id": eid,
                        "datetime_utc": dt_utc,
                        "date_ff": ev.get("date", ""),
                        "time_ff": ev.get("timeLabel", ""),
                        "currency": cur,
                        "impact": imp,
                        "event": ev.get("name", ""),
                        "actual": ev.get("actual", ""),
                        "forecast": ev.get("forecast", ""),
                        "previous": ev.get("previous", ""),
                        "revision": ev.get("revision", ""),
                        "country": ev.get("country", ""),
                        "week": token,
                        "detail_url": "https://www.forexfactory.com"
                                      + (ev.get("url") or ""),
                    })
                    kept_here += 1
            out_f.flush()
            kept_total += kept_here
            done_weeks.add(token)
            save_progress(progress_path, done_weeks, seen_ids)
            print(f"{kept_here} kept  (total {kept_total})")

            time.sleep(args.delay + random.uniform(0, args.delay))
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Re-run to resume.")
    finally:
        out_f.close()

    print(f"\nDone. {kept_total} new rows this run -> {args.out}")


if __name__ == "__main__":
    main()
