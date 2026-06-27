"""
NKY 225 Constituent History — Verified from Wayback Machine
============================================================
Scrapes archived snapshots of the Nikkei Inc. component page
(indexes.nikkei.co.jp) from the Wayback Machine to build a
verified, date-wise constituent list for 2014–2026.

Outputs
-------
nky225_constituent_history.csv   — one row per (snapshot_date, ticker)
                                   with company name and in_index flag
nky225_verified_changes.csv      — inferred additions / removals between
                                   consecutive snapshots
nky225_annual_snapshots.csv      — wide format: tickers as columns,
                                   one row per snapshot date (1/0)
"""

import re
import time
import logging
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
OUT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOTS TO FETCH
# Chosen to bracket every annual review (effective first business day Oct)
# and cover ad-hoc changes.  Format: (wayback_timestamp, label)
# ─────────────────────────────────────────────────────────────────────────────
SNAPSHOTS = [
    # 2014 baseline
    ("20141115103957", "2014-11-15", "post-2014-review"),
    # 2015
    ("20150307164656", "2015-03-07", "mid-2015"),
    ("20150907162223", "2015-09-07", "pre-2015-review"),
    ("20151214053936", "2015-12-14", "post-2015-review"),
    # 2016
    ("20160909022313", "2016-09-09", "pre-2016-review"),
    ("20161119081216", "2016-11-19", "post-2016-review"),
    # 2017
    ("20170901134427", "2017-09-01", "pre-2017-review"),
    ("20180203081646", "2018-02-03", "post-2017-review"),
    # 2018 — only one snapshot in archive; use 2019 for post-2018
    ("20190716122638", "2019-07-16", "post-2018-review"),
    # 2019
    ("20191014204620", "2019-10-14", "post-2019-review"),
    # 2020 — COVID + annual review
    ("20200317230520", "2020-03-17", "covid-march-2020"),
    ("20201020172154", "2020-10-20", "post-2020-review"),
    # 2021
    ("20210225062159", "2021-02-25", "mid-2021"),
    ("20211022063639", "2021-10-22", "post-2021-review"),
    # 2022 — TSE restructure Apr 4 + annual review Oct
    ("20220129165452", "2022-01-29", "pre-tse-restructure"),
    ("20220906191321", "2022-09-06", "pre-2022-review"),
    # 2023
    ("20230328170656", "2023-03-28", "mid-2023"),
    ("20231109042318", "2023-11-09", "post-2023-review"),
    # 2024 — annual review Oct 1
    ("20240621105509", "2024-06-21", "pre-2024-review"),
    ("20241113184135", "2024-11-13", "post-2024-review"),
    # 2025
    ("20250417151003", "2025-04-17", "mid-2025"),
]

CACHE_DIR = OUT_DIR / "_cache" / "wayback_snapshots"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_snapshot(wayback_ts: str, snapshot_date: str) -> dict[str, str]:
    """
    Fetch one archived Nikkei component page and return {code: company_name}.
    Uses a local file cache so we never re-fetch.
    """
    cache_file = CACHE_DIR / f"{wayback_ts}.json"
    if cache_file.exists():
        import json
        result = json.loads(cache_file.read_text())
        log.info("  [cache] %s → %d constituents", snapshot_date, len(result))
        return result

    url = (
        f"https://web.archive.org/web/{wayback_ts}/"
        "https://indexes.nikkei.co.jp/en/nkave/index/component"
    )
    log.info("  Fetching %s (%s) …", snapshot_date, wayback_ts)

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        log.warning("    Failed: %s", e)
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    result = {}

    # Primary parse: div.col-xs-3.col-sm-1_5 contains the code,
    # the adjacent <a> tag contains the English company name.
    rows = soup.find_all("div", class_=re.compile(r"component-list"))
    for row in rows:
        code_div = row.find("div", class_=re.compile(r"col-xs-3|col-sm-1_5"))
        name_a   = row.find("a")
        if code_div:
            code = code_div.get_text(strip=True)
            name = name_a.get_text(strip=True) if name_a else ""
            if re.match(r"^\d{4}$", code):
                result[code] = name

    # Fallback: if that selector found nothing, try table rows
    if not result:
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                code = cells[0].get_text(strip=True)
                name = cells[1].get_text(strip=True)
                if re.match(r"^\d{4}$", code):
                    result[code] = name

    log.info("    → %d constituents parsed", len(result))

    if result:
        import json
        cache_file.write_text(json.dumps(result, ensure_ascii=False))

    time.sleep(1.5)   # be polite to Wayback Machine
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INFER CHANGES BETWEEN CONSECUTIVE SNAPSHOTS
# ─────────────────────────────────────────────────────────────────────────────

def infer_changes(snapshots_data: list[tuple]) -> pd.DataFrame:
    """
    snapshots_data: [(date_str, label, {code: name}), ...]
    Returns a DataFrame of inferred add/remove events between consecutive snapshots.
    """
    rows = []
    for i in range(1, len(snapshots_data)):
        prev_date, prev_label, prev_set = snapshots_data[i-1]
        curr_date, curr_label, curr_set = snapshots_data[i]

        prev_codes = set(prev_set.keys())
        curr_codes = set(curr_set.keys())

        added   = curr_codes - prev_codes
        removed = prev_codes - curr_codes

        for code in sorted(added):
            rows.append({
                "detected_between": f"{prev_date} → {curr_date}",
                "approx_date":      curr_date,
                "ticker":           code,
                "company_name":     curr_set.get(code, ""),
                "change_type":      "ADDED",
                "from_snapshot":    prev_label,
                "to_snapshot":      curr_label,
            })
        for code in sorted(removed):
            rows.append({
                "detected_between": f"{prev_date} → {curr_date}",
                "approx_date":      curr_date,
                "ticker":           code,
                "company_name":     prev_set.get(code, ""),
                "change_type":      "REMOVED",
                "from_snapshot":    prev_label,
                "to_snapshot":      curr_label,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("NKY 225 Constituent History Builder — Wayback Machine")
    log.info("Fetching %d snapshots …", len(SNAPSHOTS))
    log.info("=" * 60)

    # ── Fetch all snapshots ───────────────────────────────────────────────────
    snapshots_data = []
    for wayback_ts, snap_date, label in SNAPSHOTS:
        constituents = fetch_snapshot(wayback_ts, snap_date)
        if constituents:
            snapshots_data.append((snap_date, label, constituents))
        else:
            log.warning("  Skipping empty snapshot %s", snap_date)

    log.info("Successfully parsed %d / %d snapshots", len(snapshots_data), len(SNAPSHOTS))

    # ── Constituent counts per snapshot ──────────────────────────────────────
    log.info("\nConstituent counts by snapshot:")
    for date, label, data in snapshots_data:
        log.info("  %s  (%s)  →  %d constituents", date, label, len(data))

    # ── Infer change events ───────────────────────────────────────────────────
    changes_df = infer_changes(snapshots_data)
    log.info("\nInferred %d change events (additions + removals)", len(changes_df))
    if len(changes_df) > 0:
        log.info("\n%s", changes_df.to_string(index=False))

    # ── Build long-format constituent history ─────────────────────────────────
    # Collect all codes ever seen
    all_codes = sorted({
        code
        for _, _, data in snapshots_data
        for code in data.keys()
    })
    log.info("\nTotal unique tickers seen across all snapshots: %d", len(all_codes))

    # Build the universe of company names (latest known name per code)
    name_map = {}
    for _, _, data in snapshots_data:
        name_map.update(data)

    # Long format: one row per (snapshot_date, ticker)
    long_rows = []
    for snap_date, label, data in snapshots_data:
        for code in all_codes:
            long_rows.append({
                "snapshot_date": snap_date,
                "snapshot_label": label,
                "ticker":         code,
                "company_name":   name_map.get(code, ""),
                "in_index":       int(code in data),
            })

    long_df = pd.DataFrame(long_rows)

    # ── Wide format: snapshot_date × ticker (1/0) ────────────────────────────
    wide_df = long_df.pivot(
        index="snapshot_date", columns="ticker", values="in_index"
    ).fillna(0).astype(int)

    # Add company name as a second header row comment (in a separate lookup df)
    name_row = pd.DataFrame(
        [[name_map.get(c, "") for c in wide_df.columns]],
        columns=wide_df.columns,
        index=["company_name"],
    )
    wide_with_names = pd.concat([name_row, wide_df])

    # ── Save ──────────────────────────────────────────────────────────────────
    out_long    = OUT_DIR / "nky225_constituent_history.csv"
    out_changes = OUT_DIR / "nky225_verified_changes.csv"
    out_wide    = OUT_DIR / "nky225_annual_snapshots.csv"

    long_df.to_csv(out_long, index=False)
    changes_df.to_csv(out_changes, index=False)
    wide_with_names.to_csv(out_wide)

    log.info("\nSaved:")
    log.info("  %s  (%d rows)", out_long.name,    len(long_df))
    log.info("  %s  (%d events)", out_changes.name, len(changes_df))
    log.info("  %s  (%d snapshots × %d tickers)", out_wide.name, len(wide_df), len(wide_df.columns))

    # ── Summary of all unique tickers + names ─────────────────────────────────
    ticker_summary = (
        pd.DataFrame(
            [(code, name_map.get(code, "")) for code in sorted(name_map.keys())],
            columns=["ticker", "company_name"],
        )
        .sort_values("ticker")
    )
    ticker_summary.to_csv(OUT_DIR / "nky225_ticker_names.csv", index=False)
    log.info("  nky225_ticker_names.csv  (%d tickers with names)", len(ticker_summary))

    return long_df, changes_df, wide_df


if __name__ == "__main__":
    long_df, changes_df, wide_df = main()
