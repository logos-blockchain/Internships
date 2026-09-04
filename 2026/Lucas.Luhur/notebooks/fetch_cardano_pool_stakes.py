"""
Fetch the Cardano pool active-stake vector for a given epoch from Koios.

Reconstructs the per-pool active stake at a historical epoch via /pool_history
(one call per pool over every pool id Koios knows) and writes it to
notebooks/data/cardano_pool_active_stake_epoch<E>.csv. The public tier allows
5,000 calls/day while a full run needs ~6,200, so a resume cache (JSON next to
the CSV) lets the script be re-run after the quota resets; with a free Koios
API key (--api-key or KOIOS_API_KEY) the vector fetches in one run. The CSV is
written only once every pool id has resolved.

Usage: python notebooks/fetch_cardano_pool_stakes.py --epoch 638 --api-key ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from consensus import gini

KOIOS = "https://api.koios.rest/api/v1"
PAGE = 1000

EPOCH = 638
API_KEY = ""

BURST_PENALTY = 60.0

_throttle = threading.Semaphore(1)
_last = [0.0]
_cache_lock = threading.Lock()
_stop = threading.Event()


class QuotaExhausted(RuntimeError):
    """429 persisted past the burst penalty window: the daily quota is gone."""


def _get(path, api_key, min_gap, tries=3):
    """
    GET a Koios path as JSON.

    A 429 that survives one burst-penalty wait is the daily quota, so it raises
    QuotaExhausted and sets the global stop flag.
    """
    for attempt in range(tries):
        if _stop.is_set():
            raise QuotaExhausted("stopped")
        with _throttle:
            wait = min_gap - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            headers = {"accept": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(KOIOS + path, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            if attempt == 0:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    pause = float(retry_after)
                except (TypeError, ValueError):
                    pause = BURST_PENALTY + 1.0
                deadline = time.time() + pause
                while time.time() < deadline and not _stop.is_set():
                    time.sleep(1.0)
                continue
            _stop.set()
            raise QuotaExhausted(f"HTTP 429 persisted for >{BURST_PENALTY:.0f}s")
        except (urllib.error.URLError, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def main():
    """Fetch the pool list, then each pool's active stake, and write the CSV."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epoch", type=int, default=EPOCH)
    ap.add_argument("--api-key", default=os.environ.get("KOIOS_API_KEY", API_KEY))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-gap", type=float, default=0.12,
                    help="seconds between requests globally (~1/rate)")
    args = ap.parse_args()

    data_dir = REPO_ROOT / "notebooks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_csv = data_dir / f"cardano_pool_active_stake_epoch{args.epoch}.csv"
    cache_path = data_dir / f"cardano_pool_history_epoch{args.epoch}_cache.json"

    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"resume cache: {len(cache)} pools already fetched")

    def flush_cache():
        """Write the resume cache to disk."""
        cache_path.write_text(json.dumps(cache))

    quota_msg = ("Koios daily quota exhausted (HTTP 429 past the 60 s burst "
                 "penalty). Public tier = 5,000 calls/day; a full run needs "
                 "~6,200. Re-run after the quota resets (cache resumes), or "
                 "pass --api-key / set KOIOS_API_KEY (free key, 50,000/day, "
                 "https://koios.rest) to finish in one go.")

    pool_ids, offset = [], 0
    while True:
        try:
            rows = _get(f"/pool_list?select=pool_id_bech32&offset={offset}"
                        f"&limit={PAGE}", args.api_key, args.min_gap)
        except QuotaExhausted:
            print(f"QUOTA EXHAUSTED before the pool list could be read. "
                  f"{quota_msg}")
            return 1
        if not rows:
            break
        pool_ids += [r["pool_id_bech32"] for r in rows]
        if len(rows) < PAGE:
            break
        offset += PAGE
    todo = [p for p in pool_ids if p not in cache]
    print(f"pool ids listed : {len(pool_ids)}  (to fetch: {len(todo)})")

    done = [0]

    def hist(pid):
        """Fetch one pool's active stake at the target epoch into the cache."""
        if _stop.is_set():
            return
        try:
            rows = _get(f"/pool_history?_pool_bech32={pid}"
                        f"&_epoch_no={args.epoch}", args.api_key, args.min_gap)
        except QuotaExhausted:
            return
        except Exception as e:
            print(f"  FAILED {pid}: {e}", flush=True)
            return
        stake = 0
        for row in rows or []:
            s = row.get("active_stake")
            if s and int(s) > 0:
                stake = int(s)
        with _cache_lock:
            cache[pid] = stake
            done[0] += 1
            if done[0] % 200 == 0:
                flush_cache()
                print(f"  {done[0]}/{len(todo)}", flush=True)

    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futs = [ex.submit(hist, pid) for pid in todo]
        # poll: on Windows a bare Future.result() blocks Ctrl+C
        while not all(f.done() for f in futs):
            fut_wait(futs, timeout=1.0)
            if _stop.is_set():
                break
    except KeyboardInterrupt:
        _stop.set()
        print("\ninterrupted: draining workers and saving the cache ...",
              flush=True)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        flush_cache()

    missing = [p for p in pool_ids if p not in cache]
    if missing:
        fetched = len(pool_ids) - len(missing)
        why = quota_msg if _stop.is_set() else "Re-run to resume."
        print(f"\nINCOMPLETE: {fetched}/{len(pool_ids)} pools cached, "
              f"{len(missing)} still unfetched. {why}")
        return 1

    stakes = sorted((v for v in cache.values() if v > 0), reverse=True)
    x = np.array(stakes, float)
    n = x.size
    alpha = x / x.sum()
    nc = int(np.searchsorted(np.cumsum(alpha), 0.5) + 1)
    g = gini(alpha)
    top_frac = nc / n
    k_share = 1.0 / (1.0 - np.log(0.5) / np.log(top_frac))
    print(f"\nepoch {args.epoch}: {n} pools with stake, "
          f"total {x.sum()/1e6:,.0f} ADA")
    print(f"pool-stake Gini  : {g:.3f}  -> k = {(1+g)/(2*g):.3f}")
    print(f"top-share        : top {100*top_frac:.1f}% hold 50%"
          f"  -> k = {k_share:.3f}")

    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# Cardano pool active stake (lovelace), Koios"
                    f" /pool_history at epoch {args.epoch},"
                    f" retrieved {retrieved}"])
        w.writerow(["active_stake_lovelace"])
        for v in x:
            w.writerow([int(v)])
    print(f"saved -> {out_csv.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
