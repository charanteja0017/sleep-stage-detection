"""Parallel downloader for Sleep-EDF Expanded (sleep-cassette).

MNE's own fetcher is sequential and PhysioNet serves a single stream at
~40 kB/s, so the full 153-recording set takes days. Several concurrent
connections get a few times that. Filenames and SHA1s come from the manifest
shipped inside MNE, and every file is checksum-verified.
"""
import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# PhysioNet's own host serves ~40 kB/s per connection. Its AWS Open Data mirror
# carries the identical files roughly 30x faster, so try that first and fall
# back. SHA1s from the manifest verify whichever one answers.
MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette",
    "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette",
)
_print_lock = threading.Lock()


def load_manifest():
    import mne.datasets.sleep_physionet as sp
    import pandas as pd

    path = os.path.join(os.path.dirname(sp.__file__), "age_records.csv")
    return pd.read_csv(path)


def sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(fname, expected_sha, out_dir, retries=2):
    dest = os.path.join(out_dir, fname)
    if os.path.exists(dest) and sha1(dest) == expected_sha:
        return fname, "cached", os.path.getsize(dest)

    tmp = f"{dest}.part"
    last = "no attempt"
    for base in MIRRORS:
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(
                    f"{base}/{fname}", headers={"User-Agent": "sleep-stage-detection/1.0"})
                with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                    while True:
                        block = r.read(1 << 16)
                        if not block:
                            break
                        f.write(block)
                got = sha1(tmp)
                if got != expected_sha:
                    os.remove(tmp)
                    last = f"sha1 mismatch ({got[:8]} != {expected_sha[:8]})"
                    continue
                os.replace(tmp, dest)
                return fname, "ok", os.path.getsize(dest)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                last = f"{type(e).__name__}: {e}"
                time.sleep(attempt)
    return fname, f"failed ({last})", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--subjects", type=int, default=None,
                    help="number of subjects to fetch (default: all 78)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_manifest()

    subjects = sorted(df.subject.unique())
    if args.subjects:
        subjects = subjects[:args.subjects]
    sel = df[df.subject.isin(subjects)]

    jobs = list(zip(sel["fname"], sel["sha"]))
    total_recordings = (sel["record type"] == "PSG").sum()
    print(f"{len(subjects)} subjects | {total_recordings} recordings | {len(jobs)} files "
          f"| {args.workers} workers", flush=True)

    t0 = time.time()
    done = bytes_got = failed = cached = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, f, s, args.out_dir) for f, s in jobs]
        for fut in as_completed(futures):
            name, status, size = fut.result()
            done += 1
            if status == "ok":
                bytes_got += size
            elif status == "cached":
                cached += 1
            else:
                failed += 1
            el = time.time() - t0
            rate = bytes_got / el / 1e6 if el > 0 else 0
            with _print_lock:
                mark = "ok  " if status in ("ok", "cached") else "FAIL"
                print(f"[{done}/{len(jobs)}] {mark} {name:<28} {status:<12} "
                      f"{rate:.2f} MB/s agg", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min | downloaded {bytes_got/1e6:.0f} MB "
          f"| cached {cached} | failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
