"""Convert Sleep-EDF Expanded PSG/Hypnogram pairs into 30s epoch arrays."""
import argparse
import glob
import os
import re

import numpy as np

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

EPOCH_SEC = 30
SFREQ = 100
EPOCH_LEN = EPOCH_SEC * SFREQ


def pair_files(raw_dir):
    """Match *-PSG.edf with its *-Hypnogram.edf by the 6-char recording id."""
    psgs = sorted(glob.glob(os.path.join(raw_dir, "**", "*-PSG.edf"), recursive=True))
    hypnos = glob.glob(os.path.join(raw_dir, "**", "*-Hypnogram.edf"), recursive=True)

    by_id = {}
    for h in hypnos:
        key = os.path.basename(h)[:6]
        by_id[key] = h

    pairs = []
    for psg in psgs:
        key = os.path.basename(psg)[:6]
        if key in by_id:
            pairs.append((psg, by_id[key]))
    return pairs


def load_recording(psg_path, hypno_path, channel="EEG Fpz-Cz", crop_wake_min=30):
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    annots = mne.read_annotations(hypno_path)

    available = raw.ch_names
    if channel not in available:
        matches = [c for c in available if "Fpz" in c or "EEG" in c]
        if not matches:
            raise ValueError(f"no EEG channel in {psg_path}: {available}")
        channel = matches[0]
    raw.pick([channel])

    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose="ERROR")

    raw.set_annotations(annots, emit_warning=False)

    # Sleep-EDF cassette files are mostly wake at the head/tail; keep only
    # crop_wake_min of wake on each side of the annotated sleep period.
    scored = [a for a in annots if a["description"] in STAGE_MAP]
    if not scored:
        raise ValueError(f"no scored epochs in {hypno_path}")
    sleep = [a for a in scored if a["description"] != "Sleep stage W"]
    if sleep:
        pad = crop_wake_min * 60
        tmin = max(raw.times[0], sleep[0]["onset"] - raw.first_time - pad)
        tmax = min(raw.times[-1], sleep[-1]["onset"] + sleep[-1]["duration"] - raw.first_time + pad)
        raw.crop(tmin=max(0.0, tmin), tmax=tmax, verbose="ERROR")

    # MNE requires unique event codes, but scoring stages 3 and 4 both map to
    # N3. Give every description its own code and merge only after epoching.
    desc_to_code = {d: i + 1 for i, d in enumerate(STAGE_MAP)}
    code_to_class = {i + 1: STAGE_MAP[d] for i, d in enumerate(STAGE_MAP)}

    # chunk_duration splits the long single annotations real hypnograms use
    # (one span can cover hours of wake) into individual 30 s epochs.
    events, event_id = mne.events_from_annotations(
        raw, event_id=desc_to_code, chunk_duration=EPOCH_SEC, verbose="ERROR",
    )
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=0.0,
        tmax=EPOCH_SEC - 1.0 / SFREQ, baseline=None, preload=True, verbose="ERROR",
    )

    x = epochs.get_data(copy=True)[:, 0, :].astype(np.float32)
    y = np.array([code_to_class[c] for c in epochs.events[:, 2]], dtype=np.int64)

    if x.shape[1] != EPOCH_LEN:
        x = x[:, :EPOCH_LEN]

    x = x * 1e6  # volts -> microvolts
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--channel", default="EEG Fpz-Cz")
    ap.add_argument("--crop-wake-min", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = pair_files(args.raw_dir)
    if not pairs:
        raise SystemExit(f"no PSG/Hypnogram pairs found under {args.raw_dir}")

    print(f"found {len(pairs)} recordings")
    total = np.zeros(5, dtype=np.int64)
    written = 0

    for i, (psg, hypno) in enumerate(pairs, 1):
        rec_id = os.path.basename(psg)[:6]
        subject = int(re.search(r"SC4(\d{2})", rec_id).group(1)) if rec_id.startswith("SC4") else i
        out_path = os.path.join(args.out_dir, f"{rec_id}.npz")
        if os.path.exists(out_path):
            # count cached files too, otherwise a rerun reports a class
            # distribution covering only whatever it happened to redo
            with np.load(out_path, allow_pickle=True) as cached:
                total += np.bincount(cached["y"], minlength=5)
            print(f"[{i}/{len(pairs)}] {rec_id} exists, skipping")
            written += 1
            continue
        try:
            x, y = load_recording(psg, hypno, args.channel, args.crop_wake_min)
        except Exception as e:
            print(f"[{i}/{len(pairs)}] {rec_id} FAILED: {e}")
            continue
        np.savez_compressed(out_path, x=x, y=y, subject=subject, rec_id=rec_id)
        counts = np.bincount(y, minlength=5)
        total += counts
        written += 1
        print(f"[{i}/{len(pairs)}] {rec_id} subj={subject} epochs={len(y)} "
              + str({c: int(v) for c, v in zip(CLASS_NAMES, counts)}))

    print(f"\nwrote {written} files to {args.out_dir}")
    print("class totals:", {c: int(v) for c, v in zip(CLASS_NAMES, total)})


if __name__ == "__main__":
    main()
