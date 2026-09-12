"""Render result figures from a training report (light and dark variants).

Palette slots and surfaces come from a validated categorical/sequential set;
see RESULTS.md. Colours are assigned by the job they do: sequential blue for
the confusion matrix (magnitude), one hue for the single-series F1 bars, and
two categorical slots where two series share an axis.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

# sequential blue ramp, light -> dark (magnitude encoding)
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

THEME = {
    "light": dict(surface="#fcfcfb", plane="#f9f9f7", primary="#0b0b0b",
                  secondary="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", plane="#0d0d0d", primary="#ffffff",
                 secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926"),
}


def style(mode):
    t = THEME[mode]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": t["primary"], "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["axis"], "grid.color": t["grid"],
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "600",
    })
    return t


def rounded_bar(ax, x, y, w, h, color, r=0.06, horizontal=True):
    """Bar with rounded data-end, square against the baseline."""
    pad = min(r, abs(w if horizontal else h) * 0.5)
    box = FancyBboxPatch((x, y), max(w - pad, 1e-9) if horizontal else w,
                         h if horizontal else max(h - pad, 1e-9),
                         boxstyle=f"round,pad=0,rounding_size={pad}",
                         mutation_aspect=1 if horizontal else 0.02,
                         linewidth=0, facecolor=color, clip_on=False)
    ax.add_patch(box)


def fig_confusion(rep, mode, out):
    t = style(mode)
    cm = np.array(rep["test"]["confusion_matrix"], dtype=float)
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    for i in range(5):
        for j in range(5):
            # ink flips on dark cells so the number stays legible
            ink = "#ffffff" if pct[i, j] > 55 else t["secondary"]
            ax.text(j, i - 0.10, f"{pct[i,j]:.0f}%", ha="center", va="center",
                    color=ink, fontsize=11, fontweight="600")
            ax.text(j, i + 0.22, f"{int(cm[i,j]):,}", ha="center", va="center",
                    color=ink, fontsize=8, alpha=0.75)

    ax.set_xticks(range(5), CLASS_NAMES)
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("predicted", labelpad=8)
    ax.set_ylabel("true", labelpad=8)
    ax.set_title("Confusion matrix — row-normalised (recall)", loc="left", pad=14,
                 color=t["primary"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)

    fig.text(0.01, 0.02, f"153 recordings · 78 subjects · {int(cm.sum()):,} test epochs",
             color=t["muted"], fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_per_class_f1(rep, mode, out):
    t = style(mode)
    f1 = [rep["test"]["per_class_f1"][c] for c in CLASS_NAMES]
    sup = [rep["test"]["support"][c] for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ypos = np.arange(5)[::-1]
    for y, v in zip(ypos, f1):
        rounded_bar(ax, 0, y - 0.22, v, 0.44, t["s1"], r=0.014)
    for y, v, n in zip(ypos, f1, sup):
        ax.text(v + 0.016, y, f"{v:.3f}", va="center", ha="left",
                color=t["primary"], fontsize=10, fontweight="600")
        # support sits inside its own bar; outside it collides with the tick label
        ax.text(0.014, y, f"{n:,} epochs", va="center", ha="left",
                color="#ffffff", fontsize=8, alpha=0.9)

    ax.set_yticks(ypos, CLASS_NAMES)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlabel("F1")
    ax.set_title("Per-stage F1 — N1 is the hard class", loc="left", pad=12,
                 color=t["primary"])
    ax.xaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(t["axis"])
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_training(rep, mode, out):
    t = style(mode)
    h = rep["history"]
    ep = [e["epoch"] for e in h]
    trl = [e["train"]["loss"] for e in h]
    vll = [e["val"]["loss"] for e in h]
    bal = [e["val"]["balanced_accuracy"] for e in h]
    kap = [e["val"]["cohen_kappa"] for e in h]
    best = rep["best_epoch"]

    # two panels rather than one dual-axis chart: loss and the 0-1 metrics
    # do not share a scale
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8))

    a1.plot(ep, trl, color=t["s1"], linewidth=2, label="train")
    a1.plot(ep, vll, color=t["s2"], linewidth=2, label="validation")
    a1.set_title("Loss", loc="left", pad=10, color=t["primary"])
    a1.set_xlabel("epoch")
    a1.legend(frameon=False, labelcolor=t["secondary"], fontsize=9)

    a2.plot(ep, bal, color=t["s1"], linewidth=2, label="balanced accuracy")
    a2.plot(ep, kap, color=t["s2"], linewidth=2, label="Cohen's κ")
    lo = min(min(bal), min(kap)); hi = max(max(bal), max(kap))
    pad = (hi - lo) * 0.9 + 0.02
    a2.set_ylim(lo - pad, hi + pad * 0.5)        # range before placing any text on it
    a2.axvline(best, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)))
    a2.annotate(f"best epoch {best}", (best, a2.get_ylim()[1]),
                xytext=(5, -11), textcoords="offset points",
                color=t["muted"], fontsize=8)
    a2.set_title("Validation metrics", loc="left", pad=10, color=t["primary"])
    a2.set_xlabel("epoch")
    a2.legend(frameon=False, labelcolor=t["secondary"], fontsize=9, loc="lower right")

    for a in (a1, a2):
        a.yaxis.grid(True, linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(length=0)
        a.spines["bottom"].set_color(t["axis"])
        a.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hypnogram(y_true, y_pred, rec_id, mode, out):
    t = style(mode)
    # conventional hypnogram order: wake at the top, deep sleep at the bottom
    order = [0, 4, 1, 2, 3]
    labels = ["W", "REM", "N1", "N2", "N3"]
    row = {c: i for i, c in enumerate(order)}
    ty = np.array([row[v] for v in y_true])
    py = np.array([row[v] for v in y_pred])
    hrs = np.arange(len(ty)) * 30 / 3600

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 4.9), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1, 0.16],
                                                  "hspace": 0.42})
    for ax, series, color, name in ((a1, ty, t["s1"], "scored by technician"),
                                    (a2, py, t["s2"], "predicted")):
        ax.step(hrs, -series, where="post", color=color, linewidth=1.6)
        ax.set_yticks(-np.arange(5), labels)
        ax.set_ylim(-4.6, 0.6)
        ax.set_title(name, loc="left", pad=6, color=t["secondary"], fontsize=10)
        ax.yaxis.grid(True, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_color(t["axis"])
        ax.spines["left"].set_visible(False)

    # a ribbon rather than per-epoch dots on the trace, which buried the signal
    dis = (ty != py)
    a3.fill_between(hrs, 0, dis.astype(float), step="post",
                    color=t["muted"], linewidth=0)
    a3.set_ylim(0, 1)
    a3.set_yticks([])
    a3.set_title("disagreement", loc="left", pad=4, color=t["muted"], fontsize=9)
    a3.spines["bottom"].set_color(t["axis"])
    a3.spines["left"].set_visible(False)
    a3.tick_params(length=0)
    a3.set_xlabel("hours from start of record")
    agree = 100 * (1 - dis.mean())
    fig.suptitle(f"Hypnogram — {rec_id}   ({agree:.1f}% epoch agreement)",
                 x=0.008, ha="left", color=t["primary"], fontsize=12,
                 fontweight="600", y=1.02)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="results/full153_report.json")
    ap.add_argument("--checkpoint", default=None, help="for the hypnogram figure")
    ap.add_argument("--recording", default=None, help=".npz of a held-out recording")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rep = json.load(open(args.report))

    made = []
    for mode in ("light", "dark"):
        sfx = "" if mode == "light" else "-dark"
        for name, fn in (("confusion-matrix", fig_confusion),
                         ("per-class-f1", fig_per_class_f1),
                         ("training-curves", fig_training)):
            p = os.path.join(args.out_dir, f"{name}{sfx}.png")
            fn(rep, mode, p)
            made.append(p)

    if args.checkpoint and args.recording:
        import torch
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.sleep_resnet import SleepResNet1D

        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = SleepResNet1D(base_width=ck["args"]["base_width"])
        model.load_state_dict(ck["model"])
        model.eval()

        d = np.load(args.recording, allow_pickle=True)
        x = d["x"].astype(np.float32)
        med = np.float32(np.median(x))
        iqr = np.float32(np.subtract(*np.percentile(x, [75, 25])))
        x = ((x - med) / (iqr + np.float32(1e-6))).astype(np.float32)
        np.clip(x, -500, 500, out=x)

        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                batch = torch.from_numpy(x[i:i + 256]).unsqueeze(1)
                preds.append(model(batch).argmax(1).numpy())
        y_pred = np.concatenate(preds)

        for mode in ("light", "dark"):
            sfx = "" if mode == "light" else "-dark"
            p = os.path.join(args.out_dir, f"hypnogram{sfx}.png")
            fig_hypnogram(d["y"], y_pred, str(d["rec_id"]), mode, p)
            made.append(p)

    for p in made:
        print(f"  {os.path.getsize(p)//1024:>4} KB  {p}")


if __name__ == "__main__":
    main()
