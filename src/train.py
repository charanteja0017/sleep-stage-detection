"""Train SleepResNet1D with mixed precision + torch.compile."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sleep_resnet import SleepResNet1D, count_parameters
from utils.dataset import make_loaders, class_weights, CLASS_NAMES
from utils.metrics import summarize, format_report


def use_non_blocking(device):
    """Async H2D copies are only safe (and only help) with pinned memory on CUDA.
    On MPS a non_blocking copy can be read before it lands, yielding garbage."""
    return device.type == "cuda"


def pick_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_setup(device, enabled=True):
    """Return (autocast_kwargs, scaler). Only CUDA gets a real GradScaler."""
    if not enabled:
        return dict(device_type=device.type, enabled=False), None
    if device.type == "cuda":
        bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
        scaler = None if bf16 else torch.amp.GradScaler("cuda")
        return dict(device_type="cuda", dtype=dtype, enabled=True), scaler
    if device.type == "cpu":
        return dict(device_type="cpu", dtype=torch.bfloat16, enabled=True), None
    # MPS autocast is fp16-only and still patchy for 1D convs; run fp32.
    return dict(device_type=device.type, enabled=False), None


class LabelSmoothedCE(nn.Module):
    def __init__(self, weight=None, smoothing=0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

    def forward(self, logits, target):
        return self.ce(logits, target)


def cosine_warmup(optimizer, warmup_steps, total_steps, min_ratio=0.02):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, prog)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, ac_kwargs, criterion=None):
    model.eval()
    nb = use_non_blocking(device)
    preds, trues, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        # keep labels on the CPU: a non_blocking round-trip to the accelerator
        # and straight back can be read before the copy lands (garbage on MPS).
        y_cpu = y.numpy().copy()
        x = x.to(device, non_blocking=nb)
        y = y.to(device, non_blocking=nb)
        with torch.autocast(**ac_kwargs):
            logits = model(x)
            if criterion is not None:
                loss_sum += criterion(logits, y).item() * y.size(0)
        preds.append(logits.float().argmax(1).cpu().numpy().copy())
        trues.append(y_cpu)
        n += y.size(0)
    m = summarize(np.concatenate(trues), np.concatenate(preds))
    m["loss"] = loss_sum / max(n, 1)
    return m


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    ac_kwargs, scaler, grad_clip=1.0, log_every=50):
    model.train()
    running, seen, correct = 0.0, 0, 0
    nb = use_non_blocking(device)
    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=nb)
        y = y.to(device, non_blocking=nb)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**ac_kwargs):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        bs = y.size(0)
        running += loss.item() * bs
        seen += bs
        correct += (logits.float().argmax(1) == y).sum().item()

        if log_every and step % log_every == 0:
            print(f"    step {step:>4}/{len(loader)}  loss {running/seen:.4f}  "
                  f"acc {correct/seen:.4f}  lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

    return {"loss": running / max(seen, 1), "acc": correct / max(seen, 1),
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base-width", type=int, default=34)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--loss-weight-scheme", default="effective",
                    choices=["effective", "inverse", "sqrt_inverse", "none"])
    # Defaults to none: stacking a balanced sampler on top of the weighted loss
    # corrects the imbalance twice. Measured over 30 subjects it cost 0.03 kappa
    # for no balanced-accuracy gain. See RESULTS.md.
    ap.add_argument("--sampler-scheme", default="none",
                    choices=["sqrt_inverse", "inverse", "none"])
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--force-compile", action="store_true",
                    help="compile even on non-CUDA devices (slow/unreliable on MPS)")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = pick_device(args.device)
    ac_kwargs, scaler = amp_setup(device, enabled=not args.no_amp)
    print(f"device: {device} | amp: {ac_kwargs.get('enabled', False)} "
          f"({ac_kwargs.get('dtype', 'fp32')}) | scaler: {scaler is not None}")

    train_loader, val_loader, test_loader, meta = make_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, sampler_scheme=None if args.sampler_scheme == "none" else args.sampler_scheme,
    )
    print("data:", json.dumps(meta, indent=2))

    model = SleepResNet1D(num_classes=5, base_width=args.base_width,
                          dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"model: SleepResNet1D  params {n_params:,}")

    # torch.compile only pays off on CUDA. Measured on this MPS box it was ~50x
    # slower and the model failed to leave chance accuracy, so it is opt-in there.
    want_compile = not args.no_compile and (device.type == "cuda" or args.force_compile)
    if want_compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile unavailable ({e}); continuing eager")
    elif not args.no_compile:
        print(f"torch.compile: skipped on {device.type} (use --force-compile to override)")

    w = None
    if args.loss_weight_scheme != "none":
        w = class_weights(meta["train_class_counts"], args.loss_weight_scheme).to(device)
        print("loss weights:", {c: round(float(v), 3) for c, v in zip(CLASS_NAMES, w)})
    criterion = LabelSmoothedCE(weight=w, smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_warmup(optimizer, warmup_steps=len(train_loader), total_steps=total_steps)

    ckpt_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
    history, best, bad_epochs = [], -1.0, 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                             device, ac_kwargs, scaler)
        va = evaluate(model, val_loader, device, ac_kwargs, criterion)
        print(f"  train loss {tr['loss']:.4f} acc {tr['acc']:.4f} ({tr['seconds']:.1f}s)")
        print(f"  val   loss {va['loss']:.4f} bal_acc {va['balanced_accuracy']:.4f} "
              f"kappa {va['cohen_kappa']:.4f}")

        history.append({"epoch": epoch, "train": tr,
                        "val": {k: v for k, v in va.items() if k != "confusion_matrix"}})

        score = va["balanced_accuracy"]
        if score > best:
            best, bad_epochs = score, 0
            state = getattr(model, "_orig_mod", model).state_dict()
            torch.save({"model": state, "args": vars(args), "epoch": epoch,
                        "val": va, "params": n_params}, ckpt_path)
            print(f"  saved best ({score:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop: no val gain in {args.patience} epochs")
                break

    print("\nloading best checkpoint for test evaluation")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    getattr(model, "_orig_mod", model).load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device, ac_kwargs)

    print(f"\nTEST (best epoch {ckpt['epoch']})")
    print(format_report(test))

    report = {"args": vars(args), "device": str(device), "params": n_params,
              "data": meta, "best_val_balanced_accuracy": best,
              "best_epoch": ckpt["epoch"], "test": test, "history": history}
    out = os.path.join(args.out_dir, f"{args.tag}_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
