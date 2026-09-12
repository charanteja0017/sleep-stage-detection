"""Sleep-staging metrics: balanced accuracy, Cohen's kappa, per-class F1."""
import numpy as np

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def confusion_matrix(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def accuracy(cm):
    return np.trace(cm) / max(cm.sum(), 1)


def balanced_accuracy(cm):
    """Mean per-class recall; classes absent from y_true are excluded."""
    support = cm.sum(axis=1)
    present = support > 0
    if not present.any():
        return 0.0
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(cm)), where=support > 0)
    return float(recall[present].mean())


def cohen_kappa(cm):
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    if np.isclose(pe, 1.0):
        return 0.0
    return float((po - pe) / (1.0 - pe))


def per_class_f1(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(cm):
    support = cm.sum(axis=1)
    f1 = per_class_f1(cm)
    return float(f1[support > 0].mean()) if (support > 0).any() else 0.0


def summarize(y_true, y_pred, num_classes=5):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return {
        "accuracy": float(accuracy(cm)),
        "balanced_accuracy": balanced_accuracy(cm),
        "cohen_kappa": cohen_kappa(cm),
        "macro_f1": macro_f1(cm),
        "per_class_f1": {CLASS_NAMES[i]: float(v) for i, v in enumerate(per_class_f1(cm))},
        "support": {CLASS_NAMES[i]: int(v) for i, v in enumerate(cm.sum(axis=1))},
        "confusion_matrix": cm.tolist(),
    }


def format_report(m):
    lines = [
        f"  accuracy          {m['accuracy']:.4f}",
        f"  balanced accuracy {m['balanced_accuracy']:.4f}",
        f"  Cohen's kappa     {m['cohen_kappa']:.4f}",
        f"  macro F1          {m['macro_f1']:.4f}",
        "",
        "  stage   F1     support",
    ]
    for name in CLASS_NAMES:
        lines.append(f"  {name:<6}  {m['per_class_f1'][name]:.3f}  {m['support'][name]:>7,}")
    lines += ["", "  confusion (rows=true, cols=pred)", "          " + "".join(f"{c:>8}" for c in CLASS_NAMES)]
    for i, row in enumerate(m["confusion_matrix"]):
        lines.append(f"  {CLASS_NAMES[i]:<6}" + "".join(f"{v:>8,}" for v in row))
    return "\n".join(lines)
