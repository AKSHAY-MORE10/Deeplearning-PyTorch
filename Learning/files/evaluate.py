"""
evaluate.py
===========
Loads the best saved model and computes full evaluation metrics on the test set:
  - Accuracy
  - Precision, Recall, F1 (per class + macro average)
  - Confusion matrix

Why these metrics matter for the Warewe job:
  - Accuracy alone is misleading on imbalanced data
  - Precision = "of all I predicted positive, how many were actually positive?"
  - Recall    = "of all actual positives, how many did I catch?"
  - F1        = harmonic mean of precision + recall (balances both)
  - The JD specifically mentions precision + recall — they will ask you to explain these

Run:
  python evaluate.py
"""

import torch
import torch.nn as nn
from collections import defaultdict

from data_utils import get_dataloaders
from model import SentimentLSTM


def compute_metrics(preds: list[int], labels: list[int], num_classes: int = 2):
    """
    Compute precision, recall, F1 per class, then macro average.
    Written from scratch (no sklearn) so you understand the math.

    Precision for class c = TP_c / (TP_c + FP_c)
    Recall    for class c = TP_c / (TP_c + FN_c)
    F1        for class c = 2 * P_c * R_c / (P_c + R_c)
    """
    # Confusion matrix: cm[true][pred] = count
    cm = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1

    metrics = {}
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes)) - tp  # predicted c, wasn't c
        fn = sum(cm[c][r] for r in range(num_classes)) - tp  # was c, predicted something else

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        metrics[c] = {"precision": precision, "recall": recall, "f1": f1,
                      "tp": tp, "fp": fp, "fn": fn}

    # Macro average: unweighted mean across classes
    macro_p  = sum(metrics[c]["precision"] for c in range(num_classes)) / num_classes
    macro_r  = sum(metrics[c]["recall"]    for c in range(num_classes)) / num_classes
    macro_f1 = sum(metrics[c]["f1"]        for c in range(num_classes)) / num_classes

    return metrics, (macro_p, macro_r, macro_f1), cm


@torch.no_grad()
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    _, test_loader, vocab = get_dataloaders()

    # ── Load model checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load("best_model.pt", map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Saved val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Saved val acc : {checkpoint['val_acc']:.4f}\n")

    model = SentimentLSTM(
        vocab_size = checkpoint["vocab_size"],
        embed_dim  = checkpoint["embed_dim"],
        hidden_dim = checkpoint["hidden_dim"],
        num_layers = checkpoint["num_layers"],
        pad_idx    = vocab.word2idx["<PAD>"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ── Collect all predictions ───────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds  = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.tolist())

    # ── Compute metrics ───────────────────────────────────────────────────────
    accuracy      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    per_class, (macro_p, macro_r, macro_f1), cm = compute_metrics(all_preds, all_labels)

    # ── Print results ─────────────────────────────────────────────────────────
    class_names = ["Negative", "Positive"]

    print("=" * 55)
    print(f"{'EVALUATION RESULTS':^55}")
    print("=" * 55)
    print(f"\n  Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)\n")

    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*45}")
    for c, name in enumerate(class_names):
        m = per_class[c]
        print(f"  {name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print(f"  {'-'*45}")
    print(f"  {'Macro avg':<12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>12} {'Pred NEG':>10} {'Pred POS':>10}")
    for c, name in enumerate(class_names):
        print(f"  {name+' (actual)':<12} {cm[c][0]:>10} {cm[c][1]:>10}")

    print(f"\n  What this means:")
    neg = per_class[0]
    pos = per_class[1]
    print(f"  → Of all reviews predicted NEGATIVE: {neg['precision']*100:.1f}% were actually negative")
    print(f"  → Of all actual NEGATIVE reviews:    {neg['recall']*100:.1f}% were correctly identified")
    print(f"  → Of all reviews predicted POSITIVE: {pos['precision']*100:.1f}% were actually positive")
    print(f"  → Of all actual POSITIVE reviews:    {pos['recall']*100:.1f}% were correctly identified")
    print("=" * 55)
    print("\nNext step: python inference.py")


if __name__ == "__main__":
    run_evaluation()
