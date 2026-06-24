"""
train.py
========
The full training loop written by hand. No HF Trainer, no Lightning.
Every single step is explicit so you can see exactly what happens.

The loop structure:
  for each epoch:
    for each batch:
      1. forward pass  → logits
      2. compute loss  → CrossEntropyLoss(logits, labels)
      3. zero gradients (important! PyTorch accumulates by default)
      4. backward pass → compute dL/dW for every parameter
      5. clip gradients (prevents exploding gradients in RNNs)
      6. optimizer step → update weights using Adam

Run:
  python train.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import json

from data_utils import get_dataloaders, MAX_VOCAB_SIZE
from model import SentimentLSTM, count_parameters

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS      = 5
LR          = 1e-3       # Adam learning rate
CLIP        = 1.0        # gradient clipping max norm
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
SAVE_PATH   = "best_model.pt"
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device, clip):
    """
    Runs one full pass over the training data.
    Returns: average loss over all batches, accuracy over all samples.
    """
    model.train()   # enable dropout, batch norm updates, etc.

    total_loss   = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        # Move data to GPU/CPU
        x, y = x.to(device), y.to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        logits = model(x)           # (batch, 2)

        # ── Loss ──────────────────────────────────────────────────────────────
        # CrossEntropyLoss = log_softmax + NLLLoss combined.
        # It expects raw logits (NOT softmaxed), and integer class labels.
        loss = criterion(logits, y)

        # ── Backward pass ─────────────────────────────────────────────────────
        optimizer.zero_grad()   # MUST clear old gradients before computing new ones
        loss.backward()         # computes dL/dW for every trainable parameter

        # ── Gradient clipping ─────────────────────────────────────────────────
        # LSTMs can have exploding gradients (product of many Jacobians).
        # Clipping rescales the gradient vector so its L2 norm ≤ clip.
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        # ── Optimizer step ────────────────────────────────────────────────────
        # Adam: adapts learning rate per parameter using first + second moments of gradients.
        # This is where the weights actually change.
        optimizer.step()

        # ── Accumulate metrics ────────────────────────────────────────────────
        total_loss    += loss.item() * x.size(0)   # weighted by batch size
        preds          = logits.argmax(dim=1)       # predicted class (0 or 1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            running_acc  = total_correct / total_samples
            running_loss = total_loss / total_samples
            print(f"    Batch {batch_idx+1:>4}/{len(loader)} | "
                  f"loss: {running_loss:.4f} | acc: {running_acc:.4f}")

    avg_loss = total_loss    / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()   # no gradient computation needed during eval — saves memory + time
def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a dataloader (validation or test).
    Returns: average loss, accuracy.
    """
    model.eval()   # disables dropout, fixes batch norm stats

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss    += loss.item() * x.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader, vocab = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SentimentLSTM(
        vocab_size = len(vocab),
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
        pad_idx    = vocab.word2idx["<PAD>"],
    ).to(device)

    print(f"\nModel: SentimentLSTM")
    print(f"  Trainable parameters: {count_parameters(model):,}")

    # ── Loss function ─────────────────────────────────────────────────────────
    # CrossEntropyLoss — standard for multi-class classification.
    # For binary classification you could also use BCEWithLogitsLoss (single output),
    # but CrossEntropyLoss with 2 outputs is equivalent and easier to extend.
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Adam: maintains per-parameter learning rates using gradient moments.
    # Good default for NLP tasks. Could also try AdamW (Adam + weight decay).
    optimizer = Adam(model.parameters(), lr=LR)

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    # Reduce learning rate by factor 0.5 if val loss doesn't improve for 1 epoch.
    # Helps squeeze out the last % of performance.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = []

    print(f"\nTraining for {EPOCHS} epochs...\n{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, CLIP
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  → Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"  → Val   loss: {val_loss:.4f}  | Val   acc: {val_acc:.4f}")
        print(f"  → LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"         : epoch,
                "model_state"   : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"      : val_loss,
                "val_acc"       : val_acc,
                "vocab_size"    : len(vocab),
                "embed_dim"     : EMBED_DIM,
                "hidden_dim"    : HIDDEN_DIM,
                "num_layers"    : NUM_LAYERS,
            }, SAVE_PATH)
            print(f"  ✓ Saved best model (val_loss improved to {val_loss:.4f})")

        # Step the scheduler
        scheduler.step(val_loss)

        # Log history
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc" : round(train_acc,  4),
            "val_loss"  : round(val_loss,   4),
            "val_acc"   : round(val_acc,    4),
        })

    # Save training history for plotting
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"History saved to: training_history.json")
    print(f"\nNext step: python evaluate.py")


if __name__ == "__main__":
    main()
