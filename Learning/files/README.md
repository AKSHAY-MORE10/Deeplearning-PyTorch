# IMDb Sentiment Classifier — Bare-bones PyTorch

Binary sentiment classification on IMDb using a hand-written training loop.
No HuggingFace Trainer. Every step is explicit.

## Architecture
```
Embedding (20K vocab, 128-dim)
    ↓ Dropout(0.3)
Bi-LSTM (256-dim hidden, 2 layers)
    ↓ take final hidden states from both directions
    ↓ Dropout(0.3)
Linear (512 → 2)
    ↓
Logits → CrossEntropyLoss during training, Softmax during inference
```

## Setup
```bash
pip install torch datasets
```

## Run order
```bash
python train.py        # ~30-40 min on RTX 4060 for 5 epochs
python evaluate.py     # precision, recall, F1, confusion matrix
python inference.py    # test on your own reviews
```

## Expected results (after 5 epochs)
| Metric        | Value       |
|---------------|-------------|
| Accuracy      | ~87–89%     |
| Macro F1      | ~0.87–0.89  |
| Val loss      | ~0.28–0.32  |

## What you'll learn from each file
- `data_utils.py` → tokenization, vocabulary building, padding, DataLoader
- `model.py` → embedding layer, LSTM cell, bidirectionality, classifier head
- `train.py` → forward pass, CrossEntropyLoss, zero_grad, backward, clip_grad, Adam step
- `evaluate.py` → precision/recall/F1 from scratch, confusion matrix
- `inference.py` → logits vs probabilities, softmax, inference mode

## Concepts to be able to explain in an interview
- Why `zero_grad()` before `backward()`? (PyTorch accumulates gradients by default)
- What does `model.eval()` change? (disables dropout, fixes batchnorm)
- Why gradient clipping in RNNs? (prevents exploding gradients from long sequences)
- BLEU vs ROUGE vs accuracy — when to use which?
- What is the difference between logits and probabilities?
