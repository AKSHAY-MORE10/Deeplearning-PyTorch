"""
inference.py
============
Load the trained model and run predictions on your own text.
Also shows the raw logits and probabilities so you understand
what the model is actually outputting.

Run:
  python inference.py
"""

import torch
import torch.nn.functional as F

from data_utils import get_dataloaders, clean_text, MAX_SEQ_LEN, Vocabulary
from model import SentimentLSTM


def predict(texts: list[str], model, vocab: Vocabulary, device) -> None:
    """
    Takes a list of raw review strings, runs them through the full pipeline,
    and prints the prediction with confidence.
    """
    model.eval()

    with torch.no_grad():
        for text in texts:
            # ── Same preprocessing as training ────────────────────────────────
            tokens = clean_text(text)
            ids    = vocab.encode(tokens)
            ids    = ids[:MAX_SEQ_LEN]
            ids    += [Vocabulary.PAD_IDX] * (MAX_SEQ_LEN - len(ids))

            # Add batch dimension: (seq_len,) → (1, seq_len)
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

            # ── Forward pass ──────────────────────────────────────────────────
            logits = model(x)          # (1, 2) — raw scores for [negative, positive]

            # ── Convert to probabilities via softmax ──────────────────────────
            # softmax(logits) → each value in [0,1], sums to 1
            probs = F.softmax(logits, dim=1).squeeze()   # (2,)
            pred  = probs.argmax().item()                 # 0 = negative, 1 = positive

            label  = "POSITIVE ✓" if pred == 1 else "NEGATIVE ✗"
            conf   = probs[pred].item() * 100

            print(f"\nText    : {text[:80]}{'...' if len(text)>80 else ''}")
            print(f"Logits  : neg={logits[0][0].item():.3f}  pos={logits[0][1].item():.3f}")
            print(f"Probs   : neg={probs[0].item():.3f}  pos={probs[1].item():.3f}")
            print(f"Result  : {label}  (confidence: {conf:.1f}%)")
            print(f"Tokens seen in vocab: {sum(1 for t in tokens if t in vocab.word2idx)}/{len(tokens)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab (we need the same vocab used during training)
    _, _, vocab = get_dataloaders()

    # Load model
    checkpoint = torch.load("best_model.pt", map_location=device)
    model = SentimentLSTM(
        vocab_size = checkpoint["vocab_size"],
        embed_dim  = checkpoint["embed_dim"],
        hidden_dim = checkpoint["hidden_dim"],
        num_layers = checkpoint["num_layers"],
        pad_idx    = vocab.word2idx["<PAD>"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print("=" * 60)
    print("IMDb Sentiment Classifier — Interactive Inference")
    print("=" * 60)

    # Test with diverse examples
    test_reviews = [
        "This movie was absolutely incredible. The acting was superb and the story kept me on the edge of my seat the entire time.",
        "Terrible waste of time. The plot made no sense and the acting was wooden. I want my two hours back.",
        "It was okay. Some parts were interesting but overall it felt a bit too long and the ending was disappointing.",
        "One of the best films I've seen in years. Brilliant cinematography and an emotional story that will stay with me forever.",
        "Boring, slow, predictable. The director clearly had no idea what they were doing.",
    ]

    predict(test_reviews, model, vocab, device)

    # Interactive mode
    print("\n" + "=" * 60)
    print("Try your own review (type 'quit' to exit):")
    while True:
        user_input = input("\nEnter review: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input:
            predict([user_input], model, vocab, device)


if __name__ == "__main__":
    main()
