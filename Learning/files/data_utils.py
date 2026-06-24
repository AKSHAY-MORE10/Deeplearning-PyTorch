"""
data_utils.py
=============
Loads IMDb dataset from HuggingFace, builds a vocabulary from scratch,
and converts raw text → padded integer sequences.

Why no HF Trainer? We want to own every step:
  raw text → tokens → integer IDs → padded tensor → model → logits → loss
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset

# ── Hyperparams you'll want to tweak ──────────────────────────────────────────
MAX_VOCAB_SIZE = 20_000   # keep only the N most common words
MAX_SEQ_LEN    = 256      # truncate/pad all sequences to this length
BATCH_SIZE     = 64
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> list[str]:
    """
    Minimal tokenizer: lowercase, strip HTML tags (IMDb has <br /> a lot),
    keep only letters and spaces, split on whitespace.

    Returns a list of tokens, e.g. ["this", "movie", "was", "great"]
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)        # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)        # keep only letters
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text.split()


class Vocabulary:
    """
    Builds a word → integer mapping from a list of tokenized sentences.

    Special tokens:
      <PAD>  → 0  (used to fill shorter sequences to MAX_SEQ_LEN)
      <UNK>  → 1  (used for words not seen during training)
    """
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = Counter()

    def build(self, tokenized_texts: list[list[str]], max_size: int = MAX_VOCAB_SIZE):
        """Count all tokens across all training texts, keep top max_size."""
        for tokens in tokenized_texts:
            self.word_counts.update(tokens)

        # most_common returns [(word, count), ...] sorted by frequency
        for word, _ in self.word_counts.most_common(max_size - 2):  # -2 for PAD + UNK
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  Vocabulary built: {len(self.word2idx):,} tokens")

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to a list of integer IDs."""
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def __len__(self):
        return len(self.word2idx)


class IMDbDataset(Dataset):
    """
    PyTorch Dataset wrapping the IMDb split.

    Each item is:
      x  → LongTensor of shape (MAX_SEQ_LEN,)  — padded/truncated token IDs
      y  → LongTensor scalar                   — 0 = negative, 1 = positive
    """
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary):
        self.vocab  = vocab
        self.labels = labels
        # Pre-process all texts once at init time (faster than doing it per __getitem__)
        self.encoded = [self._process(t) for t in texts]

    def _process(self, text: str) -> torch.Tensor:
        tokens  = clean_text(text)
        ids     = self.vocab.encode(tokens)
        # Truncate if too long
        ids     = ids[:MAX_SEQ_LEN]
        # Pad with 0s if too short
        padding = [Vocabulary.PAD_IDX] * (MAX_SEQ_LEN - len(ids))
        ids     = ids + padding
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def get_dataloaders(batch_size: int = BATCH_SIZE):
    """
    Downloads IMDb (cached after first run), builds vocab on train split,
    returns (train_loader, test_loader, vocab).

    The vocab is built ONLY on train text — never look at test data when
    building your vocabulary (that's data leakage).
    """
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    train_texts  = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts   = dataset["test"]["text"]
    test_labels  = dataset["test"]["label"]

    print(f"  Train: {len(train_texts):,} samples | Test: {len(test_texts):,} samples")

    # Build vocabulary on TRAIN only
    print("Building vocabulary...")
    vocab = Vocabulary()
    tokenized_train = [clean_text(t) for t in train_texts]
    vocab.build(tokenized_train)

    # Create Dataset objects
    print("Encoding and padding sequences...")
    train_ds = IMDbDataset(train_texts, train_labels, vocab)
    test_ds  = IMDbDataset(test_texts,  test_labels,  vocab)

    # DataLoaders — shuffle only the train set
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Batches — Train: {len(train_loader)} | Test: {len(test_loader)}")
    return train_loader, test_loader, vocab
