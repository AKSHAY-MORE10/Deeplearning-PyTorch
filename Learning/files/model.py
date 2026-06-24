"""
model.py
========
LSTM-based binary sentiment classifier, built entirely in PyTorch.

Architecture:
  Embedding → Dropout → Bi-LSTM → Dropout → Linear → (logits)

Why LSTM and not a Transformer for this exercise?
  → Forces you to understand: embedding lookup, recurrent hidden states,
    how sequence models "remember" earlier tokens via the cell state.
  → Transformers are next; nailing the LSTM first makes attention click.

Why Bidirectional?
  → A forward LSTM at position t only sees tokens 0..t.
    A backward LSTM sees tokens t..T. Concatenating both doubles the
    information available at each timestep. For classification we only
    care about the final representation, so we concat the final hidden
    states of both directions.
"""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """
    Args:
        vocab_size   : size of vocabulary (including <PAD> and <UNK>)
        embed_dim    : dimension of each token's embedding vector
        hidden_dim   : number of hidden units in each LSTM direction
        num_layers   : how many LSTM layers to stack
        dropout      : dropout probability applied after embedding + between LSTM layers
        pad_idx      : the integer ID of <PAD>; its embedding is frozen at zero
    """
    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int = 128,
        hidden_dim : int = 256,
        num_layers : int = 2,
        dropout    : float = 0.3,
        pad_idx    : int = 0,
    ):
        super().__init__()

        # ── 1. Embedding layer ────────────────────────────────────────────────
        # Maps each integer token ID → a dense vector of size embed_dim.
        # padding_idx=pad_idx tells PyTorch: keep that row as all-zeros,
        # and don't update it during backprop (makes sense — <PAD> carries no meaning).
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = pad_idx,
        )

        # ── 2. Dropout after embedding ────────────────────────────────────────
        # Randomly zeros out entire embedding vectors during training.
        # Prevents the model from over-relying on specific word embeddings.
        self.embed_dropout = nn.Dropout(dropout)

        # ── 3. Bi-directional LSTM ────────────────────────────────────────────
        # input_size  = embed_dim (what each timestep feeds in)
        # hidden_size = hidden_dim (the "memory" size per direction)
        # batch_first = True → input shape is (batch, seq_len, embed_dim)
        #                       output shape is (batch, seq_len, hidden_dim*2)
        # bidirectional = True → runs two LSTMs: one L→R, one R→L
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        # ── 4. Dropout before classifier head ────────────────────────────────
        self.fc_dropout = nn.Dropout(dropout)

        # ── 5. Linear classifier ──────────────────────────────────────────────
        # Input: concatenated final hidden states from both directions
        #        → size = hidden_dim * 2
        # Output: 2 logits (negative, positive)
        #         We output raw logits — loss fn (CrossEntropyLoss) handles softmax.
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : LongTensor of shape (batch_size, seq_len) — padded token IDs

        Returns:
            logits : FloatTensor of shape (batch_size, 2)
        """
        # x: (batch, seq_len)

        # Step 1: Embed
        embedded = self.embedding(x)           # → (batch, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        # Step 2: Run through LSTM
        # output : (batch, seq_len, hidden_dim*2)  — hidden state at each timestep
        # hidden : (num_layers*2, batch, hidden_dim) — final hidden state per layer/direction
        # cell   : (num_layers*2, batch, hidden_dim) — final cell state (we don't use this)
        output, (hidden, cell) = self.lstm(embedded)

        # Step 3: Extract the final hidden states from both directions.
        # hidden[-2] = last layer, forward direction
        # hidden[-1] = last layer, backward direction
        # Concatenate → (batch, hidden_dim*2)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Step 4: Dropout + classify
        final_hidden = self.fc_dropout(final_hidden)
        logits       = self.fc(final_hidden)         # → (batch, 2)

        return logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters — useful for understanding model size."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
