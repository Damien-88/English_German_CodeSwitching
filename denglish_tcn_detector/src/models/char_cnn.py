"""Character-level CNN feature extractor.

Implements the morphological feature extractor described in the project
paper (Section 4.2, "Character-Level Feature Extraction"): a learned
character embedding followed by parallel 1D convolutions of width 2, 3, and 4
with 64 filters each, ReLU activation, and masked global max pooling. Padding
characters are masked out of the pooling step so that artificial padding
never influences the resulting token representation, and the per-filter
pooled vectors are concatenated into a single fixed-size embedding for every
token regardless of its original character length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """Maps a batch of per-token character-id sequences to fixed-size vectors.

    Args:
        vocab_size: Size of the character vocabulary (including <PAD>/<UNK>).
        embed_dim: Dimensionality of the learned character embeddings.
        filter_sizes: Convolutional filter widths (in characters).
        num_filters: Number of filters learned per filter width.
        padding_idx: Index reserved for the <PAD> character.
        dropout: Dropout probability applied to the pooled token features.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=32,
        filter_sizes=(2, 3, 4),
        num_filters=64,
        padding_idx=0,
        dropout=0.2,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.filter_sizes = tuple(filter_sizes)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in self.filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(self.filter_sizes)

    def forward(self, char_ids):
        """
        Args:
            char_ids: LongTensor of shape (batch, seq_len, max_word_len).

        Returns:
            FloatTensor of shape (batch, seq_len, output_dim).
        """
        batch, seq_len, max_word_len = char_ids.shape
        flat_ids = char_ids.reshape(batch * seq_len, max_word_len)

        # True for real characters, False for <PAD> positions.
        char_mask = flat_ids != self.padding_idx  # (N, max_word_len)
        # Tokens that are themselves pure padding (e.g. positions added by
        # the batch collator beyond a sequence's real length) have no valid
        # characters at all and must be zeroed explicitly after pooling.
        token_has_chars = char_mask.any(dim=1)  # (N,)

        embedded = self.embedding(flat_ids)  # (N, max_word_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (N, embed_dim, max_word_len)

        pooled_features = []
        for conv, kernel_size in zip(self.convs, self.filter_sizes):
            # Symmetric ("same") padding keeps the conv output aligned with
            # the input so the character mask can be reused directly for
            # masking, regardless of whether kernel_size is even or odd.
            pad_left = (kernel_size - 1) // 2
            pad_right = kernel_size - 1 - pad_left
            padded = F.pad(embedded, (pad_left, pad_right))

            conv_out = F.relu(conv(padded))  # (N, num_filters, max_word_len)
            conv_out = conv_out.masked_fill(~char_mask.unsqueeze(1), float("-1e4"))

            pooled, _ = conv_out.max(dim=2)  # (N, num_filters)
            pooled_features.append(pooled)

        token_features = torch.cat(pooled_features, dim=1)  # (N, output_dim)
        # Zero out tokens that had no real characters instead of leaving the
        # masked_fill sentinel value (-1e4) in the representation.
        token_features = token_features * token_has_chars.unsqueeze(1).to(token_features.dtype)
        token_features = self.dropout(token_features)

        return token_features.view(batch, seq_len, self.output_dim)
