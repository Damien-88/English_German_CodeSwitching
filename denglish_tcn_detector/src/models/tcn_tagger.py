"""1D causal dilated Temporal Convolutional Network (TCN) sequence tagger.

Implements the architecture described in the project paper (Section 4.2,
"Temporal Convolutional Network"), based on Bai, Kolter, and Koltun (2018):
causal convolutions so a token's representation never depends on later
tokens, dilation rates that expand the receptive field without deepening the
network, weight-normalized convolutions for training stability, and residual
connections (with a 1x1 convolution in the shortcut when channel dimensions
differ).

The initial configuration documented in the paper/README uses 4 residual
blocks, 2 causal conv layers per block, kernel size 3, 128 channels, dilation
rates [1, 2, 4, 8], and dropout 0.2, giving a theoretical receptive field of
1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 61 tokens.

``TCNTagger`` wraps the character-level CNN feature extractor
(``char_cnn.CharCNN``) and the TCN backbone into the full tagging model shown
in Figure 1 of the paper, and also supports the character-CNN ablation study
described in Section 4.1 by swapping the CharCNN for a simple character-bag
embedding (``use_char_cnn=False``) while keeping the rest of the pipeline
identical.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .char_cnn import CharCNN


class CausalConv1d(nn.Module):
    """A 1D convolution that only looks at the current and past time steps.

    Left-only zero padding of ``(kernel_size - 1) * dilation`` guarantees the
    output at position ``t`` is a function of inputs at positions ``<= t``
    only, which is what makes the network suitable for streaming inference.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        raw_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
        )
        # Initialize before wrapping: once weight_norm's parametrization is
        # applied, `conv.weight` becomes a derived (g, v) product rather than
        # a plain leaf Parameter, so initializing afterwards would not
        # reliably update the underlying g/v parameters.
        raw_conv.weight.data.normal_(0, 0.01)
        self.conv = weight_norm(raw_conv)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """One TCN residual block: two causal dilated convs + residual shortcut."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 conv to match dimensions in the residual path when the block
        # changes the channel count (e.g. the first block).
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()
        self._init_downsample()

    def _init_downsample(self):
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of ``TemporalBlock``s with exponentially increasing dilation."""

    def __init__(self, input_dim, channels=128, num_blocks=4, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else channels
            layers.append(
                TemporalBlock(in_channels, channels, kernel_size, dilation, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.output_dim = channels

    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, channels, seq_len) for Conv1d.
        x = x.transpose(1, 2)
        x = self.network(x)
        return x.transpose(1, 2)  # (batch, seq_len, channels)

    @staticmethod
    def receptive_field(num_blocks, kernel_size, layers_per_block=2):
        """Theoretical receptive field in tokens, per Bai et al. (2018)."""
        total_dilation = sum(2 ** i for i in range(num_blocks))
        return 1 + layers_per_block * (kernel_size - 1) * total_dilation


class CharBagEmbedding(nn.Module):
    """Minimal replacement for CharCNN used in the ablation study.

    Produces a fixed-size token representation by averaging learned
    character embeddings with no convolution or local pattern extraction, so
    that disabling the character CNN removes exactly the component under
    test (morphological/subword feature extraction) while leaving the rest
    of the pipeline (padding/masking handling, output dimensionality) intact.
    """

    def __init__(self, vocab_size, output_dim, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, output_dim, padding_idx=padding_idx)

    def forward(self, char_ids):
        batch, seq_len, max_word_len = char_ids.shape
        flat_ids = char_ids.reshape(batch * seq_len, max_word_len)
        mask = (flat_ids != self.padding_idx).unsqueeze(-1)  # (N, max_word_len, 1)

        embedded = self.embedding(flat_ids) * mask
        char_counts = mask.sum(dim=1).clamp(min=1)
        token_features = embedded.sum(dim=1) / char_counts
        return token_features.view(batch, seq_len, -1)


class TCNTagger(nn.Module):
    """Full English-German code-switching tagger (Figure 1 of the paper).

    Pipeline: character embeddings -> character-level feature extractor
    -> 1D causal dilated TCN -> linear classification head. Softmax is left
    to the loss function (``nn.CrossEntropyLoss``) / caller, so this module
    returns raw logits.
    """

    def __init__(
        self,
        char_vocab_size,
        num_classes=3,
        char_embed_dim=32,
        char_filter_sizes=(2, 3, 4),
        char_num_filters=64,
        tcn_channels=128,
        tcn_num_blocks=4,
        tcn_kernel_size=3,
        dropout=0.2,
        padding_idx=0,
        use_char_cnn=True,
    ):
        super().__init__()
        self.use_char_cnn = use_char_cnn

        if use_char_cnn:
            self.char_encoder = CharCNN(
                vocab_size=char_vocab_size,
                embed_dim=char_embed_dim,
                filter_sizes=char_filter_sizes,
                num_filters=char_num_filters,
                padding_idx=padding_idx,
                dropout=dropout,
            )
            feature_dim = self.char_encoder.output_dim
        else:
            # Ablation: character CNN removed (RQ2).
            feature_dim = char_num_filters * len(char_filter_sizes)
            self.char_encoder = CharBagEmbedding(
                vocab_size=char_vocab_size,
                output_dim=feature_dim,
                padding_idx=padding_idx,
            )

        self.tcn = TemporalConvNet(
            input_dim=feature_dim,
            channels=tcn_channels,
            num_blocks=tcn_num_blocks,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(self.tcn.output_dim, num_classes)

    def forward(self, char_ids, attention_mask=None):
        """
        Args:
            char_ids: LongTensor (batch, seq_len, max_word_len).
            attention_mask: optional BoolTensor (batch, seq_len); positions
                set to False are zeroed before the TCN so right-padding added
                by the batch collator cannot leak into the causal receptive
                field computation (it wouldn't anyway, since causal
                convolutions never look ahead, but zeroing keeps padded
                feature values well-defined instead of reusing stale CharCNN
                output).

        Returns:
            FloatTensor of logits with shape (batch, seq_len, num_classes).
        """
        token_features = self.char_encoder(char_ids)  # (batch, seq_len, feature_dim)
        if attention_mask is not None:
            token_features = token_features * attention_mask.unsqueeze(-1).to(token_features.dtype)

        sequence_features = self.tcn(token_features)  # (batch, seq_len, tcn_channels)
        logits = self.classifier(sequence_features)  # (batch, seq_len, num_classes)
        return logits

    def receptive_field(self):
        return TemporalConvNet.receptive_field(
            num_blocks=len(self.tcn.network),
            kernel_size=self.tcn.network[0].conv1.conv.kernel_size[0],
        )