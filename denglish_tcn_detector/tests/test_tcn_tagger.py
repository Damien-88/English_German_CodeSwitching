import torch

from models.tcn_tagger import TCNTagger, TemporalConvNet


def _random_char_ids(batch, seq_len, max_word_len, vocab_size, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab_size, (batch, seq_len, max_word_len), generator=generator)


def test_output_shape():
    model = TCNTagger(
        char_vocab_size=30, num_classes=3, char_embed_dim=8,
        char_filter_sizes=(2, 3), char_num_filters=4,
        tcn_channels=16, tcn_num_blocks=2, tcn_kernel_size=3, dropout=0.0,
    )
    model.eval()

    char_ids = _random_char_ids(batch=2, seq_len=7, max_word_len=5, vocab_size=30)
    attention_mask = torch.ones(2, 7, dtype=torch.bool)

    with torch.no_grad():
        logits = model(char_ids, attention_mask=attention_mask)

    assert logits.shape == (2, 7, 3)


def test_causal_masking_future_tokens_do_not_affect_earlier_predictions():
    """Changing a later token must not change logits at earlier positions."""
    model = TCNTagger(
        char_vocab_size=30, num_classes=3, char_embed_dim=8,
        char_filter_sizes=(2, 3), char_num_filters=4,
        tcn_channels=16, tcn_num_blocks=2, tcn_kernel_size=3, dropout=0.0,
    )
    model.eval()

    char_ids = _random_char_ids(batch=1, seq_len=6, max_word_len=5, vocab_size=30)
    modified = char_ids.clone()
    modified[0, -1, :] = 29  # perturb only the last token

    with torch.no_grad():
        logits_original = model(char_ids)
        logits_modified = model(modified)

    # All positions except the perturbed final one should be unaffected.
    assert torch.allclose(logits_original[:, :-1, :], logits_modified[:, :-1, :], atol=1e-5)
    assert not torch.allclose(logits_original[:, -1, :], logits_modified[:, -1, :], atol=1e-5)


def test_ablation_disables_char_cnn():
    model = TCNTagger(
        char_vocab_size=30, num_classes=3, tcn_channels=16, tcn_num_blocks=2, use_char_cnn=False,
    )
    from models.tcn_tagger import CharBagEmbedding
    assert isinstance(model.char_encoder, CharBagEmbedding)


def test_receptive_field_matches_paper_configuration():
    # 4 blocks, kernel size 3, 2 layers/block, dilations [1,2,4,8] -> 61 tokens.
    assert TemporalConvNet.receptive_field(num_blocks=4, kernel_size=3) == 61