import torch

from models.char_cnn import CharCNN


def _make_char_ids(words, char_to_idx, max_word_len):
    """Encodes a list of words into a (1, seq_len, max_word_len) tensor."""
    rows = []
    for word in words:
        ids = [char_to_idx[c] for c in word[:max_word_len]]
        ids += [0] * (max_word_len - len(ids))
        rows.append(ids)
    return torch.tensor([rows], dtype=torch.long)


def test_output_shape_matches_batch_seq_and_configured_filters():
    char_to_idx = {c: i + 2 for i, c in enumerate("abcdefghij")}  # 0=<PAD>, 1=<UNK>
    model = CharCNN(vocab_size=20, embed_dim=8, filter_sizes=(2, 3, 4), num_filters=5, dropout=0.0)
    model.eval()

    char_ids = _make_char_ids(["cab", "fed"], char_to_idx, max_word_len=6)
    out = model(char_ids)

    assert out.shape == (1, 2, 5 * 3)


def test_padding_characters_do_not_change_token_representation():
    """Appending extra <PAD> characters must not change the pooled features."""
    char_to_idx = {c: i + 2 for i, c in enumerate("abcdefghij")}
    model = CharCNN(vocab_size=20, embed_dim=8, filter_sizes=(2, 3, 4), num_filters=5, dropout=0.0)
    model.eval()

    short = _make_char_ids(["cab"], char_to_idx, max_word_len=3)
    long = _make_char_ids(["cab"], char_to_idx, max_word_len=10)

    with torch.no_grad():
        out_short = model(short)
        out_long = model(long)

    assert torch.allclose(out_short, out_long, atol=1e-6)


def test_fully_padded_token_produces_zero_vector():
    model = CharCNN(vocab_size=20, embed_dim=8, filter_sizes=(2, 3, 4), num_filters=5, dropout=0.0)
    model.eval()

    # A token made entirely of <PAD> (id 0), as produced by denglisch_collate_fn
    # for positions beyond a sequence's real length.
    char_ids = torch.zeros((1, 1, 6), dtype=torch.long)

    with torch.no_grad():
        out = model(char_ids)

    assert torch.allclose(out, torch.zeros_like(out))