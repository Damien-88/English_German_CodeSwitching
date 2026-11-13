import json
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Define label mappings. PAD_LABEL_ID is intentionally kept separate from
# LABEL2ID: it is not a real classification target, only the ignore_index
# used by the loss function / collate padding, and mixing it into the same
# dict as real labels previously made it possible to accidentally treat
# "PAD" as a predictable class.
LABEL2ID = {"DE": 0, "EN": 1, "NEUT": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
PAD_LABEL_ID = -100
# Fallback label for tags that don't appear in LABEL2ID. Matches the
# fallback used by preprocess.py's clean_row() for consistency between
# the preprocessing and dataset-loading stages.
FALLBACK_LABEL = "NEUT"

class DenglischDataset(Dataset):
    """
    Dataset loader for token-level English-German code-switching data.
    Expected format per line or in a JSON list:
    {
        "tokens": ["Ich", "muss", "das", "deployment", ...],
        "labels": ["DE", "DE", "DE", "EN", ...]
    }

    Examples:
        >>> dataset = DenglischDataset("data/processed/train.json")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['char_ids', 'labels', 'length']
        >>> # Reuse the training vocabulary when loading validation data:
        >>> val_dataset = DenglischDataset("data/processed/val.json", char_vocab=dataset.char_vocab)
    """
    def __init__(
        self,
        data_path,
        char_vocab=None,
        max_word_len=20,
        min_char_freq=1,
    ):
        self.data = self._load_data(data_path)
        self.max_word_len = max_word_len

        # Initialize or use provided character vocabulary
        if char_vocab is None:
            self.char_vocab = self._build_char_vocab(self.data, min_freq=min_char_freq)
        else:
            self.char_vocab = char_vocab

        self.pad_char_idx = self.char_vocab.get("<PAD>", 0)

    def _load_data(self, path):
        """Loads samples from a standard JSON list or JSON Lines file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                # Supports JSON lines or standard JSON list
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    data = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse '{path}' as JSON or JSON Lines: {e}"
            ) from e

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list of samples in '{path}', got {type(data).__name__}."
            )

        logger.info("Loaded %d samples from %s", len(data), path)
        return data

    def _build_char_vocab(self, data, min_freq=1):
        """
        Builds a character vocabulary from the dataset.

        Characters occurring fewer than `min_freq` times are dropped (mapped
        to <UNK> at lookup time instead), which bounds vocabulary/embedding
        size on very large or noisy corpora. The default of 1 keeps every
        observed character, preserving prior behavior.
        """
        char_counts = {}

        for sample in data:
            for token in sample["tokens"]:
                for char in token:
                    char_counts[char] = char_counts.get(char, 0) + 1

        # Reserve 0 for padding, 1 for unknown
        char_vocab = {"<PAD>": 0, "<UNK>": 1}

        kept_chars = sorted(c for c, count in char_counts.items() if count >= min_freq)
        for idx, char in enumerate(kept_chars, start=2):
            char_vocab[char] = idx

        logger.info(
            "Built character vocabulary: kept %d of %d unique characters (min_freq=%d)",
            len(kept_chars), len(char_counts), min_freq,
        )
        return char_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"]
        raw_labels = sample["labels"]

        if len(tokens) != len(raw_labels):
            raise ValueError(
                f"Sample {idx} has {len(tokens)} tokens but {len(raw_labels)} labels; "
                "tokens and labels must be the same length."
            )

        labels = []
        for pos, label in enumerate(raw_labels):
            if label not in LABEL2ID:
                logger.warning(
                    "Sample %d, token %d: unrecognized label %r, defaulting to '%s'",
                    idx, pos, label, FALLBACK_LABEL,
                )
                labels.append(LABEL2ID[FALLBACK_LABEL])
            else:
                labels.append(LABEL2ID[label])

        # Convert words to character index matrices [seq_len, max_word_len]
        char_ids_list = []
        
        for token in tokens:
            char_ids = [self.char_vocab.get(c, self.char_vocab["<UNK>"]) for c in token[:self.max_word_len]]
            # Pad word to max_word_len
            char_ids += [self.pad_char_idx] * (self.max_word_len - len(char_ids))
            char_ids_list.append(char_ids)

        return {
            "char_ids": torch.tensor(char_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": torch.tensor(len(tokens), dtype=torch.long)
        }


def denglisch_collate_fn(batch, pad_token_label_id=PAD_LABEL_ID):
    """
    Collates batch samples by padding sequences to the maximum length in the batch.
    """
    batch_size = len(batch)
    max_seq_len = max(item["length"].item() for item in batch)
    max_word_len = batch[0]["char_ids"].shape[1]

    # Initialize padded tensors
    padded_char_ids = torch.zeros(batch_size, max_seq_len, max_word_len, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_seq_len), pad_token_label_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item["length"].item()
        padded_char_ids[i, :seq_len, :] = item["char_ids"]
        padded_labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = True

    return {
        "char_ids": padded_char_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }