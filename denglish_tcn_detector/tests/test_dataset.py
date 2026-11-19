import json

import torch

from dataset import LABEL2ID, PAD_LABEL_ID, DenglischDataset, denglisch_collate_fn


def _write_dataset(tmp_path, records):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_dataset_item_shapes(tmp_path):
    records = [
        {"tokens": ["Ich", "muss", "mergen"], "labels": ["DE", "DE", "EN"]},
        {"tokens": ["OK", "."], "labels": ["EN", "NEUT"]},
    ]
    path = _write_dataset(tmp_path, records)
    dataset = DenglischDataset(path, max_word_len=10)

    sample = dataset[0]
    assert sample["char_ids"].shape == (3, 10)
    assert sample["labels"].tolist() == [LABEL2ID["DE"], LABEL2ID["DE"], LABEL2ID["EN"]]
    assert sample["length"].item() == 3


def test_unrecognized_label_falls_back_to_neut(tmp_path):
    records = [{"tokens": ["foo"], "labels": ["XX"]}]
    path = _write_dataset(tmp_path, records)
    dataset = DenglischDataset(path, max_word_len=5)

    sample = dataset[0]
    assert sample["labels"].tolist() == [LABEL2ID["NEUT"]]


def test_collate_fn_pads_to_batch_max_length_and_masks_correctly(tmp_path):
    records = [
        {"tokens": ["Ich", "muss", "mergen"], "labels": ["DE", "DE", "EN"]},
        {"tokens": ["OK"], "labels": ["EN"]},
    ]
    path = _write_dataset(tmp_path, records)
    dataset = DenglischDataset(path, max_word_len=10)

    batch = [dataset[0], dataset[1]]
    collated = denglisch_collate_fn(batch)

    assert collated["char_ids"].shape == (2, 3, 10)
    assert collated["labels"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)

    # Second sequence has only 1 real token; positions 1-2 are padding.
    assert collated["attention_mask"][1].tolist() == [True, False, False]
    assert collated["labels"][1, 1:].tolist() == [PAD_LABEL_ID, PAD_LABEL_ID]
    # Padded char rows must be all <PAD> (0).
    assert torch.all(collated["char_ids"][1, 1:, :] == 0)