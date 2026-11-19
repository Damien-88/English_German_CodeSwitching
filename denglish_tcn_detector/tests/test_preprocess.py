import pandas as pd

from preprocess import clean_row, split_dataframe


def test_clean_row_maps_invalid_labels_to_fallback_and_preserves_comment_id():
    row = pd.Series({
        "tokens": ["Hallo", "world", "123"],
        "labels": ["DE", "EN", "weird-tag"],
        "comment_id": "abc123",
    })
    cleaned = clean_row(row)
    assert cleaned["tokens"] == ["Hallo", "world", "123"]
    assert cleaned["labels"] == ["DE", "EN", "NEUT"]
    assert cleaned["comment_id"] == "abc123"


def test_clean_row_truncates_mismatched_lengths():
    row = pd.Series({"tokens": ["a", "b", "c"], "labels": ["DE", "EN"]})
    cleaned = clean_row(row)
    assert cleaned["tokens"] == ["a", "b"]
    assert cleaned["labels"] == ["DE", "EN"]


def test_split_dataframe_keeps_each_comment_entirely_within_one_split():
    # Five comments, each with several sentences; a comment-level split must
    # never let the same comment_id appear in two different splits.
    rows = []
    for comment_idx in range(5):
        for sentence_idx in range(4):
            rows.append({
                "tokens": [f"tok{comment_idx}_{sentence_idx}"],
                "labels": ["DE"],
                "comment_id": f"comment_{comment_idx}",
            })
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = split_dataframe(df, train_ratio=0.6, val_ratio=0.2, seed=0)

    train_ids = set(train_df["comment_id"])
    val_ids = set(val_df["comment_id"])
    test_ids = set(test_df["comment_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    # No rows should be lost in the process.
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_split_dataframe_falls_back_to_row_level_without_comment_id():
    df = pd.DataFrame({"tokens": [["a"]] * 10, "labels": [["DE"]] * 10})
    train_df, val_df, test_df = split_dataframe(df, train_ratio=0.8, val_ratio=0.1, seed=0)
    assert len(train_df) == 8
    assert len(val_df) == 1
    assert len(test_df) == 1