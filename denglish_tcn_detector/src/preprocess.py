import json
import logging
import argparse
import io
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VALID_LABELS = {"DE", "EN", "NEUT"}
FALLBACK_LABEL = "NEUT"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw Denglisch code-switching data using Pandas."
    )
    
    parser.add_argument(
        "--raw_data", type=str, required=True, help="Path to raw input file (JSON, JSONL, or CSV)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed", help="Directory to save processed splits and vocab."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Proportion of data for training."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Proportion of data for validation."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Proportion of data for testing."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--format", type=str, default="json", choices=["json", "parquet"], help="Output format for processed data splits."
    )
    
    args = parser.parse_args()

    for name in ("train_ratio", "val_ratio", "test_ratio"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive, got {getattr(args, name)}")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= ratio_sum <= 1.01):
        parser.error(
            f"--train_ratio, --val_ratio, and --test_ratio must sum to 1.0 "
            f"(got {ratio_sum:.4f})"
        )

    return args

def load_raw_data(file_path):
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    
    if path.suffix == ".json":
        # Read the file once and try both JSON layouts against the in-memory
        # text instead of re-reading from disk on fallback.
        text = path.read_text(encoding="utf-8")
        try:
            df = pd.read_json(io.StringIO(text))
        except ValueError:
            # Fallback for JSON lines format
            df = pd.read_json(io.StringIO(text), lines=True)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    if df.empty:
        raise ValueError(f"Raw data file '{file_path}' contained no rows.")

    missing_cols = {"tokens", "labels"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Raw data file '{file_path}' is missing required column(s): {sorted(missing_cols)}"
        )

    return df

def clean_row(row):
    tokens = row.get("tokens", [])
    labels = row.get("labels", [])
    comment_id = row.get("comment_id", None)

    if len(tokens) != len(labels):
        logger.warning(
            "Row has %d tokens but %d labels; truncating to the shorter length.",
            len(tokens), len(labels),
        )

    cleaned_tokens = []
    cleaned_labels = []
    invalid_label_count = 0

    for token, label in zip(tokens, labels):
        token_str = str(token).strip()
        label_str = str(label).upper().strip()
        
        if not token_str:
            continue
            
        if label_str not in VALID_LABELS:
            invalid_label_count += 1
            label_str = FALLBACK_LABEL  # Fallback for unrecognized tags
            
        cleaned_tokens.append(token_str)
        cleaned_labels.append(label_str)

    if invalid_label_count:
        logger.warning("Row had %d invalid label(s) replaced with '%s'.", invalid_label_count, FALLBACK_LABEL)

    record = {"tokens": cleaned_tokens, "labels": cleaned_labels}
    if comment_id is not None and not (isinstance(comment_id, float) and pd.isna(comment_id)):
        record["comment_id"] = comment_id
    return record

def split_dataframe(df, train_ratio, val_ratio, seed):
    """Splits `df` into train/val/test.

    When a "comment_id" column is present, the split is performed at the
    comment (Reddit post) level rather than the sentence level: every
    sentence originating from the same comment is kept in the same split.
    This mirrors the project paper's data-leakage mitigation ("duplicate or
    closely related Reddit comments ... will be kept within the same
    split"). Without a "comment_id" column (e.g. ad-hoc datasets that are
    already one independent sample per row), the previous row-level shuffle
    is used instead.
    """
    if "comment_id" in df.columns:
        unique_comments = df["comment_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
        total_comments = len(unique_comments)
        train_end = int(total_comments * train_ratio)
        val_end = train_end + int(total_comments * val_ratio)

        train_ids = set(unique_comments.iloc[:train_end])
        val_ids = set(unique_comments.iloc[train_end:val_end])
        test_ids = set(unique_comments.iloc[val_end:])

        shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_df = shuffled_df[shuffled_df["comment_id"].isin(train_ids)].reset_index(drop=True)
        val_df = shuffled_df[shuffled_df["comment_id"].isin(val_ids)].reset_index(drop=True)
        test_df = shuffled_df[shuffled_df["comment_id"].isin(test_ids)].reset_index(drop=True)
        logger.info(
            "Split %d comments into Train/Val/Test (comment-level, no leakage): %d/%d/%d",
            total_comments, len(train_ids), len(val_ids), len(test_ids),
        )
        return train_df, val_df, test_df

    # Fall back to row-level shuffling when no grouping key is available.
    shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled_df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return (
        shuffled_df.iloc[:train_end],
        shuffled_df.iloc[train_end:val_end],
        shuffled_df.iloc[val_end:],
    )


def build_char_vocab(df):
    chars = set()

    for tokens in df["tokens"]:
        for token in tokens:
            for char in token:
                chars.add(char)
                
    # Reserve 0 for padding, 1 for unknown tokens/characters
    char_vocab = {"<PAD>": 0, "<UNK>": 1}
    
    for idx, char in enumerate(sorted(list(chars)), start=2):
        char_vocab[char] = idx
    
    return char_vocab

def main():
    args = parse_args()

    logger.info("Loading raw data from %s using Pandas...", args.raw_data)
    df = load_raw_data(args.raw_data)
    logger.info("Loaded %d raw rows.", len(df))

    logger.info("Cleaning and validating samples...")
    cleaned_records = [clean_row(row) for _, row in df.iterrows() if "tokens" in row and "labels" in row]
    clean_df = pd.DataFrame(cleaned_records)

    # Drop empty rows
    pre_drop_count = len(clean_df)
    clean_df = clean_df[clean_df["tokens"].map(len) > 0].reset_index(drop=True)
    dropped_count = pre_drop_count - len(clean_df)
    if dropped_count:
        logger.info("Dropped %d row(s) left with zero tokens after cleaning.", dropped_count)

    if clean_df.empty:
        raise ValueError("No valid samples remained after cleaning; aborting.")

    # Guard against partially-missing comment_id values (e.g. a raw dataset
    # that only annotates some rows) so that rows never get silently dropped
    # from every split in split_dataframe()'s isin() membership test.
    if "comment_id" in clean_df.columns and clean_df["comment_id"].isna().any():
        missing_mask = clean_df["comment_id"].isna()
        clean_df.loc[missing_mask, "comment_id"] = [
            f"__row_{i}__" for i in clean_df.index[missing_mask]
        ]
        logger.warning(
            "Assigned synthetic comment_id to %d row(s) missing one.", missing_mask.sum()
        )

    # Shuffle using pandas' own random_state for reproducibility.
    train_df, val_df, test_df = split_dataframe(
        clean_df, args.train_ratio, args.val_ratio, args.seed
    )

    logger.info(
        "Split sizes — Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df)
    )

    logger.info("Building character vocabulary from training split...")
    char_vocab = build_char_vocab(train_df)
    logger.info("Vocabulary size (including special tokens): %d", len(char_vocab))

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format == "parquet":
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        val_df.to_parquet(output_dir / "val.parquet", index=False)
        test_df.to_parquet(output_dir / "test.parquet", index=False)
        logger.info("Saved splits as Apache Parquet files.")
    else:
        train_df.to_json(output_dir / "train.json", orient="records", indent=2, force_ascii=False)
        val_df.to_json(output_dir / "val.json", orient="records", indent=2, force_ascii=False)
        test_df.to_json(output_dir / "test.json", orient="records", indent=2, force_ascii=False)
        logger.info("Saved splits as JSON files.")
        
    with open(output_dir / "char_vocab.json", "w", encoding="utf-8") as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=2)

    logger.info("Successfully exported all assets to %s", output_dir.absolute())

if __name__ == "__main__":
    main()