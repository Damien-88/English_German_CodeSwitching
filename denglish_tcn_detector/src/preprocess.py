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

    return {"tokens": cleaned_tokens, "labels": cleaned_labels}

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

    # Shuffle using pandas' own random_state for reproducibility.
    clean_df = clean_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    total = len(clean_df)
    train_size = int(total * args.train_ratio)
    val_size = int(total * args.val_ratio)
    # Test gets the remainder so rounding never drops samples, even if the
    # ratios don't divide `total` evenly.
    train_end = train_size
    val_end = train_size + val_size

    train_df = clean_df.iloc[:train_end]
    val_df = clean_df.iloc[train_end:val_end]
    test_df = clean_df.iloc[val_end:]

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