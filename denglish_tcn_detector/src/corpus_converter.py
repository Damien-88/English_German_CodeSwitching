"""Converts the official Denglisch Corpus (Osmelak & Wintner, 2023) CSV release
into the sentence-level JSON format expected by ``preprocess.py``.

The original corpus (https://github.com/HaifaCLG/Denglisch) is distributed as
flat per-token CSV files with the columns::

    source, user_name, sen_id, sen_num, token, categ

Each row is a single token. ``sen_id`` is the Reddit comment/post identifier
and ``sen_num`` numbers the sentences within that post, so a sentence is the
group of rows sharing the same ``(source, sen_id, sen_num)`` key and a
"comment" (the unit that must not be split across train/val/test, see Section
5.3 of the project paper) is the group sharing ``(source, sen_id)``.

This script supports the corpus's "collapsed" annotation scheme, whose
``categ`` values are documented by Osmelak & Wintner (2023) as:

    E  = English            D  = German
    SE = Shared English     SD = Shared German
    SO = Shared Other       O  = Other (punctuation, numbers, symbols)
    M  = Mixed morphology (ambiguous English/German blend)

These are mapped onto the project's three target labels as described in the
paper's Data Preparation section:

    * Unambiguous German/English tokens keep their language label.
    * "Shared" tokens are assigned using the contextual language lean already
      encoded in the annotation (Shared English -> EN, Shared German -> DE).
    * "Other" tokens (punctuation, numbers, symbols) map to NEUT.
    * "Mixed"/"Shared Other" tokens have no reliable single-language
      assignment and are excluded from the primary dataset, matching the
      paper's stated policy of dropping tokens "that cannot be reliably
      classified ... to avoid introducing arbitrary labels."

Usage:
    python src/corpus_converter.py \\
        --input data/raw/Manu_corpus_collapsed.csv \\
        --output data/raw/denglisch.json
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Mapping from the corpus's collapsed annotation scheme to the project's
# three target labels. Categories mapped to None are ambiguous and are
# excluded from the converted dataset rather than guessed at.
COLLAPSED_LABEL_MAP = {
    "E": "EN",
    "SE": "EN",
    "D": "DE",
    "SD": "DE",
    "O": "NEUT",
    "SO": None,  # Shared Other: no reliable single-language assignment.
    "M": None,   # Mixed morphology: ambiguous, excluded per project paper.
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the raw Denglisch Corpus CSV into sentence-level JSON."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a Denglisch corpus CSV file using the collapsed annotation "
             "scheme (e.g. Manu_corpus_collapsed.csv or corpus_collapsed.csv).",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/denglisch.json",
        help="Path to write the converted sentence-level JSON file.",
    )
    return parser.parse_args()


def load_corpus_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus CSV not found at: {path}")

    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    required_cols = {"source", "sen_id", "sen_num", "token", "categ"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Corpus CSV '{path}' is missing required column(s): {sorted(missing)}")
    return df


def convert_corpus(df):
    """Groups tokens into sentences and maps categories to DE/EN/NEUT labels.

    Returns a tuple of (records, exclusion_counts) where ``records`` is a
    list of ``{"tokens": [...], "labels": [...], "comment_id": str}`` dicts
    and ``exclusion_counts`` maps each original ``categ`` value to the number
    of tokens dropped for that category.
    """
    exclusion_counts = {}
    unmapped_counts = {}
    dropped_empty = 0
    records = []

    group_cols = ["source", "sen_id", "sen_num"]
    for (source, sen_id, _sen_num), group in df.groupby(group_cols, sort=False):
        tokens = []
        labels = []
        for token, categ in zip(group["token"], group["categ"]):
            token = str(token).strip()
            if not token:
                dropped_empty += 1
                continue

            if categ not in COLLAPSED_LABEL_MAP:
                unmapped_counts[categ] = unmapped_counts.get(categ, 0) + 1
                continue

            label = COLLAPSED_LABEL_MAP[categ]
            if label is None:
                exclusion_counts[categ] = exclusion_counts.get(categ, 0) + 1
                continue

            tokens.append(token)
            labels.append(label)

        if tokens:
            records.append({
                "tokens": tokens,
                "labels": labels,
                "comment_id": f"{source}:{sen_id}",
            })

    if unmapped_counts:
        logger.warning(
            "Encountered categ value(s) outside the known collapsed scheme; "
            "these tokens were dropped: %s", unmapped_counts,
        )
    if dropped_empty:
        logger.info("Dropped %d row(s) with an empty token string.", dropped_empty)

    return records, exclusion_counts


def main():
    args = parse_args()

    logger.info("Loading raw Denglisch corpus CSV from %s...", args.input)
    df = load_corpus_csv(args.input)
    logger.info("Loaded %d raw token rows.", len(df))

    records, exclusion_counts = convert_corpus(df)

    total_excluded = sum(exclusion_counts.values())
    total_kept = sum(len(r["tokens"]) for r in records)
    total_tokens = total_kept + total_excluded
    logger.info(
        "Converted %d sentences containing %d tokens (%d retained, %d excluded as "
        "ambiguous).", len(records), total_tokens, total_kept, total_excluded,
    )
    if total_tokens:
        pct = 100.0 * total_excluded / total_tokens
        logger.info("Excluded token breakdown by original category: %s (%.2f%% of all tokens)",
                     exclusion_counts, pct)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Wrote converted corpus to %s", output_path.absolute())


if __name__ == "__main__":
    main()