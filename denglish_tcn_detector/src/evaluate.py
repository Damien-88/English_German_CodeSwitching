"""Evaluation and benchmarking for the 1D-TCN English-German code-switching tagger.

Implements the evaluation procedure described in the project paper (Section
4.2, "Evaluation and Baseline Comparisons"): overall accuracy, per-class
precision/recall/F1, macro-F1, weighted-F1, a confusion matrix, inference
throughput (tokens/sec), and memory consumption.

Usage:
    python src/evaluate.py --checkpoint ../checkpoints/best_tcn.pt --test_data data/processed/test.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataset import ID2LABEL, PAD_LABEL_ID, DenglischDataset, denglisch_collate_fn
from models.tcn_tagger import TCNTagger
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    import psutil
    _PROCESS = psutil.Process()
except ImportError:  # pragma: no cover - exercised only when psutil is missing
    psutil = None
    _PROCESS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained 1D-TCN Denglisch tagger.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint saved by train.py.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test split JSON file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                         help="Device to evaluate on. Defaults to CUDA if available, else CPU.")
    parser.add_argument("--report_path", type=str, default=None,
                         help="Optional path to write the full evaluation report as JSON.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes.")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    char_vocab = checkpoint["char_vocab"]

    model = TCNTagger(
        char_vocab_size=len(char_vocab),
        num_classes=len(checkpoint["label2id"]),
        char_embed_dim=model_config["char_embed_dim"],
        char_filter_sizes=tuple(model_config["char_filter_sizes"]),
        char_num_filters=model_config["char_num_filters"],
        tcn_channels=model_config["tcn_channels"],
        tcn_num_blocks=model_config["tcn_num_blocks"],
        tcn_kernel_size=model_config["tcn_kernel_size"],
        dropout=model_config["dropout"],
        use_char_cnn=model_config["use_char_cnn"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def current_memory_mb():
    """Best-effort resident memory usage in MB, or None if unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    if _PROCESS is not None:
        return _PROCESS.memory_info().rss / (1024 ** 2)
    return None


@torch.no_grad()
def run_evaluation(model, loader, device):
    all_preds, all_labels = [], []
    total_tokens = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif _PROCESS is not None:
        _PROCESS.memory_info()  # warm up the handle before timing

    start_time = time.perf_counter()
    for batch in loader:
        char_ids = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(char_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1)

        valid_mask = labels != PAD_LABEL_ID
        all_preds.append(preds[valid_mask].cpu().numpy())
        all_labels.append(labels[valid_mask].cpu().numpy())
        total_tokens += int(valid_mask.sum().item())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    peak_memory_mb = current_memory_mb()
    throughput = total_tokens / elapsed if elapsed > 0 else float("inf")

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    return all_preds, all_labels, total_tokens, elapsed, throughput, peak_memory_mb


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    logger.info(
        "Loaded checkpoint from epoch %d (val_macro_f1=%.4f, use_char_cnn=%s)",
        checkpoint.get("epoch", -1),
        checkpoint.get("val_macro_f1", float("nan")),
        checkpoint["model_config"]["use_char_cnn"],
    )

    max_word_len = checkpoint.get("max_word_len", 20)
    test_dataset = DenglischDataset(
        args.test_data, char_vocab=checkpoint["char_vocab"], max_word_len=max_word_len
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=denglisch_collate_fn, num_workers=args.num_workers,
    )

    all_preds, all_labels, total_tokens, elapsed, throughput, peak_memory_mb = run_evaluation(
        model, test_loader, device
    )

    if len(all_labels) == 0:
        raise ValueError(f"No evaluable tokens were found in {args.test_data}.")

    label_ids = sorted(ID2LABEL.keys())
    target_names = [ID2LABEL[i] for i in label_ids]

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, labels=label_ids, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, labels=label_ids, average="weighted", zero_division=0)
    report_dict = classification_report(
        all_labels, all_preds, labels=label_ids, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        all_labels, all_preds, labels=label_ids, target_names=target_names, zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)

    logger.info("\n%s", report_text)
    logger.info("Overall accuracy: %.4f", accuracy)
    logger.info("Macro-F1: %.4f | Weighted-F1: %.4f", macro_f1, weighted_f1)
    logger.info("Confusion matrix (rows=true, cols=pred) %s:\n%s", target_names, cm)
    logger.info(
        "Throughput: %.1f tokens/sec (%d tokens in %.3fs)", throughput, total_tokens, elapsed
    )
    if peak_memory_mb is not None:
        mem_kind = "peak GPU memory" if torch.cuda.is_available() else "resident memory (RSS)"
        logger.info("%s: %.2f MB", mem_kind, peak_memory_mb)
    else:
        logger.warning("Memory consumption could not be measured (install psutil for CPU RSS tracking).")

    report = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": target_names,
        "throughput_tokens_per_sec": throughput,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_memory_mb,
    }

    report_path = args.report_path
    if report_path is None:
        checkpoint_path = Path(args.checkpoint)
        report_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_eval_report.json")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved full evaluation report to %s", report_path.absolute())


if __name__ == "__main__":
    main()