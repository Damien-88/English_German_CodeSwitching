"""Training loop for the 1D-TCN English-German code-switching tagger.

Implements the training procedure described in the project paper (Section
4.2, "Model Training"): token-level cross-entropy loss with padding
positions excluded, the Adam optimizer, mixed-precision training when CUDA is
available, validation monitoring after every epoch, and checkpointing of the
best model (by validation macro-F1) to ``checkpoints/``.

Usage:
    python src/train.py --data_dir data/processed --epochs 10 --batch_size 64 --lr 1e-3
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LABEL2ID, PAD_LABEL_ID, DenglischDataset, denglisch_collate_fn
from models.tcn_tagger import TCNTagger

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 1D-TCN Denglisch tagger.")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                         help="Directory containing train.json, val.json, and char_vocab.json.")
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Path to the YAML file with model/training hyperparameters.")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints",
                         help="Directory to save model checkpoints.")
    parser.add_argument("--checkpoint_name", type=str, default="best_tcn.pt",
                         help="Filename for the best checkpoint.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training/validation batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Adam learning rate.")
    parser.add_argument("--max_word_len", type=int, default=None,
                         help="Maximum characters per token (longer tokens are truncated).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                         help="Device to train on. Defaults to CUDA if available, else CPU.")
    parser.add_argument("--no_char_cnn", action="store_true",
                         help="Ablation: disable the character-level CNN feature extractor (RQ2).")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes.")
    return parser.parse_args()


def load_config(path):
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning("Config file '%s' not found; using built-in defaults.", path)
    return {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_setting(cli_value, cfg, *keys, default=None):
    """Returns `cli_value` if provided, else the nested config value at `keys`, else `default`."""
    if cli_value is not None:
        return cli_value
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def build_model(cfg, char_vocab_size, use_char_cnn):
    model_cfg = cfg.get("model", {})
    return TCNTagger(
        char_vocab_size=char_vocab_size,
        num_classes=len(LABEL2ID),
        char_embed_dim=model_cfg.get("char_embed_dim", 32),
        char_filter_sizes=tuple(model_cfg.get("char_filter_sizes", [2, 3, 4])),
        char_num_filters=model_cfg.get("char_num_filters", 64),
        tcn_channels=model_cfg.get("tcn_channels", 128),
        tcn_num_blocks=model_cfg.get("tcn_num_blocks", 4),
        tcn_kernel_size=model_cfg.get("tcn_kernel_size", 3),
        dropout=model_cfg.get("dropout", 0.2),
        use_char_cnn=use_char_cnn,
    )


def flatten_valid(logits, labels):
    """Flattens (batch, seq_len, ...) tensors and drops padding positions."""
    preds = logits.argmax(dim=-1).view(-1)
    flat_labels = labels.view(-1)
    valid_mask = flat_labels != PAD_LABEL_ID
    return preds[valid_mask].cpu().numpy(), flat_labels[valid_mask].cpu().numpy()


@torch.no_grad()
def run_validation(model, loader, device, criterion):
    model.eval()
    total_loss, num_batches = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        char_ids = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(char_ids, attention_mask=attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        num_batches += 1

        preds, flat_labels = flatten_valid(logits, labels)
        all_preds.append(preds)
        all_labels.append(flat_labels)

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if len(all_labels) else 0.0
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, accuracy, macro_f1


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = resolve_setting(args.seed, cfg, "seed", default=42)
    set_seed(seed)

    train_cfg = cfg.get("training", {})
    epochs = resolve_setting(args.epochs, cfg, "training", "epochs", default=10)
    batch_size = resolve_setting(args.batch_size, cfg, "training", "batch_size", default=64)
    lr = resolve_setting(args.lr, cfg, "training", "learning_rate", default=1e-3)
    max_word_len = resolve_setting(args.max_word_len, cfg, "data", "max_word_len", default=20)

    data_dir = Path(args.data_dir)
    char_vocab_path = data_dir / "char_vocab.json"
    if not char_vocab_path.exists():
        raise FileNotFoundError(
            f"Character vocabulary not found at {char_vocab_path}. Run src/preprocess.py first."
        )
    with open(char_vocab_path, "r", encoding="utf-8") as f:
        char_vocab = json.load(f)

    train_dataset = DenglischDataset(data_dir / "train.json", char_vocab=char_vocab, max_word_len=max_word_len)
    val_dataset = DenglischDataset(data_dir / "val.json", char_vocab=char_vocab, max_word_len=max_word_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=denglisch_collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=denglisch_collate_fn, num_workers=args.num_workers,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    use_char_cnn = not args.no_char_cnn
    model = build_model(cfg, char_vocab_size=len(char_vocab), use_char_cnn=use_char_cnn).to(device)
    logger.info(
        "Model parameters: %d (use_char_cnn=%s, receptive_field=%d tokens)",
        sum(p.numel() for p in model.parameters()), use_char_cnn, model.receptive_field(),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)

    use_amp = device.type == "cuda" and train_cfg.get("mixed_precision", True)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    model_cfg = cfg.get("model", {})
    model_config = {
        "char_embed_dim": model_cfg.get("char_embed_dim", 32),
        "char_filter_sizes": list(model_cfg.get("char_filter_sizes", [2, 3, 4])),
        "char_num_filters": model_cfg.get("char_num_filters", 64),
        "tcn_channels": model_cfg.get("tcn_channels", 128),
        "tcn_num_blocks": model_cfg.get("tcn_num_blocks", 4),
        "tcn_kernel_size": model_cfg.get("tcn_kernel_size", 3),
        "dropout": model_cfg.get("dropout", 0.2),
        "use_char_cnn": use_char_cnn,
    }

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, num_batches = 0.0, 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(char_ids, attention_mask=attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=running_loss / num_batches)

        train_loss = running_loss / max(num_batches, 1)
        val_loss, val_acc, val_macro_f1 = run_validation(model, val_loader, device, criterion)

        logger.info(
            "Epoch %d/%d — train_loss: %.4f | val_loss: %.4f | val_acc: %.4f | val_macro_f1: %.4f",
            epoch, epochs, train_loss, val_loss, val_acc, val_macro_f1,
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_accuracy": val_acc, "val_macro_f1": val_macro_f1,
        })

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "char_vocab": char_vocab,
                "label2id": LABEL2ID,
                "max_word_len": max_word_len,
                "epoch": epoch,
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "seed": seed,
            }, checkpoint_path)
            logger.info("Saved new best checkpoint to %s (val_macro_f1=%.4f)", checkpoint_path, val_macro_f1)

    history_path = checkpoint_dir / f"{Path(args.checkpoint_name).stem}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. Best val_macro_f1=%.4f. History saved to %s", best_val_f1, history_path)


if __name__ == "__main__":
    main()