# English_German_CodeSwitching
My capstone research project on detecting real-time english-german code-switching using lightwieght 1D Temporal Convolutional Networks.

# Real-Time English–German Code-Switching Detection Using Lightweight 1D Temporal Convolutional Networks

This repository contains the official implementation and experimentation code for the **Fall 2026 CS 687 Capstone Project**. The project investigates parameter-efficient 1D Temporal Convolutional Networks (1D-TCNs) equipped with character-level CNN feature extractors for real-time token-level language identification in English-German mixed text ("Denglisch").

## Repository Layout

```text
denglisch_tcn_detector/
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/                 # Original Denglisch/TongueSwitcher downloads
│   └── processed/           # Tokenized & mapped JSON/Parquet files
├── src/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset & character/token collators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── char_cnn.py      # Character-level CNN feature extractor
│   │   └── tcn_tagger.py    # 1D Temporal Convolutional Network sequence tagger
│   ├── train.py             # Training loop with mixed precision & logging
│   └── evaluate.py          # F1-score evaluation across DE, EN, NEUT
├── notebooks/               # Exploratory data analysis & error analysis
├── tests/                   # Unit tests for shape checking & padding
├── checkpoints/             # Saved model weights (.pt)
├── requirements.txt
└── README.md
```

## Project Overview

Standard autoregressive LLMs and heavy Transformer architectures pose latency and memory bottlenecks on resource-constrained edge devices. Recurrent architectures (like BiLSTM-CRF) encounter sequential processing constraints. This project explores whether a lightweight **1D-TCN** can match transformer-level token classification performance ($F_1$-score) while providing superior throughput (tokens/sec) and a reduced memory footprint.

### Target Classification Labels
* **DE**: German tokens and morphologically integrated forms
* **EN**: English words and loanword insertions
* **NEUT**: Punctuation, numbers, and neutral symbols

---

## Getting Started

### Prerequisites
* Python 3.10+
* CUDA-enabled GPU (recommended for training)

### Installation

1. Clone the repository and navigate into the workspace:
   ```bash
   git clone [https://github.com/Damien-88/English_German_Codeswitching.git](https://github.com/Damien-88/English_German_Codeswitching.git)
   cd denglisch-tcn-detector
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Architecture

```
Input Sentence ──> Tokenizer ──> Character Embedding 
                                      │
                                      ▼
                             [Character-Level CNN]
                                      │
                                      ▼
                           [1D Causal Dilated TCN]
                                      │
                                      ▼
                          Linear + Softmax Tagging ──> [DE, EN, NEUT]
```

* **Character-Level CNN (`src/models/char_cnn.py`)**: Extracts subword features, prefixes, and suffixes to handle productive compounding (*Komposita*) and mixed morphology (e.g., *mergen*).
* **1D-TCN Backbone (`src/models/tcn_tagger.py`)**: Utilizes causal convolutions, dilation factors, and residual blocks to widen the receptive field without sequential inference bottlenecks.

---

## Usage

### Training
To train the 1D-TCN model on preprocessed Denglisch corpus data:
```bash
python src/train.py --data_dir data/processed --epochs 10 --batch_size 64 --lr 1e-3
```

### Evaluation & Benchmarking
To evaluate token classification $F_1$-score and measure inference throughput (tokens/sec):
```bash
python src/evaluate.py --checkpoint checkpoints/best_tcn.pt --test_data data/processed/test.json
```

---

## Research Questions Addressed
* **RQ1**: TCN vs. BiLSTM-CRF vs. Transformer $F_1$-score comparison.
* **RQ2**: Impact of character-level CNN extraction on complex word boundaries (e.g., German-English compounds).
* **RQ3**: Throughput and memory footprint trade-offs on low-latency streaming pipelines.

---

## License
This project is developed for academic research under the City University of Seattle CS 687 Capstone curriculum.