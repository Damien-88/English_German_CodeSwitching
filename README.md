
# English_German_CodeSwitching

### Real-Time English–German Code-Switching Detection Using Lightweight 1D Temporal Convolutional Networks

My capstone research project on detecting real-time English-German code-switching using lightweight 1D Temporal Convolutional Networks.

This repository contains the official implementation and experimentation code for the **Fall 2026 CS 687 Capstone Project**. The project investigates parameter-efficient 1D Temporal Convolutional Networks (1D-TCNs) equipped with character-level CNN feature extractors for real-time token-level language identification in English-German mixed text ("Denglisch").

The project uses the **Denglisch Corpus of German-English Code-Switching** by Osmelak and Wintner (2023), a linguistic resource derived from Reddit interactions containing German-English code-switched text and word-level language annotations.

**Corpus source:** [HaifaCLG/Denglisch – GitHub Repository](https://github.com/HaifaCLG/Denglisch)

**Research publication:** [Osmelak and Wintner (2023) – ACL Anthology](https://aclanthology.org/2023.sigtyp-1.5/)

## Repository Layout

```text
English_German_CodeSwitching/
├── checkpoints/                 # Saved model weights (.pt)
├── denglish_tcn_detector/
│   ├── .vscode/
│   │   └── settings.json
│   ├── data/
│   │   ├── raw/                  # Original Denglisch corpus files
│   │   └── processed/            # Tokenized & mapped JSON/Parquet files
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataset.py            # PyTorch Dataset & character/token collators
│   │   ├── preprocess.py         # Cleans, validates, and splits the raw corpus
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── char_cnn.py       # Character-level CNN feature extractor
│   │   │   └── tcn_tagger.py     # 1D Temporal Convolutional Network sequence tagger
│   │   ├── train.py              # Training loop with mixed precision & logging
│   │   └── evaluate.py           # Classification metrics and benchmarking
│   ├── config.yaml               # Model and training configurations
│   ├── notebooks/                # Exploratory data analysis & error analysis
│   └── tests/                    # Unit tests for shape checking & padding
├── requirements.txt
└── README.md
```

## Project Overview

Standard autoregressive LLMs and heavy Transformer architectures pose latency and memory bottlenecks on resource-constrained edge devices. Recurrent architectures (like BiLSTM-CRF) encounter sequential processing constraints. This project explores whether a lightweight **1D-TCN** can match transformer-level token classification performance ($F_1$-score) while providing superior throughput (tokens/sec) and a reduced memory footprint.

### Target Classification Labels

- **DE**: German tokens and morphologically integrated forms.
- **EN**: English words and loanword insertions.
- **NEUT**: Punctuation, numbers, and neutral symbols.

The original corpus contains more detailed annotation categories, which will be consolidated into these three target labels during preprocessing.

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (recommended for training)

### Installation

1. Clone the repository and navigate into the workspace:

   ```bash
   git clone https://github.com/Damien-88/English_German_CodeSwitching.git
   cd English_German_CodeSwitching
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

4. Move into the project source directory, where the preprocessing, training, and evaluation scripts live:

   ```bash
   cd denglish_tcn_detector
   ```

---

## Project Architecture

```text
Input Sentence
      |
      v
   Tokenizer
      |
      v
 Character Embedding
      |
      v
[Character-Level CNN]
      |
      v
[1D Causal Dilated TCN]
      |
      v
 Linear + Softmax Tagging
      |
      v
 [DE, EN, NEUT]
```

- **Character-Level CNN (`src/models/char_cnn.py`)**: Extracts subword features, prefixes, and suffixes to handle productive compounding (*Komposita*) and mixed morphology (e.g., *mergen*). It uses character embeddings, convolutional filters of sizes 2, 3, and 4, and global max pooling to produce fixed-size token representations.
- **1D-TCN Backbone (`src/models/tcn_tagger.py`)**: Utilizes causal convolutions, dilation factors, and residual blocks to widen the receptive field without sequential inference bottlenecks.

### Initial TCN Configuration

The proposed initial architecture uses the following configurations:

| Parameter | Initial Configuration |
|---|---|
| Residual blocks | 4 |
| Convolutional layers per block | 2 |
| Kernel size | 3 |
| Channels per block | 128 |
| Dilation rates | 1, 2, 4, 8 |
| Dropout rate | 0.2 |
| Activation | ReLU |
| Weight normalization | Enabled |
| Residual connections | Enabled |

The resulting theoretical receptive field is 61 tokens, allowing the model to incorporate information from the current token and up to 60 preceding tokens.

---

## Usage

All commands below are run from `denglish_tcn_detector/` (see Installation step 4).

### Preprocessing

To clean the raw Denglisch corpus, map the original annotations to the project's three target labels, split the dataset into training, validation, and test subsets, and build the character vocabulary:

```bash
python src/preprocess.py --raw_data data/raw/denglisch.json --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

The dataset will be divided into:

- Training: 80%
- Validation: 10%
- Testing: 10%

A fixed random seed will be used to support reproducibility.

### Training

> **Note:** `train.py` currently defines the planned CLI interface below; the training loop itself is still under active development.

To train the 1D-TCN model on preprocessed Denglisch corpus data:

```bash
python src/train.py --data_dir data/processed --epochs 10 --batch_size 64 --lr 1e-3
```

The initial training configuration will use 10 epochs, a batch size of 64, a learning rate of 0.001, and the Adam optimizer. Mixed-precision training will be enabled when supported by the available hardware.

### Evaluation & Benchmarking

> **Note:** `evaluate.py` currently defines the planned CLI interface below; the evaluation logic itself is still under active development.

To evaluate token classification performance and measure inference throughput (tokens/sec):

```bash
python src/evaluate.py --checkpoint ../checkpoints/best_tcn.pt --test_data data/processed/test.json
```

The evaluation will report:

- Overall accuracy
- Per-class precision, recall, and F1-score
- Macro-F1
- Weighted-F1
- Confusion matrix
- Inference throughput (tokens/sec)
- Memory consumption

---

## Research Questions Addressed

- **RQ1**: TCN vs. BiLSTM-CRF vs. Transformer $F_1$-score comparison.
- **RQ2**: Impact of character-level CNN extraction on complex word boundaries (e.g., German-English compounds).
- **RQ3**: Throughput and memory footprint trade-offs on low-latency streaming pipelines.

---

## References

Osmelak, D., & Wintner, S. (2023). The Denglisch corpus of German-English code-switching. *Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP*, 42–51. https://doi.org/10.18653/v1/2023.sigtyp-1.5

---

## License

This project is developed for academic research under the City University of Seattle CS 687 Capstone curriculum.