
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
│   │   ├── corpus_converter.py   # Converts the raw Denglisch CSV release into sentence-level JSON
│   │   ├── dataset.py            # PyTorch Dataset & character/token collators
│   │   ├── preprocess.py         # Cleans, validates, and splits the raw corpus
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── char_cnn.py       # Character-level CNN feature extractor
│   │   │   └── tcn_tagger.py     # 1D Temporal Convolutional Network sequence tagger
│   │   ├── train.py              # Training loop with mixed precision & logging
│   │   └── evaluate.py           # Classification metrics and benchmarking
│   ├── config.yaml               # Model and training configurations
│   ├── conftest.py               # Adds src/ to sys.path for pytest
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

### Data Acquisition

The raw Denglisch Corpus files are **not** committed to this repository (they belong to Osmelak & Wintner, 2023). Download the manually annotated, "collapsed"-scheme release from the authors' GitHub repository and place it under `data/raw/`:

```bash
curl -L -o data/raw/Manu_corpus_collapsed.csv.zip https://raw.githubusercontent.com/HaifaCLG/Denglisch/main/corpus/Manu_corpus_collapsed.csv.zip
unzip data/raw/Manu_corpus_collapsed.csv.zip -d data/raw/
```

The larger, automatically annotated `corpus_collapsed.csv.zip` (~5M tokens) can be used the same way to grow the training split; see Section 5.3 of the project paper for the rationale behind prioritizing the manually annotated subset for validation/testing.

### Corpus Conversion

`src/corpus_converter.py` converts the official per-token CSV release (columns `source, user_name, sen_id, sen_num, token, categ`) into the sentence-level JSON format `preprocess.py` expects. It groups tokens into sentences by `(source, sen_id, sen_num)`, tags every sentence with a `comment_id` (`source:sen_id`) so comments never leak across splits, maps the corpus's collapsed annotation scheme onto the project's three target labels, and excludes tokens with no reliable single-language assignment:

| Original category | Meaning | Target label |
|---|---|---|
| `D`, `SD` | German / Shared-German | `DE` |
| `E`, `SE` | English / Shared-English | `EN` |
| `O` | Punctuation, numbers, symbols | `NEUT` |
| `M`, `SO` | Mixed morphology / Shared-Other (ambiguous) | *excluded* |

```bash
python src/corpus_converter.py --input data/raw/Manu_corpus_collapsed.csv --output data/raw/denglisch.json
```

The script logs the number of sentences/tokens retained and a full breakdown of excluded tokens by original category, as required by the paper's data-preparation methodology.

### Preprocessing

To clean the raw Denglisch corpus, map the original annotations to the project's three target labels, split the dataset into training, validation, and test subsets, and build the character vocabulary:

```bash
python src/preprocess.py --raw_data data/raw/denglisch.json --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

The dataset will be divided into:

- Training: 80%
- Validation: 10%
- Testing: 10%

When the input records carry a `comment_id` (as produced by `corpus_converter.py`), the split is performed at the comment level so that all sentences from the same Reddit comment stay in the same split, preventing data leakage. A fixed random seed is used throughout to support reproducibility.

### Training

To train the 1D-TCN model on preprocessed Denglisch corpus data:

```bash
python src/train.py --data_dir data/processed --epochs 10 --batch_size 64 --lr 1e-3
```

The initial training configuration uses 10 epochs, a batch size of 64, a learning rate of 0.001, and the Adam optimizer, all overridable via `config.yaml` or the corresponding CLI flags. Mixed-precision training is enabled automatically on CUDA devices. The checkpoint with the best validation macro-F1 is saved to `../checkpoints/best_tcn.pt` along with the character vocabulary, label map, and architecture hyperparameters needed to reload it. Pass `--no_char_cnn` to run the character-CNN ablation described in RQ2.

### Evaluation & Benchmarking

To evaluate token classification performance and measure inference throughput (tokens/sec):

```bash
python src/evaluate.py --checkpoint ../checkpoints/best_tcn.pt --test_data data/processed/test.json
```


The evaluation reports:

- Overall accuracy
- Per-class precision, recall, and F1-score
- Macro-F1
- Weighted-F1
- Confusion matrix
- Inference throughput (tokens/sec)
- Memory consumption (peak GPU memory, or resident set size on CPU)

A full JSON report is also written next to the checkpoint (e.g. `../checkpoints/best_tcn_eval_report.json`) for later comparison across runs/baselines.

### Testing

Unit tests cover tensor shapes, padding/masking behavior, causal-convolution correctness, label mapping, and comment-level data-leakage prevention:

```bash
python -m pytest tests
```

---

## Analysis

This section reports what was actually observed when the pipeline was run end-to-end on the
manually annotated Denglisch Corpus subset (`Manu_corpus_collapsed.csv`), combining the
exploratory notebooks (`notebooks/data_inspection.ipynb`, `notebooks/data_analysis.ipynb`) with a
real training/evaluation run. It is purely descriptive; interpretation of what these numbers mean
for the research questions is left to the Findings/Conclusion/Future Work sections (to follow).

### Raw Corpus Composition

The manually annotated subset contains 75,013 token rows across 34 subreddit sources, grouped into
962 unique Reddit comments and ~4,202 sentences — closely matching the ~950 comments/~4,200
sentences reported by Osmelak & Wintner (2023). Under the corpus's collapsed annotation scheme,
token counts are:

| Category | Meaning | Count |
|---|---|---|
| `E` | English | 29,918 |
| `D` | German | 29,730 |
| `O` | Other (punctuation/numbers) | 12,505 |
| `SO` | Shared-Other (ambiguous) | 1,108 |
| `SD` | Shared-German | 807 |
| `SE` | Shared-English | 699 |
| `M` | Mixed morphology (ambiguous) | 246 |

### Corpus Conversion (`corpus_converter.py`)

Mapping these onto the project's `DE`/`EN`/`NEUT` labels and excluding the two ambiguous
categories (`M`, `SO`) retains 73,656 of 75,010 non-empty tokens (98.19%), with 1,354 tokens
(1.81%) excluded as having no reliable single-language assignment. The resulting label
distribution is close to balanced between the two languages: `EN` 30,617, `DE` 30,537, `NEUT`
12,502. 1,509 of 4,202 sentences (35.9%) contain both German and English tokens, confirming the
corpus is a genuine code-switching resource rather than one dominated by monolingual segments.

### Processed Dataset Integrity (Train/Val/Test)

| Split | Sentences | Tokens | Unique comments | DE % | EN % | NEUT % |
|---|---|---|---|---|---|---|
| train | 3,347 | 59,196 | 769 | 41.07 | 41.77 | 17.16 |
| val | 405 | 7,042 | 96 | 45.12 | 39.38 | 15.51 |
| test | 449 | 7,418 | 97 | 41.09 | 42.05 | 16.86 |

- **No comment-level leakage**: the 769/96/97 comment sets are pairwise disjoint, and combined
  split tokens (73,656) reconcile exactly with the converter's retained-token count.
- **Label balance holds across splits**: the random comment-level split did not introduce a
  meaningful label-distribution skew between train/val/test.
- **Token length is well covered** by `max_word_len=20`: mean token length is 4.6 characters, and
  only 0.40% of training tokens are long enough to be truncated.
- **Character vocabulary** (159 entries, including `<PAD>`/`<UNK>`) covers ASCII letters/digits/
  punctuation, German-specific letters (`ä ö ü ß Ä Ö Ü`), and a long tail of other Unicode (emoji,
  accented loanwords) present in the Reddit text. Out-of-vocabulary character rates on val/test are
  negligible (0.040% / 0.012%), so `<UNK>` fallback is rarely exercised.
- **Code-switching density** is consistent across splits (31.6–36.5% of sentences mix DE/EN),
  averaging 2.3–2.7 switch points per code-switched sentence — short, localized switches rather
  than pervasive mixing, matching Osmelak & Wintner's (2023) characterization of the corpus.

### Training & Evaluation Results

A full pipeline run (preprocessing → training → evaluation) was executed on CPU with the default
architecture (4 residual blocks, 128 channels, kernel 3, dilations 1/2/4/8) for 2 of the
configured 10 epochs, to validate correctness end-to-end rather than to reach a fully converged
model. Results on the held-out test split (`checkpoints/best_tcn_eval_report.json`):

| Metric | Value |
|---|---|
| Overall accuracy | 95.03% |
| Macro-F1 | 0.9553 |
| Weighted-F1 | 0.9502 |
| DE F1 / EN F1 / NEUT F1 | 0.95 / 0.94 / 0.98 |
| Inference throughput (CPU) | ~15,063 tokens/sec |
| Peak resident memory (CPU) | ~631 MB |

Confusion matrix (rows = true, columns = predicted; order `DE, EN, NEUT`):

```text
          pred DE  pred EN  pred NEUT
true DE      2924      105         19
true EN       203     2903         13
true NEUT      11       18       1222
```

The dominant error mode is DE↔EN confusion (105 DE tokens predicted as EN, 203 EN tokens predicted
as DE), while `NEUT` is classified almost perfectly, since punctuation and numbers are rarely
ambiguous. This is consistent with prior work (Sterner & Teufel, 2023) identifying shared
vocabulary and code-switch boundaries as the primary difficulty in German-English language
identification, as opposed to clearly monolingual tokens.

Note: these are preliminary results from a 2-epoch smoke run intended to validate pipeline
correctness (data flow, loss masking, checkpointing, metric computation); they are not a tuned or
fully converged result, and no baseline (BiLSTM-CRF/Transformer) or ablation (`--no_char_cnn`)
comparison has been run yet.

---

## Research Questions Addressed

- **RQ1**: TCN vs. BiLSTM-CRF vs. Transformer $F_1$-score comparison. *(The 1D-TCN detector in this repository is fully implemented and evaluated; BiLSTM-CRF and Transformer baseline training scripts are the next planned addition for a direct comparison.)*
- **RQ2**: Impact of character-level CNN extraction on complex word boundaries (e.g., German-English compounds). Use `--no_char_cnn` with `src/train.py` to train the ablated variant for comparison.
- **RQ3**: Throughput and memory footprint trade-offs on low-latency streaming pipelines, reported by `src/evaluate.py`.

---

## References

Osmelak, D., & Wintner, S. (2023). The Denglisch corpus of German-English code-switching. *Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP*, 42–51. https://doi.org/10.18653/v1/2023.sigtyp-1.5

---

## License

This project is developed for academic research under the City University of Seattle CS 687 Capstone curriculum.