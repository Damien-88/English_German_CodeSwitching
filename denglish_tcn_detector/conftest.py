import sys
from pathlib import Path

# src/ modules (dataset.py, preprocess.py, models/...) are written to be run
# as top-level scripts (e.g. `python src/train.py`), not as a package, so
# tests need the same sys.path setup: add src/ directly rather than
# importing via `denglish_tcn_detector.src...`.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))