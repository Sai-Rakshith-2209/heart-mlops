"""Central configuration: paths, schema and feature groups."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "heart_raw.csv"
CLEAN_PATH = DATA_DIR / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
FIG_DIR = ROOT / "reports" / "figures"

# Official UCI source (Cleveland subset, the one used in virtually all literature).
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# Mirror of the same 303 Cleveland rows (ISLR "Heart.csv", string-coded categoricals).
MIRROR_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
TARGET = "target"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Valid value domains (UCI codebook) used for cleaning and API validation.
CATEGORY_DOMAINS = {
    "sex": [0, 1],
    "fbs": [0, 1],
    "exang": [0, 1],
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}
NUMERIC_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 250),
    "chol": (80, 650),
    "thalach": (50, 230),
    "oldpeak": (0.0, 10.0),
    "ca": (0, 3),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
