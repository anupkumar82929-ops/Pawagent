"""Paths and defaults — adjust DATA_ROOT if you move the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

# Default: sibling folders next to pawgent/
DATA_DIR = PROJECT_ROOT / "Mixed_breed_dataset_single"
EXCEL_PATH = PROJECT_ROOT / "Mixed_breed_dataset_normalized.xlsx"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "dog_breed_model.pt"
CLASS_NAMES_PATH = ARTIFACTS / "class_names.json"

IMAGE_SIZE = 224
TOP_K = 5
