"""Configuration constants for detector training and evaluation."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

REQUIRED_COLUMNS = ["id", "text"]
OPTIONAL_COLUMNS = ["source", "generator_model", "attack_type", "attack_owner"]

SOURCE_MAP = {
    "human": 0,
    "ai": 1,
}

DEFAULT_ATTACK_TYPE = "none"
DEFAULT_ATTACK_OWNER = "none"

EPS = 1e-12
