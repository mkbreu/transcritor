"""Pastas usadas pelo Transcritor (só biblioteca padrão: também usado pelo instalar.py)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "modelos"
DATA_DIR = ROOT / "dados"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = ROOT / "logs"
CONFIG_FILE = ROOT / "config.toml"
STATE_FILE = DATA_DIR / "estado.json"
VOICES_FILE = DATA_DIR / "vozes.json"
