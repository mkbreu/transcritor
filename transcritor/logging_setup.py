"""Registro (log): um arquivo por execução na pasta logs, guardando só os mais recentes.

O log guarda o passo a passo e os erros técnicos, mas nunca o texto das conversas.
"""

from __future__ import annotations

import contextlib
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

from .paths import LOGS_DIR

KEEP_LOGS = 30


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOGS_DIR / f"transcritor_{stamp}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    for noisy in ("urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _prune_old_logs()
    logging.getLogger("transcritor").info(
        "Início | Python %s | %s %s | %s",
        sys.version.split()[0], platform.system(), platform.version(), platform.processor(),
    )
    return log_path


def _prune_old_logs() -> None:
    logs = sorted(LOGS_DIR.glob("transcritor_*.log"))
    for old in logs[:-KEEP_LOGS]:
        with contextlib.suppress(OSError):
            old.unlink()
