"""Cache por arquivo: guarda o trabalho pesado (áudio preparado, transcrição, vozes).

Reabrir o mesmo arquivo é instantâneo (só renomear as vozes, por exemplo) e, se o
processamento for interrompido, ele continua de onde parou.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import CACHE_DIR

log = logging.getLogger(__name__)

_BLOCK = 4 * 1024 * 1024


def fingerprint(path: Path) -> str:
    """Identidade do arquivo pelo conteúdo (tamanho + início, meio e fim), sem ler tudo."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with open(path, "rb") as source:
        for offset in sorted({0, max(size // 2 - _BLOCK // 2, 0), max(size - _BLOCK, 0)}):
            source.seek(offset)
            digest.update(source.read(_BLOCK))
    return digest.hexdigest()[:24]


@dataclass(frozen=True)
class CacheEntry:
    folder: Path

    @property
    def audio_wav(self) -> Path:
        return self.folder / "audio_16k.wav"

    @property
    def asr_journal(self) -> Path:
        return self.folder / "transcricao_parcial.jsonl"

    @property
    def diarization(self) -> Path:
        return self.folder / "vozes.json"

    @property
    def transcript(self) -> Path:
        return self.folder / "transcricao.json"

    @property
    def meta(self) -> Path:
        return self.folder / "info.json"

    def read_meta(self) -> dict:
        try:
            return json.loads(self.meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_meta(self, **values) -> None:
        data = self.read_meta() | values | {"usado_em": time.time()}
        self.meta.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def entry_for(path: Path) -> CacheEntry:
    entry = CacheEntry(CACHE_DIR / fingerprint(path))
    entry.folder.mkdir(parents=True, exist_ok=True)
    entry.write_meta(arquivo=str(path))
    return entry


def cleanup(max_age_days: int) -> None:
    """Apaga do cache o que não é usado há mais de `max_age_days` dias."""
    if max_age_days <= 0 or not CACHE_DIR.exists():
        return
    limit = time.time() - max_age_days * 86400
    for folder in CACHE_DIR.iterdir():
        if not folder.is_dir():
            continue
        used = CacheEntry(folder).read_meta().get("usado_em", folder.stat().st_mtime)
        if used < limit:
            shutil.rmtree(folder, ignore_errors=True)
            log.info("Cache antigo removido: %s", folder.name)
