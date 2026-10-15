"""Catálogo e download dos modelos de IA (só biblioteca padrão: também usado pelo instalar.py).

Os modelos vêm dos lançamentos oficiais do projeto sherpa-onnx (k2-fsa) no GitHub.
O download é retomado de onde parou se a internet cair, e cada arquivo é conferido
(tamanho e SHA-256) antes de ser usado.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .paths import MODELS_DIR

_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_USER_AGENT = "Transcritor/1.0 (+sherpa-onnx models)"

# progress(rótulo, bytes_feitos, bytes_totais)
ProgressFn = Callable[[str, int, int], None]


class ModelDownloadError(RuntimeError):
    """Falha ao baixar ou conferir um modelo."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    url: str
    size: int  # tamanho exato do arquivo baixado, em bytes
    sha256: str | None
    target: str  # arquivo (ou pasta, se for .tar.bz2) dentro de modelos\
    required_files: tuple[str, ...] = ()  # arquivos obrigatórios dentro da pasta extraída

    @property
    def is_archive(self) -> bool:
        return self.url.endswith(".tar.bz2")

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.target

    def file(self, name: str) -> Path:
        return self.path / name


PARAKEET = ModelSpec(
    key="parakeet",
    label="Transcrição — NVIDIA Parakeet TDT 0.6B v3 (int8)",
    url=f"{_BASE}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
    size=487_170_055,
    sha256="5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
    target="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    required_files=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
)
VAD = ModelSpec(
    key="vad",
    label="Detecção de fala — Silero VAD",
    url=f"{_BASE}/asr-models/silero_vad.onnx",
    size=643_854,
    sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    target="silero_vad.onnx",
)
SEGMENTATION = ModelSpec(
    key="segmentation",
    label="Separação de vozes — segmentação pyannote 3.0",
    url=f"{_BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    size=6_958_444,
    sha256=None,  # o lançamento não publica o SHA-256 deste arquivo; conferimos o tamanho
    target="sherpa-onnx-pyannote-segmentation-3-0",
    required_files=("model.onnx",),
)
# "recongition" (com erro de digitação) é o nome real da tag no GitHub.
_SPEAKER_MODELS = f"{_BASE}/speaker-recongition-models"

# Testes com vozes reais (ferramentas\avaliar_separacao.py): o WeSpeaker ResNet34 quase não
# distinguia pessoas diferentes (margem -0,08); CAM++ e TitaNet separam bem (margem ~+0,20)
# e acertaram igual (93% com 3 pessoas reais, 100% no diálogo em português). O TitaNet foi
# ~20% mais rápido e foi treinado também com ligações telefônicas: é o padrão.
CAMPPLUS = ModelSpec(
    key="embedding",
    label="Impressão vocal — 3D-Speaker CAM++ (zh/en)",
    url=f"{_SPEAKER_MODELS}/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    size=28_281_164,
    sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
    target="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
)
TITANET = ModelSpec(
    key="embedding",
    label="Impressão vocal — NVIDIA TitaNet small",
    url=f"{_SPEAKER_MODELS}/nemo_en_titanet_small.onnx",
    size=40_257_283,
    sha256="ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
    target="nemo_en_titanet_small.onnx",
)
EMBEDDING = TITANET  # escolhido pela avaliação com vozes reais

ALL_MODELS: tuple[ModelSpec, ...] = (PARAKEET, VAD, SEGMENTATION, EMBEDDING)

_OK_MARKER = ".completo"


def is_installed(spec: ModelSpec) -> bool:
    if spec.is_archive:
        return (spec.path / _OK_MARKER).exists() and all(
            spec.file(name).is_file() for name in spec.required_files
        )
    return spec.path.is_file() and spec.path.stat().st_size == spec.size


def missing_models(specs: tuple[ModelSpec, ...] = ALL_MODELS) -> list[ModelSpec]:
    return [spec for spec in specs if not is_installed(spec)]


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            text = f"{num_bytes:.0f} {unit}" if unit in ("B", "KB") else f"{num_bytes:.1f} {unit}"
            return text.replace(".", ",")
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def ensure_models(
    specs: list[ModelSpec] | tuple[ModelSpec, ...],
    progress: ProgressFn | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """Baixa, confere e instala os modelos que estiverem faltando."""
    todo = [spec for spec in specs if not is_installed(spec)]
    if not todo:
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    needed = sum(spec.size * (2 if spec.is_archive else 1) for spec in todo)
    free = shutil.disk_usage(MODELS_DIR).free
    if free < needed:
        raise ModelDownloadError(
            f"Espaço insuficiente no disco: precisa de {human_size(needed)}, "
            f"há {human_size(free)} livres."
        )
    for spec in todo:
        if log:
            log(f"Baixando {spec.label} ({human_size(spec.size)})")
        _install(spec, progress)


def _install(spec: ModelSpec, progress: ProgressFn | None) -> None:
    file_name = spec.url.rsplit("/", 1)[-1]
    download_path = MODELS_DIR / file_name
    _download(spec, download_path, progress)
    if not spec.is_archive:
        return  # o próprio arquivo baixado já é o modelo
    if spec.path.exists():
        shutil.rmtree(spec.path)  # extração anterior incompleta
    _extract(download_path, spec, progress)
    missing = [name for name in spec.required_files if not spec.file(name).is_file()]
    if missing:
        raise ModelDownloadError(
            f"O pacote {file_name} não contém os arquivos esperados: {', '.join(missing)}"
        )
    (spec.path / _OK_MARKER).write_text("ok", encoding="utf-8")
    download_path.unlink(missing_ok=True)


def _download(spec: ModelSpec, dest: Path, progress: ProgressFn | None, retries: int = 4) -> None:
    if dest.is_file() and dest.stat().st_size == spec.size and _sha_ok(spec, dest):
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            _download_once(spec, part, progress)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == retries:
                raise ModelDownloadError(
                    f"Não foi possível baixar {spec.label}. Verifique a internet e tente de novo. "
                    f"Detalhe: {exc}"
                ) from exc
            time.sleep(2**attempt)
    size = part.stat().st_size
    if size != spec.size:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"{spec.label}: tamanho inesperado ({size} bytes; esperado {spec.size})."
        )
    if not _sha_ok(spec, part):
        part.unlink(missing_ok=True)
        raise ModelDownloadError(f"{spec.label}: arquivo corrompido (SHA-256 não confere).")
    part.replace(dest)


def _download_once(spec: ModelSpec, part: Path, progress: ProgressFn | None) -> None:
    start = part.stat().st_size if part.exists() else 0
    if start > spec.size:
        part.unlink()
        start = 0
    if start == spec.size:
        return
    headers = {"User-Agent": _USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(spec.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # o pedaço já baixado está completo
            return
        raise
    with response:
        if start and response.status != 206:
            start = 0  # o servidor ignorou a retomada: recomeça do zero
        done = start
        with open(part, "ab" if start else "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(spec.label, done, spec.size)


def _sha_ok(spec: ModelSpec, path: Path) -> bool:
    if spec.sha256 is None:
        return True
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest() == spec.sha256


class _CountingReader:
    """Repassa as leituras do arquivo compactado contando os bytes (para a barra de progresso)."""

    def __init__(self, raw, callback: Callable[[int], None]) -> None:
        self._raw = raw
        self._callback = callback
        self._count = 0

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self._count += len(data)
        self._callback(self._count)
        return data


def _extract(archive: Path, spec: ModelSpec, progress: ProgressFn | None) -> None:
    label = f"Descompactando {spec.label}"
    total = archive.stat().st_size

    def report(done: int) -> None:
        if progress:
            progress(label, done, total)

    # Modo "r|bz2" (sequencial): descompacta uma única vez, sem voltar no arquivo.
    with open(archive, "rb") as raw, tarfile.open(
        fileobj=_CountingReader(raw, report), mode="r|bz2"
    ) as tar:
        for member in tar:
            tar.extract(member, MODELS_DIR, filter="data")
