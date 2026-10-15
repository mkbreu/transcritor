"""Instalador do Transcritor (usa só a biblioteca padrão do Python).

O que faz (só o que estiver faltando, pode rodar quantas vezes quiser):
  1. cria o ambiente isolado .venv (não mexe no Python do sistema);
  2. instala os pacotes de requirements.txt (de novo só quando o arquivo mudar);
  3. baixa os modelos de IA para a pasta modelos\\ (conferindo tamanho e SHA-256);
  4. cria o atalho "Transcritor" na Área de Trabalho.

Uso:
  py -3.12 instalar.py                  instalação completa
  py -3.12 instalar.py --se-necessario  checagem rápida (usada pelo Transcritor.bat)
  py -3.12 instalar.py --dev            inclui as ferramentas de teste (pytest, ruff)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transcritor import models  # noqa: E402  (módulo só com biblioteca padrão)

VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS = ROOT / "requirements.txt"
DEV_REQUIREMENTS = ROOT / "requirements-dev.txt"
REQUIREMENTS_STAMP = VENV_DIR / ".requirements.sha256"
SHORTCUT_STAMP = VENV_DIR / ".atalho_criado"
LAUNCHER = ROOT / "Transcritor.bat"
ICON = ROOT / "transcritor.ico"
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


class InstallError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Instala ou verifica o Transcritor.")
    parser.add_argument("--se-necessario", action="store_true", help="checagem rápida")
    parser.add_argument("--dev", action="store_true", help="instala pytest e ruff")
    parser.add_argument("--sem-modelos", action="store_true", help="não baixa os modelos de IA")
    parser.add_argument("--sem-atalho", action="store_true", help="não cria o atalho")
    args = parser.parse_args()

    if sys.version_info < (3, 12):  # noqa: UP036  (roda com o Python do sistema, que pode ser antigo)
        print("[X] É preciso o Python 3.12 ou mais novo (encontrado: "
              f"{sys.version.split()[0]}).")
        return 1
    try:
        ensure_venv()
        ensure_requirements(dev=args.dev)
        if not args.se_necessario:
            check_imports()
            if not args.sem_modelos:
                ensure_models()
        if not args.sem_atalho:
            ensure_shortcut()
    except InstallError as exc:
        print(f"\n[X] {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n[!] Instalação interrompida. Rode de novo para continuar de onde parou.")
        return 1
    if not args.se_necessario:
        print("\n[OK] Tudo pronto. Use o atalho \"Transcritor\" na Área de Trabalho.")
    return 0


def ensure_venv() -> None:
    if VENV_PYTHON.is_file() and _runs(VENV_PYTHON, "-c", "import sys"):
        return
    if VENV_DIR.exists():
        print("Recriando o ambiente .venv (estava danificado)...")
        shutil.rmtree(VENV_DIR)
    else:
        print("Criando o ambiente isolado .venv...")
    _run([sys.executable, "-m", "venv", str(VENV_DIR)], "criar o ambiente .venv")


def ensure_requirements(dev: bool) -> None:
    files = [REQUIREMENTS, DEV_REQUIREMENTS] if dev else [REQUIREMENTS]
    stamp = _hash_files(files)
    current = REQUIREMENTS_STAMP.read_text(encoding="utf-8") if REQUIREMENTS_STAMP.exists() else ""
    if current == stamp or (not dev and current.startswith(stamp)):
        return
    print("Instalando os pacotes do Transcritor (pode levar alguns minutos)...")
    req = DEV_REQUIREMENTS if dev else REQUIREMENTS
    _run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
         "--no-input", "-q", "-r", str(req)],
        "instalar os pacotes (confira a internet)",
    )
    # Com --dev a marca guarda os dois arquivos: uma instalação comum continua válida.
    REQUIREMENTS_STAMP.write_text(_hash_files(files), encoding="utf-8")


def check_imports() -> None:
    probe = "import sherpa_onnx, av, numpy, rich"
    result = subprocess.run([str(VENV_PYTHON), "-c", probe], capture_output=True, text=True)
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["?"]
    hint = ""
    if "DLL" in detail[0]:
        hint = ("\n    Provável causa: falta o \"Microsoft Visual C++ Redistributable\". "
                f"Instale-o em {VC_REDIST_URL} e rode de novo.")
    raise InstallError(f"Os pacotes foram instalados, mas não carregam: {detail[0]}{hint}")


def ensure_models() -> None:
    missing = models.missing_models()
    if not missing:
        return
    total = sum(spec.size for spec in missing)
    print(f"\nBaixando os modelos de IA ({models.human_size(total)}) para a pasta modelos\\")
    print("Se a internet cair, rode de novo: o download continua de onde parou.")
    if models.SEGMENTATION in missing:
        print(f"Aviso de licença — separação de vozes: {models.SEGMENTATION_LICENSE_NOTICE}")
    print()
    bar = _ConsoleBar()
    try:
        models.ensure_models(missing, progress=bar.update, log=bar.log)
    except models.ModelDownloadError as exc:
        raise InstallError(str(exc)) from exc
    finally:
        bar.finish()


def ensure_shortcut() -> None:
    if SHORTCUT_STAMP.exists():
        return
    icon = str(ICON) if ICON.is_file() else r"%SystemRoot%\System32\SndVol.exe"
    script = f"""
$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$link = $ws.CreateShortcut((Join-Path $desktop 'Transcritor.lnk'))
$link.TargetPath = {_ps_quote(str(LAUNCHER))}
$link.WorkingDirectory = {_ps_quote(str(ROOT))}
$link.IconLocation = {_ps_quote(icon + ",0")}
$link.Description = 'Transcrever áudio, vídeo ou legenda SRT em texto'
$link.Save()
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Atalho é conveniência: avisa, mas não impede o uso.
        print(f"[!] Não consegui criar o atalho na Área de Trabalho: {result.stderr.strip()}")
        return
    SHORTCUT_STAMP.write_text("ok", encoding="utf-8")
    print("[OK] Atalho \"Transcritor\" criado na Área de Trabalho.")


class _ConsoleBar:
    """Barra de progresso simples para o console (sem pacotes externos)."""

    def __init__(self) -> None:
        self._label = ""
        self._last = 0.0
        self._started = time.monotonic()

    def log(self, message: str) -> None:
        self.finish()
        print(message)
        self._started = time.monotonic()

    def update(self, label: str, done: int, total: int) -> None:
        now = time.monotonic()
        if label != self._label:
            self.finish()
            if label.startswith("Descompactando"):
                print(f"{label}...")
            self._label = label
            self._started = now
        elif now - self._last < 0.25 and done < total:
            return
        self._last = now
        fraction = min(done / total, 1.0) if total else 0.0
        filled = int(fraction * 30)
        speed = done / max(now - self._started, 1e-6)
        line = (f"\r  [{'#' * filled}{'.' * (30 - filled)}] {fraction * 100:5.1f}%  "
                f"{models.human_size(done)} de {models.human_size(total)}  "
                f"({models.human_size(speed)}/s)   ")
        print(line, end="", flush=True)

    def finish(self) -> None:
        if self._label:
            print()
            self._label = ""


def _hash_files(files: list[Path]) -> str:
    return "+".join(hashlib.sha256(path.read_bytes()).hexdigest()[:16] for path in files)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _runs(python: Path, *args: str) -> bool:
    try:
        return subprocess.run([str(python), *args], capture_output=True).returncode == 0
    except OSError:
        return False


def _run(command: list[str], action: str) -> None:
    result = subprocess.run(command)
    if result.returncode != 0:
        raise InstallError(f"Não foi possível {action}.")


if __name__ == "__main__":
    sys.exit(main())
