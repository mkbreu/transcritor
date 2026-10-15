"""Janelas nativas do Windows (Explorer) para escolher arquivos e onde salvar."""

from __future__ import annotations

import contextlib
import ctypes
import tkinter as tk
from collections.abc import Iterator
from pathlib import Path
from tkinter import filedialog

from . import winsys
from .media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from .subtitles import SUBTITLE_EXTENSIONS

_dpi_configured = False


def _patterns(extensions: tuple[str, ...]) -> str:
    return " ".join(f"*{ext}" for ext in extensions)


_INPUT_TYPES = [
    ("Vídeo, áudio ou legenda", _patterns(VIDEO_EXTENSIONS + AUDIO_EXTENSIONS + SUBTITLE_EXTENSIONS)),
    ("Vídeos", _patterns(VIDEO_EXTENSIONS)),
    ("Áudios", _patterns(AUDIO_EXTENSIONS)),
    ("Legendas (SRT/VTT)", _patterns(SUBTITLE_EXTENSIONS)),
    ("Todos os arquivos", "*.*"),
]


@contextlib.contextmanager
def _hidden_root() -> Iterator[tk.Tk]:
    """Janela Tk invisível, sempre na frente, só para servir de "dona" do diálogo."""
    global _dpi_configured
    if not _dpi_configured:
        with contextlib.suppress(Exception):  # nitidez em telas com zoom
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        _dpi_configured = True
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        yield root
    finally:
        root.destroy()
        winsys.focus_console()


def ask_input_files(initial_dir: str = "") -> list[Path]:
    with _hidden_root() as root:
        chosen = filedialog.askopenfilenames(
            parent=root,
            title="Transcritor — escolha um ou mais arquivos (vídeo, áudio ou legenda SRT)",
            initialdir=_existing_dir(initial_dir),
            filetypes=_INPUT_TYPES,
        )
    return [Path(item) for item in chosen] if chosen else []


def ask_save_text(initial_dir: Path, initial_name: str) -> Path | None:
    with _hidden_root() as root:
        chosen = filedialog.asksaveasfilename(
            parent=root,
            title="Onde salvar a transcrição (TXT)?",
            initialdir=str(initial_dir),
            initialfile=initial_name,
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt")],
            confirmoverwrite=True,
        )
    return Path(chosen) if chosen else None


def ask_save_audio(initial_dir: Path, initial_name: str, extension: str) -> Path | None:
    with _hidden_root() as root:
        chosen = filedialog.asksaveasfilename(
            parent=root,
            title="Onde salvar o áudio extraído do vídeo?",
            initialdir=str(initial_dir),
            initialfile=initial_name,
            defaultextension=extension,
            filetypes=[(f"Áudio {extension}", f"*{extension}")],
            confirmoverwrite=True,
        )
    return Path(chosen) if chosen else None


def ask_folder(initial_dir: str, title: str) -> Path | None:
    with _hidden_root() as root:
        chosen = filedialog.askdirectory(
            parent=root, title=title, initialdir=_existing_dir(initial_dir), mustexist=True
        )
    return Path(chosen) if chosen else None


def _existing_dir(candidate: str) -> str:
    if candidate and Path(candidate).is_dir():
        return candidate
    return str(Path.home() / "Videos") if (Path.home() / "Videos").is_dir() else str(Path.home())
