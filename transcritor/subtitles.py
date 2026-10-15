"""Leitura robusta de legendas SRT e WebVTT.

Corrige os problemas do script antigo: BOM, arquivos em ANSI/UTF-16, quebras de linha
do Windows, linhas de texto só com números (que eram apagadas) e blocos malformados.
Também reconhece rótulos de falante já presentes na legenda.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUBTITLE_EXTENSIONS = (".srt", ".vtt")


class SubtitleError(ValueError):
    """Legenda vazia ou ilegível."""


@dataclass(slots=True)
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None = None


_TIME = r"(?:(\d{1,3}):)?(\d{1,2}):(\d{1,2})(?:[,.](\d{1,3}))?"
_TIMING_LINE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")
_INDEX_LINE = re.compile(r"^\s*\d+\s*$")

# Rótulos de falante comuns em legendas exportadas por outras ferramentas.
_SPEAKER_PATTERNS = (
    re.compile(r"^<v(?:\.[^ >]*)?\s+([^>]+)>\s*"),  # WebVTT: <v Nome>texto
    re.compile(
        r"^\[?\s*((?:speaker|falante|locutor|interlocutor|pessoa|orador)[ _-]?\w{1,3})\s*\]?\s*[:\-–]\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^\[\s*([^\]\n]{1,40}?)\s*\]\s*:\s*"),  # [Nome]: texto
)
_TAGS = re.compile(r"</?(?:i|b|u|font|c|lang|ruby|rt|span|v)(?:[.\s][^>]*)?>", re.IGNORECASE)
_VTT_INLINE_TIME = re.compile(r"<\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}>")
_ASS_TAGS = re.compile(r"\{\\[^}]*\}")
_SPACES = re.compile(r"\s+")


def read_text_file(path: Path) -> str:
    """Lê texto detectando a codificação (UTF-8 com/sem BOM, UTF-16 ou ANSI do Windows)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def parse_subtitle_file(path: Path) -> list[Cue]:
    cues = parse_subtitles(read_text_file(path))
    if not cues:
        raise SubtitleError(f"Nenhuma fala encontrada na legenda {path.name}.")
    return cues


def parse_subtitles(content: str) -> list[Cue]:
    """Converte o conteúdo de um SRT/VTT em falas com início, fim e texto."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿").split("\n")
    cues: list[Cue] = []
    i = 0
    while i < len(lines):
        match = _TIMING_LINE.match(lines[i])
        if not match:
            i += 1
            continue
        start, end = _seconds(match.groups()[:4]), _seconds(match.groups()[4:])
        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            # Número da próxima legenda colado sem linha em branco: para aqui.
            next_is_timing = i + 1 < len(lines) and _TIMING_LINE.match(lines[i + 1])
            if _TIMING_LINE.match(lines[i]) or (_INDEX_LINE.match(lines[i]) and next_is_timing):
                break
            text_lines.append(lines[i])
            i += 1
        cue = _make_cue(start, max(end, start), text_lines)
        if cue is not None:
            cues.append(cue)
    cues.sort(key=lambda cue: cue.start)
    return cues


def _make_cue(start: float, end: float, text_lines: list[str]) -> Cue | None:
    speaker: str | None = None
    cleaned: list[str] = []
    for line in text_lines:
        line = _ASS_TAGS.sub("", line)
        line = _VTT_INLINE_TIME.sub("", line)
        if speaker is None:
            speaker, line = _extract_speaker(line)
        line = _TAGS.sub("", line).strip()
        if line.startswith("- "):  # travessão de diálogo das legendas
            line = line[2:]
        if line:
            cleaned.append(line)
    text = _SPACES.sub(" ", " ".join(cleaned)).strip()
    if not text:
        return None
    return Cue(start=start, end=end, text=text, speaker=speaker)


def _extract_speaker(line: str) -> tuple[str | None, str]:
    stripped = line.strip()
    for pattern in _SPEAKER_PATTERNS:
        match = pattern.match(stripped)
        if match:
            name = _SPACES.sub(" ", match.group(1).replace("_", " ")).strip()
            if name:
                return name, stripped[match.end():]
    return None, line


def _seconds(parts: tuple[str | None, ...]) -> float:
    hours, minutes, seconds, millis = parts
    total = int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(seconds or 0)
    if millis:
        total += int(millis.ljust(3, "0")) / 1000
    return float(total)
