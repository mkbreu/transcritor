"""Montagem do TXT final (pensado para ser lido por uma IA) e gravação segura em disco."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .sentences import Paragraph, group_paragraphs
from .transcript import OVERLAP_NAME, UNKNOWN_NAME, Transcript

SEPARATOR = "=" * 64


@dataclass(slots=True)
class RenderOptions:
    show_times: bool = True  # "[início – fim]" da gravação no começo de cada parágrafo
    one_sentence_per_line: bool = False


def render_text(transcript: Transcript, options: RenderOptions | None = None) -> str:
    options = options or RenderOptions()
    names = {label: transcript.display_name(label) for label in transcript.labels}
    paragraphs = group_paragraphs(transcript.sentences, names=names)
    lines = _header(transcript, paragraphs, names, options)
    lines += ["", SEPARATOR, ""]
    long_audio = transcript.duration >= 3600
    for paragraph in paragraphs:
        lines.extend(_paragraph_lines(transcript, paragraph, options, long_audio))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _header(
    transcript: Transcript,
    paragraphs: list[Paragraph],
    names: dict[str, str],
    options: RenderOptions,
) -> list[str]:
    title = {
        "legenda": "TRANSCRIÇÃO (convertida de legenda)",
        "video": "TRANSCRIÇÃO DE VÍDEO",
    }.get(transcript.source_kind, "TRANSCRIÇÃO DE ÁUDIO")
    created = transcript.created_at or datetime.now().strftime("%d/%m/%Y às %H:%M")
    lines = [
        title,
        f"Origem: {transcript.source_name}",
        f"Duração: {format_duration(transcript.duration)}",
        f"Processado em: {created}",
    ]
    if transcript.has_speakers:
        participants = list(dict.fromkeys(names[label] for label in transcript.labels))
        me = transcript.me_name
        participants = [f"{name} (eu)" if name == me else name for name in participants]
        if participants:
            lines.append(f"Participantes: {', '.join(participants)}")
    notes = ["Transcrição automática, em ordem cronológica; pode conter pequenos erros."]
    if not transcript.has_speakers:
        notes.append("Sem identificação de quem fala; parágrafos separados pelas pausas da fala.")
    if any(p.speaker is None and not p.overlap for p in paragraphs) and transcript.has_speakers:
        notes.append(f'"{UNKNOWN_NAME}" = trecho em que não foi possível saber quem falou.')
    if any(p.speaker is None and p.overlap for p in paragraphs):
        notes.append(f'"{OVERLAP_NAME}" = duas ou mais pessoas falando ao mesmo tempo.')
    if options.show_times:
        notes.append("[início – fim] = trecho da gravação em que aquela fala acontece.")
    lines.append("Observações:")
    lines += [f"- {note}" for note in notes]
    return lines


def _paragraph_lines(
    transcript: Transcript, paragraph: Paragraph, options: RenderOptions, long_audio: bool
) -> list[str]:
    def stamp(start: float, end: float) -> str:
        if not options.show_times:
            return ""
        return f"[{format_clock(start, long_audio)} – {format_clock(end, long_audio)}] "

    name = ""
    if transcript.has_speakers:
        name = transcript.display_name(paragraph.speaker, paragraph.overlap) + ": "
    if options.one_sentence_per_line:
        return [f"{stamp(s.start, s.end)}{name}{s.text}" for s in paragraph.sentences]
    return [f"{stamp(paragraph.start, paragraph.end)}{name}{paragraph.text}"]


def format_duration(seconds: float) -> str:
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def format_clock(seconds: float, with_hours: bool = False) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if with_hours or hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def suggested_output_name(source: Path) -> str:
    return f"{source.stem} - transcrição.txt"


def unique_path(path: Path) -> Path:
    """Se o arquivo já existir, devolve "nome (2).txt", "nome (3).txt"..."""
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def write_text_atomic(path: Path, text: str) -> None:
    """Grava em um temporário e só então troca pelo definitivo: nunca deixa arquivo pela metade.

    Levanta PermissionError se o destino estiver aberto e bloqueado (ex.: no Word).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\r\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
