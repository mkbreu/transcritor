"""Transcrição local (offline) com o NVIDIA Parakeet TDT 0.6B v3 via sherpa-onnx.

Estratégia para caber em pouca memória:
  1. o detector de fala (Silero VAD) acha os trechos com voz, lendo o WAV aos poucos;
  2. os trechos são agrupados em pedaços de até ~20 s, cortados nas pausas;
  3. cada pedaço é transcrito separadamente (memória constante, qualquer duração de arquivo).
O resultado de cada pedaço é gravado no cache: se o processo cair, continua de onde parou.
"""

from __future__ import annotations

import hashlib
import json
import logging
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from . import models
from .media import SAMPLE_RATE, read_wav
from .transcript import Word

log = logging.getLogger(__name__)

ENGINE_NAME = "NVIDIA Parakeet TDT 0.6B v3 (local, offline)"
MAX_CHUNK_SECONDS = 20.0
MAX_GAP_INSIDE_CHUNK = 1.0
PAD_SECONDS = 0.2

ProgressFn = Callable[[float, float], None]


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    start: float
    end: float


def detect_speech(
    wav_path: Path, threads: int, progress: ProgressFn | None = None
) -> list[tuple[float, float]]:
    """Trechos com fala (início, fim) em segundos, usando o Silero VAD."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(models.VAD.path)
    config.silero_vad.threshold = 0.45
    config.silero_vad.min_silence_duration = 0.3
    config.silero_vad.min_speech_duration = 0.2
    config.silero_vad.max_speech_duration = MAX_CHUNK_SECONDS
    config.sample_rate = SAMPLE_RATE
    config.num_threads = 1
    window = config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=MAX_CHUNK_SECONDS * 3)

    segments: list[tuple[float, float]] = []

    def drain() -> None:
        while not vad.empty():
            segment = vad.front
            start = segment.start / SAMPLE_RATE
            segments.append((start, start + len(segment.samples) / SAMPLE_RATE))
            vad.pop()

    with wave.open(str(wav_path), "rb") as source:
        total = source.getnframes()
        block = window * 200  # ~6 s por leitura
        done = 0
        while done < total:
            raw = source.readframes(block)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            for offset in range(0, len(samples) - window + 1, window):
                vad.accept_waveform(samples[offset:offset + window])
            done += len(samples)
            drain()
            if progress:
                progress(done / SAMPLE_RATE, total / SAMPLE_RATE)
        vad.flush()
        drain()
    return segments


def plan_chunks(speech: list[tuple[float, float]], duration: float) -> list[Chunk]:
    """Agrupa os trechos de fala em pedaços de até ~20 s, sempre cortando em pausas."""
    chunks: list[Chunk] = []
    current: list[float] | None = None
    for start, end in speech:
        if current and start - current[1] <= MAX_GAP_INSIDE_CHUNK and end - current[0] <= MAX_CHUNK_SECONDS:
            current[1] = end
            continue
        if current:
            chunks.append(Chunk(len(chunks), current[0], current[1]))
        current = [start, end]
    if current:
        chunks.append(Chunk(len(chunks), current[0], current[1]))
    # Folga nas pontas para não cortar sílabas, sem passar do meio da pausa com o vizinho
    # (assim nenhum trecho de áudio é transcrito duas vezes).
    padded: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        low = (chunks[i - 1].end + chunk.start) / 2 if i else 0.0
        high = (chunk.end + chunks[i + 1].start) / 2 if i + 1 < len(chunks) else duration
        start = max(low, chunk.start - PAD_SECONDS, 0.0)
        end = min(high, chunk.end + PAD_SECONDS, duration)
        padded.append(Chunk(i, start, end))
    return padded


class ParakeetTranscriber:
    """Carrega o modelo uma vez e transcreve pedaços de áudio."""

    def __init__(self, threads: int) -> None:
        spec = models.PARAKEET
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(spec.file("encoder.int8.onnx")),
            decoder=str(spec.file("decoder.int8.onnx")),
            joiner=str(spec.file("joiner.int8.onnx")),
            tokens=str(spec.file("tokens.txt")),
            num_threads=threads,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )

    def transcribe(self, samples: np.ndarray, offset: float) -> list[Word]:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        result = stream.result
        return tokens_to_words(result.tokens, result.timestamps, result.durations, offset)

    def close(self) -> None:
        self._recognizer = None


def tokens_to_words(
    tokens: list[str], starts: list[float], durations: list[float], offset: float
) -> list[Word]:
    """Junta os pedaços de palavra do modelo (" Bo", "a", ".") em palavras com tempo.

    Um pedaço que começa com espaço inicia palavra nova; um pedaço que é só espaço
    (o modelo usa isso antes de números: " ", "4", "5") faz o pedaço seguinte iniciar uma.
    """
    words: list[Word] = []
    starts_new_word = True
    for index, token in enumerate(tokens):
        if not token:
            continue
        if not token.strip():
            starts_new_word = True
            continue
        start = offset + starts[index]
        length = durations[index] if index < len(durations) else 0.08
        end = start + max(length, 0.02)
        is_punctuation = not any(ch.isalnum() for ch in token)
        begins_word = token.startswith(" ") or not words
        if not begins_word and starts_new_word and not is_punctuation:
            begins_word = True  # ex.: "4" logo depois do pedaço " "
        if begins_word:
            words.append(Word(token.strip(), start, end))
        else:
            words[-1].text += token
            if not is_punctuation:  # a pontuação não estica o fim da palavra
                words[-1].end = end
        starts_new_word = False
    return words


def transcribe_file(
    wav_path: Path,
    duration: float,
    get_transcriber: Callable[[], ParakeetTranscriber],
    progress_file: Path,
    threads: int,
    on_progress: ProgressFn | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> list[Word]:
    """Transcreve o WAV inteiro, retomando do cache (`progress_file`) se já houver pedaços prontos.

    `get_transcriber` só é chamado se ainda houver pedaços a transcrever (o modelo é carregado
    sob demanda e reaproveitado entre arquivos; quem chama decide quando liberá-lo).
    """
    if on_stage:
        on_stage("Procurando os trechos com fala")
    speech = detect_speech(wav_path, threads, on_progress)
    chunks = plan_chunks(speech, duration)
    signature = hashlib.sha1(
        ";".join(f"{c.start:.2f}-{c.end:.2f}" for c in chunks).encode("ascii")
    ).hexdigest()[:16]
    done_chunks = _load_progress(progress_file, signature)
    log.info("%s: %d trechos de fala → %d pedaços (%d já prontos no cache)",
             wav_path.name, len(speech), len(chunks), len(done_chunks))
    pending = [chunk for chunk in chunks if chunk.index not in done_chunks]
    if on_stage:
        on_stage("Transcrevendo")
    total_audio = sum(chunk.end - chunk.start for chunk in chunks) or 1.0
    finished_audio = sum(chunk.end - chunk.start for chunk in chunks if chunk.index in done_chunks)
    if pending:
        transcriber = get_transcriber()
        with open(progress_file, "a", encoding="utf-8") as journal:
            for chunk in pending:
                samples = read_wav(wav_path, chunk.start, chunk.end)
                words = transcriber.transcribe(samples, chunk.start)
                record = {
                    "i": chunk.index,
                    "w": [[w.text, round(w.start, 3), round(w.end, 3)] for w in words],
                }
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()
                done_chunks[chunk.index] = words
                finished_audio += chunk.end - chunk.start
                if on_progress:
                    on_progress(finished_audio / total_audio * duration, duration)
    words: list[Word] = []
    for chunk in chunks:
        words.extend(done_chunks.get(chunk.index, []))
    return words


def _load_progress(path: Path, signature: str) -> dict[int, list[Word]]:
    """Lê os pedaços já transcritos. A 1ª linha guarda a "assinatura" da divisão em pedaços:
    se não bater (outra versão do programa), o diário é descartado e recomeça do zero."""
    done: dict[int, list[Word]] = {}
    header = json.dumps({"plano": signature})
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != header:
        path.write_text(header + "\n", encoding="utf-8")
        return done
    valid = [header]
    for line in lines[1:]:
        try:
            record = json.loads(line)
            done[int(record["i"])] = [Word(text, start, end) for text, start, end in record["w"]]
            valid.append(line)
        except (ValueError, KeyError, TypeError):
            # Linha cortada por uma queda: regrava só as linhas boas e continua dali.
            path.write_text("\n".join(valid) + "\n", encoding="utf-8")
            break
    return done
