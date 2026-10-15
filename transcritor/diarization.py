"""Separação de vozes ("quem falou quando") e atribuição de cada frase a uma pessoa.

Usa a segmentação pyannote + "impressões vocais" (sherpa-onnx), tudo na CPU.
"""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sherpa_onnx

from . import models
from .media import SAMPLE_RATE
from .sentences import make_sentence, split_word_groups
from .transcript import Sentence, Word, speaker_label

log = logging.getLogger(__name__)

# Limiar de agrupamento das vozes (distância). Maior = junta mais (menos vozes).
DEFAULT_THRESHOLD = 0.7
THRESHOLD_SPREAD = 0.4  # quanto a "sensibilidade" do config.toml desloca o limiar
# Passo da janela de análise (fração de 10 s). O padrão do pyannote (0,1) analisa cada
# trecho 10 vezes; 0,25 é bem mais leve com perda mínima de precisão.
WINDOW_SHIFT = 0.25

ProgressFn = Callable[[float, float], None]


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    label: str


def threshold_from_sensitivity(sensitivity: float) -> float:
    """Sensibilidade 0..1 (config.toml) → limiar. 0,5 = padrão; mais alto = separa mais vozes."""
    return DEFAULT_THRESHOLD + (0.5 - sensitivity) * THRESHOLD_SPREAD


class Diarizer:
    """Carrega os modelos uma vez e separa as vozes de vários arquivos."""

    def __init__(self, threads: int, sensitivity: float) -> None:
        self._threshold = threshold_from_sensitivity(sensitivity)
        self._threads = threads
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(self._config(-1))

    def _config(self, num_clusters: int) -> sherpa_onnx.OfflineSpeakerDiarizationConfig:
        return sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(models.SEGMENTATION.file("model.onnx")),
                    window_shift_ratio=WINDOW_SHIFT,
                ),
                num_threads=self._threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(models.EMBEDDING.path), num_threads=self._threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters, threshold=self._threshold
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )

    def run(
        self, samples: np.ndarray, num_speakers: int | None, progress: ProgressFn | None = None
    ) -> list[SpeakerTurn]:
        self._sd.set_config(self._config(num_speakers or -1))
        duration = len(samples) / SAMPLE_RATE

        def callback(done: int, total: int) -> int:
            if progress and total:
                progress(duration * done / total, duration)
            return 0

        result = self._sd.process(samples, callback=callback)
        raw = [(seg.start, seg.end, seg.speaker) for seg in result.sort_by_start_time()]
        return relabel(raw)

    def close(self) -> None:
        self._sd = None


def relabel(raw: list[tuple[float, float, int]]) -> list[SpeakerTurn]:
    """Troca os números das vozes por letras na ordem em que aparecem (A fala primeiro)."""
    order: dict[int, str] = {}
    turns = []
    for start, end, speaker in sorted(raw):
        if speaker not in order:
            order[speaker] = speaker_label(len(order))
        turns.append(SpeakerTurn(start, end, order[speaker]))
    return turns


# ------------------------------------------------------------------ palavra → voz


def assign_speakers(words: list[Word], turns: list[SpeakerTurn], snap: float = 0.5) -> None:
    """Marca em cada palavra a voz que mais fala durante ela (e se havia outra voz junto)."""
    if not turns:
        return
    starts = [turn.start for turn in turns]
    longest = max(turn.end - turn.start for turn in turns)
    for word in words:
        low = bisect.bisect_left(starts, word.start - longest)
        high = bisect.bisect_right(starts, word.end)
        cover: dict[str, float] = defaultdict(float)
        for turn in turns[low:high]:
            overlap = min(turn.end, word.end) - max(turn.start, word.start)
            if overlap > 0:
                cover[turn.label] += overlap
        if cover:
            ranked = sorted(cover.items(), key=lambda item: -item[1])
            word.speaker = ranked[0][0]
            length = max(word.end - word.start, 1e-3)
            word.overlap = len(ranked) > 1 and ranked[1][1] >= 0.5 * length
            continue
        nearest = _nearest_turn(turns, starts, word, snap)
        word.speaker = nearest.label if nearest else None
        word.overlap = False


def _nearest_turn(turns, starts, word: Word, snap: float) -> SpeakerTurn | None:
    index = bisect.bisect_left(starts, word.start)
    best, best_gap = None, snap
    for turn in turns[max(index - 3, 0):index + 2]:
        gap = max(turn.start - word.end, word.start - turn.end, 0.0)
        if gap <= best_gap:
            best, best_gap = turn, gap
    return best


def build_sentences(words: list[Word]) -> list[Sentence]:
    """Frases completas, cada uma atribuída a uma voz (ou "não identificado"/"sobreposto").

    A pontuação define as frases; se a voz muda no meio de uma frase por um trecho
    relevante (3+ palavras ou 1+ segundo), a frase é dividida ali.
    """
    sentences: list[Sentence] = []
    for group in split_word_groups(words, split_on_speaker_change=False):
        for part in _split_by_speaker(group):
            speaker, overlap = _dominant_speaker(part)
            sentences.append(make_sentence(part, speaker, overlap))
    return sentences


def _is_small(run: list[Word]) -> bool:
    return len(run) < 3 and run[-1].end - run[0].start < 1.0


def _split_by_speaker(group: list[Word]) -> list[list[Word]]:
    runs: list[list[Word]] = []
    for word in group:
        if runs and runs[-1][0].speaker == word.speaker:
            runs[-1].append(word)
        else:
            runs.append([word])
    merged: list[list[Word]] = []
    for run in runs:
        if merged and (_is_small(run) or _dominant_speaker(merged[-1])[0] == run[0].speaker):
            merged[-1].extend(run)
        else:
            merged.append(run)
    if len(merged) > 1 and _is_small(merged[0]):
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return merged


def _dominant_speaker(words: list[Word]) -> tuple[str | None, bool]:
    by_speaker: dict[str, float] = defaultdict(float)
    total = overlap = unknown = 0.0
    for word in words:
        length = max(word.end - word.start, 0.01)
        total += length
        if word.overlap:
            overlap += length
        if word.speaker is None:
            unknown += length
        else:
            by_speaker[word.speaker] += length
    if overlap >= 0.5 * total:
        return None, True
    if unknown >= 0.5 * total or not by_speaker:
        return None, False
    return max(by_speaker, key=by_speaker.__getitem__), False


# ------------------------------------------------------------------ impressões vocais


class VoiceEmbedder:
    """Calcula a "impressão vocal" de uma pessoa (usada para reconhecê-la depois)."""

    def __init__(self, threads: int) -> None:
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models.EMBEDDING.path), num_threads=threads
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        stream.input_finished()
        vector = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return vector / (np.linalg.norm(vector) or 1.0)


def voice_prints(
    samples: np.ndarray, turns: list[SpeakerTurn], embedder: VoiceEmbedder, per_speaker: int = 6
) -> dict[str, list[float]]:
    """Média das impressões vocais dos trechos mais longos e "limpos" (sem outra voz) de cada pessoa."""
    by_label: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for turn in turns:
        by_label[turn.label].append(turn)
    prints: dict[str, list[float]] = {}
    for label, own in by_label.items():
        others = [t for t in turns if t.label != label]
        clean = [t for t in own if not any(o.start < t.end and o.end > t.start for o in others)]
        chosen = sorted(clean or own, key=lambda t: t.end - t.start, reverse=True)[:per_speaker]
        vectors = []
        for turn in chosen:
            if turn.end - turn.start < 0.8:
                continue
            start = int(turn.start * SAMPLE_RATE)
            end = int(min(turn.end, turn.start + 8.0) * SAMPLE_RATE)
            vectors.append(embedder.embed(samples[start:end]))
        if vectors:
            mean = np.mean(vectors, axis=0)
            prints[label] = (mean / (np.linalg.norm(mean) or 1.0)).round(5).tolist()
    return prints
