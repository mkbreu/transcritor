"""Separação de vozes ("quem falou quando") e atribuição de cada frase a uma pessoa.

Usa a segmentação pyannote + "impressões vocais" (sherpa-onnx), tudo na CPU.
"""

from __future__ import annotations

import bisect
import itertools
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

# Refino depois do agrupamento: o sherpa-onnx agrupa por "ligação completa" (dois trechos só
# ficam juntos se TODOS os pares forem parecidos), então numa gravação longa a mesma pessoa
# se parte em vários grupos (uma gravação de uma pessoa só deu 12 "vozes"). O refino:
#  - absorve vozes com menos de 2 s de fala no total (tosse, ruído) pela voz vizinha;
#  - junta grupos cujas impressões vocais MÉDIAS são parecidas (mesma pessoa).
# Calibração com vozes reais e a segmentação Reverb (ferramentas\avaliar_separacao.py): as
# partes de uma mesma pessoa se juntam até ~0,60 e a dupla de pessoas diferentes mais
# parecida chegou a 0,44. O corte fica em 0,60: margem de 0,16 contra juntar pessoas
# diferentes (o erro sem conserto); se sobrar voz, você junta dando o mesmo nome.
MIN_SPEAKER_SECONDS = 2.0
MERGE_SIMILARITY = 0.60
MERGE_SPREAD = 0.2  # quanto a "sensibilidade" desloca a semelhança mínima para juntar

ProgressFn = Callable[[float, float], None]


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    label: str


def threshold_from_sensitivity(sensitivity: float) -> float:
    """Sensibilidade 0..1 (config.toml) → limiar. 0,5 = padrão; mais alto = separa mais vozes."""
    return DEFAULT_THRESHOLD + (0.5 - sensitivity) * THRESHOLD_SPREAD


def merge_similarity_from_sensitivity(sensitivity: float) -> float:
    """Sensibilidade 0..1 → semelhança mínima para juntar grupos. Mais alto = separa mais vozes."""
    return MERGE_SIMILARITY + (sensitivity - 0.5) * MERGE_SPREAD


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


# ------------------------------------------------------------------ refino da separação


@dataclass(slots=True)
class Separation:
    turns: list[SpeakerTurn]
    prints: dict[str, list[float]]
    detected: int  # quantas vozes a análise automática encontrou
    redone_with_hint: bool = False  # refeita procurando o número informado pelo usuário

    @property
    def found(self) -> int:
        return len({turn.label for turn in self.turns})


def separate_speakers(
    samples: np.ndarray,
    hint: int | None,
    diarizer: Diarizer,
    embedder: VoiceEmbedder,
    merge_similarity: float,
    progress: ProgressFn | None = None,
    on_redo: Callable[[int], None] | None = None,
) -> Separation:
    """Separa as vozes descobrindo sozinho quantas pessoas falam. O número informado pelo
    usuário (`hint`) é só uma pista: se a análise encontrar MENOS pessoas que ele, a separação é
    refeita procurando esse número (e o refino não junta abaixo dele); se encontrar mais, o
    resultado é mantido — sobrar voz tem conserto, misturar pessoas não."""
    raw = diarizer.run(samples, None, progress)
    turns, prints = refine_speakers(samples, raw, embedder, merge_similarity)
    detected = len({turn.label for turn in turns})
    if not hint or detected >= hint:
        return Separation(turns, prints, detected)
    if on_redo:
        on_redo(detected)
    raw = diarizer.run(samples, hint, progress)
    turns, prints = refine_speakers(samples, raw, embedder, merge_similarity, min_speakers=hint)
    return Separation(turns, prints, detected, redone_with_hint=True)


def refine_speakers(
    samples: np.ndarray,
    turns: list[SpeakerTurn],
    embedder: VoiceEmbedder,
    merge_similarity: float,
    min_speakers: int = 1,
) -> tuple[list[SpeakerTurn], dict[str, list[float]]]:
    """Corrige a separação bruta: absorve migalhas, junta grupos da mesma pessoa (sem ficar
    com menos de `min_speakers`) e renomeia as vozes (A, B...) na ordem em que aparecem."""
    turns = absorb_small_speakers(turns)
    prints = voice_prints(samples, turns, embedder)
    turns, prints = merge_similar_speakers(turns, prints, merge_similarity, min_speakers)
    return relabel_turns(turns, prints)


def absorb_small_speakers(
    turns: list[SpeakerTurn], min_seconds: float = MIN_SPEAKER_SECONDS
) -> list[SpeakerTurn]:
    """Vozes com pouquíssima fala no total (tosse, ruído) passam para a voz mais próxima no tempo."""
    totals: dict[str, float] = defaultdict(float)
    for turn in turns:
        totals[turn.label] += turn.end - turn.start
    big = [turn for turn in turns if totals[turn.label] >= min_seconds]
    if not big or len(big) == len(turns):
        return turns
    result = []
    for turn in turns:
        if totals[turn.label] >= min_seconds:
            result.append(turn)
            continue
        middle = (turn.start + turn.end) / 2
        nearest = min(big, key=lambda other: max(other.start - middle, middle - other.end, 0.0))
        result.append(SpeakerTurn(turn.start, turn.end, nearest.label))
    log.info("Vozes absorvidas por terem menos de %.0f s de fala: %s", min_seconds,
             sorted({label for label, total in totals.items() if total < min_seconds}))
    return result


def merge_similar_speakers(
    turns: list[SpeakerTurn],
    prints: dict[str, list[float]],
    min_similarity: float,
    min_speakers: int = 1,
) -> tuple[list[SpeakerTurn], dict[str, list[float]]]:
    """Junta, um par por vez, os grupos cujas impressões vocais médias são mais parecidas,
    enquanto a semelhança for de pelo menos `min_similarity` (média ponderada pelo tempo de
    fala) e restarem mais de `min_speakers` grupos."""
    durations: dict[str, float] = defaultdict(float)
    for turn in turns:
        durations[turn.label] += turn.end - turn.start
    centroids = {label: np.asarray(vector, dtype=np.float32) for label, vector in prints.items()}
    group = {label: label for label in durations}
    while len(centroids) > max(min_speakers, 1):
        pairs = itertools.combinations(sorted(centroids), 2)
        (keep, drop), similarity = max(
            ((pair, float(centroids[pair[0]] @ centroids[pair[1]])) for pair in pairs),
            key=lambda item: item[1],
        )
        if similarity < min_similarity:
            break
        if durations[drop] > durations[keep]:
            keep, drop = drop, keep  # o grupo com mais fala dá nome ao resultado
        merged = centroids[keep] * durations[keep] + centroids[drop] * durations[drop]
        centroids[keep] = merged / (np.linalg.norm(merged) or 1.0)
        durations[keep] += durations[drop]
        del centroids[drop]
        for label, target in group.items():
            if target == drop:
                group[label] = keep
        log.info("Vozes %s e %s unidas (semelhança %.2f)", keep, drop, similarity)
    merged_turns = [SpeakerTurn(t.start, t.end, group[t.label]) for t in turns]
    return merged_turns, {label: vector.round(5).tolist() for label, vector in centroids.items()}


def relabel_turns(
    turns: list[SpeakerTurn], prints: dict[str, list[float]]
) -> tuple[list[SpeakerTurn], dict[str, list[float]]]:
    """Renomeia as vozes para A, B, C... na ordem em que falam pela primeira vez."""
    order: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: t.start):
        order.setdefault(turn.label, speaker_label(len(order)))
    renamed = [SpeakerTurn(t.start, t.end, order[t.label]) for t in turns]
    return renamed, {order[label]: vector for label, vector in prints.items() if label in order}


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
    relevante (3+ palavras ou 1+ segundo), a frase é dividida ali: o pedaço interrompido
    termina com reticências e o seguinte começa com maiúscula.
    """
    sentences: list[Sentence] = []
    for group in split_word_groups(words, split_on_speaker_change=False):
        parts = _split_by_speaker(group)
        for index, part in enumerate(parts):
            speaker, overlap = _dominant_speaker(part)
            sentence = make_sentence(part, speaker, overlap)
            if index < len(parts) - 1:
                sentence.text = _close_fragment(sentence.text)
            if index > 0:
                sentence.text = _open_fragment(sentence.text)
            sentences.append(sentence)
    return sentences


def _close_fragment(text: str) -> str:
    """Frase cortada pela fala de outra pessoa: fecha com reticências (se não tiver pontuação)."""
    if text.rstrip("\"'”’»)]").endswith((".", "!", "?", "…")):
        return text
    return text.rstrip(" ,;:-–—") + "…"


def _open_fragment(text: str) -> str:
    """Continuação depois de um corte: começa com letra maiúscula."""
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1:]
    return text


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
        if not vectors:  # só falas curtas: junta as maiores (até 8 s) numa amostra única
            pieces, total = [], 0.0
            for turn in sorted(own, key=lambda t: t.end - t.start, reverse=True):
                pieces.append(samples[int(turn.start * SAMPLE_RATE):int(turn.end * SAMPLE_RATE)])
                total += turn.end - turn.start
                if total >= 8.0:
                    break
            joined = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            if len(joined) >= SAMPLE_RATE // 2:
                vectors.append(embedder.embed(joined))
        if vectors:
            mean = np.mean(vectors, axis=0)
            prints[label] = (mean / (np.linalg.norm(mean) or 1.0)).round(5).tolist()
    return prints
