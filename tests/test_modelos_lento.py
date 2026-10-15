"""Teste de ponta a ponta com os modelos de IA (lento). Rode com:  pytest -m lento

Usa o diálogo sintético gerado por ferramentas\\gerar_dialogo_teste.py (vozes do Windows).
"""

import json
from pathlib import Path

import pytest

from transcritor import asr, media, models

ROOT = Path(__file__).resolve().parents[1]
DIALOGUE = ROOT / "dados" / "teste" / "dialogo.wav"

pytestmark = [
    pytest.mark.lento,
    pytest.mark.skipif(bool(models.missing_models()), reason="modelos de IA não instalados"),
    pytest.mark.skipif(not DIALOGUE.exists(), reason="rode ferramentas\\gerar_dialogo_teste.py antes"),
]


def normalize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return cleaned.split()


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp in enumerate(hypothesis, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp))
        previous = current
    return previous[-1] / max(len(reference), 1)


def test_transcricao_do_dialogo_em_portugues(tmp_path: Path):
    truth = json.loads(DIALOGUE.with_suffix(".json").read_text(encoding="utf-8"))
    duration = media.wav_duration(DIALOGUE)
    words = asr.transcribe_file(DIALOGUE, duration, lambda: asr.ParakeetTranscriber(4),
                                tmp_path / "diario.jsonl", 4)
    hypothesis = normalize(" ".join(w.text for w in words))
    reference = normalize(" ".join(item["text"] for item in truth))
    # números falados viram algarismos ("doze" → "12"): ignora essas diferenças no cálculo
    numbers = {"quatro", "cinco", "dois", "sete", "doze", "mil", "e", "quinhentos", "reais", "três", "dez"}
    reference = [w for w in reference if w not in numbers]
    hypothesis = [w for w in hypothesis if not w.isdigit() and w not in numbers]
    assert word_error_rate(reference, hypothesis) < 0.08


def test_separacao_das_duas_vozes_do_dialogo():
    from transcritor.diarization import Diarizer

    truth = json.loads(DIALOGUE.with_suffix(".json").read_text(encoding="utf-8"))
    samples = media.load_wav_lowmem(DIALOGUE)
    turns = Diarizer(threads=4, sensitivity=0.5).run(samples, num_speakers=2)
    # cada fala do gabarito → voz que mais cobre o seu tempo
    votes = []
    for item in truth:
        cover: dict[str, float] = {}
        for turn in turns:
            overlap = min(turn.end, item["end"]) - max(turn.start, item["start"])
            if overlap > 0:
                cover[turn.label] = cover.get(turn.label, 0.0) + overlap
        votes.append((item["speaker"], max(cover, key=cover.get) if cover else None))
    mapping = {}
    for person, label in votes:
        mapping.setdefault(person, {}).setdefault(label, 0)
        mapping[person][label] += 1
    best = {person: max(labels, key=labels.get) for person, labels in mapping.items()}
    assert best["Maria"] != best["Daniel"]
    correct = sum(best[person] == label for person, label in votes)
    assert correct / len(votes) >= 0.9
