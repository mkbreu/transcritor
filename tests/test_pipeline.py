"""Etapas do processamento que não precisam dos modelos de IA (cache e montagem final)."""

from pathlib import Path

from transcritor import pipeline
from transcritor.cache import CacheEntry
from transcritor.diarization import SpeakerTurn
from transcritor.settings import Settings
from transcritor.transcript import Transcript, Word


def test_cache_da_separacao_so_vale_com_as_mesmas_opcoes(tmp_path: Path):
    entry = CacheEntry(tmp_path)
    turns = [SpeakerTurn(0.0, 1.5, "A"), SpeakerTurn(1.6, 3.0, "B")]
    pipeline._save_diarization(entry, 2, 0.7, turns, {"A": [0.1, 0.2]})
    loaded = pipeline._load_diarization(entry, 2, 0.7)
    assert loaded == (turns, {"A": [0.1, 0.2]})
    assert pipeline._load_diarization(entry, None, 0.7) is None  # outro nº de pessoas
    assert pipeline._load_diarization(entry, 2, 0.6) is None  # outra sensibilidade


def test_montagem_final_com_vozes(tmp_path: Path):
    job = pipeline.Job(source=tmp_path / "reuniao.mp4", kind="video", duration=4.0)
    job.entry = CacheEntry(tmp_path)
    job.words = [Word("Bom", 0.0, 0.3), Word("dia.", 0.35, 0.7), Word("Olá,", 1.6, 1.9),
                 Word("tudo", 2.0, 2.3), Word("bem?", 2.35, 2.8)]
    job.turns = [SpeakerTurn(0.0, 1.0, "A"), SpeakerTurn(1.5, 3.0, "B")]
    pipeline._finalize_all([job], Settings())
    assert job.transcript.has_speakers
    assert [(s.speaker, s.text) for s in job.transcript.sentences] == [("A", "Bom dia."),
                                                                       ("B", "Olá, tudo bem?")]
    assert Transcript.load(job.entry.transcript).sentences == job.transcript.sentences  # gravado no cache
    assert job.words == []  # memória liberada


def test_montagem_final_sem_separar_vozes(tmp_path: Path):
    job = pipeline.Job(source=tmp_path / "aula.mp3", kind="audio", duration=2.0)
    job.entry = CacheEntry(tmp_path)
    job.words = [Word("Uma", 0.0, 0.2), Word("frase.", 0.25, 0.6)]
    pipeline._finalize_all([job], Settings(separar_vozes=False))
    assert not job.transcript.has_speakers
    assert job.transcript.sentences[0].text == "Uma frase."


def test_gigabytes_com_virgula():
    assert pipeline.gb(1.5 * 1024**3) == "1,50"
    assert pipeline.gb(0.74 * 1024**3, 1) == "0,7"
