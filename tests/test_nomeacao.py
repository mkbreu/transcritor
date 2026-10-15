"""Fluxo de "quem é quem" com respostas simuladas (sem teclado e sem alto-falante)."""

from pathlib import Path

import numpy as np
import pytest

from transcritor import voices
from transcritor.settings import Settings, State
from transcritor.transcript import Sentence, Transcript


@pytest.fixture
def scripted(monkeypatch):
    """Substitui as perguntas do console por respostas pré-definidas e registra o que tocou."""
    answers: list[str] = []
    played: list[float] = []
    monkeypatch.setattr(voices.ui, "ask", lambda prompt: answers.pop(0) if answers else "")
    monkeypatch.setattr(voices, "play_segment", lambda path, start, end: played.append(start))
    return answers, played


def unit(vector_seed: int) -> list[float]:
    vector = np.random.default_rng(vector_seed).normal(size=64)
    return (vector / np.linalg.norm(vector)).tolist()


def conversation() -> Transcript:
    sentences = []
    for i in range(8):
        start = i * 20
        sentences.append(Sentence(f"Frase longa número {i} dita pela primeira pessoa aqui.",
                                  start, start + 5, "A"))
        sentences.append(Sentence(f"Resposta número {i} dita pela segunda pessoa agora.",
                                  start + 6, start + 10, "B"))
    return Transcript("conversa.mp4", "video", 170, sentences, "teste", has_speakers=True)


def test_nomear_ouvir_mais_exemplos_e_memorizar_minha_voz(tmp_path: Path, scripted):
    answers, played = scripted
    answers += ["1", "m", "Fulano", "", "A"]  # ouvir, mais exemplos, nome A, Enter em B, "minha voz é A"
    book = voices.VoiceBook(tmp_path / "vozes.json")
    state = State()
    transcript = conversation()
    prints = {"A": unit(1), "B": unit(2)}
    voices.name_speakers(transcript, Path("audio.wav"), prints, book, Settings(), state)
    assert transcript.speaker_names == {"A": "Fulano", "B": "Pessoa B"}
    assert transcript.me_label == "A"
    assert played == [0]  # tocou o 1º exemplo da voz A
    assert state.user_name == "Fulano"
    saved = voices.VoiceBook(tmp_path / "vozes.json")
    assert saved.me is not None and saved.me.name == "Fulano"


def test_reconhece_a_voz_conhecida_na_proxima_gravacao(tmp_path: Path, scripted):
    answers, _ = scripted
    book = voices.VoiceBook(tmp_path / "vozes.json")
    book.remember("Fulano", unit(1), is_me=True)
    book.save()
    answers += ["Atendente", ""]  # nomeia a voz A; Enter confirma a sugestão "Fulano" na voz B
    transcript = conversation()
    prints = {"A": unit(2), "B": unit(1)}  # agora a voz conhecida é a B
    voices.name_speakers(transcript, None, prints, voices.VoiceBook(tmp_path / "vozes.json"),
                         Settings(), State())
    assert transcript.speaker_names == {"A": "Atendente", "B": "Fulano"}
    assert transcript.me_label == "B"  # reconhecida sem precisar perguntar de novo


def test_mesmo_nome_junta_vozes_no_texto(scripted):
    answers, _ = scripted
    answers += ["Ana", "Ana", ""]
    transcript = conversation()
    voices.name_speakers(transcript, None, {}, voices.VoiceBook(Path("inexistente.json")),
                         Settings(reconhecer_vozes=False), State())
    from transcritor.render import render_text

    paragraphs = render_text(transcript).split("=" * 64)[1].strip().split("\n\n")
    # as duas vozes viraram a mesma pessoa: nenhum parágrafo com outro nome
    assert all(p.startswith("Ana: ") for p in paragraphs)
    assert len(paragraphs) < 16  # frases seguidas da "Ana" foram agrupadas
