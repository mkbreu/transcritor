"""Fluxo de "quem é quem" com respostas simuladas (sem teclado e sem alto-falante)."""

from pathlib import Path

import numpy as np
import pytest

from transcritor import voices
from transcritor.render import render_text
from transcritor.settings import Settings, State
from transcritor.transcript import Sentence, Transcript


@pytest.fixture
def scripted(monkeypatch):
    """Substitui as perguntas do console por respostas pré-definidas e registra o que foi perguntado."""
    answers: list[str] = []
    asked: list[str] = []
    played: list[float] = []

    def fake_ask(prompt: str) -> str:
        asked.append(prompt)
        return answers.pop(0) if answers else ""

    monkeypatch.setattr(voices.ui, "ask", fake_ask)
    monkeypatch.setattr(voices, "play_segment", lambda path, start, end: played.append(start))
    return answers, played, asked


def unit(vector_seed: int) -> list[float]:
    vector = np.random.default_rng(vector_seed).normal(size=64)
    return (vector / np.linalg.norm(vector)).tolist()


def conversation(labels: str = "AB") -> Transcript:
    sentences = []
    for i in range(8):
        for j, label in enumerate(labels):
            start = i * 20 + j * 5
            sentences.append(Sentence(f"Frase longa número {i} dita pela voz {label} aqui agora.",
                                      start, start + 4, label))
    return Transcript("conversa.mp4", "video", 170, sentences, "teste", has_speakers=True)


def test_nomear_ouvir_mais_exemplos_e_memorizar_minha_voz(tmp_path: Path, scripted):
    answers, played, _ = scripted
    answers += ["1", "m", "Fulano", "", "A"]  # ouvir, mais exemplos, nome A, Enter em B, "minha voz é A"
    book = voices.VoiceBook(tmp_path / "vozes.json")
    state = State()
    transcript = conversation()
    voices.name_speakers(transcript, Path("audio.wav"), {"A": unit(1), "B": unit(2)}, book, Settings(), state)
    assert transcript.speaker_names == {"A": "Fulano", "B": "Pessoa B"}
    assert transcript.me_labels == ["A"]
    assert played == [0]  # tocou o 1º exemplo da voz A
    assert state.user_name == "Fulano"
    saved = voices.VoiceBook(tmp_path / "vozes.json")
    assert saved.me is not None and saved.me.name == "Fulano"


def test_marcar_eu_em_varias_vozes_dispensa_a_pergunta_final(tmp_path: Path, scripted):
    answers, _, asked = scripted
    answers += ["Maykon (eu)", "eu", "João"]
    transcript = conversation("ABC")
    prints = {"A": unit(1), "B": unit(3), "C": unit(2)}
    voices.name_speakers(transcript, None, prints, voices.VoiceBook(tmp_path / "vozes.json"),
                         Settings(), State())
    assert transcript.speaker_names == {"A": "Maykon", "B": "Maykon", "C": "João"}
    assert transcript.me_labels == ["A", "B"]
    assert not any("SUA" in prompt for prompt in asked)  # não perguntou de novo qual é a sua voz
    paragraphs = render_text(transcript).split("=" * 64)[1].strip().split("\n\n")
    assert "Participantes: Maykon (eu), João" in render_text(transcript)
    assert not any(" Pessoa " in p for p in paragraphs)


def test_pergunta_final_aceita_varias_letras(tmp_path: Path, scripted):
    answers, _, _ = scripted
    answers += ["", "", "", "A e C", "Ana"]  # Enter nas 3 vozes; "A e C" são minhas; meu nome
    transcript = conversation("ABC")
    prints = {"A": unit(1), "B": unit(2), "C": unit(3)}
    voices.name_speakers(transcript, None, prints, voices.VoiceBook(tmp_path / "vozes.json"),
                         Settings(), State())
    assert transcript.me_labels == ["A", "C"]
    assert transcript.speaker_names == {"A": "Ana", "B": "Pessoa B", "C": "Ana"}


def test_reconhece_a_voz_conhecida_na_proxima_gravacao(tmp_path: Path, scripted):
    answers, _, asked = scripted
    book = voices.VoiceBook(tmp_path / "vozes.json")
    book.remember("Fulano", unit(1), is_me=True)
    book.save()
    answers += ["Atendente", ""]  # nomeia a voz A; Enter confirma a sugestão "Fulano" na voz B
    transcript = conversation()
    prints = {"A": unit(2), "B": unit(1)}  # agora a voz conhecida é a B
    voices.name_speakers(transcript, None, prints, voices.VoiceBook(tmp_path / "vozes.json"),
                         Settings(), State())
    assert transcript.speaker_names == {"A": "Atendente", "B": "Fulano"}
    assert transcript.me_labels == ["B"]  # reconhecida sem precisar perguntar de novo
    assert not any("SUA" in prompt or "Sua voz" in prompt for prompt in asked)


def test_mesmo_nome_junta_vozes_no_texto(scripted):
    answers, _, _ = scripted
    answers += ["Ana", "Ana", ""]
    transcript = conversation()
    voices.name_speakers(transcript, None, {}, voices.VoiceBook(Path("inexistente.json")),
                         Settings(reconhecer_vozes=False), State())
    paragraphs = render_text(transcript).split("=" * 64)[1].strip().split("\n\n")
    # as duas vozes viraram a mesma pessoa: nenhum parágrafo com outro nome
    assert all("] Ana: " in p for p in paragraphs)
    assert len(paragraphs) < 16  # frases seguidas da "Ana" foram agrupadas


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("Maria (eu)", ("Maria", True)),
        ("Maria (EU)", ("Maria", True)),
        ("eu", ("Fulano", True)),
        ("(eu)", ("Fulano", True)),
        ("Fulano", ("Fulano", True)),
        ("João", ("João", False)),
        ("  João   Silva ", ("João Silva", False)),
    ],
)
def test_interpretacao_do_nome_digitado(typed, expected):
    assert voices.parse_name(typed, "Fulano") == expected
