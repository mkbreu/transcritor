from pathlib import Path

import pytest

from transcritor.render import (
    RenderOptions,
    format_clock,
    format_duration,
    render_text,
    suggested_output_name,
    unique_path,
    write_text_atomic,
)
from transcritor.transcript import Sentence, Transcript


def make_transcript(**extra) -> Transcript:
    sentences = [
        Sentence("Boa tarde.", 0.0, 1.0, "A"),
        Sentence("Houve uma tentativa de estorno, porém não houve efetivação.", 1.2, 5.0, "A"),
        Sentence("Vou verificar.", 5.5, 6.5, "B"),
        Sentence("Isso não pode!", 7.0, 8.0, None, overlap=True),
        Sentence("Certo, obrigado.", 9.0, 10.0, "C"),
    ]
    values = dict(source_name="reuniao.mp4", source_kind="video", duration=3133.0,
                  sentences=sentences, engine="teste", has_speakers=True,
                  speaker_names={"A": "Fulano", "B": "Atendente", "C": "Fulano"}, me_label="A",
                  created_at="14/10/2026 às 16:20")
    values.update(extra)
    return Transcript(**values)


def test_txt_com_falantes_em_ordem_e_sem_horarios():
    text = render_text(make_transcript())
    assert "TRANSCRIÇÃO DE VÍDEO" in text
    assert "Participantes: Fulano (eu), Atendente" in text
    assert "Duração: 52 min 13 s" in text
    body = text.split("=" * 64)[1].strip().split("\n\n")
    assert body == [
        "Fulano: Boa tarde. Houve uma tentativa de estorno, porém não houve efetivação.",
        "Atendente: Vou verificar.",
        "[Vozes sobrepostas]: Isso não pode!",
        "Fulano: Certo, obrigado.",
    ]
    assert "[00:" not in text
    assert '"[Vozes sobrepostas]"' in text  # explicação no cabeçalho


def test_opcoes_horario_e_uma_frase_por_linha():
    text = render_text(make_transcript(), RenderOptions(show_times=True, one_sentence_per_line=True))
    assert "[00:00] Fulano: Boa tarde." in text  # gravação com menos de 1 h: mm:ss
    assert "[00:00] Fulano: Houve uma tentativa" in text
    long = render_text(make_transcript(duration=4000), RenderOptions(show_times=True))
    assert "[00:00:00] Fulano: Boa tarde." in long  # 1 h ou mais: hh:mm:ss


def test_legenda_sem_falantes_vira_paragrafos():
    sentences = [Sentence("Primeira frase.", 0, 2), Sentence("Segunda frase.", 2.2, 4),
                 Sentence("Depois de uma pausa longa, outra frase com bastante conteúdo aqui.", 40, 45)]
    t = Transcript("aula.srt", "legenda", 45, sentences, "Legenda", has_speakers=False)
    text = render_text(t)
    assert "Participantes" not in text
    assert "Sem identificação de quem fala" in text
    assert "Primeira frase. Segunda frase." in text


@pytest.mark.parametrize(("seconds", "expected"), [(45, "45 s"), (3133, "52 min 13 s"), (3725, "1 h 02 min")])
def test_formato_duracao(seconds, expected):
    assert format_duration(seconds) == expected


def test_formato_relogio():
    assert format_clock(195) == "03:15"
    assert format_clock(3725) == "01:02:05"


def test_nome_sugerido_nunca_e_o_original(tmp_path: Path):
    source = tmp_path / "Reunião.SRT"
    assert suggested_output_name(source) == "Reunião - transcrição.txt"


def test_unique_path(tmp_path: Path):
    first = tmp_path / "a.txt"
    first.write_text("x")
    assert unique_path(first).name == "a (2).txt"


def test_gravacao_atomica_com_crlf(tmp_path: Path):
    target = tmp_path / "saida.txt"
    write_text_atomic(target, "linha 1\nação\n")
    assert target.read_bytes() == "linha 1\r\nação\r\n".encode()
    assert list(tmp_path.iterdir()) == [target]  # nenhum temporário sobrando
