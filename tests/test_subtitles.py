from pathlib import Path

import pytest

from transcritor.subtitles import SubtitleError, parse_subtitle_file, parse_subtitles, read_text_file

BASIC = """1
00:00:01,000 --> 00:00:03,500
Houve uma tentativa de estorno,

2
00:00:03,600 --> 00:00:06,000
porém não houve efetivação.
"""


def test_basico_remove_indices_e_tempos():
    cues = parse_subtitles(BASIC)
    assert [c.text for c in cues] == ["Houve uma tentativa de estorno,", "porém não houve efetivação."]
    assert cues[0].start == 1.0 and cues[0].end == 3.5
    assert cues[1].start == pytest.approx(3.6)


def test_bom_crlf_e_ansi(tmp_path: Path):
    utf8_bom = tmp_path / "a.srt"
    utf8_bom.write_bytes(b"\xef\xbb\xbf" + BASIC.replace("\n", "\r\n").encode("utf-8"))
    ansi = tmp_path / "b.SRT"  # extensão maiúscula (o script antigo sobrescrevia o original!)
    ansi.write_bytes(BASIC.encode("cp1252"))
    utf16 = tmp_path / "c.srt"
    utf16.write_bytes(BASIC.encode("utf-16"))
    for path in (utf8_bom, ansi, utf16):
        cues = parse_subtitle_file(path)
        assert cues[0].text.startswith("Houve"), path.name  # sem o "1" vazando por causa do BOM
        assert "efetivação" in cues[1].text


def test_linha_so_com_numero_e_mantida():
    content = "1\n00:00:01,000 --> 00:00:02,000\nO ano foi\n2024\n\n2\n00:00:02,500 --> 00:00:03,000\n100\n"
    cues = parse_subtitles(content)
    assert [c.text for c in cues] == ["O ano foi 2024", "100"]


def test_blocos_sem_linha_em_branco_entre_eles():
    content = "1\n00:00:01,000 --> 00:00:02,000\nPrimeira\n2\n00:00:02,000 --> 00:00:03,000\nSegunda\n"
    assert [c.text for c in parse_subtitles(content)] == ["Primeira", "Segunda"]


def test_vtt_com_tags_e_falantes():
    content = """WEBVTT

NOTE comentário

00:01.000 --> 00:02.500
<v Maria>Bom dia, <i>tudo</i> bem?</v>

00:00:03.000 --> 00:00:04.000 align:start
<v Daniel>Tudo ótimo.
"""
    cues = parse_subtitles(content)
    assert [(c.speaker, c.text) for c in cues] == [("Maria", "Bom dia, tudo bem?"), ("Daniel", "Tudo ótimo.")]
    assert cues[0].start == 1.0


@pytest.mark.parametrize(
    ("line", "speaker", "text"),
    [
        ("Speaker 1: Olá a todos.", "Speaker 1", "Olá a todos."),
        ("[SPEAKER_00]: Olá.", "SPEAKER 00", "Olá."),
        ("Falante 2 - Pode falar.", "Falante 2", "Pode falar."),
        ("[Maria]: Oi.", "Maria", "Oi."),
        ("Então: vamos começar.", None, "Então: vamos começar."),
    ],
)
def test_rotulos_de_falante(line, speaker, text):
    cue = parse_subtitles(f"1\n00:00:00,000 --> 00:00:01,000\n{line}\n")[0]
    assert (cue.speaker, cue.text) == (speaker, text)


def test_remove_tags_de_formatacao_e_travessao():
    content = '1\n00:00:00,000 --> 00:00:01,000\n{\\an8}<font color="#fff">- Sim.</font>\n'
    assert parse_subtitles(content)[0].text == "Sim."


def test_legenda_vazia_da_erro(tmp_path: Path):
    empty = tmp_path / "vazia.srt"
    empty.write_text("só texto sem tempos", encoding="utf-8")
    with pytest.raises(SubtitleError):
        parse_subtitle_file(empty)


def test_le_ansi_com_acentos(tmp_path: Path):
    path = tmp_path / "x.srt"
    path.write_bytes("ação".encode("cp1252"))
    assert read_text_file(path) == "ação"
