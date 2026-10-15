import pytest

from transcritor.sentences import (
    MAX_SENTENCE_WORDS,
    ends_sentence,
    group_paragraphs,
    split_into_sentences,
    words_from_cues,
)
from transcritor.subtitles import Cue
from transcritor.transcript import Sentence, Word


def words(text: str, start: float = 0.0, step: float = 0.3, speaker=None) -> list[Word]:
    return [Word(token, start + i * step, start + i * step + 0.25, speaker)
            for i, token in enumerate(text.split())]


def texts(sentences) -> list[str]:
    return [s.text for s in sentences]


def test_frases_completas_juntando_legendas_cortadas():
    cues = [Cue(0, 2, "Houve uma tentativa de"), Cue(2, 4, "estorno, porém não houve"),
            Cue(4, 5, "efetivação. O senhor pode"), Cue(5, 6, "verificar?")]
    result = split_into_sentences(words_from_cues(cues))
    assert texts(result) == ["Houve uma tentativa de estorno, porém não houve efetivação.",
                             "O senhor pode verificar?"]


@pytest.mark.parametrize(
    ("text", "following", "expected"),
    [
        ("Sr.", "João", False),          # título nunca encerra
        ("Dra.", "Ana", False),
        ("Ltda.", "Depois", True),       # abreviação + maiúscula = fim de frase
        ("etc.", "e", False),
        ("pág.", "12", False),
        ("1.500,00.", "Depois", True),   # valor no fim da frase
        ("J.", "Silva", False),          # inicial de nome
        ("assim...", "e", False),        # reticências seguidas de minúscula
        ("assim...", "Então", True),
        ('"não."', "Ele", True),         # ponto dentro de aspas
        ("verdade?", "não", True),
        ("casa", "Então", False),        # sem pontuação
    ],
)
def test_fim_de_frase(text, following, expected):
    assert ends_sentence(text, following) is expected


def test_valor_monetario_nao_quebra_frase():
    result = split_into_sentences(words("O valor de R$ 1.500,00 foi estornado. Depois disso nada."))
    assert texts(result) == ["O valor de R$ 1.500,00 foi estornado.", "Depois disso nada."]


def test_pausa_longa_encerra_frase_sem_pontuacao():
    first = words("eu liguei para o banco")
    second = words("ninguém atendeu", start=first[-1].end + 4.0)
    assert len(split_into_sentences(first + second)) == 2


def test_frase_gigante_sem_pontuacao_e_dividida():
    long = words(" ".join(["palavra"] * (MAX_SENTENCE_WORDS * 2 + 5)))
    result = split_into_sentences(long)
    assert len(result) >= 3
    assert all(len(s.text.split()) <= MAX_SENTENCE_WORDS for s in result)


def test_mudanca_de_falante_da_legenda_quebra_frase():
    cues = [Cue(0, 1, "Você vem", speaker="Maria"), Cue(1, 2, "amanhã sim", speaker="João")]
    result = split_into_sentences(words_from_cues(cues), split_on_speaker_change=True)
    assert [(s.speaker, s.text) for s in result] == [("Maria", "Você vem"), ("João", "amanhã sim")]


def test_paragrafos_juntam_mesma_pessoa_e_mesmo_nome():
    sentences = [Sentence("Oi.", 0, 1, "A"), Sentence("Tudo bem?", 1.2, 2, "A"),
                 Sentence("Sim.", 2.5, 3, "B"), Sentence("E você?", 3.1, 4, "C")]
    paragraphs = group_paragraphs(sentences, names={"A": "Ana", "B": "Bia", "C": "Bia"})
    assert [(p.speaker, p.text) for p in paragraphs] == [("A", "Oi. Tudo bem?"), ("B", "Sim. E você?")]


def test_paragrafo_longo_e_quebrado():
    sentences = [Sentence(" ".join(["palavra"] * 30) + ".", i * 10, i * 10 + 9, "A") for i in range(10)]
    paragraphs = group_paragraphs(sentences)
    assert len(paragraphs) > 1
    assert all(sum(len(s.text.split()) for s in p.sentences) <= 200 for p in paragraphs)
