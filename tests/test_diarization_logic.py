"""Regras de atribuição palavra → voz e frase → voz (sem modelos de IA: dados sintéticos)."""

from transcritor.diarization import SpeakerTurn, assign_speakers, build_sentences, relabel
from transcritor.transcript import Word


def make_words(text: str, start: float, step: float = 0.3) -> list[Word]:
    return [Word(token, start + i * step, start + i * step + 0.25) for i, token in enumerate(text.split())]


def test_relabel_por_ordem_de_aparicao():
    turns = relabel([(5.0, 6.0, 7), (0.0, 1.0, 3), (2.0, 3.0, 7)])
    assert [(t.start, t.label) for t in turns] == [(0.0, "A"), (2.0, "B"), (5.0, "B")]


def test_cada_frase_vai_para_quem_falou():
    words = make_words("Boa tarde. Houve uma tentativa de estorno.", 0.0) + make_words(
        "Vou verificar no sistema.", 3.0)
    turns = [SpeakerTurn(0.0, 2.2, "A"), SpeakerTurn(2.9, 4.5, "B")]
    assign_speakers(words, turns)
    result = build_sentences(words)
    assert [(s.speaker, s.text) for s in result] == [
        ("A", "Boa tarde."), ("A", "Houve uma tentativa de estorno."), ("B", "Vou verificar no sistema.")]


def test_resposta_curta_com_pontuacao_propria_e_mantida():
    words = make_words("Você confirma?", 0.0) + make_words("Sim.", 1.0) + make_words("Ótimo, obrigado.", 2.0)
    turns = [SpeakerTurn(0.0, 0.8, "A"), SpeakerTurn(0.95, 1.3, "B"), SpeakerTurn(1.9, 3.0, "A")]
    assign_speakers(words, turns)
    assert [s.speaker for s in build_sentences(words)] == ["A", "B", "A"]


def test_erro_pequeno_na_borda_nao_quebra_a_frase():
    words = make_words("eu fui ao banco ontem e voltei hoje.", 0.0)
    # a separação "errou" uma palavra no meio (só 0,3 s de outra voz)
    turns = [SpeakerTurn(0.0, 0.85, "A"), SpeakerTurn(0.85, 1.15, "B"), SpeakerTurn(1.15, 3.0, "A")]
    assign_speakers(words, turns)
    result = build_sentences(words)
    assert [(s.speaker, s.text) for s in result] == [("A", "eu fui ao banco ontem e voltei hoje.")]


def test_mudanca_de_voz_relevante_no_meio_divide_a_frase():
    words = make_words("eu acho que devemos esperar o retorno deles e então decidir", 0.0)
    turns = [SpeakerTurn(0.0, 1.75, "A"), SpeakerTurn(1.75, 4.0, "B")]
    assign_speakers(words, turns)
    result = build_sentences(words)
    assert [s.speaker for s in result] == ["A", "B"]
    assert result[0].text == "eu acho que devemos esperar o"


def test_vozes_sobrepostas_e_trecho_sem_voz():
    overlapped = make_words("falando ao mesmo tempo aqui.", 0.0)
    silent = make_words("trecho sem voz detectada.", 10.0)
    turns = [SpeakerTurn(0.0, 1.5, "A"), SpeakerTurn(0.0, 1.5, "B")]
    assign_speakers(overlapped + silent, turns)
    result = build_sentences(overlapped + silent)
    assert (result[0].speaker, result[0].overlap) == (None, True)
    assert (result[1].speaker, result[1].overlap) == (None, False)


def test_palavra_na_lacuna_pega_a_voz_mais_proxima():
    words = [Word("então", 1.05, 1.3)]
    assign_speakers(words, [SpeakerTurn(0.0, 1.0, "A"), SpeakerTurn(2.0, 3.0, "B")])
    assert words[0].speaker == "A"
