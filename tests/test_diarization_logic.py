"""Regras de atribuição palavra → voz e frase → voz (sem modelos de IA: dados sintéticos)."""

import numpy as np

from transcritor.diarization import (
    SpeakerTurn,
    absorb_small_speakers,
    assign_speakers,
    build_sentences,
    merge_similar_speakers,
    merge_similarity_from_sensitivity,
    relabel,
    relabel_turns,
    separate_speakers,
)
from transcritor.transcript import Word


def vec(*values: float) -> list[float]:
    v = np.array(values, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


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
    # o pedaço interrompido termina com reticências; a continuação começa com maiúscula
    assert result[0].text == "eu acho que devemos esperar o…"
    assert result[1].text == "Retorno deles e então decidir"


def test_corte_depois_de_virgula_nao_duplica_pontuacao():
    words = make_words("mas olha, eu fiz tudo certo, você entende o que eu digo", 0.0)
    turns = [SpeakerTurn(0.0, 1.75, "A"), SpeakerTurn(1.75, 4.0, "B")]
    assign_speakers(words, turns)
    assert build_sentences(words)[0].text == "mas olha, eu fiz tudo certo…"


def test_migalhas_de_voz_vao_para_a_voz_vizinha():
    turns = [SpeakerTurn(0.0, 10.0, "A"), SpeakerTurn(10.2, 10.6, "B"),  # 0,4 s: tosse/ruído
             SpeakerTurn(11.0, 20.0, "A"), SpeakerTurn(30.0, 40.0, "C")]
    result = absorb_small_speakers(turns)
    assert [t.label for t in result] == ["A", "A", "A", "C"]


def test_grupos_parecidos_viram_a_mesma_pessoa_e_diferentes_nao():
    turns = [SpeakerTurn(0, 30, "A"), SpeakerTurn(30, 35, "B"), SpeakerTurn(35, 60, "C"),
             SpeakerTurn(60, 70, "B")]
    prints = {"A": vec(1, 0.1, 0), "B": vec(1, 0.3, 0.1), "C": vec(0, 1, 0.2)}  # A≈B, C diferente
    merged, centroids = merge_similar_speakers(turns, prints, min_similarity=0.65)
    assert [t.label for t in merged] == ["A", "A", "C", "A"]  # quem fala mais (A) dá o nome
    assert set(centroids) == {"A", "C"}


def test_nada_e_juntado_abaixo_do_corte():
    turns = [SpeakerTurn(0, 10, "A"), SpeakerTurn(10, 20, "B")]
    prints = {"A": vec(1, 0, 0), "B": vec(0.6, 0.8, 0)}  # semelhança 0,60
    merged, _ = merge_similar_speakers(turns, prints, min_similarity=0.65)
    assert {t.label for t in merged} == {"A", "B"}


def test_renomeia_na_ordem_em_que_falam():
    turns = [SpeakerTurn(5, 6, "C"), SpeakerTurn(0, 1, "B")]
    renamed, prints = relabel_turns(turns, {"B": [1.0], "C": [2.0]})
    assert [(t.start, t.label) for t in renamed] == [(5, "B"), (0, "A")]
    assert prints == {"A": [1.0], "B": [2.0]}


def test_sensibilidade_desloca_o_corte():
    assert merge_similarity_from_sensitivity(0.5) == 0.60
    assert merge_similarity_from_sensitivity(0.9) > 0.60 > merge_similarity_from_sensitivity(0.1)


def test_nao_junta_abaixo_do_minimo_de_pessoas():
    turns = [SpeakerTurn(0, 10, "A"), SpeakerTurn(10, 20, "B")]
    prints = {"A": vec(1, 0, 0), "B": vec(1, 0.05, 0)}  # quase idênticos
    merged, _ = merge_similar_speakers(turns, prints, min_similarity=0.6, min_speakers=2)
    assert {t.label for t in merged} == {"A", "B"}


# ---------------------------------------------------------------- pista do nº de pessoas


class FakeDiarizer:
    """Separação "bruta" combinada para cada número de pessoas pedido."""

    def __init__(self, by_hint):
        self.by_hint, self.calls = by_hint, []

    def run(self, samples, num_speakers, progress=None):
        self.calls.append(num_speakers)
        return self.by_hint[num_speakers]


class FakeEmbedder:
    """Impressão vocal = ângulo dado pelo nível do áudio (cada "pessoa" tem um nível)."""

    def embed(self, samples):
        level = float(np.mean(samples))
        return np.array([np.cos(level), np.sin(level)], dtype=np.float32)


def two_people_audio():
    samples = np.zeros(30 * 16000, dtype=np.float32)
    samples[10 * 16000:20 * 16000] = 1.4  # pessoa B: ângulo bem diferente da A (0,0)
    return samples


A_B_A = [SpeakerTurn(0, 10, "A"), SpeakerTurn(10, 20, "B"), SpeakerTurn(20, 30, "A")]
ALL_A = [SpeakerTurn(0, 10, "A"), SpeakerTurn(10, 20, "A"), SpeakerTurn(20, 30, "A")]


def test_sem_pista_detecta_sozinho():
    diarizer = FakeDiarizer({None: A_B_A})
    result = separate_speakers(two_people_audio(), None, diarizer, FakeEmbedder(), 0.6)
    assert result.found == 2 and diarizer.calls == [None]


def test_pista_maior_que_o_detectado_refaz_procurando_o_numero():
    diarizer = FakeDiarizer({None: ALL_A, 2: A_B_A})
    redo = []
    result = separate_speakers(two_people_audio(), 2, diarizer, FakeEmbedder(), 0.6, on_redo=redo.append)
    assert diarizer.calls == [None, 2] and redo == [1]
    assert result.redone_with_hint and result.detected == 1 and result.found == 2


def test_pista_menor_que_o_detectado_mantem_o_que_a_voz_mostra():
    diarizer = FakeDiarizer({None: A_B_A})
    result = separate_speakers(two_people_audio(), 1, diarizer, FakeEmbedder(), 0.6)
    assert result.found == 2 and not result.redone_with_hint  # nunca mistura pessoas à força


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
