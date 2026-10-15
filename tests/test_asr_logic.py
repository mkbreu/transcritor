"""Partes da transcrição que não dependem do modelo (junção de tokens e divisão em pedaços)."""

import itertools

from transcritor.asr import MAX_CHUNK_SECONDS, plan_chunks, tokens_to_words


def test_tokens_viram_palavras_com_pontuacao_e_numeros():
    tokens = [" C", "lar", "o", ",", " o", " proto", "c", "olo", " é", " ", "4", "5", "2", "7", "."]
    starts = [i * 0.2 for i in range(len(tokens))]
    durations = [0.2] * len(tokens)
    words = tokens_to_words(tokens, starts, durations, offset=10.0)
    assert [w.text for w in words] == ["Claro,", "o", "protocolo", "é", "4527."]
    assert words[0].start == 10.0
    # a pontuação não estica o fim da palavra
    assert words[0].end == 10.0 + 0.4 + 0.2


def test_pedacos_respeitam_limite_e_pausas():
    speech = [(0.5, 5.0), (5.4, 12.0), (12.3, 19.0), (19.2, 24.0), (30.0, 31.0)]
    chunks = plan_chunks(speech, duration=40.0)
    assert all(c.end - c.start <= MAX_CHUNK_SECONDS + 0.5 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for left, right in itertools.pairwise(chunks):
        assert left.end <= right.start  # sem sobreposição: nada é transcrito duas vezes
    assert chunks[-1].start >= 24.0  # a fala depois da pausa longa fica num pedaço separado
