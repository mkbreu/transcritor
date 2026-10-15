from pathlib import Path

import numpy as np

from transcritor import cache
from transcritor.settings import Settings, State, load_settings
from transcritor.transcript import Sentence, Transcript, default_name, speaker_label
from transcritor.voices import VoiceBook, more_samples, pick_samples, speaking_share


def test_config_valida_e_avisos(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[texto]\nmostrar_horarios = true\n[vozes]\nsensibilidade_separacao = 2\n"
        "semelhanca_minima = \"alta\"\nopcao_inexistente = 1\n[outra]\nx = 1\n",
        encoding="utf-8",
    )
    settings, warnings = load_settings(config)
    assert settings.mostrar_horarios is True
    assert settings.sensibilidade_separacao == 0.95  # limitado ao máximo
    assert settings.semelhanca_minima == Settings().semelhanca_minima  # inválido → padrão
    assert len(warnings) == 3


def test_config_com_erro_de_sintaxe_usa_padrao(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text("[texto\nmostrar_horarios = ", encoding="utf-8")
    settings, warnings = load_settings(config)
    assert settings == Settings() and warnings


def test_estado_salva_e_carrega(tmp_path: Path):
    path = tmp_path / "estado.json"
    State(last_input_dir="C:/x", user_name="Fulano").save(path)
    assert State.load(path).user_name == "Fulano"
    path.write_text("{corrompido", encoding="utf-8")
    assert State.load(path) == State()


def test_rotulos_e_nomes_padrao():
    assert [speaker_label(i) for i in (0, 1, 25, 26)] == ["A", "B", "Z", "AA"]
    assert default_name("A") == "Pessoa A"
    assert default_name("Speaker 1") == "Speaker 1"


def test_perfil_de_voz_reconhece_e_atualiza(tmp_path: Path):
    rng = np.random.default_rng(0)
    me = rng.normal(size=256)
    me /= np.linalg.norm(me)
    other = rng.normal(size=256)
    other /= np.linalg.norm(other)
    book = VoiceBook(tmp_path / "vozes.json")
    book.remember("Fulano", me.tolist(), is_me=True)
    book.save()
    book = VoiceBook(tmp_path / "vozes.json")
    noisy_me = me + rng.normal(scale=0.02, size=256)
    profile, similarity = book.best_match((noisy_me / np.linalg.norm(noisy_me)).tolist())
    assert profile.name == "Fulano" and profile.is_me and similarity > 0.9
    assert book.best_match(other.tolist())[1] < 0.3


def test_exemplos_de_frases_espalhados_e_mais_exemplos():
    sentences = [Sentence(f"frase número {i} com bastante conteúdo aqui", i * 10, i * 10 + 5, "A")
                 for i in range(12)] + [Sentence("oi", 200, 201, "B")]
    t = Transcript("x.wav", "audio", 300, sentences, "teste", has_speakers=True)
    chosen = pick_samples(t, "A")
    assert len(chosen) == 3 and chosen[0].start == 0
    assert chosen[-1].start >= 60  # espalhados pela gravação
    more = more_samples(t, "A", {id(s) for s in chosen})
    assert more and not {id(s) for s in more} & {id(s) for s in chosen}
    assert pick_samples(t, "B")[0].text == "oi"  # voz com pouca fala ainda tem exemplo
    share = speaking_share(t)
    assert share["A"][1] == 12 and share["A"][0] > share["B"][0]


def test_impressao_do_arquivo_muda_com_o_conteudo(tmp_path: Path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 10_000_000)
    first = cache.fingerprint(a)
    assert cache.fingerprint(a) == first
    a.write_bytes(b"x" * 5_000_000 + b"y" + b"x" * 4_999_999)
    assert cache.fingerprint(a) != first
