"""Partes do fluxo principal que não precisam dos modelos de IA."""

from pathlib import Path

from transcritor import app, pipeline
from transcritor import console as ui
from transcritor.settings import Settings
from transcritor.transcript import Sentence, Transcript


def make_job(tmp_path: Path, name: str = "saida.txt") -> app.Job:
    job = app.Job(source=tmp_path / "entrada.srt", kind="legenda")
    job.out_txt = tmp_path / name
    job.transcript = Transcript("entrada.srt", "legenda", 2.0, [Sentence("Olá.", 0, 1)], "teste", False)
    return job


def test_salva_e_marca_concluido(tmp_path: Path):
    job = make_job(tmp_path)
    app._save([job], Settings())
    assert job.status == "ok" and "Olá." in job.out_txt.read_text(encoding="utf-8")


def test_arquivo_bloqueado_no_modo_automatico_nao_trava(tmp_path: Path):
    job = make_job(tmp_path)
    job.out_txt.write_text("versão antiga", encoding="utf-8")
    ui.set_interactive(False)
    try:
        with open(job.out_txt, encoding="utf-8"):  # aberto em "outro programa": não dá para trocar
            app._save([job], Settings())
    finally:
        ui.set_interactive(True)
    assert job.status == "erro" and "bloqueado" in job.message
    assert job.out_txt.read_text(encoding="utf-8") == "versão antiga"  # original intacto


def test_classifica_arquivos(tmp_path: Path):
    legenda = tmp_path / "a.SRT"
    legenda.write_text("1\n00:00:00,000 --> 00:00:01,000\nOi\n", encoding="utf-8")
    falso = tmp_path / "b.mp4"
    falso.write_text("não é vídeo", encoding="utf-8")
    assert app._classify(legenda).kind == "legenda"
    assert app._classify(falso).status == "erro"
    assert app._classify(tmp_path / "nao_existe.mp3").message == "arquivo não encontrado"


def test_pasta_arrastada_vira_lista_de_arquivos(tmp_path: Path):
    for name in ("b.mp4", "a.srt", "notas.txt", "c.MP3"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    extra = tmp_path.parent / "solto.wav"
    found = app._expand_folders([tmp_path, extra])
    assert [p.name for p in found] == ["a.srt", "b.mp4", "c.MP3", "solto.wav"]


def test_nomes_reservados_no_lote_nao_colidem(tmp_path: Path):
    reserved: set[Path] = set()
    first = app._reserve(tmp_path / "a - transcrição.txt", reserved)
    second = app._reserve(tmp_path / "a - transcrição.txt", reserved)
    assert first != second and second.name == "a - transcrição (2).txt"


def test_legenda_vira_transcricao(tmp_path: Path):
    source = tmp_path / "x.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:02,000\nSpeaker 1: Bom dia.\n\n"
                      "2\n00:00:02,500 --> 00:00:04,000\nSpeaker 2: Bom dia, tudo bem?\n", encoding="utf-8")
    job = app._classify(source)
    pipeline.process_subtitle(job)
    assert job.transcript.has_speakers
    assert [s.speaker for s in job.transcript.sentences] == ["Speaker 1", "Speaker 2"]
