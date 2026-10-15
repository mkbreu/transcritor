"""Fluxo com o usuário: escolher arquivos → perguntas → processar → nomear vozes → salvar → resumo.

O processamento pesado (transcrição e separação de vozes) fica em pipeline.py.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

from rich.table import Table

from . import __version__, cache, dialogs, media, render, winsys
from . import console as ui
from .logging_setup import setup_logging
from .pipeline import Cancelled, Job, gb, process
from .settings import Settings, State, load_settings
from .subtitles import SUBTITLE_EXTENSIONS
from .voices import VoiceBook, name_speakers

log = logging.getLogger("transcritor")


def _parse_args(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Transcritor",
        description="Transcreve vídeo, áudio ou legenda SRT em texto organizado por pessoa.",
    )
    parser.add_argument("arquivos", nargs="*", help="arquivos a transcrever (sem nada, abre a janela)")
    parser.add_argument("--sem-perguntas", action="store_true",
                        help="modo automático: sem janelas nem perguntas (usa os padrões)")
    parser.add_argument("--saida", type=Path, help="pasta onde salvar (modo automático)")
    parser.add_argument("--pessoas", type=int, help="número de pessoas que falam (modo automático)")
    parser.add_argument("--prioridade-normal", action="store_true",
                        help="não reduzir a prioridade (útil se outro programa pesado estiver aberto)")
    return parser.parse_args(arguments)


def run(arguments: list[str]) -> int:
    args = _parse_args(arguments)
    log_path = setup_logging()
    settings, warnings = load_settings()
    if args.prioridade_normal:
        settings.prioridade_baixa = False
    state = State.load()
    ui.set_interactive(not args.sem_perguntas)
    ui.banner(__version__)
    for warning in warnings:
        ui.warn(warning)
        log.warning(warning)
    log.info("Configurações: %s", settings)
    cache.cleanup(settings.manter_cache_dias)
    paths = _expand_folders([Path(arg.strip().strip('"')) for arg in args.arquivos if arg.strip()])
    if args.sem_perguntas:
        return _run_automatic(paths, args, settings, state, log_path)
    while True:
        if not paths:
            ui.info("Escolha os arquivos na janela que abriu (vídeo, áudio ou legenda SRT)...")
            paths = dialogs.ask_input_files(state.last_input_dir)
            if not paths:
                ui.warn("Nenhum arquivo escolhido.")
                if ui.ask_yes_no("Quer escolher de novo?", default=False):
                    continue
                return 0
        state.last_input_dir = str(paths[0].parent)
        log.info("Arquivos: %s", [str(p) for p in paths])
        jobs = [_classify(path) for path in paths]
        started = time.monotonic()
        try:
            _prepare(jobs, settings, state)
            started = time.monotonic()
            process(jobs, settings)
        except Cancelled:
            for job in jobs:
                if job.active:
                    job.cancel("cancelado por você")
        elapsed = time.monotonic() - started
        _name_voices(jobs, settings, state)
        _save(jobs, settings)
        state.save()
        _summary(jobs, log_path, elapsed)
        if _end_menu(jobs) != "novo":
            return 0
        paths = []


def _run_automatic(
    paths: list[Path], args: argparse.Namespace, settings: Settings, state: State, log_path: Path
) -> int:
    """Modo --sem-perguntas: sem janelas; salva ao lado do original (ou em --saida)."""
    if not paths:
        ui.error("No modo --sem-perguntas, informe os arquivos na linha de comando.")
        return 3
    log.info("Arquivos (modo automático): %s", [str(p) for p in paths])
    jobs = [_classify(path) for path in paths]
    _prepare_automatic(jobs, settings, args)
    started = time.monotonic()
    process(jobs, settings)
    elapsed = time.monotonic() - started
    _name_voices(jobs, settings, state)
    _save(jobs, settings)
    _summary(jobs, log_path, elapsed)
    return 0 if all(job.status == "ok" for job in jobs) else 3


# ------------------------------------------------------------------ preparação


def _expand_folders(paths: list[Path]) -> list[Path]:
    """Pasta arrastada para o atalho → os vídeos, áudios e legendas que estão dentro dela."""
    supported = set(media.VIDEO_EXTENSIONS + media.AUDIO_EXTENSIONS + SUBTITLE_EXTENSIONS)
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded += sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in supported)
        else:
            expanded.append(path)
    return expanded


def _classify(path: Path) -> Job:
    job = Job(path)
    if not path.is_file():
        job.fail("arquivo não encontrado")
    elif path.suffix.lower() in SUBTITLE_EXTENSIONS:
        job.kind = "legenda"
    else:
        try:
            job.info = media.probe(path)
            job.kind, job.duration = job.info.kind, job.info.duration
        except media.MediaError as exc:
            job.fail(str(exc))
    return job


def _prepare(jobs: list[Job], settings: Settings, state: State) -> None:
    valid = [job for job in jobs if job.active]
    for job in jobs:
        if not job.active:
            ui.error(f"{job.source.name}: {job.message}")
    if not valid:
        return
    _choose_outputs(valid, settings, state)
    media_jobs = [job for job in valid if job.active and job.is_media]
    if media_jobs:
        ui.title("Antes de começar")
    for job in media_jobs:
        _ask_media_questions(job, settings)


def _prepare_automatic(jobs: list[Job], settings: Settings, args: argparse.Namespace) -> None:
    reserved: set[Path] = set()
    for job in jobs:
        if not job.active:
            ui.error(f"{job.source.name}: {job.message}")
            continue
        folder = args.saida or job.source.parent
        folder.mkdir(parents=True, exist_ok=True)
        job.out_txt = _reserve(folder / render.suggested_output_name(job.source), reserved)
        if job.kind == "video" and settings.salvar_audio_extraido and job.info:
            ext = media.audio_extension_for(job.info)
            job.out_audio = _reserve(folder / f"{job.source.stem} - áudio{ext}", reserved)
        if job.is_media:
            job.entry = cache.entry_for(job.source)
            job.num_speakers = args.pessoas


def _choose_outputs(valid: list[Job], settings: Settings, state: State) -> None:
    last_dir = Path(state.last_output_dir) if state.last_output_dir else None
    if len(valid) == 1:
        job = valid[0]
        initial = job.source.parent  # sugere salvar ao lado do arquivo original
        ui.info("Escolha onde salvar a transcrição (TXT)...")
        while True:
            chosen = dialogs.ask_save_text(initial, render.suggested_output_name(job.source))
            if chosen is None:
                job.cancel("nenhum local escolhido para salvar")
                ui.warn("Cancelado: nenhum local escolhido para salvar a transcrição.")
                return
            if chosen.resolve() != job.source.resolve():
                break
            ui.error("Escolha um nome diferente do arquivo original (ele nunca é sobrescrito).")
        job.out_txt = chosen
        state.last_output_dir = str(chosen.parent)
        if job.kind == "video" and settings.salvar_audio_extraido and job.info:
            ext = media.audio_extension_for(job.info)
            ui.info("Escolha onde salvar o áudio extraído do vídeo (ou cancele para não salvar)...")
            job.out_audio = dialogs.ask_save_audio(chosen.parent, f"{job.source.stem} - áudio{ext}", ext)
            if job.out_audio is None:
                ui.info("Ok: o áudio extraído não será salvo.")
        return
    ui.info(f"{len(valid)} arquivos escolhidos. Escolha a PASTA onde salvar as transcrições...")
    folder = dialogs.ask_folder(
        str(last_dir or valid[0].source.parent), "Pasta onde salvar as transcrições"
    )
    if folder is None:
        for job in valid:
            job.cancel("nenhuma pasta escolhida para salvar")
        ui.warn("Cancelado: nenhuma pasta escolhida.")
        return
    state.last_output_dir = str(folder)
    reserved: set[Path] = set()
    for job in valid:
        job.out_txt = _reserve(folder / render.suggested_output_name(job.source), reserved)
        if job.kind == "video" and settings.salvar_audio_extraido and job.info:
            ext = media.audio_extension_for(job.info)
            job.out_audio = _reserve(folder / f"{job.source.stem} - áudio{ext}", reserved)


def _reserve(path: Path, reserved: set[Path]) -> Path:
    """Nome livre no disco e ainda não usado por outro arquivo do mesmo lote."""
    candidate = render.unique_path(path)
    counter = 2
    while candidate in reserved:
        candidate = render.unique_path(path.with_name(f"{path.stem} ({counter}){path.suffix}"))
        counter += 1
    reserved.add(candidate)
    return candidate


def _ask_media_questions(job: Job, settings: Settings) -> None:
    job.entry = cache.entry_for(job.source)
    if job.entry.transcript.exists():
        when = datetime.fromtimestamp(job.entry.transcript.stat().st_mtime)
        ui.info(f"{job.source.name} já foi transcrito em {when:%d/%m/%Y às %H:%M}.")
        job.reuse = ui.ask_yes_no(
            "Reaproveitar esse resultado? (instantâneo; você ainda pode renomear as vozes)", True
        )
        if job.reuse:
            return
    if settings.separar_vozes and settings.perguntar_numero_de_pessoas:
        job.num_speakers = ui.ask_int(
            f"Quantas pessoas falam em {job.source.name}? (Enter = descobrir sozinho)", 1, 20
        )


# ------------------------------------------------------------------ vozes, gravação e resumo


def _name_voices(jobs: list[Job], settings: Settings, state: State) -> None:
    book = VoiceBook()
    for job in jobs:
        if not (job.active and job.transcript and job.transcript.has_speakers):
            continue
        wav = job.entry.audio_wav if job.entry and job.entry.audio_wav.exists() else None
        name_speakers(job.transcript, wav, job.prints, book, settings, state)
        if job.entry:
            job.transcript.save(job.entry.transcript)


def _save(jobs: list[Job], settings: Settings) -> None:
    options = render.RenderOptions(settings.mostrar_horarios, settings.uma_frase_por_linha)
    for job in jobs:
        if not (job.active and job.transcript and job.out_txt):
            continue
        text = render.render_text(job.transcript, options)
        while True:
            try:
                render.write_text_atomic(job.out_txt, text)
                job.status = "ok"
                log.info("%s: salvo em %s (%d palavras)", job.source.name, job.out_txt,
                         job.transcript.word_count())
                break
            except PermissionError:
                ui.error(f"Não consegui salvar {job.out_txt.name}: o arquivo está aberto em outro "
                         "programa (ex.: Word) ou a pasta não permite gravar.")
                if not ui.is_interactive():
                    job.fail("não foi salvo (arquivo de destino bloqueado)")
                    break
                answer = ui.ask("[T] tentar de novo (feche o arquivo antes) · [O] salvar com outro nome"
                                " · [P] pular").lower()
                if answer == "o":
                    chosen = dialogs.ask_save_text(job.out_txt.parent, job.out_txt.name)
                    if chosen:
                        job.out_txt = chosen
                elif answer == "p":
                    job.fail("não foi salvo (arquivo de destino bloqueado)")
                    break
            except OSError as exc:
                job.fail(f"não foi possível salvar: {exc}")
                break


def _summary(jobs: list[Job], log_path: Path, elapsed: float) -> None:
    ui.title("Resumo")
    table = Table(show_edge=False, header_style="bold")
    for column in ("Arquivo", "Resultado", "Duração", "Pessoas", "Palavras"):
        table.add_column(column, overflow="fold")
    results = {"ok": "[ok]Concluído[/]", "erro": "[erro]Erro[/]", "cancelado": "[aviso]Cancelado[/]"}
    for job in jobs:
        transcript = job.transcript
        people = "-"
        if transcript and transcript.has_speakers:
            people = str(len({transcript.display_name(label) for label in transcript.labels}))
        table.add_row(
            ui.escape(job.source.name),
            results.get(job.status, ui.escape(job.status)),
            render.format_duration(job.duration) if job.duration else "-",
            people,
            str(transcript.word_count()) if transcript and job.status == "ok" else "-",
        )
    ui.console.print(table)
    ui.console.print()
    for job in jobs:
        if job.status == "ok" and job.out_txt:
            ui.ok(f"Transcrição salva em: {job.out_txt}")
            if job.audio_saved:
                ui.ok(f"Áudio extraído salvo em: {job.audio_saved}")
        elif job.message:
            (ui.error if job.status == "erro" else ui.warn)(f"{job.source.name}: {job.message}")
    if elapsed >= 1:
        ui.console.print(f"  [fraco]Tempo de processamento: {render.format_duration(elapsed)}[/]")
    peak = winsys.process_memory()[1]
    if peak:
        ui.console.print(f"  [fraco]Pico de memória usado: {gb(peak)} GB[/]")
    ui.console.print(f"  [fraco]Registro desta execução (log): {ui.escape(str(log_path))}[/]")
    log.info("Resumo: %s", {job.source.name: job.status for job in jobs})


def _end_menu(jobs: list[Job]) -> str:
    done = [job for job in jobs if job.status == "ok" and job.out_txt]
    options = []
    if len(done) == 1:
        options.append("[A] abrir o TXT")
    if done:
        options.append("[P] abrir a pasta")
    options += ["[N] transcrever outros arquivos", "[Enter] sair"]
    while True:
        ui.console.print()
        answer = ui.ask("  ·  ".join(options)).lower()
        if answer == "a" and len(done) == 1:
            winsys.open_file(done[0].out_txt)
        elif answer == "p" and done:
            winsys.show_in_folder(done[0].out_txt)
        elif answer == "n":
            return "novo"
        elif not answer:
            return "sair"
        else:
            ui.warn("Opção inválida.")
