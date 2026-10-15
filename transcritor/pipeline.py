"""Processamento pesado de um lote de arquivos: legenda → frases; mídia → áudio → texto → vozes.

Cada etapa carrega o seu modelo uma vez para o lote inteiro e o libera ao terminar, para que
só um modelo ocupe a memória de cada vez. Falha em um arquivo não interrompe os outros.
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import cache, media, models, winsys
from . import console as ui
from .asr import ENGINE_NAME, ParakeetTranscriber, transcribe_file
from .diarization import (
    Diarizer,
    SpeakerTurn,
    VoiceEmbedder,
    assign_speakers,
    build_sentences,
    merge_similarity_from_sensitivity,
    separate_speakers,
    threshold_from_sensitivity,
)
from .sentences import split_into_sentences, words_from_cues
from .settings import Settings
from .subtitles import SubtitleError, parse_subtitle_file
from .transcript import Transcript, Word

log = logging.getLogger("transcritor")

# Versão do método de separação de vozes. Resultados guardados no cache com versão anterior
# não são reaproveitados (a transcrição em si é reaproveitada; só as vozes são refeitas).
# 2 = refino que absorve migalhas e junta grupos da mesma pessoa.
# 3 = segmentação Reverb; nº de pessoas informado vira pista (o programa detecta de fato).
RESULT_VERSION = 3


class Cancelled(Exception):
    """O usuário escolheu cancelar o processamento."""


@dataclass
class Job:
    """Um arquivo do lote e tudo o que foi produzido para ele."""

    source: Path
    kind: str = ""  # "legenda", "audio" ou "video"
    info: media.MediaInfo | None = None
    out_txt: Path | None = None
    out_audio: Path | None = None
    num_speakers: int | None = None
    reuse: bool = False  # aproveitar o resultado já guardado no cache
    entry: cache.CacheEntry | None = None
    words: list[Word] = field(default_factory=list)
    turns: list[SpeakerTurn] = field(default_factory=list)
    prints: dict[str, list[float]] = field(default_factory=dict)
    transcript: Transcript | None = None
    duration: float = 0.0
    status: str = "pendente"  # "ok", "erro" ou "cancelado"
    message: str = ""
    audio_saved: Path | None = None

    @property
    def active(self) -> bool:
        return self.status == "pendente"

    @property
    def is_media(self) -> bool:
        return self.kind in ("audio", "video")

    def fail(self, message: str) -> None:
        self.status, self.message = "erro", message
        log.error("%s: %s", self.source.name, message)

    def cancel(self, message: str) -> None:
        self.status, self.message = "cancelado", message
        log.info("%s: cancelado (%s)", self.source.name, message)


def process(jobs: list[Job], settings: Settings) -> None:
    for job in jobs:
        if job.active and job.kind == "legenda":
            process_subtitle(job)
    media_jobs = [job for job in jobs if job.active and job.is_media]
    if not media_jobs:
        return
    if not _ensure_models():
        for job in media_jobs:
            job.fail("os modelos de IA não estão instalados (rode instalar.py com internet)")
        return
    if settings.prioridade_baixa:
        winsys.lower_priority()
    ui.title("Processando")
    with winsys.keep_awake():
        _prepare_audio(media_jobs)
        _transcribe_all(media_jobs, settings)
        if settings.separar_vozes:
            _diarize_all(media_jobs, settings)
        _finalize_all(media_jobs, settings)


def process_subtitle(job: Job) -> None:
    try:
        cues = parse_subtitle_file(job.source)
    except (SubtitleError, OSError) as exc:
        job.fail(str(exc))
        return
    words = words_from_cues(cues)
    job.duration = max(cue.end for cue in cues)
    job.transcript = Transcript(
        source_name=job.source.name,
        source_kind="legenda",
        duration=job.duration,
        sentences=split_into_sentences(words, split_on_speaker_change=True),
        engine="Legenda SRT/VTT",
        has_speakers=any(cue.speaker for cue in cues),
        created_at=now(),
    )
    log.info("%s: legenda com %d falas → %d frases", job.source.name, len(cues),
             len(job.transcript.sentences))


def _ensure_models() -> bool:
    missing = models.missing_models()
    if not missing:
        return True
    total = sum(spec.size for spec in missing)
    ui.warn(f"Os modelos de IA ainda não foram baixados ({models.human_size(total)}).")
    if not ui.ask_yes_no("Baixar agora? (precisa de internet; se cair, continua de onde parou)"):
        return False
    try:
        with ui.stage("Baixando os modelos de IA") as bar:
            current = {"label": ""}

            def progress(label: str, done: int, size: int) -> None:
                if label != current["label"]:
                    current["label"] = label
                    bar.describe(label)
                bar.update(done, size)

            models.ensure_models(missing, progress=progress)
    except models.ModelDownloadError as exc:
        ui.error(str(exc))
        return False
    return True


def _prepare_audio(jobs: list[Job]) -> None:
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        if job.reuse or not job.active or job.entry is None or job.info is None:
            continue
        try:
            if not job.entry.audio_wav.exists():
                with ui.stage(f"[{index}/{total}] Preparando o áudio de {job.source.name}") as bar:
                    job.duration = media.decode_to_wav(job.source, job.entry.audio_wav, bar.update)
            else:
                job.duration = media.wav_duration(job.entry.audio_wav)
        except (media.MediaError, OSError) as exc:
            job.fail(f"não foi possível ler o áudio: {exc}")
            continue
        if job.out_audio:
            try:
                with ui.stage(f"[{index}/{total}] Salvando o áudio extraído") as bar:
                    job.audio_saved = media.extract_audio(job.info, job.out_audio, bar.update)
            except (media.MediaError, OSError) as exc:
                ui.warn(f"O áudio extraído não pôde ser salvo ({exc}); a transcrição continua.")
                log.warning("%s: falha ao salvar o áudio extraído: %s", job.source.name, exc)


def _check_memory(settings: Settings, purpose: str) -> None:
    """Pouca RAM livre só deixa mais lento (o Windows usa o arquivo de paginação): avisa e segue.
    Só para e pergunta se nem RAM + paginação derem conta — aí haveria risco real de falha."""
    need = settings.memoria_recomendada_gb * winsys.GB
    while True:
        status = winsys.memory_status()
        if status is None or status.available >= need:
            return
        log.warning("Pouca memória: livre %.2f GB, reservável %.2f GB (recomendado %.1f GB)",
                    status.available / winsys.GB, status.commit_available / winsys.GB,
                    settings.memoria_recomendada_gb)
        if status.commit_available >= need:
            ui.warn(f"Memória livre baixa ({gb(status.available, 1)} GB): a {purpose} vai funcionar, "
                    "mas pode ficar mais lenta. Se puder, feche Chrome, ChatGPT ou Vibe.")
            return
        ui.warn(f"Memória quase esgotada: só {gb(status.commit_available, 1)} GB disponíveis "
                f"(a {purpose} precisa de ~{gb(need, 1)} GB). Pode falhar por falta de memória.")
        ui.info("Feche programas como Chrome, ChatGPT e Vibe e digite V para verificar de novo.")
        answer = ui.ask("[V] verificar de novo · [Enter] tentar mesmo assim · [C] cancelar").lower()
        if answer in ("c", "cancelar"):
            raise Cancelled
        if answer not in ("v", "verificar"):
            return


def _transcribe_all(jobs: list[Job], settings: Settings) -> None:
    pending = [job for job in jobs if job.active and not job.reuse]
    if not pending:
        return
    _check_memory(settings, "transcrição")
    holder: dict[str, ParakeetTranscriber] = {}

    def get_transcriber() -> ParakeetTranscriber:
        if "model" not in holder:
            ui.info("Carregando o modelo de transcrição (alguns segundos)...")
            holder["model"] = ParakeetTranscriber(settings.threads)
        return holder["model"]

    try:
        for index, job in enumerate(pending, start=1):
            assert job.entry is not None
            label = f"[{index}/{len(pending)}] {job.source.name}"
            try:
                with ui.stage(f"{label}: procurando os trechos com fala") as bar:
                    job.words = transcribe_file(
                        job.entry.audio_wav, job.duration, get_transcriber, job.entry.asr_journal,
                        settings.threads, on_progress=bar.update,
                        on_stage=lambda text, bar=bar, label=label: bar.describe(f"{label}: {text.lower()}"),
                    )
                log.info("%s: %d palavras transcritas | memória pico %.0f MB", job.source.name,
                         len(job.words), winsys.process_memory()[1] / 2**20)
                if not job.words:
                    job.fail("nenhuma fala foi reconhecida no áudio")
            except MemoryError:
                job.fail("memória insuficiente: feche outros programas e tente de novo")
            except (RuntimeError, OSError, ValueError) as exc:
                log.exception("Falha na transcrição de %s", job.source.name)
                job.fail(f"falha na transcrição: {exc}")
    finally:
        holder.clear()
        gc.collect()


def _diarize_all(jobs: list[Job], settings: Settings) -> None:
    pending = [job for job in jobs if job.active and not job.reuse]
    if not pending:
        return
    options = {
        "metodo": RESULT_VERSION,
        "pessoas": None,  # preenchido por arquivo
        "limiar": threshold_from_sensitivity(settings.sensibilidade_separacao),
        "juntar": merge_similarity_from_sensitivity(settings.sensibilidade_separacao),
    }
    todo = []
    for job in pending:
        assert job.entry is not None
        cached = _load_diarization(job.entry, options | {"pessoas": job.num_speakers})
        if cached:
            job.turns, job.prints = cached
        else:
            todo.append(job)
    if not todo:
        return
    _check_memory(settings, "separação de vozes")
    ui.info("Carregando os modelos de separação de vozes...")
    ui.info("A separação primeiro analisa a gravação inteira (a barra fica \"pulsando\", sem "
            "porcentagem, por alguns minutos) e depois mostra o andamento.")
    diarizer = Diarizer(settings.threads, settings.sensibilidade_separacao)
    embedder = VoiceEmbedder(settings.threads)  # impressões vocais: refino e reconhecimento
    try:
        for index, job in enumerate(todo, start=1):
            assert job.entry is not None
            try:
                samples = media.load_wav_lowmem(job.entry.audio_wav)
                label = f"[{index}/{len(todo)}] {job.source.name}"
                with ui.stage(f"{label}: separando as vozes") as bar:
                    result = separate_speakers(
                        samples, job.num_speakers, diarizer, embedder, options["juntar"], bar.update,
                        on_redo=lambda detected, bar=bar, label=label, hint=job.num_speakers: (
                            bar.describe(f"{label}: a voz indicou {detected}; refazendo para {hint} pessoas")
                        ),
                    )
                job.turns, job.prints = result.turns, result.prints
                del samples
                gc.collect()
                _save_diarization(job.entry, options | {"pessoas": job.num_speakers},
                                  job.turns, job.prints)
                _report_separation(job, result.found, result.detected, result.redone_with_hint)
                log.info("%s: detectadas %d vozes, resultado final %d (refeita com a pista: %s), "
                         "%d trechos | memória pico %.0f MB", job.source.name, result.detected,
                         result.found, result.redone_with_hint, len(job.turns),
                         winsys.process_memory()[1] / 2**20)
            except MemoryError:
                ui.warn(f"{job.source.name}: memória insuficiente para separar as vozes; "
                        "o texto sai sem identificar quem fala.")
                job.turns = []
            except (RuntimeError, OSError, ValueError) as exc:
                log.exception("Falha na separação de vozes de %s", job.source.name)
                ui.warn(f"{job.source.name}: a separação de vozes falhou ({exc}); "
                        "o texto sai sem identificar quem fala.")
                job.turns = []
    finally:
        diarizer.close()
        del diarizer, embedder
        gc.collect()


def _report_separation(job: Job, found: int, detected: int, redone: bool) -> None:
    """Explica na tela quantas pessoas foram identificadas e como isso se compara à pista dada."""
    people = f"{found} {'pessoa identificada' if found == 1 else 'pessoas identificadas'} pela voz"
    hint = job.num_speakers
    if redone:
        ui.ok(f"{job.source.name}: {people}. A análise automática achou {detected}, menos que as "
              f"{hint} que você informou; a separação foi refeita procurando {hint}.")
    elif hint and found > hint:
        ui.warn(f"{job.source.name}: {people} — você informou {hint}. Confira nos exemplos: se "
                "duas vozes forem da mesma pessoa, dê o mesmo nome a elas.")
    else:
        ui.ok(f"{job.source.name}: {people}.")


def _load_diarization(
    entry: cache.CacheEntry, options: dict
) -> tuple[list[SpeakerTurn], dict[str, list[float]]] | None:
    """Separação já feita com o mesmo método e as mesmas opções? Então reaproveita."""
    try:
        data = json.loads(entry.diarization.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for key, value in options.items():
        saved = data.get(key)
        same = (abs(saved - value) <= 1e-6 if isinstance(value, float) and isinstance(saved, float)
                else saved == value)
        if not same:
            return None
    turns = [SpeakerTurn(start, end, label) for start, end, label in data["trechos"]]
    return turns, data.get("impressoes", {})


def _save_diarization(
    entry: cache.CacheEntry,
    options: dict,
    turns: list[SpeakerTurn],
    prints: dict[str, list[float]],
) -> None:
    data = options | {
        "trechos": [[round(t.start, 3), round(t.end, 3), t.label] for t in turns],
        "impressoes": prints,
    }
    entry.diarization.write_text(json.dumps(data), encoding="utf-8")


def _finalize_all(jobs: list[Job], settings: Settings) -> None:
    for job in jobs:
        if not job.active or job.entry is None:
            continue
        if job.reuse:
            try:
                job.transcript = Transcript.load(job.entry.transcript)
                cached = json.loads(job.entry.diarization.read_text(encoding="utf-8"))
                job.prints = cached.get("impressoes", {})
            except (OSError, ValueError, KeyError, TypeError):
                if job.transcript is None:
                    job.fail("o resultado guardado está danificado; transcreva de novo")
            continue
        if settings.separar_vozes and job.turns:
            assign_speakers(job.words, job.turns)
            sentences = build_sentences(job.words)
            has_speakers = True
        else:
            sentences = split_into_sentences(job.words, split_on_speaker_change=False)
            has_speakers = False
        job.transcript = Transcript(
            source_name=job.source.name,
            source_kind=job.kind,
            duration=job.duration,
            sentences=sentences,
            engine=ENGINE_NAME,
            has_speakers=has_speakers,
            created_at=now(),
            version=RESULT_VERSION,
        )
        job.transcript.save(job.entry.transcript)
        job.words = []


def now() -> str:
    return datetime.now().strftime("%d/%m/%Y às %H:%M")


def gb(num_bytes: float, decimals: int = 2) -> str:
    """Bytes → "1,25" (gigabytes com vírgula decimal)."""
    return f"{num_bytes / winsys.GB:.{decimals}f}".replace(".", ",")
