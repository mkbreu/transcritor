"""Áudio e vídeo: identificar o tipo, extrair o áudio e preparar o som para a transcrição.

Usa o PyAV (FFmpeg embutido): não precisa instalar o FFmpeg no Windows.
"""

from __future__ import annotations

import contextlib
import io
import logging
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from av.stream import Disposition

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".m4v", ".3gp", ".flv", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts",
)
AUDIO_EXTENSIONS = (
    ".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac", ".wma", ".amr", ".aiff", ".aif",
    ".3ga", ".weba", ".mka",
)
SAMPLE_RATE = 16_000  # taxa usada pelos modelos de IA

# progress(segundos_processados, segundos_totais)
ProgressFn = Callable[[float, float], None]


class MediaError(RuntimeError):
    """Arquivo de mídia ilegível ou sem áudio."""


@dataclass(slots=True)
class MediaInfo:
    path: Path
    kind: str  # "audio" ou "video"
    duration: float  # pode ser 0 se o arquivo não informar (é corrigido ao decodificar)
    codec: str
    channels: int
    sample_rate: int


def probe(path: Path) -> MediaInfo:
    """Descobre se o arquivo é áudio ou vídeo e lê as características do som."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise MediaError("o arquivo não tem nenhuma faixa de áudio")
            stream = container.streams.audio[0]
            has_video = any(
                not (video.disposition & Disposition.attached_pic)  # capa de MP3 não é vídeo
                for video in container.streams.video
            )
            duration = 0.0
            if container.duration:
                duration = container.duration / av.time_base
            elif stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            return MediaInfo(
                path=path,
                kind="video" if has_video else "audio",
                duration=duration,
                codec=stream.codec_context.name,
                channels=stream.codec_context.channels or 1,
                sample_rate=stream.codec_context.sample_rate or 0,
            )
    except MediaError:
        raise
    except (av.FFmpegError, OSError) as exc:
        log.warning("Falha ao abrir %s: %s", path, exc)
        raise MediaError(
            "não foi possível abrir como áudio ou vídeo (arquivo danificado ou formato não suportado)"
        ) from exc


def decode_to_wav(path: Path, dest: Path, progress: ProgressFn | None = None) -> float:
    """Converte o som do arquivo para WAV 16 kHz mono (formato dos modelos). Devolve a duração."""
    info = probe(path)
    tmp = dest.with_name(dest.stem + ".parcial.wav")
    written = 0
    bad_packets = 0
    try:
        with av.open(str(path)) as container, wave.open(str(tmp), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.InvalidDataError:
                    bad_packets += 1  # trecho corrompido: pula e continua
                    continue
                for frame in frames:
                    for converted in resampler.resample(frame):
                        data = converted.to_ndarray()
                        out.writeframes(data.tobytes())
                        written += data.shape[-1]
                if progress:
                    progress(written / SAMPLE_RATE, info.duration)
            for converted in resampler.resample(None):
                data = converted.to_ndarray()
                out.writeframes(data.tobytes())
                written += data.shape[-1]
        tmp.replace(dest)
    except av.FFmpegError as exc:
        raise MediaError(f"falha ao ler o áudio ({exc})") from exc
    finally:
        tmp.unlink(missing_ok=True)
    if bad_packets:
        log.warning("%s: %d trechos corrompidos foram ignorados", path.name, bad_packets)
    if written == 0:
        raise MediaError("nenhum som foi encontrado no arquivo")
    return written / SAMPLE_RATE


def audio_extension_for(info: MediaInfo) -> str:
    """Extensão do áudio extraído: copia sem perda quando possível, senão gera .m4a."""
    return ".mp3" if info.codec == "mp3" else ".m4a"


def extract_audio(info: MediaInfo, dest: Path, progress: ProgressFn | None = None) -> Path:
    """Salva o áudio de um vídeo como arquivo próprio (cópia sem perda se for AAC ou MP3)."""
    fmt = "mp3" if dest.suffix.lower() == ".mp3" else "ipod"
    can_copy = (fmt == "ipod" and info.codec in ("aac", "alac")) or (
        fmt == "mp3" and info.codec == "mp3"
    )
    tmp = dest.with_name(dest.stem + ".parcial" + dest.suffix)
    try:
        if can_copy:
            _remux(info, tmp, fmt, progress)
        else:
            _transcode_aac(info, tmp, progress)
        tmp.replace(dest)
    except av.FFmpegError as exc:
        raise MediaError(f"falha ao salvar o áudio extraído ({exc})") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _remux(info: MediaInfo, dest: Path, fmt: str, progress: ProgressFn | None) -> None:
    with av.open(str(info.path)) as source, av.open(str(dest), "w", format=fmt) as target:
        in_stream = source.streams.audio[0]
        out_stream = target.add_stream_from_template(in_stream)
        for packet in source.demux(in_stream):
            if packet.dts is None:
                continue
            if progress and packet.pts is not None and packet.time_base:
                progress(float(packet.pts * packet.time_base), info.duration)
            packet.stream = out_stream
            target.mux(packet)


def _transcode_aac(info: MediaInfo, dest: Path, progress: ProgressFn | None) -> None:
    rate = info.sample_rate if info.sample_rate in (16000, 22050, 24000, 32000, 44100, 48000) else 48000
    layout = "mono" if info.channels == 1 else "stereo"
    with av.open(str(info.path)) as source, av.open(str(dest), "w", format="ipod") as target:
        in_stream = source.streams.audio[0]
        out_stream = target.add_stream("aac", rate=rate, layout=layout)
        out_stream.bit_rate = 96_000 if layout == "mono" else 128_000
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        for packet in source.demux(in_stream):
            try:
                frames = packet.decode()
            except av.InvalidDataError:
                continue
            for frame in frames:
                if progress and frame.time is not None:
                    progress(frame.time, info.duration)
                frame.pts = None
                for converted in resampler.resample(frame):
                    target.mux(out_stream.encode(converted))
        for converted in resampler.resample(None):
            target.mux(out_stream.encode(converted))
        target.mux(out_stream.encode(None))


# ---------------------------------------------------------------- leitura do WAV preparado


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def read_wav(path: Path, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Lê um trecho do WAV 16 kHz mono como float32 entre -1 e 1."""
    with wave.open(str(path), "rb") as source:
        total = source.getnframes()
        first = max(0, min(int(start * SAMPLE_RATE), total))
        last = total if end is None else max(first, min(int(end * SAMPLE_RATE), total))
        source.setpos(first)
        raw = source.readframes(last - first)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load_wav_lowmem(path: Path) -> np.ndarray:
    """Lê o WAV inteiro como float32 sem manter uma cópia extra em memória (via memmap)."""
    with wave.open(str(path), "rb") as source:
        frames = source.getnframes()
    offset = path.stat().st_size - frames * 2  # o bloco de dados é o último do arquivo
    pcm = np.memmap(path, dtype=np.int16, mode="r", offset=offset, shape=(frames,))
    try:
        return pcm.astype(np.float32) / 32768.0
    finally:
        del pcm


def play_segment(path: Path, start: float, end: float, max_seconds: float = 15.0) -> None:
    """Toca um trecho do áudio no alto-falante (espera terminar)."""
    import winsound  # só existe no Windows

    end = min(end, start + max_seconds)
    samples = read_wav(path, max(0.0, start - 0.25), end + 0.35)
    if samples.size == 0:
        return
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    with contextlib.suppress(RuntimeError):  # sem placa de som: ignora
        winsound.PlaySound(buffer.getvalue(), winsound.SND_MEMORY)
