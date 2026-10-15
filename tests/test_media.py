"""Mídia com arquivos gerados na hora (sem depender de gravações pessoais)."""

import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from transcritor import media

RATE = 44_100


def tone(seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(tone(2.0).tobytes())
    return path


@pytest.fixture
def mp4_file(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    pcm = tone(2.0, 16_000)
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=2)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = container.add_stream("aac", rate=16_000, layout="mono")
        black = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
        for i in range(4):
            frame = black.reformat(format="yuv420p")
            frame.pts, frame.time_base = i, Fraction(1, 2)
            container.mux(video.encode(frame))
        container.mux(video.encode(None))
        for start in range(0, len(pcm), 1024):
            block = pcm[start:start + 1024].reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(block, format="s16", layout="mono")
            frame.sample_rate, frame.pts = 16_000, start
            container.mux(audio.encode(frame))
        container.mux(audio.encode(None))
    return path


def test_detecta_audio_e_video(wav_file: Path, mp4_file: Path):
    assert media.probe(wav_file).kind == "audio"
    info = media.probe(mp4_file)
    assert info.kind == "video" and info.codec == "aac"
    assert media.audio_extension_for(info) == ".m4a"


def test_decodifica_para_16k_mono(wav_file: Path, tmp_path: Path):
    out = tmp_path / "16k.wav"
    duration = media.decode_to_wav(wav_file, out)
    assert duration == pytest.approx(2.0, abs=0.05)
    with wave.open(str(out)) as source:
        assert (source.getframerate(), source.getnchannels()) == (16_000, 1)
    samples = media.load_wav_lowmem(out)
    assert samples.dtype == np.float32 and abs(samples).max() <= 1.0
    assert len(media.read_wav(out, 0.5, 1.0)) == 8_000


def test_extrai_audio_do_video_sem_perda(mp4_file: Path, tmp_path: Path):
    info = media.probe(mp4_file)
    out = media.extract_audio(info, tmp_path / "audio.m4a")
    extracted = media.probe(out)
    assert extracted.kind == "audio" and extracted.codec == "aac"
    assert extracted.duration == pytest.approx(2.0, abs=0.2)


def test_converte_audio_que_nao_e_aac(wav_file: Path, tmp_path: Path):
    out = media.extract_audio(media.probe(wav_file), tmp_path / "convertido.m4a")
    assert media.probe(out).codec == "aac"


def test_arquivo_invalido_gera_mensagem_clara(tmp_path: Path):
    fake = tmp_path / "falso.mp4"
    fake.write_text("isto não é um vídeo", encoding="utf-8")
    with pytest.raises(media.MediaError, match="não foi possível abrir"):
        media.probe(fake)
