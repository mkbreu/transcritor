"""Gera um diálogo de teste em português com as vozes do Windows (Maria e Daniel).

Serve para testar a transcrição e a separação de vozes sem usar gravações pessoais.
Cria: <saida>.wav (16 kHz mono), <saida>.json (gabarito: quem falou o quê e quando)
e, com --video, também <saida>.mp4 (imagem preta + o mesmo áudio).

Uso:
  .venv\\Scripts\\python ferramentas\\gerar_dialogo_teste.py dados\\teste\\dialogo
  .venv\\Scripts\\python ferramentas\\gerar_dialogo_teste.py dados\\teste\\longo --minutos 58
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

RATE = 16_000
# Vozes "OneCore" do Windows (API WinRT). Atenção: pelo System.Speech (SAPI) a voz
# "Microsoft Daniel" aparece na lista, mas o Windows fala tudo com a voz da Maria.
VOICES = {"Maria": "Microsoft Maria", "Daniel": "Microsoft Daniel"}

DIALOGUE = [
    ("Maria", "Boa tarde. Meu nome é Maria e eu estou ligando por causa de uma cobrança no meu cartão."),
    ("Daniel", "Boa tarde, Maria. Eu sou o Daniel. Pode me informar o número do protocolo, por favor?"),
    ("Maria", "Claro. O protocolo é quatro cinco dois sete. Houve uma tentativa de estorno, porém não houve efetivação."),
    ("Daniel", "Entendi. Deixa eu verificar aqui no sistema. Só um momento."),
    ("Maria", "Tudo bem."),
    ("Daniel", "Aqui consta que o estorno foi solicitado no dia doze de março, no valor de mil e quinhentos reais."),
    ("Maria", "Isso mesmo. Mas até hoje o dinheiro não voltou para a minha conta."),
    ("Daniel", "Peço desculpas pelo transtorno. O prazo normal é de até cinco dias úteis, e já passou bastante tempo."),
    ("Maria", "Pois é. Eu já liguei três vezes e ninguém resolveu. Eu quero isso por escrito, por favor."),
    ("Daniel", "Sem problemas. Vou abrir uma reclamação formal e enviar a confirmação para o seu e-mail."),
    ("Maria", "E quanto tempo vai demorar para eu receber uma resposta?"),
    ("Daniel", "A equipe responsável tem até dez dias para responder. Se não houver retorno, a senhora pode procurar a ouvidoria."),
    ("Maria", "Certo. Então anota aí que eu não autorizei nenhuma outra cobrança depois do cancelamento."),
    ("Daniel", "Anotado. Mais alguma coisa em que eu possa ajudar?"),
    ("Maria", "Não, era só isso. Obrigada, Daniel."),
    ("Daniel", "Eu que agradeço. Tenha uma boa tarde."),
]


_WINRT_TTS = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result }
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
foreach ($job in $jobs) {
    $synth.Voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
        Where-Object { $_.DisplayName -eq $job.voice } | Select-Object -First 1
    if (-not $synth.Voice) { throw "Voz não encontrada: $($job.voice)" }
    $text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($job.text))
    $stream = Await ($synth.SynthesizeTextToStreamAsync($text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    $null = Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
    $bytes = New-Object byte[] ([int]$stream.Size); $reader.ReadBytes($bytes)
    [IO.File]::WriteAllBytes($job.out, $bytes)
}
"""


def synthesize(lines: list[tuple[str, str]], folder: Path) -> list[Path]:
    """Gera um WAV por fala com as vozes OneCore do Windows (Windows PowerShell 5.1 + WinRT)."""
    outputs = [folder / f"fala_{index:03d}.wav" for index in range(len(lines))]
    entries = []
    for (speaker, text), output in zip(lines, outputs, strict=True):
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        entries.append(f"@{{voice='{VOICES[speaker]}'; text='{encoded_text}'; out='{output}'}}")
    script = "$jobs = @(" + ",".join(entries) + ")\n" + _WINRT_TTS
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                   check=True, capture_output=True)
    return outputs


def read_pcm(path: Path) -> np.ndarray:
    """Lê o WAV da fala já convertido para 16 kHz mono (as vozes podem vir em outra taxa)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transcritor import media

    converted = path.with_name(path.stem + "_16k.wav")
    media.decode_to_wav(path, converted)
    with wave.open(str(converted), "rb") as source:
        return np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)


def build(output: Path, minutes: float, video: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        clips = [read_pcm(path) for path in synthesize(DIALOGUE, Path(tmp))]
    pieces: list[np.ndarray] = []
    truth: list[dict] = []
    cursor = 0
    target = minutes * 60 * RATE if minutes else 0
    while True:
        for (speaker, text), clip in zip(DIALOGUE, clips, strict=True):
            gap = np.zeros(int(rng.uniform(0.35, 1.1) * RATE), dtype=np.int16)
            pieces += [gap, clip]
            cursor += len(gap)
            truth.append({"speaker": speaker, "text": text,
                          "start": cursor / RATE, "end": (cursor + len(clip)) / RATE})
            cursor += len(clip)
        if cursor >= target:
            break
    pcm = np.concatenate([*pieces, np.zeros(RATE // 2, dtype=np.int16)])
    wav_path = output.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())
    output.with_suffix(".json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{wav_path} ({len(pcm) / RATE / 60:.1f} min, {len(truth)} falas)")
    if video:
        make_video(pcm, output.with_suffix(".mp4"))


def make_video(pcm: np.ndarray, path: Path) -> None:
    """Vídeo simples (imagem preta, 2 quadros/s) com o áudio do diálogo em AAC."""
    import av

    with av.open(str(path), "w") as container:
        vstream = container.add_stream("mpeg4", rate=2)
        vstream.width, vstream.height, vstream.pix_fmt = 320, 240, "yuv420p"
        astream = container.add_stream("aac", rate=RATE, layout="mono")
        black = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="rgb24")
        seconds = int(len(pcm) / RATE) + 1
        for index in range(seconds * 2):
            frame = black.reformat(format="yuv420p")
            frame.pts = index
            frame.time_base = Fraction(1, 2)
            container.mux(vstream.encode(frame))
        container.mux(vstream.encode(None))
        block = 1024
        for start in range(0, len(pcm), block):
            chunk = pcm[start:start + block].reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
            frame.sample_rate = RATE
            frame.pts = start
            container.mux(astream.encode(frame))
        container.mux(astream.encode(None))
    print(f"{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("saida", type=Path, help="caminho sem extensão (ex.: dados\\teste\\dialogo)")
    parser.add_argument("--minutos", type=float, default=0, help="repete o diálogo até essa duração")
    parser.add_argument("--video", action="store_true", help="também cria um .mp4")
    args = parser.parse_args()
    build(args.saida, args.minutos, args.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
