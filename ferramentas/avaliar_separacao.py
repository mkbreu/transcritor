"""Compara modelos de separação de vozes com gravações reais (dados\\teste\\reais).

Teste 1: "impressão vocal" — quão diferentes ficam pessoas diferentes (margem).
Teste 2: conversa montada com 3 pessoas reais intercaladas (gabarito exato de quem fala).
Teste 3: conversas reais de 2 e 4 pessoas — quantas vozes o sistema encontra sozinho.

Uso: .venv\\Scripts\\python ferramentas\\avaliar_separacao.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx as so

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transcritor import media, models, winsys  # noqa: E402

REAL = ROOT / "dados" / "teste" / "reais"
CAND = ROOT / "modelos" / "_candidatos"


def _first_existing(*paths: Path) -> Path:
    return next((p for p in paths if p.exists()), paths[0])


# Procura cada modelo na pasta de candidatos (testes) ou na pasta definitiva (modelos\).
EMBEDDINGS = {
    name: _first_existing(CAND / file, models.MODELS_DIR / file)
    for name, file in {
        "wespeaker-r34LM": "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "campplus-zh_en": models.CAMPPLUS.target,
        "titanet-small": models.TITANET.target,
    }.items()
}
EMBEDDINGS = {name: path for name, path in EMBEDDINGS.items() if path.exists()}
SEGMENTERS = {
    "pyannote-3.0": models.SEGMENTATION.file("model.onnx"),
    "reverb-v1": CAND / "sherpa-onnx-reverb-diarization-v1" / "model.onnx",
}
SEGMENTERS = {name: path for name, path in SEGMENTERS.items() if path.exists()}
SPEAKERS = ("fangjun", "leijun", "liudehua")


def load16k(path: Path) -> np.ndarray:
    wav16 = path.with_name(path.stem + ".16k.wav")
    if not wav16.exists():
        media.decode_to_wav(path, wav16)
    return media.read_wav(wav16)


def embed(extractor: so.SpeakerEmbeddingExtractor, samples: np.ndarray) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(16000, samples)
    stream.input_finished()
    vector = np.array(extractor.compute(stream))
    return vector / np.linalg.norm(vector)


def test_embeddings() -> None:
    print("\n=== Teste 1: impressão vocal (pessoas reais) ===")
    clips = {spk: [load16k(p) for p in sorted(REAL.glob(f"{spk}-*.wav")) if ".16k." not in p.name]
             for spk in SPEAKERS}
    for name, path in EMBEDDINGS.items():
        ext = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=4))
        vectors = [(spk, embed(ext, clip)) for spk in SPEAKERS for clip in clips[spk]]
        same, diff = [], []
        for (s1, v1), (s2, v2) in itertools.combinations(vectors, 2):
            (same if s1 == s2 else diff).append(float(v1 @ v2))
        print(f"{name:16s} mesma pessoa: média {np.mean(same):.2f} (mín {min(same):.2f}) | "
              f"pessoas diferentes: média {np.mean(diff):.2f} (máx {max(diff):.2f}) | "
              f"margem {min(same) - max(diff):+.2f}", flush=True)


def build_conversation() -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    rng = np.random.default_rng(3)
    items = [(spk, load16k(p)) for spk in SPEAKERS
             for p in sorted(REAL.glob(f"{spk}-*.wav")) if ".16k." not in p.name]
    order = list(rng.permutation(len(items)))
    # reordena para quase sempre alternar quem fala (como numa conversa)
    arranged = [items[order.pop(0)]]
    while order:
        pick = next((i for i in order if items[i][0] != arranged[-1][0]), order[0])
        order.remove(pick)
        arranged.append(items[pick])
    pieces, truth, cursor = [], [], 0
    for spk, clip in arranged:
        gap = np.zeros(int(rng.uniform(0.25, 0.9) * 16000), dtype=np.float32)
        pieces += [gap, clip]
        cursor += len(gap)
        truth.append((cursor / 16000, (cursor + len(clip)) / 16000, spk))
        cursor += len(clip)
    return np.concatenate(pieces), truth


def diarize(samples, seg_model, emb_model, num_clusters, threshold, shift=0.1):
    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model), window_shift_ratio=shift),
            num_threads=4),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=4),
        clustering=so.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = so.OfflineSpeakerDiarization(config)
    start = time.time()
    result = sd.process(samples)
    segments = [(s.start, s.end, s.speaker) for s in result.sort_by_start_time()]
    return result.num_speakers, segments, time.time() - start


def accuracy(segments, truth, speakers=SPEAKERS) -> float:
    """Fração do tempo de fala atribuída à pessoa certa (com o melhor casamento voz ↔ pessoa)."""
    votes = []
    for start, end, spk in truth:
        cover: dict[int, float] = {}
        for s, e, label in segments:
            overlap = min(e, end) - max(s, start)
            if overlap > 0:
                cover[label] = cover.get(label, 0.0) + overlap
        votes.append((spk, max(cover, key=cover.get) if cover else -1, end - start))
    labels = sorted({v[1] for v in votes})
    best = 0.0
    for perm in itertools.permutations(labels, min(len(labels), len(speakers))):
        mapping = dict(zip(perm, speakers, strict=False))
        hit = sum(d for spk, lab, d in votes if mapping.get(lab) == spk)
        best = max(best, hit)
    return best / sum(v[2] for v in votes)


def tts_dialogue() -> tuple[np.ndarray, list[tuple[float, float, str]]] | None:
    """Diálogo em português (Maria e Daniel, vozes do Windows) com gabarito, se existir."""
    import json

    wav = ROOT / "dados" / "teste" / "dialogo.wav"
    if not wav.exists():
        return None
    truth = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    return media.read_wav(wav), [(t["start"], t["end"], t["speaker"]) for t in truth]


def test_matrix(configs: list[str], thresholds: list[float]) -> None:
    """Cada config "segmentação:impressão:passo" → acerto com nº de pessoas conhecido e automático."""
    print("\n=== Matriz: acerto de quem fala (gabarito) e tempo ===", flush=True)
    conversation = build_conversation()
    tests = {"3 pessoas reais": (conversation, 3)}
    dialogue = tts_dialogue()
    if dialogue:
        tests["diálogo pt-BR (TTS)"] = (dialogue, 2)
    for name, ((samples, _truth), people) in tests.items():
        print(f"- {name}: {len(samples) / 16000:.0f} s, {people} pessoas", flush=True)
    for config in configs:
        seg_name, emb_name, shift = config.split(":")
        for name, ((samples, truth), people) in tests.items():
            speakers = sorted({t[2] for t in truth})
            row = []
            for nc, th in [(people, 0.5)] + [(-1, th) for th in thresholds]:
                found, segs, secs = diarize(samples, SEGMENTERS[seg_name], EMBEDDINGS[emb_name], nc, th,
                                            float(shift))
                tag = f"n={nc}" if nc > 0 else f"t={th}"
                row.append(f"{tag}: {found}v {accuracy(segs, truth, speakers) * 100:3.0f}% ({secs:.0f}s)")
            print(f"{seg_name}+{emb_name} passo {shift} | {name:20s} | " + " | ".join(row), flush=True)


def _combos(segmenters: list[str], embeddings: list[str]):
    return [(s, SEGMENTERS[s], e, EMBEDDINGS[e]) for s in segmenters for e in embeddings]


def test_conversation(segmenters, embeddings, thresholds) -> None:
    print("\n=== Teste 2: conversa montada com 3 pessoas reais (gabarito) ===", flush=True)
    samples, truth = build_conversation()
    print(f"duração {len(samples) / 16000:.0f} s, {len(truth)} falas", flush=True)
    for sname, smodel, ename, emodel in _combos(segmenters, embeddings):
        row = []
        for nc, th in [(-1, th) for th in thresholds] + [(3, 0.5)]:
            found, segs, secs = diarize(samples, smodel, emodel, nc, th)
            tag = f"n={nc}" if nc > 0 else f"t={th}"
            row.append(f"{tag}: {found}v {accuracy(segs, truth) * 100:4.0f}%")
        print(f"{sname:13s} + {ename:16s} | " + " | ".join(row) + f" | {secs:.0f}s/rodada", flush=True)


def test_real_conversations(segmenters, embeddings, thresholds) -> None:
    print("\n=== Teste 3: conversas reais (vozes encontradas sozinho) ===", flush=True)
    files = {"1-two-speakers-en.wav": 2, "2-two-speakers-en.wav": 2, "3-two-speakers-en.wav": 2,
             "0-four-speakers-zh.wav": 4}
    audio = {name: load16k(REAL / name) for name in files}
    for sname, smodel, ename, emodel in _combos(segmenters, embeddings):
        row = []
        for th in thresholds:
            started = time.time()
            counts = [diarize(audio[name], smodel, emodel, -1, th, 0.25)[0] for name in files]
            ok = sum(c == e for c, e in zip(counts, files.values(), strict=True))
            row.append(f"t={th}: {counts} ({ok}/4, {time.time() - started:.0f}s)")
        print(f"{sname:13s} + {ename:16s} | " + " | ".join(row), flush=True)
    print("esperado:", list(files.values()), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--testes", default="1,2,3")
    parser.add_argument("--segmentacao", default=",".join(SEGMENTERS))
    parser.add_argument("--impressao", default=",".join(EMBEDDINGS))
    parser.add_argument("--limiares", default="0.5,0.7,0.9")
    parser.add_argument("--matriz", default="", help='ex.: "pyannote-3.0:campplus-zh_en:0.25,..."')
    parser.add_argument("--prioridade-normal", action="store_true",
                        help="não reduzir a prioridade (se outro programa estiver ocupando a CPU)")
    args = parser.parse_args()
    if args.matriz:
        if not args.prioridade_normal:
            winsys.lower_priority()
        test_matrix(args.matriz.split(","), [float(x) for x in args.limiares.split(",")])
        print(f"\npico de memória: {winsys.process_memory()[1] / 2**20:.0f} MB", flush=True)
        raise SystemExit(0)
    chosen_tests = set(args.testes.split(","))
    segs = args.segmentacao.split(",")
    embs = args.impressao.split(",")
    limits = [float(x) for x in args.limiares.split(",")]
    winsys.lower_priority()
    if "1" in chosen_tests:
        test_embeddings()
    if "2" in chosen_tests:
        test_conversation(segs, embs, limits)
    if "3" in chosen_tests:
        test_real_conversations(segs, embs, limits)
    print(f"\npico de memória: {winsys.process_memory()[1] / 2**20:.0f} MB", flush=True)
