"""Dar nome às vozes: exemplos de frases com o tempo, ouvir o trecho e reconhecer vozes já conhecidas.

Os perfis de voz (só "impressões vocais", nunca o áudio) ficam em dados\\vozes.json.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import console as ui
from .media import play_segment
from .paths import VOICES_FILE
from .render import format_clock
from .settings import Settings, State
from .transcript import Sentence, Transcript, default_name

log = logging.getLogger(__name__)

MAX_PRINTS_PER_PROFILE = 12


@dataclass
class VoiceProfile:
    name: str
    is_me: bool = False
    prints: list[list[float]] = field(default_factory=list)
    updated: str = ""

    def centroid(self) -> np.ndarray:
        mean = np.mean(np.asarray(self.prints, dtype=np.float32), axis=0)
        return mean / (np.linalg.norm(mean) or 1.0)


class VoiceBook:
    """Perfis de voz conhecidos (o seu e, se configurado, o de outras pessoas)."""

    def __init__(self, path: Path = VOICES_FILE) -> None:
        self.path = path
        self.profiles: list[VoiceProfile] = []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.profiles = [VoiceProfile(**item) for item in data.get("perfis", []) if item.get("prints")]
        except (OSError, ValueError, TypeError):
            pass

    @property
    def me(self) -> VoiceProfile | None:
        return next((p for p in self.profiles if p.is_me), None)

    def find(self, name: str) -> VoiceProfile | None:
        key = name.casefold()
        return next((p for p in self.profiles if p.name.casefold() == key), None)

    def best_match(self, vector: list[float]) -> tuple[VoiceProfile, float] | None:
        if not self.profiles or not vector:
            return None
        target = np.asarray(vector, dtype=np.float32)
        scored = [(p, float(p.centroid() @ target)) for p in self.profiles]
        return max(scored, key=lambda item: item[1])

    def remember(self, name: str, vector: list[float], is_me: bool) -> None:
        profile = self.find(name)
        if profile is None:
            profile = VoiceProfile(name=name)
            self.profiles.append(profile)
        if is_me:
            for other in self.profiles:
                other.is_me = other is profile
        profile.prints = [*profile.prints, vector][-MAX_PRINTS_PER_PROFILE:]
        profile.updated = datetime.now().isoformat(timespec="seconds")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"perfis": [asdict(p) for p in self.profiles]}
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


# ------------------------------------------------------------------ exemplos de frases


def _candidates(transcript: Transcript, label: str) -> list[Sentence]:
    own = [s for s in transcript.sentences if s.speaker == label and not s.overlap]
    return [s for s in own if len(s.text.split()) >= 5] or own


def more_samples(
    transcript: Transcript, label: str, exclude: set[int], count: int = 3
) -> list[Sentence]:
    """Mais exemplos: as frases mais longas da voz que ainda não foram mostradas."""
    rest = [s for s in _candidates(transcript, label) if id(s) not in exclude]
    return sorted(sorted(rest, key=lambda s: -len(s.text))[:count], key=lambda s: s.start)


def pick_samples(transcript: Transcript, label: str, count: int = 3) -> list[Sentence]:
    """A primeira frase longa da voz e outras espalhadas pela gravação (as mais longas de cada parte)."""
    good = _candidates(transcript, label)
    if not good:
        return []
    first = next((s for s in good if len(s.text.split()) >= 6), good[0])
    chosen = [first]
    remaining = [s for s in good if s.start > first.end]
    buckets = count - 1
    if remaining and buckets:
        span_start, span_end = remaining[0].start, remaining[-1].end
        width = (span_end - span_start) / buckets or 1.0
        for b in range(buckets):
            low, high = span_start + b * width, span_start + (b + 1) * width
            inside = [s for s in remaining if low <= s.start < high or (b == buckets - 1 and s.start >= low)]
            if inside:
                chosen.append(max(inside, key=lambda s: len(s.text)))
    return sorted({id(s): s for s in chosen}.values(), key=lambda s: s.start)


def speaking_share(transcript: Transcript) -> dict[str, tuple[float, int]]:
    """Por voz: (fração do tempo de fala, número de frases)."""
    totals: dict[str, list[float]] = {}
    for sentence in transcript.sentences:
        if sentence.speaker is None:
            continue
        item = totals.setdefault(sentence.speaker, [0.0, 0])
        item[0] += max(sentence.end - sentence.start, 0.0)
        item[1] += 1
    grand = sum(value[0] for value in totals.values()) or 1.0
    return {label: (value[0] / grand, int(value[1])) for label, value in totals.items()}


# ------------------------------------------------------------------ nomeação interativa


def name_speakers(
    transcript: Transcript,
    wav_path: Path | None,
    prints: dict[str, list[float]],
    book: VoiceBook,
    settings: Settings,
    state: State,
) -> None:
    """Pergunta o nome de cada voz (com exemplos e trechos para ouvir) e atualiza os perfis."""
    labels = transcript.labels
    if not labels:
        return
    suggestions = _recognize(labels, prints, book, settings) if settings.reconhecer_vozes else {}
    my_name = book.me.name if book.me else state.user_name
    shares = speaking_share(transcript)
    plural = "vozes" if len(labels) > 1 else "voz"
    ui.title(f"Quem é quem — {transcript.source_name}")
    ui.info(f"Encontrei {len(labels)} {plural}. Para cada uma, digite o nome da pessoa.")
    if wav_path:
        ui.info("Digite o número de um exemplo para OUVIR o trecho; M mostra mais exemplos.")
    ui.info("Enter mantém o nome sugerido. O mesmo nome em duas vozes junta as duas.")
    ui.info('Para marcar a SUA voz, escreva (eu) depois do nome — ex.: Maria (eu) — ou só: eu')
    mine: set[str] = set()
    for label in labels:
        share, count = shares.get(label, (0.0, 0))
        current = transcript.speaker_names.get(label)
        suggestion = suggestions.get(label)
        default = current or (suggestion[0] if suggestion else default_name(label))
        default_is_mine = label in transcript.me_labels or _same_name(default, my_name)
        ui.console.print()
        header = f"VOZ {label}  —  {share * 100:.0f}% da fala ({count} frases)"
        if suggestion and not current:
            header += f"  —  parece ser {suggestion[0]} (semelhança {suggestion[1] * 100:.0f}%)"
        ui.console.print(f"  [destaque]{ui.escape(header)}[/]")
        name, is_mine = _ask_name(transcript, label, wav_path, default, default_is_mine, my_name)
        transcript.speaker_names[label] = name
        if is_mine:
            mine.add(label)
            my_name = my_name or name  # quem digitar só "eu" depois recebe este nome
    _update_profiles(transcript, prints, book, settings, state, mine, my_name)


_ME_MARK = re.compile(r"\(\s*eu\s*\)", re.IGNORECASE)
_ME_ALONE = {"eu", "eu mesmo", "eu mesma", "sou eu", "minha voz"}


def parse_name(answer: str, my_name: str) -> tuple[str, bool]:
    """Nome digitado → (nome, é a voz do usuário?). "Maria (eu)" e "eu" marcam a própria voz."""
    text = " ".join(answer.split())
    if _ME_MARK.search(text):
        name = " ".join(_ME_MARK.sub(" ", text).split())
        return name or my_name or "Eu", True
    if text.casefold() in _ME_ALONE:
        return my_name or "Eu", True
    return text, _same_name(text, my_name)


def _same_name(first: str, second: str) -> bool:
    return bool(first and second) and first.casefold() == second.casefold()


def _ask_name(
    transcript: Transcript, label: str, wav_path: Path | None, default: str, default_is_mine: bool,
    my_name: str,
) -> tuple[str, bool]:
    samples = pick_samples(transcript, label)
    shown = {id(s) for s in samples}
    _print_samples(samples)
    while True:
        answer = ui.ask(f'Nome para a VOZ {label} (Enter = "{default}"):')
        if not answer:
            return default, default_is_mine
        if answer.isdigit() and wav_path and 1 <= int(answer) <= len(samples):
            sample = samples[int(answer) - 1]
            ui.info(f"Tocando o trecho de {format_clock(sample.start)}...")
            play_segment(wav_path, sample.start, sample.end)
            continue
        if answer.casefold() == "m":
            more = more_samples(transcript, label, shown)
            if not more:
                ui.warn("Não há mais exemplos desta voz.")
                continue
            samples = more
            shown.update(id(s) for s in more)
            _print_samples(samples)
            continue
        if answer.isdigit():
            ui.warn("Não há trecho com esse número para ouvir.")
            continue
        return parse_name(answer, my_name)


def _print_samples(samples: list[Sentence]) -> None:
    for index, sample in enumerate(samples, start=1):
        text = sample.text if len(sample.text) <= 120 else sample.text[:117].rstrip() + "..."
        ui.console.print(f"    {index}) [fraco]\\[{format_clock(sample.start)}][/] “{ui.escape(text)}”")


def _recognize(
    labels: list[str], prints: dict[str, list[float]], book: VoiceBook, settings: Settings
) -> dict[str, tuple[str, float]]:
    """Sugere nomes para vozes parecidas com perfis salvos (cada perfil vai para uma voz só)."""
    candidates = []
    for label in labels:
        match = book.best_match(prints.get(label, []))
        if match and match[1] >= settings.semelhanca_minima:
            candidates.append((match[1], label, match[0].name))
    used_names: set[str] = set()
    suggestions: dict[str, tuple[str, float]] = {}
    for similarity, label, name in sorted(candidates, reverse=True):
        if name not in used_names:
            suggestions[label] = (name, similarity)
            used_names.add(name)
    if suggestions:
        log.info("Vozes reconhecidas: %s", {k: f"{v[0]} ({v[1]:.2f})" for k, v in suggestions.items()})
    return suggestions


def _update_profiles(
    transcript: Transcript, prints: dict[str, list[float]], book: VoiceBook, settings: Settings,
    state: State, mine: set[str], my_name: str,
) -> None:
    labels = transcript.labels
    names = transcript.speaker_names
    had_profile = book.me is not None
    me_labels = [lb for lb in labels if lb in mine or _same_name(names.get(lb, ""), my_name)]
    asked = False
    if not me_labels and prints and settings.reconhecer_vozes:
        me_labels = _ask_which_are_mine(labels, had_profile)
        asked = True
    if me_labels:
        # Todas as vozes do usuário ficam com o mesmo nome (assim viram uma pessoa só no texto).
        typed = [names[lb] for lb in me_labels if names.get(lb) not in (default_name(lb), "Eu")]
        chosen = typed[0] if typed else my_name
        if not chosen and asked:
            chosen = ui.ask('Como você quer aparecer no texto? (Enter = "Eu"):')
        for label in me_labels:
            names[label] = chosen or "Eu"
    transcript.me_labels = me_labels
    if me_labels:
        state.user_name = names[me_labels[0]]
    if not prints or not settings.reconhecer_vozes:
        return
    changed = False
    my_prints = [prints[lb] for lb in me_labels if prints.get(lb)]
    if my_prints:
        mean = np.mean(np.asarray(my_prints, dtype=np.float32), axis=0)
        mean /= np.linalg.norm(mean) or 1.0
        book.remember(names[me_labels[0]], mean.round(5).tolist(), is_me=True)
        changed = True
        if not had_profile:
            ui.ok(f'Sua voz foi memorizada como "{names[me_labels[0]]}": nas próximas gravações '
                  "ela já vem com o seu nome.")
    if settings.memorizar_outras_vozes:
        for label in labels:
            name = names.get(label, "")
            if label not in me_labels and prints.get(label) and name and name != default_name(label):
                book.remember(name, prints[label], is_me=False)
                changed = True
    if changed:
        book.save()


def _ask_which_are_mine(labels: list[str], had_profile: bool) -> list[str]:
    """Pergunta quais vozes são do usuário (pode ser mais de uma: "A C"). Enter = nenhuma."""
    question = ("Sua voz está nesta gravação? Digite a(s) letra(s) da(s) sua(s) voz(es), ex.: A ou A C "
                "(Enter = não está):" if had_profile else
                "Alguma destas vozes é a SUA? Digite a(s) letra(s), ex.: A ou A C (Enter = nenhuma):")
    options = {label.casefold(): label for label in labels}
    while True:
        answer = ui.ask(question).casefold()
        if not answer:
            return []
        answer = re.sub(r"(?<=\S)\s+e\s+(?=\S)", " ", answer)  # "A e C" → "A C"
        tokens = [t for t in re.split(r"[\s,;/]+", answer) if t]
        if len(tokens) == 1 and tokens[0] not in options and all(ch in options for ch in tokens[0]):
            tokens = list(tokens[0])  # "ac" → A e C
        if tokens and all(token in options for token in tokens):
            return list(dict.fromkeys(options[token] for token in tokens))
        ui.warn(f"Digite letras entre: {', '.join(labels)} (ex.: A ou A C).")
