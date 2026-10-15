"""Modelo de dados da transcrição (palavras, frases e o resultado completo)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

UNKNOWN_NAME = "[Não identificado]"
OVERLAP_NAME = "[Vozes sobrepostas]"


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None  # rótulo interno da voz ("A", "B"...); None = desconhecido
    overlap: bool = False  # True = mais de uma voz ao mesmo tempo


@dataclass(slots=True)
class Sentence:
    text: str
    start: float
    end: float
    speaker: str | None = None  # None = falante não identificado
    overlap: bool = False


@dataclass
class Transcript:
    source_name: str
    source_kind: str  # "audio", "video" ou "legenda"
    duration: float
    sentences: list[Sentence]
    engine: str
    has_speakers: bool
    speaker_names: dict[str, str] = field(default_factory=dict)  # rótulo → nome escolhido
    me_labels: list[str] = field(default_factory=list)  # vozes que são do próprio usuário
    created_at: str = ""
    version: int = 1  # versão do método que gerou o resultado (ver pipeline.RESULT_VERSION)

    @property
    def me_name(self) -> str | None:
        """Nome com que o usuário aparece no texto (se alguma voz foi marcada como dele)."""
        return self.display_name(self.me_labels[0]) if self.me_labels else None

    @property
    def labels(self) -> list[str]:
        """Rótulos das vozes na ordem em que aparecem pela primeira vez."""
        seen: dict[str, None] = {}
        for sentence in self.sentences:
            if sentence.speaker is not None:
                seen.setdefault(sentence.speaker)
        return list(seen)

    def display_name(self, label: str | None, overlap: bool = False) -> str:
        if label is None:
            return OVERLAP_NAME if overlap else UNKNOWN_NAME
        return self.speaker_names.get(label) or default_name(label)

    def word_count(self) -> int:
        return sum(len(sentence.text.split()) for sentence in self.sentences)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Transcript:
        data = json.loads(text)
        data["sentences"] = [Sentence(**item) for item in data["sentences"]]
        old_me = data.pop("me_label", None)  # formato antigo: uma voz só
        if old_me and not data.get("me_labels"):
            data["me_labels"] = [old_me]
        return cls(**data)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Transcript:
        return cls.from_json(path.read_text(encoding="utf-8"))


def default_name(label: str) -> str:
    """Nome padrão de uma voz: "Pessoa A" para rótulos da separação; senão o próprio rótulo."""
    is_letter_label = label.isascii() and label.isalpha() and label.isupper() and len(label) <= 2
    return f"Pessoa {label}" if is_letter_label else label


def speaker_label(index: int) -> str:
    """0 → "A", 1 → "B", ..., 25 → "Z", 26 → "AA"."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters
