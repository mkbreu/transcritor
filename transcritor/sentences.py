"""Divisão do texto em frases completas (português do Brasil) e em parágrafos.

Funciona sobre uma lista de palavras com tempo (vindas da transcrição ou de uma legenda),
para que o mesmo código sirva aos dois modos.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .subtitles import Cue
from .transcript import Sentence, Word

# Títulos e tratamentos: nunca terminam frase ("o Sr. João").
_TITLES = {
    "sr", "sra", "srta", "srs", "sras", "dr", "dra", "drs", "dras", "prof", "profa", "profs",
    "exmo", "exma", "ilmo", "ilma", "eng", "adv", "des", "min", "gov", "pres", "dep", "sen",
    "gen", "ten", "sgt", "pe", "fr", "sto", "sta",
}
# Abreviações que só terminam a frase se a próxima palavra começar com maiúscula.
_ABBREVIATIONS = {
    "av", "r", "al", "rod", "n", "nº", "n°", "no", "núm", "num", "art", "arts", "inc", "par",
    "pág", "pag", "págs", "p", "pp", "fl", "fls", "vol", "ed", "ex", "obs", "tel", "cel", "cap",
    "ltda", "cia", "etc", "aprox", "máx", "mín", "jan", "fev", "abr", "jun", "jul", "ago",
    "set", "out", "nov", "dez", "seg", "qua", "qui", "sex", "sáb", "dom", "vs", "km", "kg",
}
_TERMINATORS = (".", "!", "?", "…")
_CLOSING = "\"'”’»)]"
_OPENING = "\"'“‘«(["

PAUSE_FORCES_SENTENCE = 3.0  # segundos de silêncio que encerram a frase mesmo sem ponto
MAX_SENTENCE_WORDS = 80


@dataclass(slots=True)
class Paragraph:
    speaker: str | None
    overlap: bool
    sentences: list[Sentence]

    @property
    def start(self) -> float:
        return self.sentences[0].start

    @property
    def end(self) -> float:
        return self.sentences[-1].end

    @property
    def text(self) -> str:
        return " ".join(sentence.text for sentence in self.sentences)


def words_from_cues(cues: Iterable[Cue]) -> list[Word]:
    """Quebra cada legenda em palavras, distribuindo o tempo pelo tamanho de cada palavra."""
    words: list[Word] = []
    for cue in cues:
        tokens = cue.text.split()
        if not tokens:
            continue
        total_chars = sum(len(token) for token in tokens)
        duration = max(cue.end - cue.start, 0.0)
        cursor = cue.start
        for token in tokens:
            length = duration * len(token) / total_chars
            words.append(Word(token, cursor, cursor + length, speaker=cue.speaker))
            cursor += length
    return words


def split_into_sentences(words: list[Word], split_on_speaker_change: bool = True) -> list[Sentence]:
    """Agrupa palavras em frases completas, respeitando abreviações (e mudanças de voz)."""
    return [
        make_sentence(group, group[0].speaker, group[0].overlap)
        for group in split_word_groups(words, split_on_speaker_change)
    ]


def split_word_groups(words: list[Word], split_on_speaker_change: bool = True) -> list[list[Word]]:
    """Como `split_into_sentences`, mas devolve as palavras de cada frase."""
    groups: list[list[Word]] = []
    current: list[Word] = []
    for index, word in enumerate(words):
        current.append(word)
        following = words[index + 1] if index + 1 < len(words) else None
        if following is None or _breaks_after(word, following, split_on_speaker_change):
            groups.extend(_limit_length(current))
            current = []
    return groups


def make_sentence(words: list[Word], speaker: str | None, overlap: bool = False) -> Sentence:
    text = " ".join(word.text for word in words).strip()
    return Sentence(text=text, start=words[0].start, end=words[-1].end,
                    speaker=speaker, overlap=overlap)


def _breaks_after(word: Word, following: Word, split_on_speaker_change: bool) -> bool:
    if split_on_speaker_change and (word.speaker, word.overlap) != (
        following.speaker, following.overlap
    ):
        return True
    if following.start - word.end >= PAUSE_FORCES_SENTENCE:
        return True
    return ends_sentence(word.text, following.text)


def ends_sentence(text: str, next_text: str | None) -> bool:
    core = text.rstrip(_CLOSING)
    if not core.endswith(_TERMINATORS):
        return False
    next_upper = next_text is None or _starts_upper(next_text)
    if core.endswith(("..", "…")):  # reticências: só encerra se a próxima começar maiúscula
        return next_upper
    if core.endswith("."):
        stem = core[:-1].lstrip(_OPENING).lower()
        if stem in _TITLES:
            return False
        if stem in _ABBREVIATIONS:
            return next_text is not None and next_upper and not next_text[:1].isdigit()
        if len(stem) == 1 and stem.isalpha():  # iniciais: "J. Silva"
            return False
    return True


def _starts_upper(text: str) -> bool:
    stripped = text.lstrip(_OPENING)
    return bool(stripped) and (stripped[0].isupper() or stripped[0].isdigit())


def _limit_length(words: list[Word]) -> list[list[Word]]:
    """Frases longas demais (sem pontuação) são quebradas na maior pausa."""
    if len(words) <= MAX_SENTENCE_WORDS:
        return [words]
    cut = _best_cut(words)
    return _limit_length(words[:cut]) + _limit_length(words[cut:])


def _best_cut(words: list[Word]) -> int:
    low = MAX_SENTENCE_WORDS // 3
    high = len(words) - MAX_SENTENCE_WORDS // 3

    def score(i: int) -> float:
        gap = words[i].start - words[i - 1].end
        punctuation = 0.3 if words[i - 1].text.endswith((",", ";", ":")) else 0.0
        return gap + punctuation

    return max(range(low, high + 1), key=score)


def group_paragraphs(
    sentences: list[Sentence],
    names: dict[str, str] | None = None,
    pause_break: float = 2.5,
    soft_limit_words: int = 90,
    hard_limit_words: int = 170,
) -> list[Paragraph]:
    """Junta frases seguidas da mesma pessoa em parágrafos (quebrando os muito longos).

    `names` mapeia rótulo → nome: vozes com o mesmo nome ficam no mesmo parágrafo.
    """
    names = names or {}
    paragraphs: list[Paragraph] = []
    for sentence in sentences:
        key = _speaker_key(sentence, names)
        last = paragraphs[-1] if paragraphs else None
        if last is not None and _speaker_key(last.sentences[-1], names) == key:
            words_so_far = sum(len(s.text.split()) for s in last.sentences)
            gap = sentence.start - last.sentences[-1].end
            too_long = words_so_far >= hard_limit_words or (
                words_so_far >= soft_limit_words and gap >= 0.6
            )
            long_pause = gap >= pause_break and words_so_far >= 25
            if not (too_long or long_pause):
                last.sentences.append(sentence)
                continue
        paragraphs.append(Paragraph(sentence.speaker, sentence.overlap, [sentence]))
    return paragraphs


def _speaker_key(sentence: Sentence, names: dict[str, str]) -> tuple[str | None, bool]:
    if sentence.speaker is None:
        return None, sentence.overlap
    return names.get(sentence.speaker, sentence.speaker), False
