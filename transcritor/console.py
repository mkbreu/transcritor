"""Tudo o que aparece na janela do console: mensagens, perguntas, barras de progresso e resumo.

Usa só símbolos que a fonte padrão do console do Windows 10 mostra (sem emojis).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

console = Console(
    theme=Theme({
        "ok": "bold green",
        "aviso": "bold yellow",
        "erro": "bold red",
        "destaque": "bold cyan",
        "fraco": "grey62",
    }),
    highlight=False,
)


def banner(version: str) -> None:
    console.print(Panel.fit(
        "[destaque]TRANSCRITOR[/]  [fraco]v" + version + "[/]\n"
        "Vídeo, áudio ou legenda SRT  →  texto organizado por pessoa, pronto para a IA",
        border_style="cyan",
    ))


def title(text: str) -> None:
    console.print()
    console.rule(f"[destaque]{escape(text)}[/]", style="cyan")


# As funções abaixo recebem texto comum (nomes de arquivo podem ter colchetes): tudo é
# "escapado" para não ser confundido com as marcações de cor do rich.
def info(text: str) -> None:
    console.print(f"  {escape(text)}")


def ok(text: str) -> None:
    console.print(f"  [ok]\\[OK][/] {escape(text)}")


def warn(text: str) -> None:
    console.print(f"  [aviso]\\[!][/] {escape(text)}")


def error(text: str) -> None:
    console.print(f"  [erro]\\[X][/] {escape(text)}")


_interactive = True


def set_interactive(enabled: bool) -> None:
    """Desligado (modo --sem-perguntas): toda pergunta usa a resposta padrão."""
    global _interactive
    _interactive = enabled


def is_interactive() -> bool:
    return _interactive


def ask(prompt: str) -> str:
    """Pergunta livre. Devolve o texto digitado (sem espaços nas pontas)."""
    if not _interactive:
        console.print(f"  [destaque]>[/] {escape(prompt)} [fraco](automático: padrão)[/]")
        return ""
    try:
        return console.input(f"  [destaque]>[/] {escape(prompt)} ").strip()
    except EOFError:
        return ""


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "[S/n]" if default else "[s/N]"
    while True:
        answer = ask(f"{prompt} {hint}").lower()
        if not answer:
            return default
        if answer in ("s", "sim", "y"):
            return True
        if answer in ("n", "nao", "não", "no"):
            return False
        warn("Responda S (sim) ou N (não).")


def ask_int(prompt: str, low: int, high: int) -> int | None:
    """Número entre `low` e `high`; Enter devolve None."""
    while True:
        answer = ask(prompt)
        if not answer:
            return None
        if answer.isdigit() and low <= int(answer) <= high:
            return int(answer)
        warn(f"Digite um número de {low} a {high}, ou só Enter.")


def wait_enter(prompt: str = "Pressione Enter para fechar.") -> None:
    ask(prompt)


class StageProgress:
    """Barra de progresso de uma etapa (atualizada por callbacks em segundos de áudio)."""

    def __init__(self, progress: Progress, description: str) -> None:
        self._progress = progress
        self._task = progress.add_task(escape(description), total=None)
        self._last = 0.0

    def describe(self, description: str) -> None:
        self._progress.update(self._task, description=escape(description), completed=0, total=None)

    def update(self, done: float, total: float) -> None:
        now = time.monotonic()
        if now - self._last < 0.2 and done < total:
            return
        self._last = now
        self._progress.update(self._task, completed=done, total=total or None)

    def finish(self) -> None:
        task = self._progress.tasks[self._task]
        self._progress.update(self._task, completed=task.total or 1, total=task.total or 1)


@contextlib.contextmanager
def stage(description: str) -> Iterator[StageProgress]:
    with Progress(
        SpinnerColumn("line"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("[fraco]· falta[/]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        tracker = StageProgress(progress, description)
        yield tracker
        tracker.finish()
