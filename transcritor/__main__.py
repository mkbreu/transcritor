"""Ponto de entrada: python -m transcritor [arquivos...]

Códigos de saída: 0 = terminou normalmente; 3 = erro ou interrupção já mostrados na tela
(o Transcritor.bat não pausa de novo nesses casos).
"""

from __future__ import annotations

import contextlib
import logging
import sys

from . import console as ui


def main() -> int:
    try:
        from .app import run

        return run(sys.argv[1:])
    except KeyboardInterrupt:
        logging.getLogger("transcritor").warning("Interrompido pelo usuário (Ctrl+C)")
        ui.console.print()
        ui.warn("Interrompido. O que já foi transcrito fica guardado: rode de novo para continuar "
                "de onde parou.")
    except Exception as exc:  # último recurso: nunca fechar a janela sem explicar
        logging.getLogger("transcritor").exception("Erro inesperado")
        ui.console.print()
        ui.error("Ocorreu um erro inesperado e o processamento parou.")
        ui.info(f"Detalhe: {type(exc).__name__}: {exc}")
        ui.info('Os detalhes técnicos foram gravados no registro mais recente da pasta "logs".')
    with contextlib.suppress(KeyboardInterrupt):
        ui.wait_enter()
    return 3


if __name__ == "__main__":
    sys.exit(main())
