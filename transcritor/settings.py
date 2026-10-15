"""Configurações (config.toml, editável pelo usuário) e estado (lembranças entre execuções)."""

from __future__ import annotations

import json
import logging
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .paths import CONFIG_FILE, STATE_FILE

log = logging.getLogger(__name__)


@dataclass
class Settings:
    # [texto]
    mostrar_horarios: bool = False
    uma_frase_por_linha: bool = False
    # [vozes]
    separar_vozes: bool = True
    perguntar_numero_de_pessoas: bool = True
    sensibilidade_separacao: float = 0.5
    reconhecer_vozes: bool = True
    semelhanca_minima: float = 0.55
    memorizar_outras_vozes: bool = False
    # [audio]
    salvar_audio_extraido: bool = True
    # [desempenho]
    threads: int = 4
    prioridade_baixa: bool = True
    memoria_recomendada_gb: float = 1.2
    # [manutencao]
    manter_cache_dias: int = 30


_SECTIONS = ("texto", "vozes", "audio", "desempenho", "manutencao")


def load_settings(path: Path = CONFIG_FILE) -> tuple[Settings, list[str]]:
    """Lê o config.toml. Devolve as configurações e uma lista de avisos (valores ignorados)."""
    settings = Settings()
    warnings: list[str] = []
    if not path.exists():
        return settings, warnings
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        return settings, [f"config.toml com erro ({exc}); usando os valores padrão."]
    known = {f.name: f for f in fields(Settings)}
    for section, values in data.items():
        if section not in _SECTIONS or not isinstance(values, dict):
            warnings.append(f"Seção desconhecida no config.toml: [{section}]")
            continue
        for key, value in values.items():
            spec = known.get(key)
            if spec is None:
                warnings.append(f"Opção desconhecida no config.toml: {key}")
                continue
            expected = type(getattr(settings, key))
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if type(value) is not expected:
                warnings.append(
                    f"Valor inválido para {key} no config.toml: {value!r}; usando o padrão."
                )
                continue
            setattr(settings, key, value)
    settings.threads = max(1, min(settings.threads, 16))
    settings.sensibilidade_separacao = min(max(settings.sensibilidade_separacao, 0.05), 0.95)
    settings.semelhanca_minima = min(max(settings.semelhanca_minima, 0.1), 0.99)
    return settings, warnings


@dataclass
class State:
    """Lembranças entre execuções (pastas usadas por último, seu nome...)."""

    last_input_dir: str = ""
    last_output_dir: str = ""
    user_name: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> State:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path = STATE_FILE) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # lembrança é conveniência: não interrompe o trabalho
            log.warning("Não foi possível salvar o estado: %s", exc)
