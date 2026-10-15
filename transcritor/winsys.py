"""Recursos do Windows: impedir suspensão, prioridade baixa, memória e abrir arquivos/pastas."""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import subprocess
from collections.abc import Iterator
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_IS_WINDOWS = os.name == "nt"
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
GB = 1024**3


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    total: int
    available: int  # memória física livre agora
    commit_available: int  # quanto ainda pode ser reservado (RAM + arquivo de paginação)


def memory_status() -> MemoryStatus | None:
    if not _IS_WINDOWS:
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return MemoryStatus(status.ullTotalPhys, status.ullAvailPhys, status.ullAvailPageFile)


def process_memory() -> tuple[int, int]:
    """(memória em uso agora, pico de memória) deste processo, em bytes."""
    if not _IS_WINDOWS:
        return 0, 0
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(_current_process(), ctypes.byref(counters), counters.cb)
    if not ok:
        return 0, 0
    return counters.WorkingSetSize, counters.PeakWorkingSetSize


def _current_process() -> int:
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    return kernel32.GetCurrentProcess()


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    """Impede o Windows de suspender enquanto um processamento longo estiver rodando."""
    if not _IS_WINDOWS:
        yield
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def lower_priority() -> None:
    """Roda com prioridade "abaixo do normal": o PC continua utilizável durante o trabalho."""
    if not _IS_WINDOWS:
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    if not kernel32.SetPriorityClass(_current_process(), _BELOW_NORMAL_PRIORITY_CLASS):
        log.warning("Não foi possível reduzir a prioridade do processo.")


def focus_console() -> None:
    """Traz a janela do console de volta para a frente (depois de uma janela de seleção)."""
    if not _IS_WINDOWS:
        return
    with contextlib.suppress(Exception):
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)


def open_file(path: Path) -> None:
    os.startfile(str(path))  # type: ignore[attr-defined]  (só existe no Windows)


def show_in_folder(path: Path) -> None:
    """Abre o Explorer na pasta do arquivo, já com ele selecionado."""
    subprocess.Popen(["explorer", "/select,", str(path)])
