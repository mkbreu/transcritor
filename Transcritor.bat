@echo off
rem ======================================================================
rem  Transcritor - de dois cliques para escolher os arquivos,
rem  ou arraste videos, audios ou legendas SRT para cima deste icone.
rem ======================================================================
setlocal
chcp 65001 >nul
title Transcritor
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem Python do sistema: usado so para criar/verificar o ambiente isolado .venv
set "PY_SISTEMA="
py -3.12 --version >nul 2>&1 && set "PY_SISTEMA=py -3.12"
if not defined PY_SISTEMA python --version >nul 2>&1 && set "PY_SISTEMA=python"
if not defined PY_SISTEMA (
    echo [X] Não encontrei o Python 3.12 neste computador.
    echo     Instale pelo site https://www.python.org/downloads/ e abra o Transcritor de novo.
    echo.
    pause
    exit /b 1
)

%PY_SISTEMA% instalar.py --se-necessario
if errorlevel 1 goto falhou

".venv\Scripts\python.exe" -m transcritor %*
if errorlevel 3 exit /b 3
if errorlevel 1 goto falhou
exit /b 0

:falhou
echo.
echo [X] O Transcritor encontrou um problema. Leia a mensagem acima.
echo     Os detalhes técnicos ficam no registro mais recente da pasta "logs".
echo.
pause
exit /b 1