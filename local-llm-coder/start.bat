@echo off
REM One-click launcher for Gurasees Ollama on Windows. Double-click this file.
title Gurasees Ollama
cd /d "%~dp0"

REM Find Ollama: on PATH, or in its default install folder.
set "OLLAMA=ollama"
where ollama >nul 2>nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
  ) else (
    echo Ollama was not found. Install it from https://ollama.com/download, then run this again.
    pause
    exit /b 1
  )
)

if not exist ".venv" (
  echo Creating Python environment, one time only...
  python -m venv .venv
  if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from https://python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

"%OLLAMA%" list | findstr /i "qwen2.5-coder:3b" >nul
if errorlevel 1 (
  echo Downloading the coding model, one time only, about 2 GB...
  "%OLLAMA%" pull qwen2.5-coder:3b
)

"%OLLAMA%" list | findstr /i "gurasees-coder" >nul
if errorlevel 1 (
  echo Building the gurasees-coder model...
  "%OLLAMA%" create gurasees-coder -f Modelfile.gurasees
)

start "" http://localhost:8000
python server.py
pause
