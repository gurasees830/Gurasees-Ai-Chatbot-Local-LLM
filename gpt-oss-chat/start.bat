@echo off
REM One-click launcher for Gurasees's Deep Learning Local LLM Chatbot on Windows. Double-click this file.
title Gurasees's Deep Learning Local LLM Chatbot
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
REM Keep the app's Python packages up to date on every start.
pip install -q -U -r requirements.txt

REM Start the Ollama server if it isn't running yet.
"%OLLAMA%" list >nul 2>nul
if errorlevel 1 (
  echo Starting Ollama...
  start "" /min "%OLLAMA%" serve
  timeout /t 8 /nobreak >nul
)

"%OLLAMA%" list | findstr /i "gpt-oss:20b" >nul
if errorlevel 1 (
  echo Downloading gpt-oss:20b, one time only, about 13 GB...
  "%OLLAMA%" pull gpt-oss:20b
)
REM Vision model: lets every chat model see photos, screenshots and video frames.
"%OLLAMA%" list | findstr /i "gemma3:4b" >nul
if errorlevel 1 (
  echo Downloading gemma3:4b for reading images, one time only, about 3 GB...
  "%OLLAMA%" pull gemma3:4b
)
REM Image generation lives in its own Python environment.
if not exist ".venv-image" (
  echo Setting up image generation, one time only...
  call setup_image.bat quiet
)

start "" http://localhost:8001
python server.py
pause
