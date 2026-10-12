@echo off
REM One-time setup for image generation in Gurasees's Deep Learning Local LLM Chatbot.
REM PyTorch needs Python 3.12 (the chat server's Python is too new for it), so images get their own environment.
title Image generation setup
cd /d "%~dp0"

set "UV="
where uv >nul 2>nul && set "UV=uv"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"

if defined UV (
  "%UV%" venv .venv-image --python 3.12
  if errorlevel 1 goto fail
  echo Installing PyTorch with CUDA, about 3 GB, one time only...
  "%UV%" pip install --python .venv-image\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
  if errorlevel 1 goto fail
  "%UV%" pip install --python .venv-image\Scripts\python.exe -U diffusers transformers accelerate safetensors fastapi uvicorn pillow
  if errorlevel 1 goto fail
) else (
  py -3.12 -m venv .venv-image
  if errorlevel 1 (
    echo Python 3.12 or uv is needed. Install uv: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    goto fail
  )
  echo Installing PyTorch with CUDA, about 3 GB, one time only...
  .venv-image\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  if errorlevel 1 goto fail
  .venv-image\Scripts\python.exe -m pip install -U diffusers transformers accelerate safetensors fastapi uvicorn pillow
  if errorlevel 1 goto fail
)
echo.
echo Image generation is installed. The image model (~2.5 GB) downloads the first time you create an image.
if not "%1"=="quiet" pause
exit /b 0

:fail
echo Image generation setup failed. See the messages above.
if exist .venv-image rmdir /s /q .venv-image
if not "%1"=="quiet" pause
exit /b 1
