@echo off
REM Share the chatbot online from this PC for free: password-protected, through a Cloudflare Tunnel.
REM Everything (models, images, files) still runs on this PC, so keep it on while others use the site.
title Gurasees's Chatbot - public link
cd /d "%~dp0"

where cloudflared >nul 2>nul
if errorlevel 1 (
  echo cloudflared is not installed. Install it once with:
  echo     winget install Cloudflare.cloudflared
  echo Then close this window, open a new one and run this file again.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo Run start.bat once first to set up the app.
  pause
  exit /b 1
)

set /p APP_PASSWORD=Choose a password for your visitors:
if "%APP_PASSWORD%"=="" (
  echo A password is required before the site goes online.
  pause
  exit /b 1
)

REM Make sure Ollama is running.
set "OLLAMA=ollama"
where ollama >nul 2>nul || set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
"%OLLAMA%" list >nul 2>nul
if errorlevel 1 (
  start "" /min "%OLLAMA%" serve
  timeout /t 8 /nobreak >nul
)

echo.
echo Starting the chat server and the tunnel. Look for the https://....trycloudflare.com address below:
echo share that link and the password. Close this window to take the site offline.
echo.
start "Cloudflare Tunnel" cloudflared tunnel --url http://127.0.0.1:8001
call .venv\Scripts\activate.bat
python server.py
pause
