@echo off
echo.
echo === cloudflared tunnel to http://127.0.0.1 port 8000 ===
echo Before: npm run build in frontend, then uvicorn on port 8000.
echo.

if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" set "PATH=%ProgramFiles(x86)%\cloudflared;%PATH%"
if exist "%ProgramFiles%\cloudflared\cloudflared.exe" set "PATH=%ProgramFiles%\cloudflared;%PATH%"

where cloudflared >nul 2>&1
if errorlevel 1 (
  echo cloudflared not found in PATH.
  echo Install: winget install Cloudflare.cloudflared
  echo Or: https://github.com/cloudflare/cloudflared/releases
  exit /b 1
)

cloudflared tunnel --url http://127.0.0.1:8000
