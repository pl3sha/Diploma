$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\uvicorn.exe")) {
    Write-Host "Сначала выполните: scripts\setup.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "frontend\dist\index.html")) {
    Write-Host "Сначала соберите фронтенд: scripts\setup.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Запуск http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "Первый старт может скачать SD v1.5 с HuggingFace (~4 ГБ)" -ForegroundColor DarkGray
& .venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000
