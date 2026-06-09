param(
    [switch]$Research,
    [switch]$Diploma,
    [switch]$DownloadModel,
    [switch]$All
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Не найден $Name. Установите и добавьте в PATH."
    }
}

function Invoke-VenvPython($Args) {
    if (Test-Path ".venv\Scripts\python.exe") {
        & .venv\Scripts\python.exe @Args
    } else {
        & python @Args
    }
}

function Install-PyTorch {
    $Pip = ".venv\Scripts\pip.exe"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "  PyTorch + CUDA (cu124) ..."
        & $Pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    } else {
        Write-Host "  PyTorch CPU (nvidia-smi не найден) ..."
        & $Pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    }
}

Write-Host "=== 8-bit Character Generator — установка ===" -ForegroundColor Cyan
Write-Host "Корень проекта: $Root"

Require-Command python
Require-Command node
Require-Command npm

if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/4] Создание виртуального окружения .venv ..."
    python -m venv .venv
} else {
    Write-Host "`n[1/4] Виртуальное окружение .venv уже есть"
}

Write-Host "[2/4] Установка PyTorch и Python-зависимостей ..."
Install-PyTorch
& .venv\Scripts\pip.exe install -r backend\requirements.txt

Write-Host "[3/4] Сборка фронтенда ..."
Push-Location frontend
npm install
npm run build
Pop-Location

if ($Research -or $All) {
    Write-Host "`n[research] Установка зависимостей исследований ..."
    & .venv\Scripts\pip.exe install -r research\requirements.txt
    Write-Host "[research] Разбивка датасета train/val (70/30) ..."
    Invoke-VenvPython @("research\split_dataset.py")
}

if ($Diploma -or $All) {
    Write-Host "`n[diploma] Установка зависимостей для генерации ВКР ..."
    & .venv\Scripts\pip.exe install -r scripts\requirements.txt
}

if ($DownloadModel -or $All) {
    Write-Host "`n[model] Загрузка Stable Diffusion v1.5 с HuggingFace (~4 ГБ) ..."
    Invoke-VenvPython @(
        "-c",
        "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', safety_checker=None); print('OK')"
    )
}

Write-Host "`n=== Готово ===" -ForegroundColor Green
Write-Host "Запуск:  scripts\run.ps1"
Write-Host "Режимы:  scripts\setup.ps1 -Research | -DownloadModel | -All"
