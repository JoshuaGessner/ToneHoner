# Build ToneHoner server and client single-file executables with PyInstaller
# Usage: run in PowerShell from repo root after activating your venv
#   .\build_executables.ps1

$ErrorActionPreference = 'Stop'

# Ensure PyInstaller is available
Write-Host "Checking PyInstaller..."
$pyi = (python -m pip show pyinstaller) 2>$null
if (-not $pyi) {
  Write-Host "Installing PyInstaller..."
  python -m pip install pyinstaller | Out-Null
}

# Clean previous builds
Write-Host "Cleaning build/ and dist/ ..."
if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist)  { Remove-Item dist  -Recurse -Force }

# Build Server (GUI, no console)
Write-Host "Building ToneHoner-Server (onefile)..." -ForegroundColor Cyan
python -m PyInstaller -y .\tonehoner_server.spec

# Build Client (console)
Write-Host "Building ToneHoner-Client (onefile)..." -ForegroundColor Cyan
python -m PyInstaller -y .\tonehoner_client.spec

Write-Host "Done. Executables in .\\dist\\" -ForegroundColor Green
Get-ChildItem .\dist\
