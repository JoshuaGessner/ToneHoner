# ToneHoner - Build All Script
# Builds both client and server executables and installers with icons

param(
    [switch]$SkipExecutables,
    [switch]$SkipInstallers,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ToneHoner Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-Command {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Function to find Inno Setup Compiler
function Find-ISCC {
    # Check if iscc is in PATH
    $isccPath = Get-Command iscc -ErrorAction SilentlyContinue
    if ($isccPath) {
        return $isccPath.Source
    }
    
    # Check common installation paths
    $commonPaths = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 5\ISCC.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    return $null
}

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-Command "pyinstaller")) {
    Write-Host "[X] PyInstaller not found. Installing..." -ForegroundColor Red
    pip install pyinstaller
}

$global:IsccPath = Find-ISCC
if (-not $global:IsccPath -and -not $SkipInstallers) {
    Write-Host "[X] Inno Setup Compiler (iscc) not found." -ForegroundColor Red
    Write-Host "  Install Inno Setup from: https://jrsoftware.org/isdl.php" -ForegroundColor Yellow
    Write-Host "  Or use Chocolatey: choco install innosetup" -ForegroundColor Yellow
    $SkipInstallers = $true
} elseif ($global:IsccPath) {
    Write-Host "[OK] Found Inno Setup at: $global:IsccPath" -ForegroundColor Green
}

Write-Host "[OK] Prerequisites checked" -ForegroundColor Green
Write-Host ""

# Check if icons exist
if (-not (Test-Path "client_icon.ico") -or -not (Test-Path "server_icon.ico")) {
    Write-Host "Generating icons..." -ForegroundColor Yellow
    python generate_icons.py
    Write-Host "[OK] Icons generated" -ForegroundColor Green
    Write-Host ""
}

# Clean build directories if requested
if ($Clean) {
    Write-Host "Cleaning build directories..." -ForegroundColor Yellow
    Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "dist" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[OK] Build directories cleaned" -ForegroundColor Green
    Write-Host ""
}

# Create dist/installers directory
New-Item -ItemType Directory -Path "dist\installers" -Force | Out-Null

# Build executables
if (-not $SkipExecutables) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Building Executables" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Building ToneHoner Client..." -ForegroundColor Yellow
    pyinstaller tonehoner_client.spec
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Client executable built successfully" -ForegroundColor Green
    } else {
        Write-Host "[X] Client build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    
    Write-Host "Building ToneHoner Server..." -ForegroundColor Yellow
    pyinstaller tonehoner_server.spec
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Server executable built successfully" -ForegroundColor Green
    } else {
        Write-Host "[X] Server build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# Build installers
if (-not $SkipInstallers) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Building Installers" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Building ToneHoner Client Installer..." -ForegroundColor Yellow
    & $global:IsccPath tonehoner_client_setup.iss
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Client installer built successfully" -ForegroundColor Green
    } else {
        Write-Host "[X] Client installer build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    
    Write-Host "Building ToneHoner Server Installer..." -ForegroundColor Yellow
    & $global:IsccPath tonehoner_server_setup.iss
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Server installer built successfully" -ForegroundColor Green
    } else {
        Write-Host "[X] Server installer build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipExecutables) {
    Write-Host "Executables:" -ForegroundColor Yellow
    if (Test-Path "dist\ToneHoner-Client\ToneHoner-Client.exe") {
        $clientDir = Get-ChildItem "dist\ToneHoner-Client" -Recurse | Measure-Object -Property Length -Sum
        $clientSize = $clientDir.Sum / 1MB
        $clientSizeRounded = [math]::Round($clientSize, 2)
        Write-Host "  [OK] dist\ToneHoner-Client\ ($clientSizeRounded MB)" -ForegroundColor Green
    }
    if (Test-Path "dist\ToneHoner-Server\ToneHoner-Server.exe") {
        $serverDir = Get-ChildItem "dist\ToneHoner-Server" -Recurse | Measure-Object -Property Length -Sum
        $serverSize = $serverDir.Sum / 1MB
        $serverSizeRounded = [math]::Round($serverSize, 2)
        Write-Host "  [OK] dist\ToneHoner-Server\ ($serverSizeRounded MB)" -ForegroundColor Green
    }
    Write-Host ""
}

if (-not $SkipInstallers) {
    Write-Host "Installers:" -ForegroundColor Yellow
    Get-ChildItem "dist\installers\*.exe" | ForEach-Object {
        $size = $_.Length / 1MB
        $sizeRounded = [math]::Round($size, 2)
        Write-Host "  [OK] $($_.FullName) ($sizeRounded MB)" -ForegroundColor Green
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[OK] Build completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Optional: Open dist folder
$openDist = Read-Host "Open dist folder? (y/n)"
if ($openDist -eq "y" -or $openDist -eq "Y") {
    $distPath = (Get-Item "dist").FullName
    Start-Process "explorer.exe" -ArgumentList $distPath
}
