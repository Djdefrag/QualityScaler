# OpenCV Installation for Windows (gocv)
# This script helps install OpenCV for gocv usage

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "OpenCV Installation Guide for gocv on Windows" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Check if OpenCV is already installed
$opencvPaths = @(
    "C:\opencv",
    "C:\opencv\build",
    "C:\Program Files\opencv",
    "$env:LOCALAPPDATA\opencv"
)

$foundOpenCV = $false
foreach ($path in $opencvPaths) {
    if (Test-Path "$path\include\opencv2\opencv.hpp") {
        Write-Host "[OK] OpenCV found at: $path" -ForegroundColor Green
        $foundOpenCV = $true
        break
    }
}

if (-not $foundOpenCV) {
    Write-Host "[!] OpenCV C++ library not found on system" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To use gocv, you need to install OpenCV C++ library:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Download and install manually"
    Write-Host "  1. Visit: https://opencv.org/releases/"
    Write-Host "  2. Download 'Windows' version (e.g., opencv-4.9.0-windows.exe)"
    Write-Host "  3. Extract to C:\opencv (or another location)"
    Write-Host "  4. Set environment variables:"
    Write-Host "     PKG_CONFIG_PATH=C:\opencv\build\x64\vc16\lib\pkgconfig"
    Write-Host "     PATH=%PATH%;C:\opencv\build\x64\vc16\bin"
    Write-Host ""
    Write-Host "Option 2: Use package manager (chocolatey)"
    Write-Host "  choco install opencv"
    Write-Host ""
    Write-Host "Option 3: Use vcpkg"
    Write-Host "  vcpkg install opencv:x64-windows"
    Write-Host ""
} else {
    Write-Host "[OK] OpenCV is available for gocv" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now build the project with gocv support." -ForegroundColor Green
    Write-Host ""
    Write-Host "Build command:" -ForegroundColor Cyan
    Write-Host "  go build -tags custom -ldflags `-H windowsgui -s -w` -o qualityscaler-go.exe ./cmd/qualityscaler-go/" -ForegroundColor White
}

Write-Host ""
Write-Host "Note: If OpenCV is not installed, the application will automatically" -ForegroundColor Yellow
Write-Host "      fall back to pure Go image processing (slower but functional)." -ForegroundColor Yellow
Write-Host ""
