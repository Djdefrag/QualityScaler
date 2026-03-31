<#
.SYNOPSIS
    Regenerate MinGW import library for qualityscaler_tensorrt.dll
.DESCRIPTION
    This script regenerates:
      1. qualityscaler_tensorrt.def  (export list)
      2. internal/core/libqualityscaler_tensorrt.dll.a  (MinGW import library)
    Run it after you drop a new qualityscaler_tensorrt.dll into the project root.
.EXAMPLE
    PS> .\scripts\update-tensorrt-import-lib.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "=== Regenerating .def / .dll.a for qualityscaler_tensorrt.dll ===" -ForegroundColor Cyan

# 1. generate .def
Write-Host "1/2  gendef qualityscaler_tensorrt.dll"
gendef qualityscaler_tensorrt.dll
if (-not (Test-Path qualityscaler_tensorrt.def)) {
    Write-Error ".def generation failed"
    exit 1
}

# 2. generate import library
Write-Host "2/2  dlltool -> internal/core/libqualityscaler_tensorrt.dll.a"
New-Item -ItemType Directory -Force internal/core | Out-Null
dlltool -d qualityscaler_tensorrt.def -l internal/core/libqualityscaler_tensorrt.dll.a -k
if (-not (Test-Path internal/core/libqualityscaler_tensorrt.dll.a)) {
    Write-Error "import library generation failed"
    exit 1
}

Write-Host "=== Done ===" -ForegroundColor Green