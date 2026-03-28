@echo off
REM ==================================================
REM TensorRT Engine Validation Script
REM ==================================================

setlocal enabledelayedexpansion

echo ========================================
echo TensorRT Engine Validation Tool
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PowerShell not found
    echo This script requires PowerShell to run
    pause
    exit /b 1
)

REM Run PowerShell validation script
powershell -ExecutionPolicy Bypass -File "%~dp0validate_tensorrt_engines.ps1" %*

if errorlevel 1 (
    echo.
    echo [WARNING] Validation failed with errors
    pause
    exit /b 1
) else (
    echo.
    echo [SUCCESS] All engines validated successfully
    pause
    exit /b 0
)
