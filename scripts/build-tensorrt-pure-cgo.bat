@echo off
REM Build QualityScaler Go with TensorRT support (Pure CGO)
REM This script compiles the Go application with pure CGO TensorRT bindings
REM No Python dependency required

echo.
echo ========================================
echo QualityScaler Go + TensorRT (Pure CGO)
echo ========================================
echo.

REM Check if TensorRT engines directory exists
if not exist "AI-tensorrt" (
    echo [ERROR] TensorRT engines directory not found: AI-tensorrt
    echo.
    echo Please ensure TensorRT engines are built first.
    echo See: TENSORRT_PURE_CGO_README.md
    pause
    exit /b 1
)

REM Check for .engine files
set HAS_ENGINES=0
for %%f in (AI-tensorrt\*.engine) do set HAS_ENGINES=1
if "%HAS_ENGINES%"=="0" (
    echo [ERROR] No TensorRT engines found in AI-tensorrt directory
    echo.
    echo Please build TensorRT engines first.
    pause
    exit /b 1
)

echo [Build] TensorRT engines found, enabling CGO build...
echo.

REM Set CGO flags for TensorRT
set TENSORRT_PATH=C:\TensorRT-10.16.0.72

REM Check for CUDA symlink (recommended) or use short path
if exist "C:\CUDA\include" (
    set CUDA_PATH=C:\CUDA
    echo [Build] Using CUDA symlink: C:\CUDA
) else if exist "C:\PROGRA~1\NVIDIA~2\CUDA\v13.1\include" (
    set CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.1
    echo [Build] Using CUDA short path: C:\PROGRA~1\NVIDIA~2\CUDA\v13.1
) else (
    echo.
    echo [ERROR] CUDA not accessible via symlink or short path!
    echo.
    echo Please create a symlink (run as Administrator):
    echo   mklink /D "C:\CUDA" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    echo.
    echo Or run setup-cuda-link.bat as Administrator.
    echo.
    pause
    exit /b 1
)

echo [Build] Using CUDA: %CUDA_PATH%
echo [Build] Using TensorRT: %TENSORRT_PATH%
echo.

REM Set CGO compiler flags (paths without spaces)
set CGO_CXXFLAGS=-std=c++17 -O3 -I%TENSORRT_PATH%\include -I%CUDA_PATH%\include
set CGO_LDFLAGS=-L%TENSORRT_PATH%\lib -L%CUDA_PATH%\lib\x64 -lnvinfer_10 -lcudart

REM Add CUDA and TensorRT to PATH for runtime
set PATH=%PATH%;%TENSORRT_PATH%\bin;%TENSORRT_PATH%\lib;%CUDA_PATH%\bin

echo [Build] Building Go application with CGO...
echo.

echo Environment variables:
echo   CGO_ENABLED=%CGO_ENABLED%
echo   CGO_CXXFLAGS=%CGO_CXXFLAGS%
echo   CGO_LDFLAGS=%CGO_LDFLAGS%
echo.

REM Build the Go application
echo Building...
go build -v -tags=no_gocv -ldflags="-s -w" -o qualityscaler-trt-go.exe ./cmd/qualityscaler-go

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    echo.
    echo Common issues:
    echo 1. TensorRT not installed or not in PATH
    echo 2. CGO environment not configured correctly
    echo 3. Missing TensorRT libraries
    echo.
    echo Check TENSORRT_PURE_CGO_README.md for setup instructions
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Build completed!
echo ========================================
echo.
echo Output: qualityscaler-trt-go.exe
echo.
echo To run the application:
echo   qualityscaler-trt-go.exe
echo.
echo The application will automatically:
echo 1. Try to use TensorRT engines
echo 2. Fall back to ONNX Runtime if TensorRT is not available
echo.
pause
