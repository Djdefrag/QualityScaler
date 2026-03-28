# Build QualityScaler Go with TensorRT support (Pure CGO)
Write-Host ""
Write-Host "========================================"
Write-Host "QualityScaler Go + TensorRT (Pure CGO)"
Write-Host "========================================"
Write-Host ""

# Check if TensorRT engines directory exists
if (-not (Test-Path "AI-tensorrt")) {
    Write-Host "[ERROR] TensorRT engines directory not found: AI-tensorrt" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ensure TensorRT engines are built first."
    Write-Host "See: TENSORRT_PURE_CGO_README.md"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for .engine files
$engineFiles = Get-ChildItem "AI-tensorrt\*.engine" -ErrorAction SilentlyContinue
if ($engineFiles.Count -eq 0) {
    Write-Host "[ERROR] No TensorRT engines found in AI-tensorrt directory" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build TensorRT engines first."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[Build] TensorRT engines found, enabling CGO build..."
Write-Host ""

# Build TensorRT DLL first
Write-Host "[Build] Building TensorRT wrapper DLL..."
$dllBuildResult = & "$PSScriptRoot\..\internal\tensorrt_dll\build.bat" 2>&1
$dllExitCode = $LASTEXITCODE

if ($dllExitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] DLL build failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Build output:"
    Write-Host $dllBuildResult
    Write-Host ""
    Write-Host "Please check:"
    Write-Host "1. TensorRT is installed at C:\TensorRT-10.16.0.72"
    Write-Host "2. CUDA is installed at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    Write-Host "3. Visual Studio MSVC is installed"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[Build] DLL build completed successfully"
Write-Host ""

# Check if DLL was created
if (-not (Test-Path "qualityscaler_tensorrt.dll")) {
    Write-Host "[ERROR] DLL not found after build!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[Build] DLL found: qualityscaler_tensorrt.dll"
Write-Host ""

# Clear all CGO environment variables first to prevent inheriting stale flags
$env:CGO_ENABLED = $null
$env:CGO_CFLAGS = $null
$env:CGO_CXXFLAGS = $null
$env:CGO_FFLAGS = $null
$env:CGO_LDFLAGS = $null

# Set CGO compiler flags (DLL mode - link with pre-built DLL)
$env:CGO_ENABLED = "1"
$env:CGO_CXXFLAGS = "-std=c++17 -O3"
$env:CGO_LDFLAGS = "-L. -lqualityscaler_tensorrt"

Write-Host "[Build] CGO flags configured (DLL mode)"

Write-Host "[Build] Building Go application with CGO..."
Write-Host ""

Write-Host "Environment variables:"
Write-Host "  CGO_ENABLED=$($env:CGO_ENABLED)"
Write-Host "  CGO_CXXFLAGS=$($env:CGO_CXXFLAGS)"
Write-Host "  CGO_LDFLAGS=$($env:CGO_LDFLAGS)"
Write-Host ""

# Build the Go application
Write-Host "Building with tensorrt tag..."
go build -v -tags="tensorrt,no_gocv" -ldflags="-s -w" -o qualityscaler-trt-go.exe ./cmd/qualityscaler-go
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:"
    Write-Host "1. TensorRT not installed or not in PATH"
    Write-Host "2. CGO environment not configured correctly"
    Write-Host "3. Missing TensorRT libraries"
    Write-Host ""
    Write-Host "Check TENSORRT_PURE_CGO_README.md for setup instructions"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host "[SUCCESS] Build completed!"
Write-Host "========================================"
Write-Host ""
Write-Host "Output: qualityscaler-trt-go.exe"
Write-Host ""
Write-Host "To run the application:"
Write-Host "  qualityscaler-trt-go.exe"
Write-Host ""
Write-Host "The application will automatically:"
Write-Host "1. Try to use TensorRT engines"
Write-Host "2. Fall back to ONNX Runtime if TensorRT is not available"
Write-Host ""
Read-Host "Press Enter to exit"
