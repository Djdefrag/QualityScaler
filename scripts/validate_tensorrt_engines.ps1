# TensorRT Engine Validation Script
# Validates all TensorRT .engine files in AI-tensorrt directory

param(
    [string]$TensorRTDir = "AI-tensorrt",
    [switch]$Verbose,
    [switch]$Detailed
)

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Validate-EngineFile {
    param([string]$FilePath)

    $fileName = Split-Path $FilePath -Leaf

    Write-Host "`nValidating: $fileName" -ForegroundColor Cyan

    # Check file exists
    if (-not (Test-Path $FilePath)) {
        Write-ColorOutput "  [ERROR] File not found!" -Color Red
        return $false
    }

    # Get file info
    $fileInfo = Get-Item $FilePath
    $fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)

    Write-ColorOutput "  Size: $fileSizeMB MB" -Color Gray

    # Check minimum size (TensorRT engines are typically > 1MB)
    if ($fileInfo.Length -lt 1MB) {
        Write-ColorOutput "  [ERROR] File too small (expected > 1MB)" -Color Red
        return $false
    }

    # Check file extension
    if ($FilePath -notmatch '\.engine$') {
        Write-ColorOutput "  [WARNING] Unexpected file extension" -Color Yellow
    }

    # Check file header (basic validation)
    try {
        $fileStream = [System.IO.File]::OpenRead($FilePath)
        $header = [byte[]]::new(12)
        $null = $fileStream.Read($header, 0, 12)
        $fileStream.Close()

        if ($Verbose) {
            $headerHex = ($header | ForEach-Object { "{0:X2}" -f $_ }) -join " "
            Write-ColorOutput "  Header (hex): $headerHex" -Color DarkGray
        }

        # Validate header size
        if ($header.Length -lt 12) {
            Write-ColorOutput "  [ERROR] Invalid header size" -Color Red
            return $false
        }
    }
    catch {
        Write-ColorOutput "  [ERROR] Failed to read file header: $_" -Color Red
        return $false
    }

    # Check if file is readable
    try {
        $null = [System.IO.File]::ReadAllBytes($FilePath)
        Write-ColorOutput "  [OK] File is readable" -Color Green
    }
    catch {
        Write-ColorOutput "  [ERROR] File is not readable: $_" -Color Red
        return $false
    }

    # Extract model name from filename
    if ($Detailed) {
        $modelName = $fileName -replace '_fp16\.engine$', '' -replace '_fp32\.engine$', '' -replace '\.engine$', ''

        Write-ColorOutput "  Model name: $modelName" -Color Gray

        # Check if corresponding ONNX model exists
        $onnxPath = Join-Path (Split-Path (Split-Path $FilePath -Parent) -Parent) "AI-onnx\$($modelName)_fp16.onnx"
        if (Test-Path $onnxPath) {
            Write-ColorOutput "  [INFO] Corresponding ONNX model exists" -Color Gray
        }
        else {
            Write-ColorOutput "  [WARNING] Corresponding ONNX model not found" -Color Yellow
        }
    }

    Write-ColorOutput "  [SUCCESS] Validation passed" -Color Green
    return $true
}

function Main {
    Write-Host "`n" + "="*80 -ForegroundColor Cyan
    Write-Host "TensorRT Engine Validation Tool" -ForegroundColor Cyan
    Write-Host "="*80 -ForegroundColor Cyan

    # Check if directory exists
    if (-not (Test-Path $TensorRTDir)) {
        Write-ColorOutput "`n[ERROR] Directory not found: $TensorRTDir" -Color Red
        Write-ColorOutput "  Run convert-to-tensorrt.bat to generate engines" -Color Yellow
        exit 1
    }

    # Get all engine files
    $engineFiles = Get-ChildItem -Path $TensorRTDir -Filter "*.engine" -File

    if ($engineFiles.Count -eq 0) {
        Write-ColorOutput "`n[WARNING] No .engine files found in $TensorRTDir" -Color Yellow
        exit 0
    }

    Write-ColorOutput "`nFound $($engineFiles.Count) engine file(s)" -Color Cyan

    # Validate each engine
    $validCount = 0
    $invalidCount = 0
    $results = @()

    foreach ($engineFile in $engineFiles) {
        $result = Validate-EngineFile -FilePath $engineFile.FullName
        if ($result) {
            $validCount++
        }
        else {
            $invalidCount++
        }
        $results += [PSCustomObject]@{
            Name = $engineFile.Name
            SizeMB = [math]::Round($engineFile.Length / 1MB, 2)
            Valid = $result
        }
    }

    # Summary
    Write-Host "`n" + "="*80 -ForegroundColor Cyan
    Write-Host "Validation Summary" -ForegroundColor Cyan
    Write-Host "="*80 -ForegroundColor Cyan

    Write-ColorOutput "`nTotal engines: $($engineFiles.Count)" -Color White
    Write-ColorOutput "Valid: $validCount" -Color Green
    Write-ColorOutput "Invalid: $invalidCount" -Color Red

    # Detailed results table
    if ($Detailed) {
        Write-Host "`nDetailed Results:" -ForegroundColor Cyan
        $results | Format-Table -AutoSize
    }

    # Exit code
    if ($invalidCount -gt 0) {
        Write-ColorOutput "`n[WARNING] Some engines failed validation" -Color Yellow
        exit 1
    }
    else {
        Write-ColorOutput "`n[SUCCESS] All engines are valid!" -Color Green
        exit 0
    }
}

# Run main function
Main
