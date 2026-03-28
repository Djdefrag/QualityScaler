# =============================================================================
# Prepare-Icon.ps1  --  Icon preparation tool
#
# Features:
#   1. (Optional) Convert JPG/PNG source to multi-size ICO (256/48/32/16 px)
#      using PowerShell built-in System.Drawing, no external dependencies
#   2. Use rsrc tool to package ICO into Windows .syso resource file
#      go build will automatically link *.syso in the same directory, embedding icon into exe
#
# Usage:
#   .\scripts\Prepare-Icon.ps1                          # Default parameters, generate app.syso
#   .\scripts\Prepare-Icon.ps1 -SkipConvert             # Skip JPG->ICO (use existing ICO)
#   .\scripts\Prepare-Icon.ps1 -Force                   # Force regenerate (overwrite existing files)
#   .\scripts\Prepare-Icon.ps1 -OutputSyso "path\to\app.syso"  # Specify syso output path
#
# Prerequisites: Go environment required (rsrc installed automatically via go install)
# =============================================================================

[CmdletBinding()]
param(
    # Icon source file (JPG / PNG)
    [string]$SourceImage  = "",

    # Output ICO file path
    [string]$OutputIco    = "",

    # Output .syso file path (default to cmd/qualityscaler-fyne/ directory)
    [string]$OutputSyso   = "",

    # Skip JPG->ICO conversion step (directly use existing ICO file)
    [switch]$SkipConvert,

    # Force regenerate even if target file exists
    [switch]$Force
)

# Set UTF-8 output to avoid encoding issues
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8
$ErrorActionPreference    = "Stop"

$Root = (Get-Location).Path

# Fill default paths
if (-not $SourceImage) { $SourceImage = Join-Path $Root "Assets\dist_icon\miao.jpg" }
if (-not $OutputIco)   { $OutputIco   = Join-Path $Root "Assets\miao.ico" }
if (-not $OutputSyso)  { $OutputSyso  = Join-Path $Root "cmd\qualityscaler-fyne\app.syso" }

# Utility functions
function Write-Step([string]$msg) { Write-Host "  >> $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg)   { Write-Host "     [OK] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "     [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "     [FAIL] $msg" -ForegroundColor Red }

# =============================================================================
# Step 1: JPG/PNG -> ICO conversion
# =============================================================================
function Convert-ImageToIco {
    param([string]$SrcPath, [string]$DstPath)

    Write-Step "Converting image to ICO: $SrcPath -> $DstPath"

    Add-Type -AssemblyName System.Drawing

    $srcBmp = [System.Drawing.Bitmap]::new($SrcPath)

    # Target sizes: 256px with PNG compression (Vista+ standard), others with DIB
    $sizes = @(256, 48, 32, 16)

    # Collect image data for each size into memory streams
    $entries = @()
    foreach ($sz in $sizes) {
        $resized = [System.Drawing.Bitmap]::new($sz, $sz)
        $g = [System.Drawing.Graphics]::FromImage($resized)
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g.DrawImage($srcBmp, 0, 0, $sz, $sz)
        $g.Dispose()

        $ms = [System.IO.MemoryStream]::new()

        if ($sz -eq 256) {
            # 256x256 saved as PNG compressed format (Windows Vista+ ICO standard)
            $resized.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
        } else {
            # Other sizes saved as 32bpp DIB (BITMAPINFOHEADER)
            $bmpMs = [System.IO.MemoryStream]::new()
            $resized.Save($bmpMs, [System.Drawing.Imaging.ImageFormat]::Bmp)
            $bmpBytes = $bmpMs.ToArray()
            $bmpMs.Dispose()
            # BMP file header is 14 bytes, ICO needs DIB data without file header
            # Also multiply image height by 2 (ICO DIB includes AND mask, height field must be doubled)
            $dibBytes = $bmpBytes[14..($bmpBytes.Length - 1)]
            # Fix height field in DIB header (offset 8, 4 bytes, little-endian) to double it
            $origHeight = [BitConverter]::ToInt32($dibBytes, 8)
            $newHeight  = $origHeight * 2
            $heightBytes = [BitConverter]::GetBytes([int32]$newHeight)
            $dibBytes[8]  = $heightBytes[0]
            $dibBytes[9]  = $heightBytes[1]
            $dibBytes[10] = $heightBytes[2]
            $dibBytes[11] = $heightBytes[3]
            $ms.Write($dibBytes, 0, $dibBytes.Length)
        }

        $entries += [PSCustomObject]@{
            Size  = $sz
            Data  = $ms.ToArray()
        }
        $ms.Dispose()
        $resized.Dispose()
    }
    $srcBmp.Dispose()

    # Write ICO file
    # ICO format:
    #   6 bytes  ICONDIR (reserved=0, type=1, count=N)
    #   N * 16 bytes  ICONDIRENTRY (width,height,colorCount,reserved,planes,bitCount,bytesInRes,imageOffset)
    #   Image data
    $iconCount = $entries.Count
    $headerSize = 6
    $dirEntrySize = 16
    $dataOffset = $headerSize + $dirEntrySize * $iconCount

    $ms = [System.IO.MemoryStream]::new()
    $bw = [System.IO.BinaryWriter]::new($ms)

    # ICONDIR
    $bw.Write([uint16]0)       # reserved
    $bw.Write([uint16]1)       # type = 1 (icon)
    $bw.Write([uint16]$iconCount)

    # Calculate offsets for each entry
    $offsets = @()
    $cur = $dataOffset
    foreach ($e in $entries) {
        $offsets += $cur
        $cur += $e.Data.Length
    }

    # ICONDIRENTRY
    for ($i = 0; $i -lt $iconCount; $i++) {
        $e  = $entries[$i]
        $sz = $e.Size
        # width / height: 256 stored as 0 (ICO spec)
        $bw.Write([byte]$(if ($sz -eq 256) { 0 } else { $sz }))   # width
        $bw.Write([byte]$(if ($sz -eq 256) { 0 } else { $sz }))   # height
        $bw.Write([byte]0)          # colorCount (0 = 256+ colors)
        $bw.Write([byte]0)          # reserved
        if ($sz -eq 256) {
            $bw.Write([uint16]0)    # planes (PNG entry: 0)
            $bw.Write([uint16]0)    # bitCount (PNG entry: 0)
        } else {
            $bw.Write([uint16]1)    # planes
            $bw.Write([uint16]32)   # bitCount (32bpp)
        }
        $bw.Write([uint32]$e.Data.Length)   # bytesInRes
        $bw.Write([uint32]$offsets[$i])     # imageOffset
    }

    # Image data
    foreach ($e in $entries) {
        $bw.Write($e.Data)
    }

    $bw.Flush()
    $icoBytes = $ms.ToArray()
    $bw.Dispose()
    $ms.Dispose()

    [System.IO.File]::WriteAllBytes($DstPath, $icoBytes)
    Write-OK "ICO generated: $DstPath ($([math]::Round($icoBytes.Length / 1KB, 1)) KB, $iconCount sizes: $($sizes -join ', ') px)"
}

# =============================================================================
# Step 2: Install/Find rsrc tool
# =============================================================================
function Get-RsrcPath {
    # First search from PATH
    $fromPath = Get-Command "rsrc.exe" -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }

    # Then search from GOPATH/bin
    $gopath = (& go env GOPATH 2>$null)
    if ($gopath) {
        $candidate = Join-Path $gopath "bin\rsrc.exe"
        if (Test-Path $candidate) { return $candidate }
    }

    return $null
}

function Install-Rsrc {
    Write-Step "Installing rsrc tool (go install github.com/akavel/rsrc@latest)..."
    & go install github.com/akavel/rsrc@latest
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "rsrc installation failed (exit $LASTEXITCODE)"
        throw "rsrc installation failed"
    }
    Write-OK "rsrc installed successfully"
}

# =============================================================================
# Step 3: ICO -> .syso (rsrc)
# =============================================================================
function New-AppSyso {
    param([string]$IcoPath, [string]$SysoPath)

    Write-Step "Generating app.syso: $IcoPath -> $SysoPath"

    $rsrc = Get-RsrcPath
    if (-not $rsrc) {
        Install-Rsrc
        $rsrc = Get-RsrcPath
        if (-not $rsrc) {
            Write-Fail "rsrc.exe not found after installation, check if GOPATH/bin is in PATH"
            throw "rsrc.exe not found"
        }
    }
    Write-OK "rsrc: $rsrc"

    # Ensure output directory exists
    $sysoDir = Split-Path $SysoPath -Parent
    if (-not (Test-Path $sysoDir)) {
        New-Item -ItemType Directory -Path $sysoDir -Force | Out-Null
    }

    & $rsrc -ico $IcoPath -o $SysoPath
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "rsrc execution failed (exit $LASTEXITCODE)"
        throw "rsrc execution failed"
    }

    $sizeKB = [math]::Round((Get-Item $SysoPath).Length / 1KB, 1)
    Write-OK "app.syso generated: $SysoPath ($sizeKB KB)"
}

# =============================================================================
# Main flow
# =============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QualityScaler Icon Preparation Tool" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SourceImage : $SourceImage"
Write-Host "  OutputIco   : $OutputIco"
Write-Host "  OutputSyso  : $OutputSyso"
Write-Host "  SkipConvert : $($SkipConvert.IsPresent)"
Write-Host "  Force       : $($Force.IsPresent)"
Write-Host ""

# --- Step 1: JPG/PNG -> ICO ---
if ($SkipConvert) {
    Write-Step "Skipping image conversion (-SkipConvert)"
    if (-not (Test-Path $OutputIco)) {
        Write-Fail "ICO file not found: $OutputIco"
        Write-Host "         Please provide an ICO file, or remove -SkipConvert to let script auto-convert" -ForegroundColor Yellow
        exit 1
    }
    Write-OK "Using existing ICO: $OutputIco"
} elseif ((Test-Path $OutputIco) -and -not $Force) {
    Write-Step "ICO file exists, skipping conversion (use -Force to regenerate)"
    Write-OK "Existing ICO: $OutputIco"
} else {
    if (-not (Test-Path $SourceImage)) {
        Write-Fail "Source image not found: $SourceImage"
        exit 1
    }
    Convert-ImageToIco -SrcPath $SourceImage -DstPath $OutputIco
}

# --- Step 2: ICO -> .syso ---
if ((Test-Path $OutputSyso) -and -not $Force) {
    Write-Step "app.syso exists, skipping generation (use -Force to regenerate)"
    Write-OK "Existing syso: $OutputSyso"
} else {
    New-AppSyso -IcoPath $OutputIco -SysoPath $OutputSyso
}

Write-Host ""
Write-Host "  Icon preparation completed!" -ForegroundColor Green
Write-Host "  ICO  : $OutputIco" -ForegroundColor White
Write-Host "  syso : $OutputSyso" -ForegroundColor White
Write-Host ""
