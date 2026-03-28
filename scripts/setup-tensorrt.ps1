# TensorRT 环境配置脚本
# 用于检查和配置 TensorRT 环境

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TensorRT 环境配置工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 配置路径（用户实际安装版本）
$TensorRTPath = "C:\TensorRT-10.16.0.72"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$AssetsDir = Join-Path $ProjectDir "Assets"
$TRTDir = Join-Path $ProjectDir "AI-tensorrt"

Write-Host "项目目录: $ProjectDir" -ForegroundColor Gray
Write-Host "Assets目录: $AssetsDir" -ForegroundColor Gray
Write-Host "TensorRT路径: $TensorRTPath" -ForegroundColor Gray
Write-Host ""

# 检查 TensorRT 安装
Write-Host "[1/5] 检查 TensorRT 安装..." -ForegroundColor Yellow
if (-not (Test-Path $TensorRTPath)) {
    Write-Host "  [错误] 未找到 TensorRT" -ForegroundColor Red
    Write-Host "  请下载并安装 TensorRT: https://developer.nvidia.com/tensorrt" -ForegroundColor Gray
    Write-Host "  解压到: $TensorRTPath" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  提示: 也可以修改脚本中的 `$TensorRTPath` 变量指向你的 TensorRT 安装路径" -ForegroundColor Cyan
    Read-Host "按回车键退出"
    exit 1
}
Write-Host "  [OK] TensorRT 已安装" -ForegroundColor Green
Write-Host ""

# 检查 trtexec
Write-Host "[2/5] 检查 trtexec 工具..." -ForegroundColor Yellow
$trtexecPath = Join-Path $TensorRTPath "bin\trtexec.exe"
if (-not (Test-Path $trtexecPath)) {
    Write-Host "  [错误] 未找到 trtexec.exe" -ForegroundColor Red
    Write-Host "  路径: $trtexecPath" -ForegroundColor Gray
    Read-Host "按回车键退出"
    exit 1
}
Write-Host "  [OK] trtexec.exe 已找到" -ForegroundColor Green
Write-Host ""

# 检查并复制 TensorRT DLL
Write-Host "[3/5] 检查并复制 TensorRT DLL..." -ForegroundColor Yellow

$requiredDLLs = @(
    "nvinfer.dll",
    "nvonnxparser.dll"
)

$dllFound = $true
foreach ($dll in $requiredDLLs) {
    $sourcePath = Join-Path $TensorRTPath "bin\$dll"
    $destPath = Join-Path $AssetsDir $dll

    if (Test-Path $sourcePath) {
        if (-not (Test-Path $destPath)) {
            Write-Host "  复制 $dll..." -ForegroundColor Gray
            Copy-Item $sourcePath $destPath -Force
            Write-Host "    [OK] 已复制" -ForegroundColor Green
        } else {
            Write-Host "  $dll 已存在" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  [警告] 未找到 $dll" -ForegroundColor Yellow
        $dllFound = $false
    }
}

if (-not $dllFound) {
    Write-Host ""
    Write-Host "  部分DLL缺失，但不影响编译（运行时需要）" -ForegroundColor Yellow
}
Write-Host ""

# 创建 AI-tensorrt 目录
Write-Host "[4/5] 检查 TensorRT 模型目录..." -ForegroundColor Yellow
if (-not (Test-Path $TRTDir)) {
    New-Item -ItemType Directory -Path $TRTDir | Out-Null
    Write-Host "  [OK] 已创建目录: $TRTDir" -ForegroundColor Green
} else {
    Write-Host "  [OK] 目录已存在" -ForegroundColor Green
}
Write-Host ""

# 检查现有引擎文件
Write-Host "[5/5] 检查现有 TensorRT 引擎..." -ForegroundColor Yellow
$engineFiles = Get-ChildItem -Path $TRTDir -Filter "*.engine" -ErrorAction SilentlyContinue

if ($engineFiles.Count -eq 0) {
    Write-Host "  未找到 TensorRT 引擎文件" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  要启用 TensorRT，需要转换 ONNX 模型:" -ForegroundColor Cyan
    Write-Host "  运行: scripts\convert-to-tensorrt.bat" -ForegroundColor Gray
} else {
    Write-Host "  找到 $($engineFiles.Count) 个引擎文件:" -ForegroundColor Green
    foreach ($file in $engineFiles) {
        $sizeMB = [math]::Round($file.Length / 1MB, 2)
        Write-Host "    - $($file.Name) ($sizeMB MB)" -ForegroundColor Gray
    }
}
Write-Host ""

# 状态总结
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "配置状态总结" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "✓ TensorRT 已安装" -ForegroundColor Green
Write-Host "✓ trtexec 工具可用" -ForegroundColor Green
Write-Host "✓ Assets 目录就绪" -ForegroundColor Green

if ($engineFiles.Count -gt 0) {
    Write-Host "✓ TensorRT 引擎已就绪" -ForegroundColor Green
    Write-Host ""
    Write-Host "状态: 可以使用 TensorRT 推理！" -ForegroundColor Green
} else {
    Write-Host "○ TensorRT 引擎未生成" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "下一步: 运行 scripts\convert-to-tensorrt.bat 转换模型" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "当前后端: ONNX Runtime (TensorRT 未完全集成)" -ForegroundColor Gray
Write-Host "性能预期: TensorRT 可提升 2-5 倍速度" -ForegroundColor Gray
Write-Host ""

Read-Host "按回车键退出"
