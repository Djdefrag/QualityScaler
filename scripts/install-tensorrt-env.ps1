# TensorRT 10.16.0.16 永久环境变量配置脚本
# 需要管理员权限

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TensorRT 10.16.0.72 永久环境变量配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[错误] 需要管理员权限" -ForegroundColor Red
    Write-Host "请以管理员身份运行此脚本:" -ForegroundColor Yellow
    Write-Host "  右键点击脚本 -> '以管理员身份运行'" -ForegroundColor Gray
    Read-Host "按回车键退出"
    exit 1
}

$TensorRTPath = "C:\TensorRT-10.16.0.72"

# 检查 TensorRT 是否存在
if (-not (Test-Path $TensorRTPath)) {
    Write-Host "[错误] 未找到 TensorRT: $TensorRTPath" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

Write-Host "[1/3] 备份当前 PATH..." -ForegroundColor Yellow
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
Write-Host "  [OK] 已备份" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] 添加 TensorRT 到系统 PATH..." -ForegroundColor Yellow
$pathsToAdd = @(
    "$TensorRTPath\bin",
    "$TensorRTPath\lib"
)

foreach ($path in $pathsToAdd) {
    if ($machinePath -notlike "*$path*") {
        Write-Host "  添加: $path" -ForegroundColor Gray
        $machinePath += ";$path"
    } else {
        Write-Host "  已存在: $path" -ForegroundColor DarkGray
    }
}

[Environment]::SetEnvironmentVariable("Path", $machinePath, "Machine")
Write-Host "  [OK] PATH 已更新" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] 设置 TensorRT_DIR 环境变量..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("TensorRT_DIR", $TensorRTPath, "Machine")
Write-Host "  [OK] TensorRT_DIR = $TensorRTPath" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "配置完成！" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "已永久配置以下环境变量:" -ForegroundColor Green
Write-Host "  PATH += $TensorRTPath\bin" -ForegroundColor Gray
Write-Host "  PATH += $TensorRTPath\lib" -ForegroundColor Gray
Write-Host "  TensorRT_DIR = $TensorRTPath" -ForegroundColor Gray
Write-Host ""
Write-Host "⚠ 重要提示:" -ForegroundColor Yellow
Write-Host "  1. 需要重启 PowerShell 或命令提示符" -ForegroundColor Yellow
Write-Host "  2. 或重新启动计算机以使更改生效" -ForegroundColor Yellow
Write-Host ""
Write-Host "验证方法（重启后）:" -ForegroundColor Cyan
Write-Host "  trtexec --version" -ForegroundColor Gray
Write-Host "  echo \$env:TensorRT_DIR" -ForegroundColor Gray
Write-Host ""

Read-Host "按回车键退出"
