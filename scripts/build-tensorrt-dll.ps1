<#
.SYNOPSIS
    Build qualityscaler_tensorrt.dll from C++ source
.DESCRIPTION
    Uses CMake + MSVC/MinGW to compile the TensorRT wrapper DLL.
    The resulting DLL is placed in the project root, ready for Go CGO.
.EXAMPLE
    PS> .\scripts\build-tensorrt-dll.ps1
#>
param(
    [ValidateSet("Release","Debug")]
    [string]$Config = "Release",

    [ValidateSet(" MSVC","MinGW")]
    [string]$Generator = "MSVC"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dllDir = "$root\internal\tensorrt_dll"
$buildDir = "$dllDir\build"

Write-Host "=== Building qualityscaler_tensorrt.dll ($Config / $Generator) ===" -ForegroundColor Cyan

# 1. 准备构建目录
Remove-Item -Recurse -Force -ErrorAction Ignore $buildDir
New-Item -ItemType Directory -Force $buildDir | Out-Null
Set-Location $buildDir

# 2. 选择生成器
if ($Generator -eq "MSVC") {
    $cmakeGen = "Visual Studio 17 2022"
    $cmakeArch = "-A x64"
} else {
    $cmakeGen = "MinGW Makefiles"
    $cmakeArch = ""
}

# 3. 配置
cmake -G "$cmakeGen" $cmakeArch `
      -DCMAKE_BUILD_TYPE=$Config `
      -DTENSORRT_ROOT="C:/TensorRT-10.16.0.72" `
      -DCUDA_ROOT="C:/CUDA" `
      ..

# 4. 编译
cmake --build . --config $Config --parallel

# 5. 安装到项目根目录
cmake --install . --config $Config

# 6. 验证
Set-Location $root
if (Test-Path "qualityscaler_tensorrt.dll") {
    Write-Host "=== DLL build succeeded ===" -ForegroundColor Green
    Write-Host "Output: $root\qualityscaler_tensorrt.dll"
} else {
    Write-Error "DLL not found after build"
    exit 1
}

# 7. 顺手更新导入库（可选，但推荐）
Write-Host "=== Regenerating MinGW import library ==="
& "$root\scripts\update-tensorrt-import-lib.ps1"