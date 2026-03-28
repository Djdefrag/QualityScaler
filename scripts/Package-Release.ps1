# =============================================================================
# QualityScaler 多版本打包脚本 (PowerShell)
# 版本：onnx-cpu（无N卡兼容版） / onnx-cuda（ONNX+CUDA版） / tensorrt-gpu（N卡专业版） / full（全量版）
# 用法：
#   .\scripts\Package-Release.ps1                        # 打包全部四个版本
#   .\scripts\Package-Release.ps1 -Edition onnx-cpu      # 只打包无N卡兼容版
#   .\scripts\Package-Release.ps1 -Edition onnx-cuda     # 只打包 ONNX+CUDA版
#   .\scripts\Package-Release.ps1 -Edition tensorrt-gpu  # 只打包 N卡专业版（自动重新编译 TensorRT DLL）
#   .\scripts\Package-Release.ps1 -Edition full         # 只打包全量版
#   .\scripts\Package-Release.ps1 -SkipBuild             # 跳过编译，只重新打包
# =============================================================================

[CmdletBinding()]
param(
    [string]$Version   = "1.0.0",
    [string]$AppName   = "QualityScaler",
    [ValidateSet("onnx-cpu", "onnx-cuda", "tensorrt-gpu", "full", "all")]
    [string]$Edition   = "all",
    [switch]$SkipBuild
)

# 设置 UTF-8 输出，避免中文乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8
$ErrorActionPreference    = "Continue"

$Root      = (Get-Location).Path
$DistDir   = Join-Path $Root "dist\release"

# 提示：如果你的 TensorRT 安装路径不同，请在此处修改为你的实际 bin 目录路径
$env:TensorRT = "C:\TensorRT-10.16.0.72\bin"

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "  >> $msg" -ForegroundColor Cyan
}

function Write-OK([string]$msg) {
    Write-Host "     [OK] $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
    Write-Host "     [WARN] $msg" -ForegroundColor Yellow
}

function Write-Fail([string]$msg) {
    Write-Host "     [FAIL] $msg" -ForegroundColor Red
}

function Copy-IfExists([string]$src, [string]$destDir) {
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $destDir -Force
        Write-OK (Split-Path $src -Leaf)
    } else {
        Write-Warn "未找到: $src，跳过"
    }
}

function Copy-DirIfExists([string]$src, [string]$destDir) {
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $destDir -Recurse -Force
        Write-OK (Split-Path $src -Leaf)
    } else {
        Write-Warn "未找到目录: $src，跳过"
    }
}

# -----------------------------------------------------------------------------
# 构建函数
# -----------------------------------------------------------------------------

# 图标准备（生成 app.syso）
function Invoke-PrepareIcon {
    $prepareIconScript = Join-Path $Root "scripts\Prepare-Icon.ps1"
    if (-not (Test-Path $prepareIconScript)) {
        Write-Warn "未找到 Prepare-Icon.ps1，跳过图标嵌入（exe 将使用默认图标）"
        return
    }

    Write-Step "准备 exe 图标（Prepare-Icon.ps1）..."
    $sysoPath = Join-Path $Root "cmd\qualityscaler-fyne\app.syso"
    # 按优先级查找 ICO 文件：Assets\miao.ico -> Assets\dist_icon\miao.ico -> 从 jpg 自动转换
    $icoPath = $null
    foreach ($candidate in @("Assets\miao.ico", "Assets\dist_icon\miao.ico")) {
        $full = Join-Path $Root $candidate
        if (Test-Path $full) { $icoPath = $full; break }
    }

    if ($icoPath) {
        # ICO 已就绪，仅生成 syso（跳过图片转换）
        & powershell -ExecutionPolicy Bypass -NoProfile -File $prepareIconScript `
            -OutputIco $icoPath -OutputSyso $sysoPath -SkipConvert -Force
    } else {
        # ICO 不存在，从源图自动转换
        & powershell -ExecutionPolicy Bypass -NoProfile -File $prepareIconScript `
            -OutputSyso $sysoPath -Force
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Prepare-Icon.ps1 执行失败（exit $LASTEXITCODE），继续构建但 exe 可能没有图标"
    }
}

function Invoke-Build([string]$edition, [string]$buildTags, [string]$outExe) {
    if ($SkipBuild) {
        if (Test-Path $outExe) {
            Write-OK "跳过构建，使用已有 $outExe"
            return $true
        } else {
            Write-Warn "指定 -SkipBuild 但 $outExe 不存在，将重新构建"
        }
    }

    # 保存原始 CGO 环境变量，用于构建后恢复
    $origCgoEnabled  = $env:CGO_ENABLED
    $origCgoCflags   = $env:CGO_CFLAGS
    $origCgoCxxflags = $env:CGO_CXXFLAGS
    $origCgoFflags   = $env:CGO_FFLAGS
    $origCgoLdflags  = $env:CGO_LDFLAGS

    if ($edition -eq "tensorrt-gpu") {
        # TensorRT 版需要 CGO 环境
        $trtPath = "C:\TensorRT-10.16.0.72"
        if (-not (Test-Path $trtPath)) {
            Write-Fail "未找到 TensorRT 安装目录: $trtPath"
            Write-Host "         请安装 TensorRT 或修改路径后重试，跳过 tensorrt-gpu 版本" -ForegroundColor Yellow
            return $false
        }


        $env:PATH = "$trtPath\bin;C:\CUDA\bin;" + $env:PATH

        # 清空所有 CGO 变量，防止继承会话中遗留 flags。
        # 注意：TensorRT 的 include/lib 已在 internal/core/ai_tensorrt_cgo.go 的 #cgo 指令中定义，
        # 这里若再注入 CGO_CXXFLAGS/CGO_LDFLAGS，在部分 Go 版本上会触发 cmd/go 的 cgo panic（index out of range）。
        Write-Step "设置 TensorRT CGO 环境（清除遗留 CGO flags）"
        $env:CGO_ENABLED  = "1"
        $env:CGO_CFLAGS   = ""
        $env:CGO_FFLAGS   = ""
        $env:CGO_CXXFLAGS = ""
        $env:CGO_LDFLAGS  = ""
        Write-OK "CGO flags 已清空，使用源码内 #cgo 配置"
    }

    # 生成 app.syso（为 exe 嵌入图标）
    $sysoPath = Join-Path $Root "cmd\qualityscaler-fyne\app.syso"
    Invoke-PrepareIcon

    Write-Step "构建 $edition 版本 (tags: $buildTags) -> $outExe"
    $buildArgs = @("build", "-tags", $buildTags, "-ldflags", "-s -w -H windowsgui", "-o", $outExe, "./cmd/qualityscaler-fyne")
    try {
        & go @buildArgs
        $exitCode = $LASTEXITCODE
    } finally {
        # 无论构建成功与否，均清理临时 .syso，避免污染源码树
        if (Test-Path $sysoPath) {
            Remove-Item $sysoPath -Force -ErrorAction SilentlyContinue
            Write-OK "已清理临时 app.syso"
        }
        # 恢复原始 CGO 环境变量，避免影响后续 onnx 等版本的构建
        $env:CGO_ENABLED  = $origCgoEnabled
        $env:CGO_CFLAGS   = $origCgoCflags
        $env:CGO_CXXFLAGS = $origCgoCxxflags
        $env:CGO_FFLAGS   = $origCgoFflags
        $env:CGO_LDFLAGS  = $origCgoLdflags
    }

    if ($exitCode -ne 0) {
        Write-Fail "$edition 版本构建失败（exit $exitCode），跳过打包"
        return $false
    }
    Write-OK "构建成功: $outExe"
    return $true
}

# -----------------------------------------------------------------------------
# 打包函数（单版本）
# -----------------------------------------------------------------------------
function Build-Edition {
    param(
        [string]   $EditionName,    # onnx-cpu / onnx-cuda / tensorrt-gpu
        [string]   $DisplayName,    # 显示名称（中文）
        [string]   $BuildTags,      # Go build tags
        [string[]] $DllList,        # 需要复制的 DLL 相对路径列表
        [string[]] $ModelDirs,      # 需要复制的模型目录列表
        [string]   $ReleaseNote     # 内嵌到 ZIP 的版本说明
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Magenta
    Write-Host "  打包版本: $DisplayName ($EditionName)" -ForegroundColor Magenta
    Write-Host "============================================================" -ForegroundColor Magenta

    $exeName    = "${AppName}-${EditionName}.exe"
    $portableDir = Join-Path $DistDir "portable-$EditionName"
    $zipPath    = Join-Path $DistDir "${AppName}-${Version}-${EditionName}.zip"

    # 清理旧目录
    if (Test-Path $portableDir) {
        Remove-Item -Path $portableDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $portableDir -Force | Out-Null

    # 构建
    $exeOut = Join-Path $Root $exeName
    $ok = Invoke-Build -edition $EditionName -buildTags $BuildTags -outExe $exeOut
    if (-not $ok) { return }

    # 复制可执行文件
    Write-Step "复制可执行文件"
    Copy-Item -Path $exeOut -Destination (Join-Path $portableDir "${AppName}.exe") -Force
    Write-OK "${AppName}.exe"

    # 复制 Assets（图标等，排除 DLL）
    Write-Step "复制资源文件"
    $assetsTarget = Join-Path $portableDir "Assets"
    New-Item -ItemType Directory -Path $assetsTarget -Force | Out-Null
    Get-ChildItem -Path (Join-Path $Root "Assets") -File | Where-Object { $_.Extension -ne ".dll" } | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $assetsTarget -Force
        Write-OK $_.Name
    }

    # 复制 DLL
    Write-Step "复制依赖 DLL"
    foreach ($dll in $DllList) {
        if ([System.IO.Path]::IsPathRooted($dll)) {
            $src = $dll
        } else {
            $src = Join-Path $Root $dll
        }
        Copy-IfExists $src $portableDir
    }

    # 复制模型目录
    if ($ModelDirs.Count -gt 0) {
        Write-Step "复制 AI 模型"
        foreach ($dir in $ModelDirs) {
            $src = Join-Path $Root $dir
            Copy-DirIfExists $src $portableDir
        }
    }

    # 写入 RELEASE_INFO.txt
    $releaseInfoPath = Join-Path $portableDir "RELEASE_INFO.txt"
    $releaseNote | Out-File -FilePath $releaseInfoPath -Encoding UTF8
    Write-OK "RELEASE_INFO.txt"

    # 压缩为 ZIP
    Write-Step "生成 ZIP: ${AppName}-${Version}-${EditionName}.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path "$portableDir\*" -DestinationPath $zipPath -Force
    $sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-OK "输出: $zipPath ($sizeMB MB)"
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QualityScaler 多版本打包工具 v$Version" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Edition : $Edition"
Write-Host "  SkipBuild: $($SkipBuild.IsPresent)"
Write-Host ""

# 确保 dist/release 目录存在
New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

# ---- 版本一：onnx-cpu（无N卡兼容版，纯 CPU ONNX）----
$onnxCpuNote = @"
QualityScaler - 无N卡兼容版（ONNX CPU）
版本: $Version
构建类型: onnx-cpu

【适用人群】
  - 没有 NVIDIA 显卡的用户
  - 希望最简单安装、无需配置驱动的用户

【依赖要求】
  - Windows 10/11 x64
  - 无需安装 CUDA、TensorRT 或任何显卡驱动

【包含文件】
  - QualityScaler.exe      主程序
  - Assets/                图标等资源
  - onnxruntime.dll        ONNX Runtime CPU 推理
  - onnxruntime_providers_shared.dll
  - AI-onnx/               ONNX 超分辨率模型

【注意】
  此版本仅使用 CPU 推理，处理速度较慢，但兼容性最好。
  如有 NVIDIA 显卡，建议使用 onnx-cuda 或 tensorrt-gpu 版本。
"@

# ---- 版本二：onnx-cuda（ONNX+CUDA加速版）----
$onnxCudaNote = @"
QualityScaler - ONNX+CUDA 加速版
版本: $Version
构建类型: onnx-cuda

【适用人群】
  - 拥有 NVIDIA 显卡（GTX 10 系及以上）的用户
  - 不想配置 TensorRT 的用户

【依赖要求】
  - Windows 10/11 x64
  - NVIDIA 显卡 + 最新驱动（推荐 ≥ 526.x）
  - CUDA 运行时（ONNX Runtime 已内置，无需单独安装 CUDA SDK）

【包含文件】
  - QualityScaler.exe               主程序
  - Assets/                         图标等资源
  - onnxruntime.dll                 ONNX Runtime
  - onnxruntime_providers_shared.dll
  - onnxruntime_providers_cuda.dll  CUDA 加速 Provider
  - AI-onnx/                        ONNX 超分辨率模型

【注意】
  程序启动时会自动检测 CUDA 是否可用，不可用时自动回退到 CPU 推理。
  如需极致性能，建议使用 tensorrt-gpu 版本。
"@

# ---- 版本三：tensorrt-gpu（N卡专业版）----
$trtGpuNote = @"
QualityScaler - N卡专业版（TensorRT）
版本: $Version
构建类型: tensorrt-gpu

【适用人群】
  - 拥有 NVIDIA 显卡且追求最高性能的用户
  - 愿意配置 TensorRT 环境的专业用户

【依赖要求】
  - Windows 10/11 x64
  - NVIDIA 显卡（RTX 20 系及以上推荐）
  - CUDA 12.x（https://developer.nvidia.com/cuda-downloads）
  - TensorRT 10.x（https://developer.nvidia.com/tensorrt）
    安装后请将 TensorRT\bin 和 TensorRT\lib 加入 PATH

【包含文件】
  - QualityScaler.exe               主程序（TensorRT 主推理，ONNX备选）
  - Assets/                         图标等资源
  - onnxruntime.dll                 ONNX Runtime（备用）
  - onnxruntime_providers_shared.dll
  - onnxruntime_providers_cuda.dll
  - qualityscaler_tensorrt.dll       TensorRT 推理后端（内置）
  - AI-tensorrt/                    TensorRT Engine 文件（预编译）

【注意】
  本版本已内置 qualityscaler_tensorrt.dll，无需安装 TensorRT。
  TensorRT Engine 为预编译产物，与 GPU 架构绑定；如无法加载请使用 onnx-cuda 版本。
  程序启动时：优先使用 TensorRT，失败则回退到 ONNX Runtime，最终回退到 CPU。
"@

# ---- 版本四：full（全量版）----
$fullNote = @"
QualityScaler - 全量版（包含所有运行时和模型）
版本: $Version
构建类型: full（含所有资源）

【适用人群】
  - 不确定自己硬件环境的用户
  - 希望一次性下载后自动适配最佳推理方式的用户
  - 网络条件有限、希望一次获取所有组件的用户

【依赖要求】
  - Windows 10/11 x64
  - NVIDIA 显卡 + 最新驱动（推荐，用于启用 CUDA/TensorRT）

【包含文件】
  - QualityScaler.exe               主程序（TensorRT 主推理，ONNX备选）
  - Assets/                         图标等资源
  - onnxruntime.dll                 ONNX Runtime
  - onnxruntime_providers_shared.dll
  - onnxruntime_providers_cuda.dll  CUDA Provider
  - qualityscaler_tensorrt.dll       TensorRT 推理后端（内置）
  - AI-onnx/                        ONNX 超分辨率模型
  - AI-tensorrt/                    TensorRT Engine 文件（预编译）

【注意】
  本版本已内置所有运行时 DLL（包括 qualityscaler_tensorrt.dll），无需安装 TensorRT。
  程序启动时自动检测可用硬件，按优先级选择：
    1. TensorRT（若有 N 卡且 GPU 架构匹配）
    2. ONNX CUDA（若有 N 卡驱动）
    3. ONNX CPU（回退方案）
  体积较大但无需再次下载，推荐给不确定硬件环境的用户。
"@

# -----------------------------------------------------------------------------
# 根据 -Edition 参数决定打包哪些版本
# -----------------------------------------------------------------------------
$buildAll = ($Edition -eq "all")

if ($buildAll -or $Edition -eq "onnx-cpu") {
    Build-Edition `
        -EditionName "onnx-cpu" `
        -DisplayName "无N卡兼容版（ONNX CPU）" `
        -BuildTags "no_gocv" `
        -DllList @(
            "Assets\onnxruntime.dll",
            "Assets\onnxruntime_providers_shared.dll"
        ) `
        -ModelDirs @("AI-onnx") `
        -ReleaseNote $onnxCpuNote
}

if ($buildAll -or $Edition -eq "onnx-cuda") {
    Build-Edition `
        -EditionName "onnx-cuda" `
        -DisplayName "ONNX+CUDA 加速版" `
        -BuildTags "no_gocv" `
        -DllList @(
            "Assets\onnxruntime.dll",
            "Assets\onnxruntime_providers_shared.dll",
            "Assets\onnxruntime_providers_cuda.dll"
        ) `
        -ModelDirs @("AI-onnx") `
        -ReleaseNote $onnxCudaNote
}

if ($buildAll -or $Edition -eq "tensorrt-gpu") {
    Build-Edition `
        -EditionName "tensorrt-gpu" `
        -DisplayName "N卡专业版（TensorRT）" `
        -BuildTags "tensorrt no_gocv" `
        -DllList @(
            "Assets\onnxruntime.dll",
            "Assets\onnxruntime_providers_shared.dll",
            "Assets\onnxruntime_providers_cuda.dll",
            "qualityscaler_tensorrt.dll",
            "$env:TensorRT\nvinfer_10.dll"
        ) `
        -ModelDirs @("AI-tensorrt") `
        -ReleaseNote $trtGpuNote
}

if ($buildAll -or $Edition -eq "full") {
    Build-Edition `
        -EditionName "full" `
        -DisplayName "全量版（所有运行时和模型）" `
        -BuildTags "tensorrt no_gocv" `
        -DllList @(
            "Assets\onnxruntime.dll",
            "Assets\onnxruntime_providers_shared.dll",
            "Assets\onnxruntime_providers_cuda.dll",
            "qualityscaler_tensorrt.dll",
            "$env:TensorRT\nvinfer_10.dll"
        ) `
        -ModelDirs @("AI-onnx", "AI-tensorrt") `
        -ReleaseNote $fullNote
}

# -----------------------------------------------------------------------------
# 汇总输出
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  打包完成！输出文件：" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Get-ChildItem -Path $DistDir -Filter "*.zip" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.Name)  ($sizeMB MB)" -ForegroundColor White
}
Write-Host ""
Write-Host "  输出目录: $DistDir" -ForegroundColor DarkGray
Write-Host ""
Read-Host "按回车键退出"
