# =============================================================================
# QualityScaler 安装包生成脚本（Inno Setup）
# 将 dist/release/portable-{edition}/ 目录打包成标准 Windows .exe 安装程序
#
# 用法：
#   .\scripts\Build-Installer.ps1                           # 生成全部四个版本安装包
#   .\scripts\Build-Installer.ps1 -Edition onnx-cpu         # 只生成无N卡兼容版安装包
#   .\scripts\Build-Installer.ps1 -Edition onnx-cuda        # 只生成 ONNX+CUDA 版安装包
#   .\scripts\Build-Installer.ps1 -Edition tensorrt-gpu     # 只生成 N卡专业版安装包
#   .\scripts\Build-Installer.ps1 -Edition full             # 只生成全量版安装包
#   .\scripts\Build-Installer.ps1 -SkipPortable             # 跳过便携包生成，直接使用已有目录
#
# 前提条件：需要安装 Inno Setup 6.x
#   下载地址：https://jrsoftware.org/isdl.php
# =============================================================================

[CmdletBinding()]
param(
    [string]$Version   = "1.0.0",
    [string]$AppName   = "QualityScaler",
    [ValidateSet("onnx-cpu", "onnx-cuda", "tensorrt-gpu", "full", "all")]
    [string]$Edition   = "all",
    [switch]$SkipPortable
)

# 设置 UTF-8 输出，避免中文乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8
$ErrorActionPreference    = "Continue"

$Root    = (Get-Location).Path
$DistDir = Join-Path $Root "dist\release"

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

# -----------------------------------------------------------------------------
# 检测 Inno Setup 编译器 iscc.exe
# -----------------------------------------------------------------------------
function Find-Iscc {
    # 1. 优先查 PATH
    $fromPath = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }

    # 2. 常见安装路径（32位 / 64位 Program Files）
    $candidates = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
        "C:\Program Files\Inno Setup 5\ISCC.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

# -----------------------------------------------------------------------------
# 确保 ChineseSimplified.isl 存在于 Inno Setup Languages 目录
# 若不存在则自动从 GitHub 下载；下载失败时设 $script:ChineseIslAvailable = $false
# -----------------------------------------------------------------------------
$script:ChineseIslAvailable = $true

function Ensure-ChineseSimplifiedIsl {
    $langDir = Join-Path (Split-Path $script:IsccPath -Parent) "Languages"
    $islPath = Join-Path $langDir "ChineseSimplified.isl"

    if (Test-Path $islPath) {
        Write-OK "ChineseSimplified.isl 已存在，跳过下载"
        return
    }

    Write-Step "下载 ChineseSimplified.isl..."
    $url = "https://raw.githubusercontent.com/jrsoftware/issrc/main/Files/Languages/Unofficial/ChineseSimplified.isl"
    try {
        Invoke-WebRequest -Uri $url -OutFile $islPath -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        Write-OK "ChineseSimplified.isl 下载成功: $islPath"
    } catch {
        Write-Warn "下载 ChineseSimplified.isl 失败: $_"
        Write-Warn "安装包将降级为英文界面（不影响功能）"
        $script:ChineseIslAvailable = $false
    }
}

# -----------------------------------------------------------------------------
# 生成单个版本的 Inno Setup .iss 脚本内容
# -----------------------------------------------------------------------------
function New-IssContent {
    param(
        [string]$EditionName,    # onnx-cpu / onnx-cuda / tensorrt-gpu / full
        [string]$DisplayName,    # 显示名称（英文，用于快捷方式）
        [string]$Description     # 版本描述（用于安装程序标题）
    )

    $portableDir   = Join-Path $DistDir "portable-$EditionName"
    # 按优先级查找图标：Assets\miao.ico -> Assets\dist_icon\miao.ico -> Assets\logo.ico
    $iconFile = $null
    foreach ($candidate in @("Assets\miao.ico", "Assets\dist_icon\miao.ico", "Assets\logo.ico")) {
        $full = Join-Path $Root $candidate
        if (Test-Path $full) { $iconFile = $full; break }
    }
    if (-not $iconFile) { $iconFile = Join-Path $Root "Assets\logo.ico" }
    # Inno Setup 需要反斜杠绝对路径
    $sourcePath    = $portableDir
    $outputDir     = $DistDir
    $outputBase    = "${AppName}-${Version}-${EditionName}-setup"
    # 不同版本安装到独立子目录，避免相互覆盖
    $installSubDir = "${AppName}-${EditionName}"

    # 动态构建 [Languages] 节：仅当语言文件可用时才追加中文行
    $langSection = '[Languages]' + "`r`n" + 'Name: "english"; MessagesFile: "compiler:Default.isl"'
    if ($script:ChineseIslAvailable) {
        $langSection += "`r`n" + 'Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"'
    }

    return @"
; =============================================================================
; QualityScaler $DisplayName 安装脚本（由 Build-Installer.ps1 自动生成）
; =============================================================================

[Setup]
AppName=${AppName} ($DisplayName)
AppVersion=$Version
AppPublisher=LingXin
AppPublisherURL=https://github.com/Djdefrag/QualityScaler
AppSupportURL=https://github.com/Djdefrag/QualityScaler/issues
AppUpdatesURL=https://github.com/Djdefrag/QualityScaler/releases
DefaultDirName={autopf}\$installSubDir
DefaultGroupName=$AppName
AllowNoIcons=yes
OutputDir=$outputDir
OutputBaseFilename=$outputBase
SetupIconFile=$iconFile
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\${AppName}.exe
UninstallDisplayName=${AppName} ($DisplayName)

$langSection

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "$sourcePath\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\$AppName ($DisplayName)"; Filename: "{app}\${AppName}.exe"
Name: "{group}\{cm:UninstallProgram,$AppName ($DisplayName)}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\$AppName ($DisplayName)"; Filename: "{app}\${AppName}.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\${AppName}.exe"; Description: "{cm:LaunchProgram,$AppName ($DisplayName)}"; Flags: nowait postinstall skipifsilent
"@
}

# -----------------------------------------------------------------------------
# 为单个版本生成安装包
# -----------------------------------------------------------------------------
function Build-InstallerEdition {
    param(
        [string]$EditionName,
        [string]$DisplayName,
        [string]$Description
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Magenta
    Write-Host "  生成安装包: $DisplayName ($EditionName)" -ForegroundColor Magenta
    Write-Host "============================================================" -ForegroundColor Magenta

    $portableDir  = Join-Path $DistDir "portable-$EditionName"
    $setupExePath = Join-Path $DistDir "${AppName}-${Version}-${EditionName}-setup.exe"

    # 确认便携目录存在
    if (-not (Test-Path $portableDir)) {
        if ($SkipPortable) {
            Write-Fail "便携目录不存在: $portableDir"
            Write-Host "         使用了 -SkipPortable 但目录不存在，请先运行 Package-Release.ps1" -ForegroundColor Yellow
            return $false
        }

        Write-Step "便携目录不存在，调用 Package-Release.ps1 生成..."
        $psScript = Join-Path $Root "scripts\Package-Release.ps1"
        & powershell -ExecutionPolicy Bypass -NoProfile -File $psScript -Edition $EditionName -SkipBuild
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Package-Release.ps1 执行失败（exit $LASTEXITCODE），跳过 $EditionName"
            return $false
        }
        if (-not (Test-Path $portableDir)) {
            Write-Fail "Package-Release.ps1 执行后仍未找到目录: $portableDir"
            return $false
        }
    }

    Write-OK "便携目录: $portableDir"

    # 生成临时 .iss 文件
    Write-Step "生成 Inno Setup 脚本"
    $issContent  = New-IssContent -EditionName $EditionName -DisplayName $DisplayName -Description $Description
    $tempIssPath = Join-Path $env:TEMP "${AppName}-${EditionName}-$([System.Guid]::NewGuid().ToString('N')).iss"
    $issContent | Out-File -FilePath $tempIssPath -Encoding UTF8
    Write-OK "临时 .iss: $tempIssPath"

    # 调用 iscc.exe 编译
    Write-Step "调用 iscc.exe 编译安装包..."
    try {
        & $script:IsccPath $tempIssPath
        $exitCode = $LASTEXITCODE
    } finally {
        # 无论成功与否，删除临时文件
        if (Test-Path $tempIssPath) { Remove-Item $tempIssPath -Force }
    }

    if ($exitCode -ne 0) {
        Write-Fail "iscc.exe 编译失败（exit $exitCode），跳过 $EditionName"
        return $false
    }

    if (Test-Path $setupExePath) {
        $sizeMB = [math]::Round((Get-Item $setupExePath).Length / 1MB, 1)
        Write-OK "输出: $setupExePath ($sizeMB MB)"
        return $true
    } else {
        Write-Fail "未找到预期输出文件: $setupExePath"
        return $false
    }
}

# =============================================================================
# 主流程
# =============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QualityScaler 安装包生成工具 v$Version" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Edition     : $Edition"
Write-Host "  SkipPortable: $($SkipPortable.IsPresent)"
Write-Host ""

# 确保输出目录存在
New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

# 检测 iscc.exe
Write-Step "检测 Inno Setup 编译器 (iscc.exe)"
$script:IsccPath = Find-Iscc
if (-not $script:IsccPath) {
    Write-Fail "未找到 Inno Setup 编译器 iscc.exe"
    Write-Host ""
    Write-Host "  请先安装 Inno Setup 6.x：" -ForegroundColor Yellow
    Write-Host "    https://jrsoftware.org/isdl.php" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  安装后重新运行本脚本即可。" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
Write-OK "iscc.exe: $script:IsccPath"

# 确保 ChineseSimplified.isl 存在（若缺失则自动下载）
Ensure-ChineseSimplifiedIsl

# 根据 -Edition 参数决定处理哪些版本
$buildAll = ($Edition -eq "all")
$results  = @()

if ($buildAll -or $Edition -eq "onnx-cpu") {
    $ok = Build-InstallerEdition `
        -EditionName "onnx-cpu" `
        -DisplayName "CPU Edition" `
        -Description "无N卡兼容版（ONNX CPU 推理）"
    $results += [PSCustomObject]@{ Edition = "onnx-cpu"; Success = $ok }
}

if ($buildAll -or $Edition -eq "onnx-cuda") {
    $ok = Build-InstallerEdition `
        -EditionName "onnx-cuda" `
        -DisplayName "CUDA Edition" `
        -Description "ONNX+CUDA 加速版"
    $results += [PSCustomObject]@{ Edition = "onnx-cuda"; Success = $ok }
}

if ($buildAll -or $Edition -eq "tensorrt-gpu") {
    $ok = Build-InstallerEdition `
        -EditionName "tensorrt-gpu" `
        -DisplayName "TensorRT Edition" `
        -Description "N卡专业版（TensorRT 推理）"
    $results += [PSCustomObject]@{ Edition = "tensorrt-gpu"; Success = $ok }
}

if ($buildAll -or $Edition -eq "full") {
    $ok = Build-InstallerEdition `
        -EditionName "full" `
        -DisplayName "Full Edition" `
        -Description "全量版（包含所有运行时和模型）"
    $results += [PSCustomObject]@{ Edition = "full"; Success = $ok }
}

# =============================================================================
# 汇总输出
# =============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  生成完成！安装包文件：" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

$anySuccess = $false
Get-ChildItem -Path $DistDir -Filter "*-setup.exe" -ErrorAction SilentlyContinue | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.Name)  ($sizeMB MB)" -ForegroundColor White
    $anySuccess = $true
}

if (-not $anySuccess) {
    Write-Host "  （无安装包生成）" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "  输出目录: $DistDir" -ForegroundColor DarkGray

# 输出失败汇总
$failedEditions = $results | Where-Object { -not $_.Success }
if ($failedEditions.Count -gt 0) {
    Write-Host ""
    Write-Host "  以下版本生成失败：" -ForegroundColor Yellow
    $failedEditions | ForEach-Object { Write-Host "    - $($_.Edition)" -ForegroundColor Yellow }
}

Write-Host ""
Read-Host "按回车键退出"
