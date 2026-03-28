# QualityScaler Go

基于 **Go + Fyne GUI** 的 AI 图像/视频超分辨率工具。

AI 推理后端优先使用 **TensorRT**（最快，需 GPU 编译环境），自动回退至 **ONNX Runtime**（支持 CUDA GPU 或 CPU），再回退至普通插值缩放。

---

## 目录

- [系统要求](#系统要求)
- [运行时 DLL 准备](#运行时-dll-准备)
- [路径一：标准 ONNX 构建（推荐新手）](#路径一标准-onnx-构建推荐新手)
- [路径二：TensorRT 加速构建](#路径二tensorrt-加速构建)
  - [1. 解决 CUDA 路径空格问题](#1-解决-cuda-路径空格问题)
  - [2. 构建 TensorRT Wrapper DLL](#2-构建-tensorrt-wrapper-dll)
  - [3. 转换模型：ONNX → TensorRT Engine](#3-转换模型onnx--tensorrt-engine)
  - [4. 设置临时环境变量](#4-设置临时环境变量)
  - [5. 编译应用](#5-编译应用)
- [运行应用](#运行应用)
- [脚本参考](#脚本参考)
- [目录结构](#目录结构)
- [常见问题排查](#常见问题排查)

---

## 队列管理

QualityScaler GUI 提供内存队列管理功能：

### 特性

- **内存队列**：队列数据仅保存在内存中，程序关闭后自动清空
- **实时增删**：支持实时添加/删除队列文件
- **自动移除**：完成处理的文件会自动从队列中移除
- **自然排序**：文件按自然顺序排序（如 `1.jpg`, `2.jpg`, `10.jpg`）

### 操作

- **添加文件**：点击"添加文件"按钮，支持多选
- **清空列表**：点击"清空列表"按钮清除所有文件
- **查看进度**：实时显示当前处理进度和剩余时间

### 注意事项

⚠️ **队列不持久化**：关闭程序后队列会清空。如需保存待处理文件列表，请使用批处理脚本或其他方式记录文件路径。

---

## 系统要求

| 组件 | 版本 / 说明 |
|------|-------------|
| **Go** | 1.25 或更高（`go.mod` 指定） |
| **操作系统** | Windows 10/11 x64 |
| **Visual C++ Redistributable** | 2015–2022（运行 ONNX Runtime 必须） |
| **NVIDIA 显卡驱动** | 支持 CUDA 的驱动（GPU 加速可选） |
| **CUDA Toolkit** | v13.1（TensorRT 构建必须），安装路径：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1` |
| **TensorRT** | 10.16.0.72，安装路径：`C:\TensorRT-10.16.0.72` |
| **Visual Studio / Build Tools** | 含 MSVC `cl.exe`（TensorRT 构建必须） |
| **CGO_ENABLED** | 1（TensorRT 构建必须；标准 ONNX 路径无需 C 编译器） |

> **只想快速试用？** 跳至「路径一」，无需安装 CUDA 或 TensorRT，即可使用 CPU/CUDA ONNX Runtime。

---

## 运行时 DLL 准备

无论哪条路径，都需要将 ONNX Runtime DLL 放入 `Assets/` 目录：

```
Assets/
├── onnxruntime.dll                  # 必须
├── onnxruntime_providers_shared.dll # 必须
└── onnxruntime_providers_cuda.dll   # 可选，GPU 加速
```

下载地址：[ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)

> 应用启动时会自动搜索 `Assets/` 目录，无需手动设置 PATH。

### 快速安装（PowerShell 模板）

**下载并安装 ONNX Runtime GPU DLL：**

```powershell
$version = "1.17.1"
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-win-x64-gpu-$version.zip"
Invoke-WebRequest -Uri $url -OutFile "$env:TEMP\ort.zip" -UseBasicParsing
Expand-Archive "$env:TEMP\ort.zip" -DestinationPath "$env:TEMP\ort" -Force
Copy-Item "$env:TEMP\ort\onnxruntime-win-x64-gpu-$version\lib\onnxruntime*.dll" -Destination ".\Assets\" -Force
```

**验证 DLL 是否齐全：**

```powershell
@("onnxruntime.dll","onnxruntime_providers_shared.dll","onnxruntime_providers_cuda.dll") | ForEach-Object {
    $p = ".\Assets\$_"
    if (Test-Path $p) { Write-Host "[OK ] $_" -ForegroundColor Green }
    else              { Write-Host "[MISS] $_" -ForegroundColor Red }
}
```

**安装 CUDA/cuDNN 依赖 DLL（CUDA GPU 加速需要）：**

```powershell
# 通过 pip 安装 NVIDIA runtime wheel，再从 site-packages 复制 DLL
python -m pip install --upgrade `
    nvidia-cuda-runtime-cu12 `
    nvidia-cublas-cu12 `
    "nvidia-cudnn-cu12==8.9.7.29"
# 找到安装位置后手动将所需 .dll 复制到 Assets\ 或系统 PATH 目录
python -c "import site; print(site.getsitepackages()[0])"
```

或用 `check-dlls.bat` 进行快速检查：

```bat
check-dlls.bat
```

---

## 路径一：标准 ONNX 构建（推荐新手）

不需要 CUDA Toolkit 或 TensorRT，也不需要 C++ 编译器。

### 构建

```powershell
go build -tags=no_gocv -ldflags="-s -w" -o qualityscaler-go.exe ./cmd/qualityscaler-fyne
```

### 运行

```powershell
.\qualityscaler-go.exe
```

AI 后端将自动选择：
- 若 `Assets/onnxruntime_providers_cuda.dll` 存在且 CUDA 驱动可用 → **CUDA GPU**
- 否则 → **CPU ONNX**
- 若 `Assets/onnxruntime.dll` 不存在 → **CPU 插值（无 AI）**

---

## 路径二：TensorRT 加速构建

TensorRT 推理速度最快，但需要完整的 CUDA + TensorRT + MSVC 环境，并且需要预先将 ONNX 模型转换为 `.engine` 文件。

### ⚠️ 重要：直接 `go build` 会失败

**直接运行以下命令会报错：**

```bash
go build -v -tags=no_gocv -o test_trt.exe ./cmd/qualityscaler-fyne
```

**错误原因：**
1. 缺少 CGO 环境变量（`CGO_CXXFLAGS` 和 `CGO_LDFLAGS`）
2. CUDA/TensorRT 路径未配置或包含空格
3. 缺少必要的 DLL 运行时依赖

**正确做法：** 按以下步骤完整配置环境，然后使用推荐脚本编译。

---

### 1. 解决 CUDA 路径空格问题

CUDA 默认安装在含有空格的路径下，会导致 CGO 编译错误：

```
runtime/cgo: invalid flag in go:cgo_ldflag: Files\\NVIDIA
```

**解决方法：以管理员身份创建符号链接**

```bat
REM 方法一：运行现成脚本（需管理员权限）
setup-cuda-link.bat

REM 方法二：手动创建（CMD 管理员）
mklink /D "C:\CUDA" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"

REM 方法三：手动创建（PowerShell 管理员）
New-Item -ItemType SymbolicLink -Path "C:\CUDA" -Target "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
```

验证：

```powershell
dir C:\CUDA\include\cuda_runtime_api.h
```

### 2. 准备 TensorRT 引擎文件

将 `AI-onnx/` 中的 `.onnx` 模型批量转换为 `.engine`，存入 `AI-tensorrt/`：

```bat
scripts\convert-to-tensorrt.bat
```

转换参数（写在脚本中，可按需修改）：

| 参数 | 值 |
|------|----|
| 最小输入尺寸 | 64×64 |
| 最优输入尺寸 | 256×256 |
| 最大输入尺寸 | 1024×1024 |
| 精度 | FP16 |
| 优化级别 | 2 |

> 转换时间较长（每个模型数分钟），结果与 GPU 型号绑定，不可跨 GPU 迁移。

### 3. 配置 CGO 环境变量

编译 TensorRT 版本前，必须设置以下 CGO 环境变量。这些变量告诉 Go 编译器如何找到 CUDA 和 TensorRT 的头文件和库文件。

#### 方法一：使用 PowerShell 脚本（推荐，最简单）

脚本会自动配置所有环境变量并执行构建：

```powershell
# 首次使用需设置执行策略（仅当前会话有效）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 运行构建脚本（自动配置环境 + 编译）
.\scripts\build-tensorrt-pure-cgo.ps1
```

#### 方法二：使用批处理脚本

```bat
.\scripts\build-tensorrt-pure-cgo.bat
```

> 注意：批处理脚本在某些环境下可能出现 `: was unexpected at this time.` 错误，此时请使用 PowerShell 脚本。

#### 方法三：手动配置环境变量（高级用户）

如果你想了解每个环境变量的具体含义，或需要在自定义环境下编译，可以手动设置：

##### 环境变量说明

| 变量 | 作用 | 示例值 |
|------|------|--------|
| `CGO_ENABLED` | 启用 CGO（C 语言绑定） | `1` |
| `CGO_CXXFLAGS` | C++ 编译器标志（头文件路径） | `-std=c++17 -O3 -IC:\TensorRT-10.16.0.72\include -IC:\CUDA\include` |
| `CGO_LDFLAGS` | 链接器标志（库文件路径和库名） | `-LC:\TensorRT-10.16.0.72\lib -LC:\CUDA\lib\x64 -lnvinfer_10 -lcudart` |
| `PATH` | 运行时 DLL 搜索路径 | `C:\TensorRT-10.16.0.72\bin;C:\CUDA\bin;...` |

##### 参数详解

**`CGO_CXXFLAGS` 参数说明：**
- `-std=c++17`：使用 C++17 标准
- `-O3`：最高优化级别
- `-IC:\TensorRT-10.16.0.72\include`：添加 TensorRT 头文件路径（`-I` 表示 include 目录）
- `-IC:\CUDA\include`：添加 CUDA 头文件路径

**`CGO_LDFLAGS` 参数说明：**
- `-LC:\TensorRT-10.16.0.72\lib`：添加 TensorRT 库文件路径（`-L` 表示 library 目录）
- `-LC:\CUDA\lib\x64`：添加 CUDA 库文件路径
- `-lnvinfer_10`：链接 TensorRT 核心库（`-l` 表示链接 libnvinfer_10.lib）
- `-lcudart`：链接 CUDA 运行时库

**注意：TensorRT 推理只需要 `nvinfer_10` 和 `cudart` 两个库，不需要 cuDNN、cuBLAS、cuRAND 等额外库。**

##### PowerShell 配置示例

```powershell
# 清除可能存在的旧 CGO 配置（防止污染）
$env:CGO_ENABLED = $null
$env:CGO_CFLAGS = $null
$env:CGO_CXXFLAGS = $null
$env:CGO_FFLAGS = $null
$env:CGO_LDFLAGS = $null

# 设置 TensorRT 和 CUDA 路径（根据实际安装路径调整）
$env:TENSORRT_PATH = "C:\TensorRT-10.16.0.72"
$env:CUDA_PATH = "C:\CUDA"

# 设置 CGO 编译环境
$env:CGO_ENABLED = "1"
$env:CGO_CXXFLAGS = "-std=c++17 -O3 -IC:/TensorRT-10.16.0.72/include -IC:/CUDA/include"
$env:CGO_LDFLAGS = "-LC:/TensorRT-10.16.0.72/lib -LC:/CUDA/lib/x64 -lnvinfer_10 -lcudart"

# 添加运行时 DLL 到 PATH（运行程序时需要）
$env:PATH = "$($env:TENSORRT_PATH)\bin;$($env:TENSORRT_PATH)\lib;$($env:CUDA_PATH)\bin;$($env:PATH)"

# 验证配置是否正确
Write-Host "CGO_ENABLED:   $($env:CGO_ENABLED)"
Write-Host "CGO_CXXFLAGS:  $($env:CGO_CXXFLAGS)"
Write-Host "CGO_LDFLAGS:   $($env:CGO_LDFLAGS)"

# 开始编译
go build -v -tags=no_gocv -ldflags="-s -w" -o qualityscaler-trt-go.exe ./cmd/qualityscaler-fyne
```

##### CMD 配置示例

```bat
@echo off
setlocal enabledelayedexpansion

REM 清除可能存在的旧 CGO 配置
set CGO_ENABLED=
set CGO_CFLAGS=
set CGO_CXXFLAGS=
set CGO_FFLAGS=
set CGO_LDFLAGS=

REM 设置 TensorRT 和 CUDA 路径（根据实际安装路径调整）
set TENSORRT_PATH=C:\TensorRT-10.16.0.72
set CUDA_PATH=C:\CUDA

REM 设置 CGO 编译环境
set CGO_ENABLED=1
set "CGO_CXXFLAGS=-std=c++17 -O3 -I%TENSORRT_PATH%\include -I!CUDA_PATH!\include"
set "CGO_LDFLAGS=-L%TENSORRT_PATH%\lib -L!CUDA_PATH!\lib\x64 -lnvinfer_10 -lcudart"

REM 添加运行时 DLL 到 PATH（运行程序时需要）
set "PATH=%PATH%;%TENSORRT_PATH%\bin;%TENSORRT_PATH%\lib;!CUDA_PATH!\bin"

REM 验证配置是否正确
echo CGO_ENABLED:   %CGO_ENABLED%
echo CGO_CXXFLAGS:  %CGO_CXXFLAGS%
echo CGO_LDFLAGS:   %CGO_LDFLAGS%

REM 开始编译
go build -v -tags=no_gocv -ldflags="-s -w" -o qualityscaler-trt-go.exe ./cmd/qualityscaler-fyne
```

##### 验证环境变量是否设置成功

**PowerShell：**
```powershell
echo $env:CGO_ENABLED
echo $env:CGO_CXXFLAGS
echo $env:CGO_LDFLAGS
```

**CMD：**
```bat
echo %CGO_ENABLED%
echo %CGO_CXXFLAGS%
echo %CGO_LDFLAGS%
```

##### 路径调整说明

如果你的 TensorRT 或 CUDA 安装路径不同，请修改对应的路径变量：

**TensorRT 常见安装路径：**
- `C:\TensorRT-10.16.0.72`（本文档默认）
- `C:\TensorRT-10.x.x.x`（其他版本）

**CUDA 常见安装路径：**
- `C:\CUDA`（符号链接，推荐）
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`（原始路径，需创建符号链接）
- `C:\PROGRA~1\NVIDIA~2\CUDA\v13.1`（短路径格式）

### 4. 编译应用

**推荐方法：使用构建脚本**

```powershell
# PowerShell（推荐）
.\scripts\build-tensorrt-pure-cgo.ps1

# 或批处理
.\scripts\build-tensorrt-pure-cgo.bat
```

**手动构建（仅在使用方法三手动配置环境变量后）：**

```bash
go build -v -tags=no_gocv -ldflags="-s -w" -o qualityscaler-trt-go.exe ./cmd/qualityscaler-fyne
```

构建成功后输出 `qualityscaler-trt-go.exe`。

### 5. 环境配置检查

如遇问题，可运行环境检查脚本：

```powershell
# PowerShell
.\scripts\setup-tensorrt.ps1
```

此脚本会检查：
- TensorRT 安装是否完整
- CUDA 路径是否正确
- TensorRT 引擎文件是否存在
- 必要的 DLL 是否就绪

---

### 常见编译错误及解决方案

#### 错误 1：`NvInfer.h: No such file or directory`

**原因：** `CGO_CXXFLAGS` 未正确设置

**解决：**
```powershell
$env:CGO_CXXFLAGS = "-std=c++17 -O3 -IC:\TensorRT-10.16.0.72\include -IC:\CUDA\include"
```

#### 错误 2：`cannot find -lnvinfer_10`

**原因：** `CGO_LDFLAGS` 未正确设置或 TensorRT lib 路径错误

**解决：**
```powershell
$env:CGO_LDFLAGS = "-LC:\TensorRT-10.16.0.72\lib -LC:\CUDA\lib\x64 -lnvinfer_10 -lcudart"
```

#### 错误 3：`lld: error: unable to find library -lcudnn`

**原因：** 环境变量 `CGO_LDFLAGS` 中包含了多余的 `-lcudnn -lcurand -lcublas` 标志，但系统中没有对应的库文件

**解决：**
1. 清除环境变量中的多余标志，只保留必需的 `-lnvinfer_10 -lcudart`
2. 使用 `.\scripts\build-tensorrt-pure-cgo.ps1` 脚本会自动处理此问题
3. 如果手动编译，确保 `CGO_LDFLAGS` 不包含不需要的库

#### 错误 4：`runtime/cgo: invalid flag in go:cgo_ldflag: Files\\NVIDIA`

**原因：** CUDA 路径包含空格

**解决：** 以管理员身份创建 `C:\CUDA` 符号链接（见步骤 1）

#### 错误 5：`: was unexpected at this time.`（批处理脚本）

**原因：** 批处理脚本在 PowerShell 中运行时变量展开问题

**解决：**
1. 使用 PowerShell 脚本：`.\scripts\build-tensorrt-pure-cgo.ps1`
2. 或在 CMD 环境中运行批处理脚本
3. 使用 `cmd /c .\scripts\build-tensorrt-pure-cgo.bat`

---

## 打包发布

QualityScaler 提供四个发行版本，适配不同硬件环境：

| 版本 | 文件名后缀 | 说明 | 默认配置 | 适用人群 |
|------|-----------|------|---------|---------|
| 无N卡兼容版 | `onnx-cpu` | 纯 CPU 推理，体积最小 | 质量/单线程/x264 | 无 NVIDIA 显卡的用户 |
| ONNX+CUDA版 | `onnx-cuda` | ONNX Runtime + CUDA 加速 | 平衡/双线程/x264 | 有 N 卡、不想配置 TensorRT |
| N卡专业版 | `tensorrt-gpu` | TensorRT 主推理，ONNX 备选 | 极限性能/4线程/nvenc | 追求最高性能的 N 卡用户 |
| **全量版** | `full` | 包含所有模型和运行时，自动适配 | 极限性能/4线程/nvenc | 不确定硬件环境的用户 |

### 打包全部四个版本（推荐）

```powershell
# PowerShell（推荐）
scripts\Package-Release.ps1

# 批处理
scripts\build-and-package.bat
```

**输出文件：**
```
dist\release\
├── QualityScaler-1.0.0-onnx-cpu.zip       # 无N卡兼容版（ONNX）
├── QualityScaler-1.0.0-onnx-cuda.zip      # ONNX+CUDA 加速版
├── QualityScaler-1.0.0-tensorrt-gpu.zip   # N卡专业版（TensorRT）
└── QualityScaler-1.0.0-full.zip           # 全量版（包含所有）
```

### 只打包指定版本

```powershell
# PowerShell
scripts\Package-Release.ps1 -Edition onnx-cpu       # 无N卡兼容版
scripts\Package-Release.ps1 -Edition onnx-cuda      # ONNX+CUDA 版
scripts\Package-Release.ps1 -Edition tensorrt-gpu   # N卡专业版
scripts\Package-Release.ps1 -Edition full           # 全量版（推荐给不确定硬件的用户）

# 批处理
scripts\build-and-package.bat onnx-cpu
scripts\build-and-package.bat tensorrt-gpu
scripts\build-and-package.bat full                  # 全量版
```

### 仅打包（跳过编译）

如已有编译好的可执行文件，可跳过构建步骤：

```powershell
scripts\Package-Release.ps1 -SkipBuild
scripts\Package-Release.ps1 -Edition onnx-cuda -SkipBuild

# 批处理（package-release.bat 默认添加 -SkipBuild）
scripts\package-release.bat
scripts\package-release.bat onnx-cpu
```

### TensorRT 版本注意事项

`tensorrt-gpu` 和 `full` 版本需要 CGO 编译环境：
- TDM-GCC 或 Visual Studio Build Tools（C++ 编译器）
- TensorRT 10.x 安装在 `C:\TensorRT-10.16.0.72`（编译时需要）
- CUDA 12.x

**注意**：打包后不再需要用户安装 TensorRT，`qualityscaler_tensorrt.dll` 已内置。

打包脚本在构建前会自动检查依赖，如未满足则跳过该版本，不影响其他版本打包。

---

## 运行应用

### 直接运行（开发调试）

```bat
REM 使用 go run，自动设置 CGO 环境（TensorRT 路径）
run-with-tensorrt.bat
```

### 运行已构建的可执行文件

```bat
REM ONNX Runtime 版
qualityscaler-go.exe

REM TensorRT 版（运行时需要 TensorRT DLL 在 PATH 中）
run-tensorrt.bat
REM 或手动指定路径后运行：
REM set PATH=C:\TensorRT-10.16.0.72\bin;C:\TensorRT-10.16.0.72\lib;C:\CUDA\bin;%PATH%
REM qualityscaler-trt-go.exe
```

### 应用 AI 后端优先级

```
TensorRT (GPU)          ← 最快，需 .engine 文件 + qualityscaler_tensorrt.dll
    ↓ 失败或不可用
ONNX Runtime CUDA (GPU) ← 需 onnxruntime_providers_cuda.dll
    ↓ 失败或不可用
ONNX Runtime CPU        ← 需 onnxruntime.dll
    ↓ DLL 不存在
CPU 插值缩放（无 AI）
```

---

## 脚本参考

| 脚本 | 说明 |
|------|------|
| `setup-cuda-link.bat` | 创建 `C:\CUDA` 符号链接，**需管理员权限**，一次性操作 |
| `scripts\build-tensorrt-pure-cgo.bat` | 批处理脚本：构建 TensorRT 版可执行文件，自动检测 CUDA 路径 |
| `scripts\build-tensorrt-pure-cgo.ps1` | **推荐**：PowerShell 脚本：构建 TensorRT 版，自动配置环境 |
| `run-with-tensorrt.bat` | `go run` 开发调试，自动设置 CGO 环境变量 |
| `run-tensorrt.bat` | 运行已构建的 `qualityscaler-trt-go.exe` |
| `check-dlls.bat` | 检查 `Assets/` 中 ONNX Runtime DLL 是否齐全 |
| `scripts\setup-tensorrt.ps1` | 检查并配置 TensorRT 环境（DLL、引擎文件） |
| `scripts\convert-to-tensorrt.bat` | 批量将 `AI-onnx/*.onnx` 转换为 `AI-tensorrt/*.engine` |
| `scripts\validate_tensorrt_engines.bat` | 验证已生成的 `.engine` 文件 |
| `scripts\validate_tensorrt_engines.ps1` | PowerShell 版引擎验证脚本 |
| `scripts\Package-Release.ps1` | 打包发布（所有版本或指定版本） |
| `scripts\Build-Installer.ps1` | 构建 Windows 安装程序 |

> ONNX Runtime DLL 安装与验证的命令模板请参见[运行时 DLL 准备](#运行时-dll-准备)章节。

---

### 重新生成 TensorRT 导入库（DLL 更新后必做）

若你更换了 `qualityscaler_tensorrt.dll`，必须重新生成 MinGW 导入库，否则 CGO 链接阶段会报 `cannot find -lqualityscaler_tensorrt`：

```powershell
.\scripts\update-tensorrt-import-lib.ps1
```

脚本会自动刷新：
- `qualityscaler_tensorrt.def`（导出符号列表）  
- `internal\core\libqualityscaler_tensorrt.dll.a`（MinGW 链接用导入库）

---

## 目录结构

```
QualityScaler/
├── cmd/
│   └── qualityscaler-fyne/     # 主程序入口
├── internal/
│   ├── core/                   # AI 推理、图像/视频处理核心逻辑
│   ├── gui_fyne/               # Fyne GUI 应用层
│   ├── app/                    # 配置与应用状态
│   └── tensorrt_dll/           # TensorRT C++ Wrapper 源码及构建脚本
├── AI-onnx/                    # ONNX 模型文件 (*.onnx)
├── AI-tensorrt/                # TensorRT 引擎文件 (*.engine)，转换后生成
├── Assets/                     # 运行时 DLL（onnxruntime*.dll 等）
├── scripts/                    # 辅助脚本（环境安装、模型转换、验证）
├── third_party/                # 第三方依赖（lxn/walk 替换）
├── go.mod / go.sum
├── qualityscaler_tensorrt.dll  # TensorRT Wrapper DLL（构建后生成）
├── build_dll.bat               # 构建 TensorRT Wrapper DLL
├── build-tensorrt-pure-cgo.bat # 构建 TensorRT 版应用
├── run-with-tensorrt.bat       # 开发调试运行
├── run-tensorrt.bat            # 运行已构建的 TensorRT 版
├── check-dlls.bat              # 检查 DLL
├── check-tensorrt-env.bat      # 检查环境
└── setup-cuda-link.bat         # 创建 CUDA 符号链接
```

---

## 常见问题排查

### 错误：`invalid flag in go:cgo_ldflag: Files\\NVIDIA`

**原因**：CUDA 安装路径含有空格，CGO 无法正确解析。

**解决**：以管理员身份运行 `setup-cuda-link.bat` 创建 `C:\CUDA` 符号链接，然后重新编译。

---

### 错误：`cuda_runtime_api.h: No such file or directory`

**原因**：`CGO_CXXFLAGS` 未正确设置，或 CUDA 符号链接不存在。

**解决**：
1. 确认 `C:\CUDA` 符号链接存在：`dir C:\CUDA\include`
2. 确认 `CGO_CXXFLAGS` 包含 `-IC:\CUDA\include`

---

### 错误：`cannot find -lnvinfer_10`

**原因**：`CGO_LDFLAGS` 中 TensorRT lib 路径不对，或 TensorRT 未安装。

**解决**：
1. 确认 TensorRT 安装目录：`dir C:\TensorRT-10.16.0.72\lib\nvinfer_10.lib`
2. 检查 `CGO_LDFLAGS` 是否包含 `-LC:\TensorRT-10.16.0.72\lib`

---

### 错误：`onnxruntime.dll not found` / AI 后端降级为 CPU 插值

**原因**：`Assets/` 目录中缺少 ONNX Runtime DLL。

**解决**：
1. 运行 `check-dlls.bat` 查看缺少哪些文件
2. 从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载对应版本，放入 `Assets/`

---

### 错误：MSVC 编译器未找到（构建 DLL 时）

**原因**：未安装 Visual Studio 或 Build Tools，或 `vswhere.exe` 不存在。

**解决**：安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选「C++ 桌面开发」工作负载。

---

### TensorRT 可用但推理失败

**原因**：`.engine` 文件与当前 GPU 型号不匹配（跨 GPU 迁移），或引擎已过期。

**解决**：删除 `AI-tensorrt/` 目录中的 `.engine` 文件，重新运行 `scripts\convert-to-tensorrt.bat` 在本机转换。

---

### 运行时找不到 TensorRT DLL（`qualityscaler-trt-go.exe` 启动失败）

**原因**：TensorRT 的运行时 DLL（`nvinfer_10.dll` 等）不在 PATH 中。

**解决**：运行 `run-tensorrt.bat`（已自动追加 PATH），或手动设置：

```bat
set PATH=C:\TensorRT-10.16.0.72\bin;C:\TensorRT-10.16.0.72\lib;C:\CUDA\bin;%PATH%
qualityscaler-trt-go.exe
```
