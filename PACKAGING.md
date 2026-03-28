# 打包发布指南

QualityScaler 提供四个发行版本，对应不同的硬件和运行时环境。

---

## 版本说明

| 版本 | 文件名 | Build Tags | DLL 依赖 | 模型 | 默认性能模式 | 默认编码器 |
|------|--------|-----------|---------|------|-------------|-----------|
| **无N卡兼容版**（ONNX CPU） | `*-onnx-cpu.zip` | `no_gocv` | onnxruntime.dll + shared | AI-onnx/ | Quality | x264 |
| **ONNX+CUDA 加速版** | `*-onnx-cuda.zip` | `no_gocv` | onnxruntime + cuda + shared | AI-onnx/ | Balanced | x264 |
| **N卡专业版**（TensorRT） | `*-tensorrt-gpu.zip` | `tensorrt no_gocv` | onnxruntime + cuda + shared + tensorrt | AI-tensorrt/ | Extreme Performance | h264_nvenc |
| **全量版** | `*-full.zip` | `tensorrt no_gocv` | onnxruntime + cuda + shared + tensorrt | AI-onnx/ + AI-tensorrt/ | Extreme Performance | h264_nvenc |

> `onnx-cpu` 和 `onnx-cuda` 使用相同可执行文件，区别仅在于是否打包 CUDA DLL。
> `tensorrt-gpu` 和 `full` 使用相同的可执行文件（CGO 编译），区别在于模型目录。
> `tensorrt-gpu` 和 `full` 已内置 `qualityscaler_tensorrt.dll`，用户无需安装 TensorRT。
> **不同版本有不同的默认配置**，程序会自动检测版本并应用最优默认值。

---

## 各版本默认配置详情

### onnx-cpu（无N卡兼容版）
- **性能模式**: Quality（质量优先）
- **AI 线程**: OFF（CPU 是瓶颈，多线程无意义）
- **混合**: High
- **视频编码器**: x264
- **GPU 显存**: 4 GB
- **适用**: 无 N 卡用户，追求最佳兼容性

### onnx-cuda（ONNX+CUDA 加速版）
- **性能模式**: Balanced（平衡）
- **AI 线程**: 2 threads
- **混合**: High
- **视频编码器**: x264
- **GPU 显存**: 8 GB
- **适用**: 有 N 卡但不想配置 TensorRT 的用户

### tensorrt-gpu（N卡专业版）
- **性能模式**: Extreme Performance（极限性能）
- **AI 线程**: 4 threads
- **混合**: OFF（TensorRT 无需混合）
- **视频编码器**: h264_nvenc（GPU 硬件加速）
- **GPU 显存**: 12 GB
- **适用**: 追求最高性能的 N 卡用户

### full（全量版）
- **性能模式**: Extreme Performance（极限性能）
- **AI 线程**: 4 threads
- **混合**: OFF
- **视频编码器**: h264_nvenc
- **GPU 显存**: 12 GB
- **适用**: 不确定硬件环境的用户，自动选择最佳配置

---

## 快速使用

### 打包全部四个版本

```powershell
# PowerShell（推荐）
scripts\Package-Release.ps1

# 批处理入口
scripts\build-and-package.bat
```

### 打包指定版本

```powershell
scripts\Package-Release.ps1 -Edition onnx-cpu
scripts\Package-Release.ps1 -Edition onnx-cuda
scripts\Package-Release.ps1 -Edition tensorrt-gpu
scripts\Package-Release.ps1 -Edition full         # 全量版
scripts\Package-Release.ps1 -Edition all          # 默认
```

批处理方式：

```bat
scripts\build-and-package.bat onnx-cpu
scripts\build-and-package.bat onnx-cuda
scripts\build-and-package.bat tensorrt-gpu
scripts\build-and-package.bat full                # 全量版
```

### 跳过编译（只重新打包）

适用于已编译好可执行文件、只想重新打包的场景：

```powershell
scripts\Package-Release.ps1 -SkipBuild
scripts\Package-Release.ps1 -Edition full -SkipBuild
```

批处理（`package-release.bat` 默认添加 `-SkipBuild`）：

```bat
scripts\package-release.bat
scripts\package-release.bat full
```

### 修改版本号

```powershell
scripts\Package-Release.ps1 -Version "2.0.0"
```

---

## 各版本包含文件

### onnx-cpu（无N卡兼容版）

```
QualityScaler-{version}-onnx-cpu.zip
├── QualityScaler.exe
├── Assets/                             # 图标等资源（不含 DLL）
├── onnxruntime.dll
├── onnxruntime_providers_shared.dll
├── AI-onnx/                            # ONNX 模型
└── RELEASE_INFO.txt                    # 版本说明
```

### onnx-cuda（ONNX+CUDA 加速版）

```
QualityScaler-{version}-onnx-cuda.zip
├── QualityScaler.exe
├── Assets/
├── onnxruntime.dll
├── onnxruntime_providers_shared.dll
├── onnxruntime_providers_cuda.dll      # CUDA Provider
├── AI-onnx/
└── RELEASE_INFO.txt
```

### tensorrt-gpu（N卡专业版）

```
QualityScaler-{version}-tensorrt-gpu.zip
├── QualityScaler.exe                   # CGO 编译，内置 TRT 绑定
├── Assets/
├── onnxruntime.dll
├── onnxruntime_providers_shared.dll
├── onnxruntime_providers_cuda.dll
├── qualityscaler_tensorrt.dll          # TensorRT 推理后端（内置）
├── AI-tensorrt/                        # TensorRT Engine 文件
└── RELEASE_INFO.txt
```

> 已内置 `qualityscaler_tensorrt.dll`，用户无需安装 TensorRT。

### full（全量版）

```
QualityScaler-{version}-full.zip
├── QualityScaler.exe                   # CGO 编译，TensorRT 主
├── Assets/
├── onnxruntime.dll
├── onnxruntime_providers_shared.dll
├── onnxruntime_providers_cuda.dll
├── qualityscaler_tensorrt.dll          # TensorRT 推理后端（内置）
├── AI-onnx/                            # ONNX 模型
├── AI-tensorrt/                        # TensorRT Engine 文件
└── RELEASE_INFO.txt
```

> 已内置 `qualityscaler_tensorrt.dll`，用户无需安装 TensorRT。
> 包含所有模型，程序启动时自动选择最佳推理方式：
> 1. TensorRT（若有环境且 GPU 匹配）
> 2. ONNX CUDA（若有 N 卡驱动）
> 3. ONNX CPU（回退）

---

## TensorRT 版本构建要求

`tensorrt-gpu` 和 `full` 版本需要 CGO 编译环境，打包脚本在构建前会自动检查：

| 依赖 | 要求 |
|------|------|
| C++ 编译器 | TDM-GCC 或 Visual Studio Build Tools |
| TensorRT | 安装在 `C:\TensorRT-10.16.0.72` |
| CUDA | `C:\CUDA\bin` 在 PATH 中 |

若检查失败，脚本会跳过 `tensorrt-gpu` 和 `full` 版本并继续打包其他版本。

---

## 编码说明

- PowerShell 脚本自动设置 UTF-8 输出，无编码问题
- 批处理脚本开头添加 `chcp 65001`，切换到 UTF-8
- 推荐使用 `pwsh`（PowerShell 7+），脚本自动检测并优先使用

---

## 输出目录结构

```
dist/release/
├── QualityScaler-1.0.0-onnx-cpu.zip
├── QualityScaler-1.0.0-onnx-cuda.zip
├── QualityScaler-1.0.0-tensorrt-gpu.zip
├── QualityScaler-1.0.0-full.zip       # 新增全量版
├── portable-onnx-cpu/                 # 打包用临时目录（可删除）
├── portable-onnx-cuda/
├── portable-tensorrt-gpu/
└── portable-full/                      # 新增
```
