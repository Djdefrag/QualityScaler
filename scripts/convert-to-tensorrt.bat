@echo off
setlocal enabledelayedexpansion
REM ==================================================
REM TensorRT 10.16 批量模型转换工具 (全模型支持)
REM ==================================================

set TENSORRT_PATH=C:\TensorRT-10.16.0.72
set PROJECT_DIR=%~dp0..
set ONNX_DIR=%PROJECT_DIR%\AI-onnx
set TRT_DIR=%PROJECT_DIR%\AI-tensorrt

REM 创建TensorRT目录
if not exist "%TRT_DIR%" mkdir "%TRT_DIR%"

REM 设置PATH
set PATH=%TENSORRT_PATH%\bin;%PATH%

echo ========================================
echo TensorRT 10.16 批量模型转换工具
echo ========================================
echo.

REM 检查trtexec
if not exist "%TENSORRT_PATH%\bin\trtexec.exe" (
    echo [错误] 未找到 trtexec.exe
    echo 请确保 TensorRT 已正确安装
    echo 路径: %TENSORRT_PATH%\bin\trtexec.exe
    pause
    exit /b 1
)

set /a count=1
for %%F in ("%ONNX_DIR%\*.onnx") do (
    set "filename=%%~nF"
    echo [!count!] 正在转换: !filename!
    echo       参数: 动态维度 64x64 - 1024x1024, FP16, 优化级别2
    
    "%TENSORRT_PATH%\bin\trtexec.exe" ^
        --onnx="%%F" ^
        --saveEngine="%TRT_DIR%\!filename!.engine" ^
        --minShapes=input:1x3x64x64 ^
        --optShapes=input:1x3x256x256 ^
        --maxShapes=input:1x3x1024x1024 ^
        --fp16 ^
        --builderOptimizationLevel=2 ^
        --memPoolSize=workspace:4096 > "%TRT_DIR%\!filename!_build.log" 2>&1

    if errorlevel 1 (
        echo       [错误] !filename! 转换失败!
        echo       详情请查看: "%TRT_DIR%\!filename!_build.log"
    ) else (
        echo       [成功] 生成: !filename!.engine
        del "%TRT_DIR%\!filename!_build.log"
    )
    
    set /a count+=1
    echo.
)

echo ========================================
echo 所有转换任务已完成！
echo ========================================
echo.
echo 生成的引擎文件列表:
dir "%TRT_DIR%\*.engine" /b
echo.
pause
