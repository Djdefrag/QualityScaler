@echo off
setlocal

:: Find MSVC cl.exe
set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set "VS_PATH=%%i"
)

if not defined VS_PATH (
  echo [ERROR] Visual Studio MSVC compiler not found.
  exit /b 1
)

:: Setup VC environment
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

:: Setup Paths
set TRT_INC=C:\TensorRT-10.16.0.72\include
set TRT_LIB=C:\TensorRT-10.16.0.72\lib
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64

echo [BUILD] Compiling TensorRT Wrapper DLL...
cl.exe /LD /EHsc /std:c++17 /O2 ^
  /I"%TRT_INC%" /I"%CUDA_INC%" ^
  ai_tensorrt.cpp ^
  /link /LIBPATH:"%TRT_LIB%" /LIBPATH:"%CUDA_LIB%" nvinfer_10.lib cudart.lib ^
  /OUT:../../qualityscaler_tensorrt.dll

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed
    exit /b 1
)

echo [OK] DLL generated successfully: qualityscaler_tensorrt.dll
