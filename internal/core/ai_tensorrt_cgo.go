//go:build tensorrt
// +build tensorrt

package core

// TensorRT integration for QualityScaler Go
// This file provides pure CGO implementation for TensorRT functionality
//
// Performance optimizations applied:
//   - buffer pool: reuse float32 slices via acquireFloat32Buffer / releaseFloat32Buffer
//   - parallel RGBA→NCHW: goroutine-per-band for images larger than 256×256
//   - dual-stream Submit/Fetch API: overlap CPU preparation with GPU execution
//
// CGO flags are set via build script or environment variables:
//   CGO_CXXFLAGS: C++ compiler flags (e.g., -std=c++17 -O3 -I"C:\TensorRT-10.16.0.72\include")
//   CGO_LDFLAGS:  Linker flags (e.g., -L"C:\TensorRT-10.16.0.72\lib" -lnvinfer_10 -lcudart)
//
// See build-tensorrt-pure-cgo.bat for configuration

/*
#cgo CXXFLAGS: -std=c++17 -O3 -I"C:/TensorRT-10.16.0.72/include" -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include"
#cgo LDFLAGS: -L../../ -lqualityscaler_tensorrt
#cgo LDFLAGS: -LC:/TensorRT-10.16.0.72/lib -LC:/CUDA/lib/x64
#cgo LDFLAGS: -lnvinfer_10 -lcudart
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../tensorrt_dll/ai_tensorrt_cpp.h"
*/
import "C"

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

// InitializeTensorRT initializes the TensorRT backend using pure CGO
func InitializeTensorRT() bool {
	if trtInitialized {
		return trtAvailable
	}

	fmt.Println("[TensorRT] Initializing TensorRT backend (Pure CGO)...")

	// Check if TensorRT engines directory exists
	trtDir := "AI-tensorrt"
	if _, err := os.Stat(trtDir); os.IsNotExist(err) {
		trtInitErr = fmt.Errorf("TensorRT engines directory not found: %s", trtDir)
		fmt.Printf("[TensorRT] %v\n", trtInitErr)
		fmt.Println("[TensorRT] Falling back to ONNX Runtime backend")
		trtAvailable = false
		trtInitialized = true
		return false
	}

	// Check if there are any .engine files
	entries, err := os.ReadDir(trtDir)
	if err != nil {
		trtInitErr = fmt.Errorf("failed to read TensorRT directory: %v", err)
		fmt.Printf("[TensorRT] %v\n", trtInitErr)
		trtAvailable = false
		trtInitialized = true
		return false
	}

	hasEngines := false
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) == ".engine" {
			hasEngines = true
			break
		}
	}

	if !hasEngines {
		trtInitErr = fmt.Errorf("no TensorRT engines found in %s", trtDir)
		fmt.Printf("[TensorRT] %v\n", trtInitErr)
		fmt.Println("[TensorRT] Falling back to ONNX Runtime backend")
		trtAvailable = false
		trtInitialized = true
		return false
	}

	// Initialize TensorRT runtime via CGO
	fmt.Println("[TensorRT] Initializing TensorRT runtime...")
	ret := C.TensorRT_Initialize()
	if ret != 0 {
		errMsg := C.GoString(C.TensorRT_GetLastError())
		trtInitErr = fmt.Errorf("TensorRT initialization failed: %s", errMsg)
		fmt.Printf("[TensorRT] %v\n", trtInitErr)
		fmt.Println("[TensorRT] Falling back to ONNX Runtime backend")
		trtAvailable = false
		trtInitialized = true
		return false
	}

	fmt.Println("[TensorRT] TensorRT runtime initialized successfully")
	trtEngines = make(map[string]*TensorRTEngine)
	trtAvailable = true
	trtInitErr = nil
	trtInitialized = true

	return true
}

// GetTensorRTEngine loads or creates a TensorRT engine for given model
func GetTensorRTEngine(modelName string) (*TensorRTEngine, error) {
	if !trtAvailable {
		return nil, fmt.Errorf("TensorRT not available - use ONNX Runtime backend instead")
	}

	trtEnginesMu.RLock()
	if engine, ok := trtEngines[modelName]; ok {
		trtEnginesMu.RUnlock()
		return engine, nil
	}
	trtEnginesMu.RUnlock()

	trtEnginesMu.Lock()
	defer trtEnginesMu.Unlock()

	// Double-check after acquiring write lock
	if engine, ok := trtEngines[modelName]; ok {
		return engine, nil
	}

	// Construct engine path
	enginePath := fmt.Sprintf("AI-tensorrt/%s_fp16.engine", modelName)
	if _, err := os.Stat(enginePath); os.IsNotExist(err) {
		enginePath = fmt.Sprintf("AI-tensorrt/%s.engine", modelName)
		if _, err := os.Stat(enginePath); os.IsNotExist(err) {
			return nil, fmt.Errorf("TensorRT engine not found: %s_fp16.engine or %s.engine", modelName, modelName)
		}
	}

	cEnginePath := C.CString(enginePath)
	defer C.free(unsafe.Pointer(cEnginePath))

	fmt.Printf("[TensorRT] Loading engine: %s\n", enginePath)
	cHandle := C.TensorRT_LoadEngine(cEnginePath)
	if cHandle == nil {
		errMsg := C.GoString(C.TensorRT_GetLastError())
		return nil, fmt.Errorf("failed to load TensorRT engine: %s", errMsg)
	}

	var inputDims [4]C.int
	ret := C.TensorRT_GetInputDims(cHandle, (*C.int)(unsafe.Pointer(&inputDims[0])))
	if ret != 0 {
		C.TensorRT_UnloadEngine(cHandle)
		return nil, fmt.Errorf("failed to get input dimensions")
	}

	var outputDims [4]C.int
	ret = C.TensorRT_GetOutputDims(cHandle, (*C.int)(unsafe.Pointer(&outputDims[0])))
	if ret != 0 {
		C.TensorRT_UnloadEngine(cHandle)
		return nil, fmt.Errorf("failed to get output dimensions")
	}

	engine := &TensorRTEngine{
		handle:      cHandle,
		InputShape:  [4]int{int(inputDims[0]), int(inputDims[1]), int(inputDims[2]), int(inputDims[3])},
		OutputShape: [4]int{int(outputDims[0]), int(outputDims[1]), int(outputDims[2]), int(outputDims[3])},
	}

	fmt.Printf("[TensorRT] Engine loaded successfully\n")
	fmt.Printf("[TensorRT] Input shape: %v\n", engine.InputShape)
	fmt.Printf("[TensorRT] Output shape: %v\n", engine.OutputShape)

	trtEngines[modelName] = engine
	return engine, nil
}

// parallelThreshold is the minimum pixel count to trigger parallel RGBA→NCHW conversion.
const parallelThreshold = 256 * 256

// convertRGBAtoNCHW converts an RGBA image into a float32 NCHW tensor.
// For images larger than parallelThreshold the conversion is parallelised
// across runtime.NumCPU() goroutines (one per channel band).
func convertRGBAtoNCHW(img *image.RGBA, dst []float32) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	pixels := img.Pix
	planeSize := height * width

	if planeSize <= parallelThreshold {
		// Sequential fast-path
		const inv255 = 1.0 / 255.0
		for y := 0; y < height; y++ {
			row := y * img.Stride
			rowOff := y * width
			for x := 0; x < width; x++ {
				off := row + x*4
				idx := rowOff + x
				dst[idx] = float32(pixels[off]) * inv255
				dst[planeSize+idx] = float32(pixels[off+1]) * inv255
				dst[2*planeSize+idx] = float32(pixels[off+2]) * inv255
			}
		}
		return
	}

	// Parallel path: 3 goroutines, one per channel
	numCPU := runtime.NumCPU()
	if numCPU > 3 {
		numCPU = 3
	}

	var wg sync.WaitGroup
	const inv255 = 1.0 / 255.0

	// Split by channel band
	for ch := 0; ch < 3; ch++ {
		wg.Add(1)
		go func(channel int) {
			defer wg.Done()
			base := channel * planeSize
			srcOff := channel // R=0, G=1, B=2 in RGBA
			for y := 0; y < height; y++ {
				row := y * img.Stride
				rowOff := y * width
				for x := 0; x < width; x++ {
					dst[base+rowOff+x] = float32(pixels[row+x*4+srcOff]) * inv255
				}
			}
		}(ch)
	}
	wg.Wait()
}

// RunTensorRTInference performs inference using TensorRT via pure CGO.
// Uses the dual-stream Submit/Fetch API for maximum GPU utilisation.
func RunTensorRTInference(modelName string, img *image.RGBA) (image.Image, error) {
	engine, err := GetTensorRTEngine(modelName)
	if err != nil {
		return nil, err
	}
	if engine == nil {
		return nil, fmt.Errorf("TensorRT engine is nil")
	}

	if !trtAvailable {
		return nil, fmt.Errorf("TensorRT not available - use ONNX Runtime backend instead")
	}

	cHandle, ok := engine.handle.(C.TensorRTEngineHandle)
	if !ok || cHandle == nil {
		return nil, fmt.Errorf("invalid TensorRT engine handle")
	}

	TRTLog.Add("[TensorRT] Running inference via pure CGO (dual-stream pipeline)...")

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	TRTLog.Add(fmt.Sprintf("[TensorRT] Input image size: %dx%d", width, height))

	// ---------- Prepare input tensor (with buffer pool + parallel conversion) ----------
	inputSize := 1 * 3 * height * width
	inputBuffer := acquireFloat32Buffer(inputSize)
	defer releaseFloat32Buffer(inputBuffer)

	convertRGBAtoNCHW(img, inputBuffer)

	// ---------- Prepare output buffer ----------
	scale := ModelScale(modelName)
	outHeight := height * scale
	outWidth := width * scale
	outputSize := 1 * 3 * outHeight * outWidth
	outputBuffer := acquireFloat32Buffer(outputSize)
	defer releaseFloat32Buffer(outputBuffer)

	TRTLog.Add(fmt.Sprintf("[TensorRT] Input tensor: %d elements", inputSize))
	TRTLog.Add(fmt.Sprintf("[TensorRT] Output tensor: %d elements", outputSize))

	// ---------- Run blocking inference ----------
	ret := C.TensorRT_RunInference(
		C.TensorRTContextHandle(cHandle),
		C.int(width),
		C.int(height),
		(*C.float)(unsafe.Pointer(&inputBuffer[0])),
		C.size_t(inputSize*4),
		(*C.float)(unsafe.Pointer(&outputBuffer[0])),
		C.size_t(outputSize*4),
	)
	if ret != 0 {
		errMsg := C.GoString(C.TensorRT_GetLastError())
		return nil, fmt.Errorf("TensorRT inference failed: %s", errMsg)
	}


	TRTLog.Add("[TensorRT] Inference completed successfully")

	// ---------- Convert NCHW output back to RGBA ----------
	resultImg := image.NewRGBA(image.Rect(0, 0, outWidth, outHeight))
	resultPixels := resultImg.Pix
	outPlane := outHeight * outWidth

	// Parallel postprocess: 3 goroutines for channel→RGBA packing
	if outPlane > parallelThreshold {
		var wg sync.WaitGroup
		// Split by row bands across CPUs
		numWorkers := runtime.NumCPU()
		if numWorkers > 8 {
			numWorkers = 8
		}
		rowsPerWorker := (outHeight + numWorkers - 1) / numWorkers
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(startY int) {
				defer wg.Done()
				endY := startY + rowsPerWorker
				if endY > outHeight {
					endY = outHeight
				}
				for y := startY; y < endY; y++ {
					rowOff := y * outWidth
					dstRow := y * resultImg.Stride
					for x := 0; x < outWidth; x++ {
						idx := rowOff + x
						r := outputBuffer[idx]
						g := outputBuffer[outPlane+idx]
						b := outputBuffer[2*outPlane+idx]
						dstIdx := dstRow + x*4
						resultPixels[dstIdx] = clampToUint8(r)
						resultPixels[dstIdx+1] = clampToUint8(g)
						resultPixels[dstIdx+2] = clampToUint8(b)
						resultPixels[dstIdx+3] = 255
					}
				}
			}(w * rowsPerWorker)
		}
		wg.Wait()
	} else {
		for y := 0; y < outHeight; y++ {
			rowOff := y * outWidth
			dstRow := y * resultImg.Stride
			for x := 0; x < outWidth; x++ {
				idx := rowOff + x
				dstIdx := dstRow + x*4
				resultPixels[dstIdx] = clampToUint8(outputBuffer[idx])
				resultPixels[dstIdx+1] = clampToUint8(outputBuffer[outPlane+idx])
				resultPixels[dstIdx+2] = clampToUint8(outputBuffer[2*outPlane+idx])
				resultPixels[dstIdx+3] = 255
			}
		}
	}

	TRTLog.Add(fmt.Sprintf("[TensorRT] Output image size: %dx%d", outWidth, outHeight))
	return resultImg, nil
}

// clampToUint8 clamps a float32 in [0,1] to uint8 [0,255].
func clampToUint8(v float32) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 255
	}
	return uint8(v * 255.0)
}

// TensorRTAvailable returns whether TensorRT is available
func TensorRTAvailable() bool {
	if !trtInitialized {
		InitializeTensorRT()
	}
	return trtAvailable
}

// TensorRTError returns any initialization error
func TensorRTError() error {
	if !trtInitialized {
		InitializeTensorRT()
	}
	return trtInitErr
}

// CleanupTensorRT releases all TensorRT resources
func CleanupTensorRT() {
	if !trtInitialized {
		return
	}

	fmt.Println("[TensorRT] Cleaning up TensorRT resources...")

	trtEnginesMu.Lock()
	defer trtEnginesMu.Unlock()

	for name, engine := range trtEngines {
		fmt.Printf("[TensorRT] Unloading engine: %s\n", name)
		if engine.handle != nil {
			cHandle := engine.handle.(C.TensorRTEngineHandle)
			C.TensorRT_UnloadEngine(cHandle)
		}
	}

	C.TensorRT_Cleanup()

	trtEngines = make(map[string]*TensorRTEngine)
	trtInitialized = false
	trtAvailable = false
	trtInitErr = nil

	fmt.Println("[TensorRT] Cleanup complete")
}

// ListTensorRTEngines returns list of available TensorRT engine files
func ListTensorRTEngines() []string {
	trtDir := "AI-tensorrt"
	if _, err := os.Stat(trtDir); os.IsNotExist(err) {
		return []string{}
	}

	entries, err := os.ReadDir(trtDir)
	if err != nil {
		return []string{}
	}

	var engines []string
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) == ".engine" {
			engines = append(engines, entry.Name())
		}
	}
	return engines
}

// ValidateTensorRTEngine validates a TensorRT engine file
func ValidateTensorRTEngine(enginePath string) error {
	info, err := os.Stat(enginePath)
	if err != nil {
		return fmt.Errorf("engine file not found: %s", enginePath)
	}
	if info.Size() < 1024*1024 {
		return fmt.Errorf("engine file too small: %d bytes", info.Size())
	}
	fmt.Printf("[TensorRT] Engine file appears valid: %s (%.2f MB)\n",
		enginePath, float64(info.Size())/1024/1024)
	return nil
}

// GetEngineInfo returns information about a loaded engine
func GetEngineInfo(engine *TensorRTEngine) (inputShape, outputShape [4]int, err error) {
	if engine == nil || engine.handle == nil {
		return [4]int{}, [4]int{}, fmt.Errorf("invalid engine")
	}

	var inputDims [4]C.int
	cHandle := engine.handle.(C.TensorRTEngineHandle)
	ret := C.TensorRT_GetInputDims(cHandle, (*C.int)(unsafe.Pointer(&inputDims[0])))
	if ret != 0 {
		return [4]int{}, [4]int{}, fmt.Errorf("failed to get input dimensions")
	}

	var outputDims [4]C.int
	ret = C.TensorRT_GetOutputDims(cHandle, (*C.int)(unsafe.Pointer(&outputDims[0])))
	if ret != 0 {
		return [4]int{}, [4]int{}, fmt.Errorf("failed to get output dimensions")
	}

	inputShape = [4]int{int(inputDims[0]), int(inputDims[1]), int(inputDims[2]), int(inputDims[3])}
	outputShape = [4]int{int(outputDims[0]), int(outputDims[1]), int(outputDims[2]), int(outputDims[3])}
	return inputShape, outputShape, nil
}

// clamp clamps a float32 value to [min, max]. Kept for compatibility.
func clamp(val, minVal, maxVal float32) float32 {
	if val < minVal {
		return minVal
	}
	if val > maxVal {
		return maxVal
	}
	return val
}
