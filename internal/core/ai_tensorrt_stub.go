//go:build !tensorrt && !full
// +build !tensorrt,!full

package core

import (
	"image"
)

// Stub implementations for TensorRT when not built with tensorrt tag
// These allow the rest of the code to compile without TensorRT support

// InitializeTensorRT initializes the TensorRT backend (stub - always fails)
func InitializeTensorRT() bool {
	trtInitialized = true
	trtAvailable = false
	trtInitErr = nil
	return false
}

// GetTensorRTEngine loads or creates a TensorRT engine for given model (stub - always returns error)
func GetTensorRTEngine(modelName string) (*TensorRTEngine, error) {
	return nil, trtInitErr
}

// RunTensorRTInference performs inference using TensorRT (stub - always returns error)
func RunTensorRTInference(modelName string, img *image.RGBA) (image.Image, error) {
	return nil, trtInitErr
}

// TensorRTAvailable returns whether TensorRT is available (stub - always false)
func TensorRTAvailable() bool {
	if !trtInitialized {
		InitializeTensorRT()
	}
	return trtAvailable
}

// TensorRTError returns any initialization error (stub - always nil)
func TensorRTError() error {
	if !trtInitialized {
		InitializeTensorRT()
	}
	return trtInitErr
}

// CleanupTensorRT releases all TensorRT resources (stub - no-op)
func CleanupTensorRT() {
}

// ListTensorRTEngines returns list of available TensorRT engine files (stub - always empty)
func ListTensorRTEngines() []string {
	return []string{}
}

// ValidateTensorRTEngine validates a TensorRT engine file (stub - always returns error)
func ValidateTensorRTEngine(enginePath string) error {
	return trtInitErr
}

// GetEngineInfo returns information about a loaded engine (stub - returns empty shapes)
func GetEngineInfo(engine *TensorRTEngine) (inputShape, outputShape [4]int, err error) {
	return [4]int{}, [4]int{}, trtInitErr
}
