package core

import (
	"sync"
)

// TensorRT Engine state (shared between real and stub implementations)
var (
	trtInitialized bool
	trtAvailable   bool
	trtInitErr     error
	trtEngines     map[string]*TensorRTEngine
	trtEnginesMu   sync.RWMutex
)

// TensorRTEngine wraps a TensorRT engine handle
// In real implementation (tensorrt tag), handle is C.TensorRTEngineHandle
// In stub implementation (!tensorrt), this struct is not used for actual inference
type TensorRTEngine struct {
	handle      interface{} // Holds C.TensorRTEngineHandle when built with tensorrt tag
	InputShape  [4]int
	OutputShape [4]int
}
