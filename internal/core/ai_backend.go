package core

import (
	"fmt"
	"image"
	"sync"
)

// AIBackendType represents the AI inference backend
type AIBackendType int

const (
	BackendAuto     AIBackendType = iota // Auto-select best available
	BackendONNX                          // ONNX Runtime
	BackendTensorRT                      // TensorRT
)

var (
	// Current backend selection
	currentBackend   AIBackendType
	preferredBackend AIBackendType = BackendAuto
	backendLock      sync.RWMutex
)

// SetPreferredBackend sets the preferred AI backend
func SetPreferredBackend(backend AIBackendType) {
	backendLock.Lock()
	defer backendLock.Unlock()
	preferredBackend = backend
}

// GetPreferredBackend returns the preferred AI backend
func GetPreferredBackend() AIBackendType {
	backendLock.RLock()
	defer backendLock.RUnlock()
	return preferredBackend
}

// GetCurrentBackend returns the currently active AI backend
func GetCurrentBackend() AIBackendType {
	backendLock.RLock()
	defer backendLock.RUnlock()
	return currentBackend
}

// BackendName returns the human-readable name of a backend
func BackendName(backend AIBackendType) string {
	switch backend {
	case BackendTensorRT:
		return "TensorRT (GPU)"
	case BackendONNX:
		return "ONNX Runtime"
	default:
		return "Auto"
	}
}

// RunAIInferenceAuto performs AI inference using the best available backend
func RunAIInferenceAuto(img image.Image, modelName string) (image.Image, error) {
	return RunAIInferenceWithBackend(img, modelName, preferredBackend)
}

// RunAIInferenceWithBackend performs AI inference using specified backend
func RunAIInferenceWithBackend(img image.Image, modelName string, backend AIBackendType) (image.Image, error) {
	// Auto-select backend
	if backend == BackendAuto {
		backend = selectBestBackend()
	}

	backendLock.Lock()
	currentBackend = backend
	backendLock.Unlock()

	switch backend {
	case BackendTensorRT:
		if TensorRTAvailable() {
			imgRGBA := toRGBA(img)
			result, err := RunTensorRTInference(modelName, imgRGBA)
			if err != nil {
				fmt.Printf("[TensorRT] inference error, falling back to ONNX: %v\n", err)
				return RunAIInference(img, modelName)
			}
			return result, nil
		}
		// TRT not available, fall back to ONNX Runtime
		return RunAIInference(img, modelName)

	case BackendONNX:
		return RunAIInference(img, modelName)

	default:
		return nil, fmt.Errorf("unknown AI backend: %d", backend)
	}
}

// selectBestBackend selects the best available backend
func selectBestBackend() AIBackendType {
	// Prefer TensorRT if available, otherwise ONNX Runtime
	if TensorRTAvailable() {
		return BackendTensorRT
	}
	return BackendONNX
}

// InitializeAllBackends initializes all AI backends
func InitializeAllBackends() bool {
	// Try to initialize all backends
	onnxOk := InitializeAI()
	trtOk := InitializeTensorRT()

	// Log backend availability
	fmt.Printf("AI Backend Status:\n")
	fmt.Printf("  ONNX Runtime: %v\n", onnxOk)
	fmt.Printf("  TensorRT:     %v\n", trtOk)

	// Select best backend
	if trtOk {
		fmt.Printf("  Selected:      TensorRT (high performance)\n")
	} else if onnxOk {
		fmt.Printf("  Selected:      ONNX Runtime\n")
	} else {
		fmt.Printf("  Selected:      None (CPU resize only)\n")
		return false
	}

	return true
}

// RunAIInferenceBatchAuto performs batch AI inference using best available backend
func RunAIInferenceBatchAuto(imgs []*image.RGBA, modelName string) ([]image.Image, error) {
	backend := preferredBackend
	if backend == BackendAuto {
		backend = selectBestBackend()
	}

	results := make([]image.Image, len(imgs))

	switch backend {
	case BackendTensorRT:
		if TensorRTAvailable() {
			_, err := GetTensorRTEngine(modelName)
			if err == nil {
				for i, img := range imgs {
					result, err := RunTensorRTInference(modelName, img)
					if err != nil {
						return nil, fmt.Errorf("TensorRT inference failed on image %d: %w", i, err)
					}
					results[i] = result
				}
				return results, nil
			}
			fmt.Printf("TensorRT engine error: %v, falling back to ONNX\n", err)
		}
		// Fall through to ONNX
		fallthrough

	case BackendONNX:
		session, err := GetAISession(modelName)
		if err != nil {
			return nil, err
		}
		for i, img := range imgs {
			result, err := RunAIInferenceWithSession(session, img, modelName)
			if err != nil {
				return nil, err
			}
			results[i] = result
		}
		return results, nil

	default:
		return nil, fmt.Errorf("unknown backend: %d", backend)
	}
}
