package core

import (
	"fmt"
	"image"
	stddraw "image/draw"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortInitialized bool
	ortLibPath     string
	aiBackend      = "CPU"
	aiInitErr      string
	aiCudaErr      string
	aiMissingDLLs  []string
	aiSessions     = make(map[string]*ort.DynamicAdvancedSession)
	aiSessionsMu   sync.RWMutex
	floatPoolsMu   sync.Mutex
	floatPools     = make(map[int]*sync.Pool)
	// Limit pool size to prevent memory bloat
	maxFloatPools = 128
	floatPoolCount int
)

func acquireFloat32Buffer(size int) []float32 {
	if size <= 0 {
		return nil
	}

	floatPoolsMu.Lock()
	pool, ok := floatPools[size]
	if !ok {
		// Create new pool only if under limit
		if floatPoolCount < maxFloatPools {
			pool = &sync.Pool{New: func() interface{} { return make([]float32, size) }}
			floatPools[size] = pool
			floatPoolCount++
		} else {
			// If pool limit reached, don't pool this size
			floatPoolsMu.Unlock()
			return make([]float32, size)
		}
	}
	floatPoolsMu.Unlock()

	v := pool.Get()
	b, _ := v.([]float32)
	if cap(b) < size {
		return make([]float32, size)
	}
	return b[:size]
}

func releaseFloat32Buffer(buf []float32) {
	if len(buf) == 0 {
		return
	}

	size := cap(buf)

	floatPoolsMu.Lock()
	pool, ok := floatPools[size]
	if !ok {
		// No pool exists for this size, just discard
		floatPoolsMu.Unlock()
		return
	}
	floatPoolsMu.Unlock()

	buf = buf[:size]
	pool.Put(buf)
}

func AIBackend() string {
	return aiBackend
}

func AIStatusDetail() string {
	if len(aiMissingDLLs) > 0 {
		return "missing runtime DLLs: " + strings.Join(aiMissingDLLs, ", ")
	}
	if aiInitErr != "" {
		return aiInitErr
	}
	if aiCudaErr != "" {
		return aiCudaErr
	}
	return "OK"
}

func MissingRuntimeDLLs() []string {
	out := make([]string, len(aiMissingDLLs))
	copy(out, aiMissingDLLs)
	return out
}

func WarmupAISession(modelName string) error {
	if trtAvailable {
		// No warmup needed for TensorRT or it's handled differently
		return nil
	}
	_, err := GetAISession(modelName)
	return err
}

func requiredRuntimeDLLs() []string {
	if runtime.GOOS != "windows" {
		return nil
	}
	// Only require main DLL and shared provider; CUDA is optional
	return []string{
		"onnxruntime.dll",
		"onnxruntime_providers_shared.dll",
	}
}

func findRuntimeDLL(name string) (string, bool) {
	// Try multiple search paths
	candidates := []string{
		name,
		filepath.Join("Assets", name),
		filepath.Join("..", "Assets", name),
		filepath.Join("..", "..", "Assets", name),
	}
	for _, p := range candidates {
		if absPath, err := filepath.Abs(p); err == nil {
			if info, err := os.Stat(absPath); err == nil && !info.IsDir() {
				return absPath, true
			}
		}
	}
	return "", false
}

func fileInfo(path string) int64 {
	if info, err := os.Stat(path); err == nil {
		return info.Size()
	}
	return 0
}

// InitializeAI attempts to load the ONNX Runtime library.
// Returns true if successful, false otherwise.
func InitializeAI() bool {
	if ortInitialized {
		return true
	}

	// First try to initialize TensorRT (fastest)
	trtInitialized = false
	trtAvailable = false
	if InitializeTensorRT() {
		if trtAvailable {
			ortInitialized = true
			return true
		}
	}

	// Fallback to ONNX Runtime
	aiMissingDLLs = aiMissingDLLs[:0]
	for _, dll := range requiredRuntimeDLLs() {
		if _, ok := findRuntimeDLL(dll); !ok {
			aiMissingDLLs = append(aiMissingDLLs, dll)
		}
	}

	libName := "onnxruntime.dll"
	if runtime.GOOS == "linux" {
		libName = "libonnxruntime.so"
	} else if runtime.GOOS == "darwin" {
		libName = "libonnxruntime.dylib"
	}

	// Find DLL before setting PATH
	if p, ok := findRuntimeDLL(libName); ok {
		ortLibPath = p
	}
	if ortLibPath == "" {
		aiInitErr = fmt.Sprintf("missing %s in app root or Assets", libName)
		fmt.Printf("AI Error: %s not found in app root or Assets. Falling back to CPU resize.\n", libName)
		return false
	}

	// Set PATH to include DLL directory BEFORE loading
	if absLib, err := filepath.Abs(ortLibPath); err == nil {
		ortDir := filepath.Dir(absLib)
		fmt.Printf("AI: DLL directory: %s\n", ortDir)
		pathEnv := os.Getenv("PATH")
		// Add DLL directory to PATH if not already there
		if !strings.Contains(strings.ToLower(pathEnv), strings.ToLower(ortDir)) {
			_ = os.Setenv("PATH", ortDir+";"+pathEnv)
		}
		// Also try to add Assets directory directly
		assetsDir := filepath.Join(filepath.Dir(absLib), "Assets")
		if _, err := os.Stat(assetsDir); err == nil {
			if !strings.Contains(strings.ToLower(pathEnv), strings.ToLower(assetsDir)) {
				_ = os.Setenv("PATH", assetsDir+";"+os.Getenv("PATH"))
			}
		}
	}

	// Verify DLL exists and is accessible
	if _, err := os.Stat(ortLibPath); err != nil {
		aiInitErr = fmt.Sprintf("cannot access %s: %v", ortLibPath, err)
		fmt.Printf("AI Error: %v\n", aiInitErr)
		return false
	}

	fmt.Printf("AI: Attempting to load ONNX Runtime from: %s\n", ortLibPath)

	// On Windows, try to preload DLLs using LoadLibrary
	if runtime.GOOS == "windows" {
		// Load required DLLs in dependency order
		requiredDLLs := []string{
			"onnxruntime_providers_shared.dll",
			"onnxruntime.dll",
		}
		for _, dll := range requiredDLLs {
			if _, ok := findRuntimeDLL(dll); ok {
				fmt.Printf("AI: Preloading %s...\n", dll)
				// Note: We don't call loadDLL() here as onnxruntime-go
				// will handle the final load properly
			}
		}
	}

	ort.SetSharedLibraryPath(ortLibPath)

	if err := ort.InitializeEnvironment(); err != nil {
		aiInitErr = err.Error()
		// Provide more helpful error message
		fmt.Printf("AI Error: Failed to initialize ONNX environment: %v\n", err)
		fmt.Printf("  Library path: %s\n", ortLibPath)
		fmt.Printf("  File size: %d bytes\n", fileInfo(ortLibPath))
		if len(aiMissingDLLs) > 0 {
			fmt.Printf("  Missing dependencies: %v\n", aiMissingDLLs)
		}
		fmt.Printf("  Tip: Ensure Visual C++ Redistributable 2015-2022 is installed\n")
		fmt.Printf("  Falling back to CPU resize.\n")
		return false
	}

	ortInitialized = true
	aiBackend = "CPU (ONNX)"
	aiInitErr = ""
	fmt.Printf("AI: ONNX Runtime initialized from %s\n", ortLibPath)
	return true
}

// GetAIBackend returns the current AI backend being used
func GetAIBackend() string {
	if trtAvailable {
		return "TensorRT (GPU)"
	}
	return aiBackend
}

func GetAISession(modelName string) (*ort.DynamicAdvancedSession, error) {
	if !ortInitialized {
		return nil, fmt.Errorf("ONNX runtime not initialized")
	}

	aiSessionsMu.RLock()
	if session, ok := aiSessions[modelName]; ok {
		aiSessionsMu.RUnlock()
		return session, nil
	}
	aiSessionsMu.RUnlock()

	aiSessionsMu.Lock()
	defer aiSessionsMu.Unlock()

	if session, ok := aiSessions[modelName]; ok {
		return session, nil
	}

	session, err := createSession(modelName)
	if err != nil {
		return nil, err
	}
	aiSessions[modelName] = session
	return session, nil
}

// NewAISessionForWorker creates an uncached session for dedicated worker goroutines.
func NewAISessionForWorker(modelName string) (*ort.DynamicAdvancedSession, error) {
	if !ortInitialized {
		return nil, fmt.Errorf("ONNX runtime not initialized")
	}
	return createSession(modelName)
}

func createSession(modelName string) (*ort.DynamicAdvancedSession, error) {

	modelPath := filepath.Join("AI-onnx", modelName+"_fp16.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", modelPath)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()

	_ = options.SetCpuMemArena(true)
	_ = options.SetMemPattern(true)
	_ = options.SetInterOpNumThreads(1)
	_ = options.SetIntraOpNumThreads(1) // Set to 1 to match Python's ORT_SEQUENTIAL

	// Try CUDA provider first; if this fails, ORT keeps CPU fallback.
	if cudaOpts, err := ort.NewCUDAProviderOptions(); err == nil {
		defer cudaOpts.Destroy()
		_ = cudaOpts.Update(map[string]string{
			"device_id":                  "0",
			"arena_extend_strategy":      "kNextPowerOfTwo",
			"cudnn_conv_algo_search":     "HEURISTIC", // Use HEURISTIC for faster warmup (was EXHAUSTIVE)
			"do_copy_in_default_stream":  "1",
			"cudnn_conv_use_max_workspace": "1",        // Use maximum workspace for better performance
			"cudnn_conv_bias_prefetch":    "1",        // Pre-fetch bias data
			"cudnn_conv_allow_tf32":       "1",        // Use TF32 for faster computation on Ampere+ GPUs
		})
		if err := options.AppendExecutionProviderCUDA(cudaOpts); err == nil {
			aiBackend = "CUDA (GPU 0)"
			aiCudaErr = ""
			fmt.Println("AI: CUDA provider enabled with optimized settings (max workspace, bias prefetch, TF32).")
		} else {
			aiCudaErr = err.Error()
			fmt.Printf("AI Warning: CUDA provider unavailable, fallback to CPU ONNX: %v\n", err)
		}
	} else {
		aiCudaErr = err.Error()
		fmt.Printf("AI Warning: Failed to create CUDA options, fallback to CPU ONNX: %v\n", err)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"input"}, []string{"output"}, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create session for %s: %w", modelName, err)
	}
	return session, nil
}

// RunAIInference performs AI upscaling on the given image.
// Automatically uses TensorRT if available, otherwise falls back to ONNX Runtime.
func RunAIInference(img image.Image, modelName string) (image.Image, error) {
	// Try TensorRT first if available
	if trtAvailable {
		imgRGBA := toRGBA(img)
		result, err := RunTensorRTInference(modelName, imgRGBA)
		if err == nil {
			TRTLog.Add(fmt.Sprintf("[TensorRT] Inference completed for %s", modelName))
			return result, nil
		}
		fmt.Printf("[TensorRT] Inference failed, falling back to ONNX: %v\n", err)
	}

	// Fallback to ONNX Runtime
	var session *ort.DynamicAdvancedSession
	if !trtAvailable {
		var err error
		session, err = GetAISession(modelName)
		if err != nil {
			return nil, err
		}
	}
	if trtAvailable {
		// If TensorRT was available but failed, we just returned error earlier or fallback here.
		// Wait, if trtAvailable is true but inference failed, we might still want ONNX.
		// But in this implementation, if trtAvailable is true, we didn't init ONNX!
		// Let's just return error if TRT fails and ONNX is not initialized.
		if !ortInitialized {
			return nil, fmt.Errorf("TensorRT inference failed and ONNX Runtime is not initialized")
		}
		var err error
		session, err = GetAISession(modelName)
		if err != nil {
			return nil, err
		}
	}

	return RunAIInferenceWithSession(session, img, modelName)
}

// RunAIInferenceWithSession performs AI upscaling on the given image using the provided session.
// For images with alpha channel, it separates RGB and Alpha, processes RGB with AI, and upscales Alpha with fast interpolation.
func RunAIInferenceWithSession(session *ort.DynamicAdvancedSession, img image.Image, modelName string) (image.Image, error) {
	if session == nil && !trtAvailable {
		return nil, fmt.Errorf("AI session is nil and TensorRT is not available")
	}

	// Check if image has alpha channel
	imgRGBA := toRGBA(img)

	// Check if image has meaningful alpha channel (not all 255)
	hasAlpha := checkAlphaChannel(imgRGBA)

	if !hasAlpha {
	// No alpha channel, use the standard fast path
	var result image.Image
	var err error
	if session != nil {
		result, err = runAIInferenceRGBOnly(session, imgRGBA, modelName)
	} else if trtAvailable {
		// Fallback to auto inference which handles TensorRT natively
		// RunAIInference Auto logic separates alpha, so call it directly
		result, err = RunAIInference(imgRGBA, modelName)
	} else {
		// Neither session nor TRT available
		return nil, fmt.Errorf("AI session is nil and TensorRT is not available")
	}
	return result, err
	}

	// Image has alpha channel - separate and process
	AILog.Add("[AIInference] Image has alpha channel - separating RGB and Alpha for optimization")
	return runAIInferenceWithAlpha(session, imgRGBA, modelName)
}

// checkAlphaChannel checks if the image has a meaningful (non-opaque) alpha channel
func checkAlphaChannel(img *image.RGBA) bool {
	// Sample check: check every 100th pixel to speed up
	stride := img.Stride
	height := img.Bounds().Dy()
	width := img.Bounds().Dx()
	step := max(100, (width*height)/1000) // Check at most 1000 pixels

	for y := 0; y < height; y += step {
		row := y * stride
		for x := 0; x < width; x += step {
			if img.Pix[row+x*4+3] != 255 {
				return true // Found non-opaque pixel
			}
		}
	}
	return false // All checked pixels are opaque
}

// runAIInferenceWithAlpha separates RGB and Alpha, processes them separately
func runAIInferenceWithAlpha(session *ort.DynamicAdvancedSession, img *image.RGBA, modelName string) (image.Image, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	scale := ModelScale(modelName)

	AILog.Add(fmt.Sprintf("[AIInference] Input size: %dx%d | Output size: %dx%d (x%d)", width, height, width*scale, height*scale, scale))

	// Step 1: Extract RGB image (strip alpha)
	rgbImg := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		srcRow := y * img.Stride
		dstRow := y * rgbImg.Stride
		for x := 0; x < width; x++ {
			srcIdx := srcRow + x*4
			dstIdx := dstRow + x*4
			rgbImg.Pix[dstIdx] = img.Pix[srcIdx]     // R
			rgbImg.Pix[dstIdx+1] = img.Pix[srcIdx+1] // G
			rgbImg.Pix[dstIdx+2] = img.Pix[srcIdx+2] // B
			rgbImg.Pix[dstIdx+3] = 255                // A (opaque)
		}
	}

	// Step 2: Process RGB with AI
	var rgbUpscaled image.Image
	var err error
	if session != nil {
		rgbUpscaled, err = runAIInferenceRGBOnly(session, rgbImg, modelName)
	} else if trtAvailable {
		// Fallback to TensorRT
		// RunAIInference checks for alpha again, but since alpha is 255 here, it will take the fast path
		rgbUpscaled, err = RunAIInference(rgbImg, modelName)
	} else {
		return nil, fmt.Errorf("AI session is nil and TensorRT is not available")
	}
	if err != nil {
		return nil, err
	}

	// Step 3: Upscale alpha channel with fast interpolation (Lanczos)
	outWidth := width * scale
	outHeight := height * scale
	alphaUpscaled := upsampleAlphaChannel(img, outWidth, outHeight)

	// Step 4: Merge RGB and Alpha
	result := image.NewRGBA(image.Rect(0, 0, outWidth, outHeight))
	rgbUpscaledRGBA := toRGBA(rgbUpscaled)
	for y := 0; y < outHeight; y++ {
		srcRow := y * rgbUpscaledRGBA.Stride
		alphaRow := y * alphaUpscaled.Stride
		dstRow := y * result.Stride
		for x := 0; x < outWidth; x++ {
			idx := x * 4
			dstIdx := dstRow + idx
			srcIdx := srcRow + idx
			alphaIdx := alphaRow + idx
			result.Pix[dstIdx] = rgbUpscaledRGBA.Pix[srcIdx]     // R
			result.Pix[dstIdx+1] = rgbUpscaledRGBA.Pix[srcIdx+1] // G
			result.Pix[dstIdx+2] = rgbUpscaledRGBA.Pix[srcIdx+2] // B
			result.Pix[dstIdx+3] = alphaUpscaled.Pix[alphaIdx+3] // A
		}
	}

	return result, nil
}

// upsampleAlphaChannel upscales the alpha channel using fast interpolation
func upsampleAlphaChannel(img *image.RGBA, outWidth, outHeight int) *image.RGBA {
	// Create a grayscale image from alpha channel
	grayImg := image.NewRGBA(img.Bounds())
	for i := range img.Pix {
		if (i+1)%4 == 0 {
			// Alpha channel
			grayImg.Pix[i] = img.Pix[i]
		} else {
			// RGB channels - set to same value as alpha
			grayImg.Pix[i] = img.Pix[(i/4)*4+3]
		}
	}

	// Use standard library resize (fast enough for single channel)
	result := image.NewRGBA(image.Rect(0, 0, outWidth, outHeight))
	scaleX := float64(outWidth) / float64(img.Bounds().Dx())
	scaleY := float64(outHeight) / float64(img.Bounds().Dy())

	for y := 0; y < outHeight; y++ {
		srcY := float64(y) / scaleY
		y0 := int(srcY)
		y1 := min(y0+1, img.Bounds().Dy()-1)
		fy := srcY - float64(y0)

		dstRow := y * result.Stride

		for x := 0; x < outWidth; x++ {
			srcX := float64(x) / scaleX
			x0 := int(srcX)
			x1 := min(x0+1, img.Bounds().Dx()-1)
			fx := srcX - float64(x0)

			// Bilinear interpolation
			dstIdx := dstRow + x*4

			// Get alpha values from 4 neighbors
			idx00 := y0*img.Stride + x0*4 + 3
			idx01 := y0*img.Stride + x1*4 + 3
			idx10 := y1*img.Stride + x0*4 + 3
			idx11 := y1*img.Stride + x1*4 + 3

			v00 := float64(img.Pix[idx00])
			v01 := float64(img.Pix[idx01])
			v10 := float64(img.Pix[idx10])
			v11 := float64(img.Pix[idx11])

			// Bilinear interpolation
			alpha := v00*(1-fx)*(1-fy) + v01*fx*(1-fy) + v10*(1-fx)*fy + v11*fx*fy

			result.Pix[dstIdx] = 255   // R
			result.Pix[dstIdx+1] = 255 // G
			result.Pix[dstIdx+2] = 255 // B
			result.Pix[dstIdx+3] = uint8(alpha) // A
		}
	}

	return result
}

// runAIInferenceRGBOnly performs AI upscaling on RGB-only images (no alpha)
func runAIInferenceRGBOnly(session *ort.DynamicAdvancedSession, img *image.RGBA, modelName string) (image.Image, error) {
	if session == nil {
		return nil, fmt.Errorf("runAIInferenceRGBOnly requires a valid ONNX session")
	}

	// Calculate scale factor
	scale := ModelScale(modelName)

	// 1. Preprocess: Image to Tensor (NCHW RGB float32 0-1)
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	inputData := acquireFloat32Buffer(1 * 3 * height * width)
	defer releaseFloat32Buffer(inputData)
	planeSize := height * width
	const inv255 = 1.0 / 255.0

	// Preprocess timing
	preprocessStart := time.Now()

	// Sequential preprocessing (simpler and often faster for this workload)
	for y := 0; y < height; y++ {
		row := y * img.Stride
		rowOffset := y * width
		for x := 0; x < width; x++ {
			px := row + x*4
			idx := rowOffset + x
			inputData[idx] = float32(img.Pix[px]) * inv255
			inputData[planeSize+idx] = float32(img.Pix[px+1]) * inv255
			inputData[2*planeSize+idx] = float32(img.Pix[px+2]) * inv255
		}
	}

	preprocessTime := time.Since(preprocessStart)

	// 2. Prepare Tensors
	// Input tensor
	inputShape := ort.NewShape(1, 3, int64(height), int64(width))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Output tensor
	// Based on RealESRGAN/BSRGAN/etc, the output shape is usually calculated exactly by scale.
	outHeight := height * scale
	outWidth := width * scale
	outputData := acquireFloat32Buffer(1 * 3 * outHeight * outWidth)
	defer releaseFloat32Buffer(outputData)
	outputShape := ort.NewShape(1, 3, int64(outHeight), int64(outWidth))

	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run Inference
	// session.Run expects []Value for inputs and outputs
	startTime := time.Now()
	
	// Add progress monitoring for long-running inferences
	var progressTimer *time.Timer
	var progressDone chan struct{}
	if width*height > 2000000 { // Only for large images (>2MP)
		progressDone = make(chan struct{})
		progressTimer = time.AfterFunc(5*time.Second, func() {
			select {
			case <-progressDone:
				return
			default:
				AILog.Add(fmt.Sprintf("[AIInference] Still processing large image (%dx%d = %0.2fMP)...",
					width, height, float64(width*height)/1000000))
			}
		})
		defer func() {
			if progressTimer != nil {
				progressTimer.Stop()
				close(progressDone)
			}
		}()
	}
	
	err = session.Run([]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor})
	inferTime := time.Since(startTime)
	
	if progressTimer != nil {
		progressTimer.Stop()
		close(progressDone)
	}

	AILog.Add(fmt.Sprintf("[AIInference] Preprocess: %v | Inference: %v | Postprocess: calculating | Total so far: %v",
		preprocessTime, inferTime, preprocessTime+inferTime))

	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// 3. Postprocess: Tensor to Image
	// Since we provided outputData to NewTensor, ONNX Runtime should write directly to it.
	// No need to call GetData() which would create an unnecessary copy.
	resultData := outputData

	postprocessStart := time.Now()

	outImg := image.NewRGBA(image.Rect(0, 0, outWidth, outHeight))
	outPlaneSize := outHeight * outWidth

	// Sequential postprocessing
	for y := 0; y < outHeight; y++ {
		row := y * outImg.Stride
		rowOffset := y * outWidth
		for x := 0; x < outWidth; x++ {
			idx := rowOffset + x
			rVal := resultData[idx]
			gVal := resultData[outPlaneSize+idx]
			bVal := resultData[2*outPlaneSize+idx]

			// Clamp and scale in one step
			px := row + x*4
			if rVal <= 0 {
				outImg.Pix[px] = 0
			} else if rVal >= 1 {
				outImg.Pix[px] = 255
			} else {
				outImg.Pix[px] = uint8(rVal * 255)
			}

			if gVal <= 0 {
				outImg.Pix[px+1] = 0
			} else if gVal >= 1 {
				outImg.Pix[px+1] = 255
			} else {
				outImg.Pix[px+1] = uint8(gVal * 255)
			}

			if bVal <= 0 {
				outImg.Pix[px+2] = 0
			} else if bVal >= 1 {
				outImg.Pix[px+2] = 255
			} else {
				outImg.Pix[px+2] = uint8(bVal * 255)
			}

			outImg.Pix[px+3] = 255
		}
	}

	postprocessTime := time.Since(postprocessStart)
	AILog.Add(fmt.Sprintf("[AIInference] Postprocess: %v | Total: %v (Preprocess: %v, Inference: %v, Postprocess: %v)",
		postprocessTime, preprocessTime+inferTime+postprocessTime, preprocessTime, inferTime, postprocessTime))

	return outImg, nil
}

func toRGBA(src image.Image) *image.RGBA {
	if rgba, ok := src.(*image.RGBA); ok {
		return rgba
	}
	b := src.Bounds()
	rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	stddraw.Draw(rgba, rgba.Bounds(), src, b.Min, stddraw.Src)
	return rgba
}
