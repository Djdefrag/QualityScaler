#ifndef AI_TENSORRT_CPP_H
#define AI_TENSORRT_CPP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque handle for TensorRT Engine
typedef void* TensorRTEngineHandle;

// Opaque handle for TensorRT Context
typedef void* TensorRTContextHandle;

// Opaque handle for GPU buffers
typedef void* TensorRTBufferHandle;

// Error callback function type
typedef void (*ErrorCallback)(const char* error);

#ifdef _WIN32
#  define TENSORRT_API __declspec(dllexport)
#else
#  define TENSORRT_API
#endif

// Initialize TensorRT runtime
// Returns 0 on success, -1 on failure
TENSORRT_API int TensorRT_Initialize();

// Cleanup TensorRT runtime
TENSORRT_API void TensorRT_Cleanup();

// Load a TensorRT engine from file
// Returns engine handle on success, NULL on failure
TENSORRT_API TensorRTEngineHandle TensorRT_LoadEngine(const char* enginePath);

// Unload an engine and free resources
TENSORRT_API void TensorRT_UnloadEngine(TensorRTEngineHandle engine);

// Get engine input dimensions (NCHW)
// Returns 0 on success, -1 on failure
TENSORRT_API int TensorRT_GetInputDims(TensorRTEngineHandle engine, int* dims);

// Get engine output dimensions (NCHW)
// Returns 0 on success, -1 on failure
TENSORRT_API int TensorRT_GetOutputDims(TensorRTEngineHandle engine, int* dims);

// Create execution context for an engine
// Returns context handle on success, NULL on failure
TENSORRT_API TensorRTContextHandle TensorRT_CreateContext(TensorRTEngineHandle engine);

// Destroy execution context
TENSORRT_API void TensorRT_DestroyContext(TensorRTContextHandle context);

// Run inference on single image
// input: float32 NCHW tensor [1, 3, height, width] values in range [0, 1]
// inputSize: size of input buffer in bytes
// output: float32 NCHW tensor [1, 3, outHeight, outWidth] values in range [0, 1]
// outputSize: size of output buffer in bytes (must be allocated by caller)
// Returns 0 on success, -1 on failure
TENSORRT_API int TensorRT_RunInference(TensorRTContextHandle context,
                          int width, int height,
                          const float* input, size_t inputSize,
                          float* output, size_t outputSize);

// Get last error message
TENSORRT_API const char* TensorRT_GetLastError();

#ifdef __cplusplus
}
#endif

#endif // AI_TENSORRT_CPP_H
