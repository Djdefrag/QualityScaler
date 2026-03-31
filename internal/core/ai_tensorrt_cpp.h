#ifndef AI_TENSORRT_CPP_H
#define AI_TENSORRT_CPP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque handle for TensorRT Engine (contains context + dual-slot pipeline)
typedef void* TensorRTEngineHandle;

// Opaque handle for TensorRT Context (alias for Engine handle)
typedef void* TensorRTContextHandle;

// Opaque handle for GPU buffers (unused, kept for ABI compatibility)
typedef void* TensorRTBufferHandle;

// Error callback function type
typedef void (*ErrorCallback)(const char* error);

#ifdef _WIN32
#  define TENSORRT_API __declspec(dllexport)
#else
#  define TENSORRT_API
#endif

// Initialize TensorRT / CUDA runtime.
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_Initialize();

// Cleanup TensorRT runtime.
TENSORRT_API void TensorRT_Cleanup();

// Load a TensorRT engine from file.
// Allocates dual-slot pinned-memory pipeline internally.
// Returns engine handle on success, NULL on failure.
TENSORRT_API TensorRTEngineHandle TensorRT_LoadEngine(const char* enginePath);

// Unload engine and free all resources (pinned memory, GPU buffers, streams).
TENSORRT_API void TensorRT_UnloadEngine(TensorRTEngineHandle engine);

// Get engine input dimensions (NCHW, 4 ints).
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_GetInputDims(TensorRTEngineHandle engine, int* dims);

// Get engine output dimensions (NCHW, 4 ints).
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_GetOutputDims(TensorRTEngineHandle engine, int* dims);

// Create execution context — context is embedded in the engine; returns same handle.
TENSORRT_API TensorRTContextHandle TensorRT_CreateContext(TensorRTEngineHandle engine);

// Destroy context (no-op; destroyed with engine).
TENSORRT_API void TensorRT_DestroyContext(TensorRTContextHandle context);

// ---------------------------------------------------------------------------
// Non-blocking Submit / Fetch API  (preferred for high-throughput pipelines)
// ---------------------------------------------------------------------------

// Submit one inference job to the GPU asynchronously.
//   input      : float32 NCHW tensor [1,3,height,width] values in [0,1]
//   inputSize  : size of input  buffer in bytes
//   outputSize : size of output buffer in bytes (caller pre-calculates)
//   slotIdOut  : receives the slot index to pass to FetchInference
//
// The call copies input into a pinned buffer, enqueues H2D + inference + D2H
// on an internal CUDA stream, and returns immediately (non-blocking).
// The caller must later call TensorRT_FetchInference to retrieve results.
//
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_SubmitInference(TensorRTContextHandle context,
                                          int width, int height,
                                          const float* input, size_t inputSize,
                                          size_t outputSize,
                                          int* slotIdOut);

// Wait for the submitted job in slotId to finish, copy results into output.
//   output     : caller-allocated float32 buffer (at least outputSize bytes)
//   outputSize : bytes to copy from pinned output buffer
//
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_FetchInference(TensorRTContextHandle context,
                                         int slotId,
                                         float* output, size_t outputSize);

// ---------------------------------------------------------------------------
// Blocking (legacy) inference — kept for backward compatibility.
// Internally calls Submit then Fetch on slot-0.
//   input      : float32 NCHW tensor [1,3,height,width] values in [0,1]
//   inputSize  : size of input  buffer in bytes
//   output     : float32 NCHW tensor [1,3,outH,outW]   values in [0,1]
//   outputSize : size of output buffer in bytes (must be pre-allocated)
// Returns 0 on success, -1 on failure.
// ---------------------------------------------------------------------------
TENSORRT_API int TensorRT_RunInference(TensorRTContextHandle context,
                                       int width, int height,
                                       const float* input, size_t inputSize,
                                       float* output, size_t outputSize);

// Get last error message (thread-local per-process; not fully thread-safe for multi-engine errors).
TENSORRT_API const char* TensorRT_GetLastError();

#ifdef __cplusplus
}
#endif

#endif // AI_TENSORRT_CPP_H
