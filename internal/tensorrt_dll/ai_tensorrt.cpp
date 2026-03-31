// TensorRT 10.x compatible implementation
#include "ai_tensorrt_cpp.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <memory>
#include <vector>
#include <cstring>
#include <mutex>

using namespace nvinfer1;

// Logger for TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;
static std::string gLastError;

// Smart pointer deleters for TensorRT 10.x
struct TrtDestroyer {
    template <class T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

template <class T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer>;

// Helper struct to hold engine and context together
struct TensorRTEngineObj {
    TrtUniquePtr<ICudaEngine> engine;
    TrtUniquePtr<IExecutionContext> context;
    void* gpuInputBuffer = nullptr;
    void* gpuOutputBuffer = nullptr;
    int inputDims[4] = {0};
    int outputDims[4] = {0};
    size_t inputSize = 0;
    size_t outputSize = 0;
    size_t currentAllocatedInputSize = 0;
    size_t currentAllocatedOutputSize = 0;
    cudaStream_t stream = nullptr;
    std::string inputName;
    std::string outputName;
    std::mutex mu;  // per-engine mutex: IExecutionContext is not thread-safe
};

// Set error message
static void setError(const char* msg) {
    gLastError = msg;
}

// Read engine file
static std::vector<char> readEngineFile(const char* enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        setError("Failed to open engine file");
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        setError("Failed to read engine file");
        return {};
    }

    return buffer;
}

extern "C" {

#ifdef __MINGW32__
// Dummy implementations for MSVC runtime functions required by CUDA 13.1 import libs under MinGW
void __GSHandlerCheck() {}
void __security_check_cookie(uintptr_t cookie) {}
uintptr_t __security_cookie = 0;
#endif

int TensorRT_Initialize() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        setError("Failed to initialize CUDA");
        return -1;
    }
    return 0;
}

void TensorRT_Cleanup() {
    // Global cleanup if needed
}

TensorRTEngineHandle TensorRT_LoadEngine(const char* enginePath) {
    // Read engine file
    auto engineData = readEngineFile(enginePath);
    if (engineData.empty()) {
        return nullptr;
    }

    // Create runtime (TensorRT 10.x uses unique_ptr)
    auto runtime = TrtUniquePtr<IRuntime>(createInferRuntime(gLogger));
    if (!runtime) {
        setError("Failed to create TensorRT runtime");
        return nullptr;
    }

    // Deserialize engine
    auto engine = TrtUniquePtr<ICudaEngine>(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size())
    );
    if (!engine) {
        setError("Failed to deserialize engine");
        return nullptr;
    }

    // Create context
    auto context = TrtUniquePtr<IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        setError("Failed to create execution context");
        return nullptr;
    }

    // Create engine object
    TensorRTEngineObj* engineObj = new TensorRTEngineObj();
    engineObj->engine = std::move(engine);
    engineObj->context = std::move(context);

    int32_t numIO = 0;
    try {
        numIO = engineObj->engine->getNbIOTensors();
        std::cout << "[TensorRT] Engine reports " << numIO << " IO tensors." << std::endl;
    } catch (...) {
        std::cerr << "Warning: getNbIOTensors threw an exception" << std::endl;
    }

    std::string inputTensorName;
    std::string outputTensorName;

    if (numIO > 0 && numIO <= 10) {
        try {
            const char* inName = engineObj->engine->getIOTensorName(0);
            if (inName) inputTensorName = inName;
            
            if (numIO > 1) {
                const char* outName = engineObj->engine->getIOTensorName(numIO - 1);
                if (outName) outputTensorName = outName;
            }
        } catch (...) {
            std::cerr << "Warning: getIOTensorName threw an exception" << std::endl;
        }
    }

    if (inputTensorName.empty()) inputTensorName = "input";
    if (outputTensorName.empty()) outputTensorName = "output";

    engineObj->inputName = inputTensorName;
    engineObj->outputName = outputTensorName;

    Dims inputDims;
    Dims outputDims;
    inputDims.nbDims = 0;
    outputDims.nbDims = 0;

    // Use specific valid names first, otherwise TensorRT crashes internally if invalid name is passed.
    try {
        inputDims = engineObj->engine->getTensorShape("input");
        if (inputDims.nbDims > 0) inputTensorName = "input";
    } catch(...) {}

    if (inputDims.nbDims <= 0) {
        try {
            inputDims = engineObj->engine->getTensorShape(inputTensorName.c_str());
        } catch(...) {}
    }

    try {
        outputDims = engineObj->engine->getTensorShape("output");
        if (outputDims.nbDims > 0) outputTensorName = "output";
    } catch(...) {}
    
    if (outputDims.nbDims <= 0) {
        try {
            outputDims = engineObj->engine->getTensorShape("output0");
            if (outputDims.nbDims > 0) outputTensorName = "output0";
        } catch(...) {}
    }

    if (outputDims.nbDims <= 0) {
        try {
            outputDims = engineObj->engine->getTensorShape(outputTensorName.c_str());
        } catch(...) {}
    }

    if (inputDims.nbDims <= 0 || outputDims.nbDims <= 0) {
        std::cerr << "Error: Invalid tensor dimensions" << std::endl;
        delete engineObj;
        return nullptr;
    }

    // Store dimensions (NCHW)
    for (int i = 0; i < 4 && i < inputDims.nbDims; i++) {
        engineObj->inputDims[i] = inputDims.d[i];
    }
    for (int i = 0; i < 4 && i < outputDims.nbDims; i++) {
        engineObj->outputDims[i] = outputDims.d[i];
    }

    // Calculate buffer sizes
    engineObj->inputSize = 1;
    for (int i = 0; i < 4; i++) {
        if (engineObj->inputDims[i] > 0) {
            engineObj->inputSize *= engineObj->inputDims[i];
        }
    }
    engineObj->inputSize *= sizeof(float);

    engineObj->outputSize = 1;
    for (int i = 0; i < 4; i++) {
        if (engineObj->outputDims[i] > 0) {
            engineObj->outputSize *= engineObj->outputDims[i];
        }
    }
    engineObj->outputSize *= sizeof(float);

    // Create CUDA stream
    cudaError_t status = cudaStreamCreate(&engineObj->stream);
    if (status != cudaSuccess) {
        setError("Failed to create CUDA stream");
        delete engineObj;
        return nullptr;
    }

    return engineObj;
}

void TensorRT_UnloadEngine(TensorRTEngineHandle handle) {
    if (!handle) return;

    TensorRTEngineObj* engineObj = static_cast<TensorRTEngineObj*>(handle);

    if (engineObj->stream) {
        cudaStreamSynchronize(engineObj->stream);
        cudaStreamDestroy(engineObj->stream);
    }

    if (engineObj->gpuInputBuffer) {
        cudaFree(engineObj->gpuInputBuffer);
    }

    if (engineObj->gpuOutputBuffer) {
        cudaFree(engineObj->gpuOutputBuffer);
    }

    // Smart pointers will automatically clean up engine and context
    delete engineObj;
}

int TensorRT_GetInputDims(TensorRTEngineHandle engine, int* dims) {
    if (!engine || !dims) return -1;
    TensorRTEngineObj* engineObj = static_cast<TensorRTEngineObj*>(engine);
    memcpy(dims, engineObj->inputDims, sizeof(int) * 4);
    return 0;
}

int TensorRT_GetOutputDims(TensorRTEngineHandle engine, int* dims) {
    if (!engine || !dims) return -1;
    TensorRTEngineObj* engineObj = static_cast<TensorRTEngineObj*>(engine);
    memcpy(dims, engineObj->outputDims, sizeof(int) * 4);
    return 0;
}

TensorRTContextHandle TensorRT_CreateContext(TensorRTEngineHandle engine) {
    // Context is already created with engine
    return engine;
}

void TensorRT_DestroyContext(TensorRTContextHandle context) {
    // Context will be destroyed with engine
}

int TensorRT_RunInference(TensorRTContextHandle context,
                          int width, int height,
                          const float* input, size_t inputSize,
                          float* output, size_t outputSize) {
    if (!context || !input || !output) {
        setError("Invalid parameters for inference");
        return -1;
    }

    TensorRTEngineObj* engineObj = static_cast<TensorRTEngineObj*>(context);

    // Lock for thread-safe inference (TensorRT IExecutionContext is not thread-safe)
    std::lock_guard<std::mutex> lock(engineObj->mu);

    // Allocate or reallocate input buffer if needed
    if (inputSize > engineObj->currentAllocatedInputSize) {
        if (engineObj->gpuInputBuffer) cudaFree(engineObj->gpuInputBuffer);
        cudaError_t status = cudaMalloc(&engineObj->gpuInputBuffer, inputSize);
        if (status != cudaSuccess) {
            setError("Failed to allocate GPU input buffer");
            engineObj->gpuInputBuffer = nullptr;
            engineObj->currentAllocatedInputSize = 0;
            return -1;
        }
        engineObj->currentAllocatedInputSize = inputSize;
    }

    // Allocate or reallocate output buffer if needed
    if (outputSize > engineObj->currentAllocatedOutputSize) {
        if (engineObj->gpuOutputBuffer) cudaFree(engineObj->gpuOutputBuffer);
        cudaError_t status = cudaMalloc(&engineObj->gpuOutputBuffer, outputSize);
        if (status != cudaSuccess) {
            setError("Failed to allocate GPU output buffer");
            engineObj->gpuOutputBuffer = nullptr;
            engineObj->currentAllocatedOutputSize = 0;
            return -1;
        }
        engineObj->currentAllocatedOutputSize = outputSize;
    }

    // TensorRT 10.x: Set dynamic shape
    Dims inputShape;
    inputShape.nbDims = 4;
    inputShape.d[0] = 1;
    inputShape.d[1] = 3;
    inputShape.d[2] = height;
    inputShape.d[3] = width;
    
    try {
        engineObj->context->setInputShape(engineObj->inputName.c_str(), inputShape);
    } catch (...) {
        std::cerr << "Warning: setInputShape threw an exception" << std::endl;
    }

    // Copy input to GPU
    cudaError_t status = cudaMemcpyAsync(engineObj->gpuInputBuffer, input, inputSize,
                                         cudaMemcpyHostToDevice, engineObj->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy input to GPU");
        return -1;
    }

    // TensorRT 10.x: Set tensor addresses
    try {
        engineObj->context->setTensorAddress(engineObj->inputName.c_str(), engineObj->gpuInputBuffer);
        engineObj->context->setTensorAddress(engineObj->outputName.c_str(), engineObj->gpuOutputBuffer);
    } catch (...) {
        std::cerr << "Warning: setTensorAddress threw an exception" << std::endl;
        return -1;
    }

    // Execute inference using TensorRT 10.x API
    bool success = engineObj->context->enqueueV3(engineObj->stream);
    if (!success) {
        setError("Inference execution failed");
        return -1;
    }

    // Copy output from GPU
    status = cudaMemcpyAsync(output, engineObj->gpuOutputBuffer, outputSize,
                              cudaMemcpyDeviceToHost, engineObj->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy output from GPU");
        return -1;
    }

    // Synchronize stream
    status = cudaStreamSynchronize(engineObj->stream);
    if (status != cudaSuccess) {
        setError("CUDA stream synchronization failed");
        return -1;
    }

    return 0;
}

const char* TensorRT_GetLastError() {
    return gLastError.c_str();
}

} // extern "C"
