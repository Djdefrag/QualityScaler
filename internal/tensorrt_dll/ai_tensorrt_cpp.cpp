// NOTE:
// Keep CMake target source name (`ai_tensorrt_cpp.cpp`) for compatibility,
// but reuse the canonical TensorRT implementation in `ai_tensorrt.cpp`.
// This avoids having two divergent wrappers (one old API, one new API).

#include "ai_tensorrt.cpp"
