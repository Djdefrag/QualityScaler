package core

import (
	"runtime"
	"strings"
)

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type Options struct {
	OutputPath         string
	AIModel            string
	AIThreads          int
	InputScalePercent  int
	OutputScalePercent int
	GPU                string
	VRAMGB             int
	BlendingFactor     float64
	KeepFrames         bool
	ImageExtension     string
	VideoExtension     string
	VideoCodec         string
	PerformanceMode    string
	// Performance tuning options
	BatchSize          int
	EnableParallel     bool
}

func ApplyPerformanceMode(opts Options) Options {
	mode := strings.ToLower(strings.TrimSpace(opts.PerformanceMode))

	// Set defaults
	if opts.BatchSize <= 0 {
		opts.BatchSize = 8
	}

	// Enable parallel processing by default
	opts.EnableParallel = true

	// 🔥 优化: 基于CPU核心数设置默认线程数
	numCPU := runtime.NumCPU()

	switch mode {
	case "extreme performance":
		// 极限性能: 使用较少的AI线程避免GPU上下文切换开销
		// 过多的线程会导致GPU上下文切换，反而降低性能
		opts.AIThreads = minInt(4, numCPU) // 最多4个AI线程
		if opts.AIThreads < 2 {
			opts.AIThreads = 2
		}
		opts.BlendingFactor = 0
		opts.VideoCodec = "h264_nvenc"
		opts.KeepFrames = false
		// TensorRT supports larger batches, ONNX only supports 1
		opts.BatchSize = 1
	case "balanced":
		// 平衡模式: 使用CPU核心数，但限制上限
		opts.AIThreads = minInt(2, numCPU) // 最多2个AI线程
		if opts.AIThreads < 1 {
			opts.AIThreads = 1
		}
		// TensorRT supports larger batches, ONNX only supports 1
		opts.BatchSize = 1
	case "quality":
		// 质量模式: 单线程处理，确保最佳质量
		opts.AIThreads = 1
		if opts.VideoCodec == "h264_nvenc" {
			opts.VideoCodec = "x264"
		}
		// TensorRT supports larger batches, ONNX only supports 1
		opts.BatchSize = 1
	}

	// Clamp thread count to CPU cores * 2
	maxThreads := runtime.NumCPU() * 2
	if opts.AIThreads > maxThreads {
		opts.AIThreads = maxThreads
	}
	if opts.AIThreads < 1 {
		opts.AIThreads = 1
	}

	return opts
}
