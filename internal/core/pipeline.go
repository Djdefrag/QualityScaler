package core

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"qualityscaler-go/internal/app"
)

type StatusFn func(msg string)
type ProgressFn func(percent float64, stageETA string, fileETA string, totalETA string)
type FileCompletedFn func(filePath string)

func ProcessBatch(ctx context.Context, files []string, opts Options, ffmpegPath string, status StatusFn, progress ProgressFn) error {
	return ProcessBatchWithCallback(ctx, files, opts, ffmpegPath, status, progress, nil)
}

func ProcessBatchWithCallback(ctx context.Context, files []string, opts Options, ffmpegPath string, status StatusFn, progress ProgressFn, onFileCompleted FileCompletedFn) error {
	total := len(files)
	if total == 0 {
		if status != nil {
			status("No input files")
		}
		return nil
	}

	startTime := time.Now()
	currentFileIdx := -1
	currentFileStart := startTime
	stageStart := startTime
	stageStartFilePercent := 0.0
	currentFilePercent := 0.0
	currentStageKey := "init"

	resetForFile := func(fileIdx int) {
		if fileIdx != currentFileIdx {
			currentFileIdx = fileIdx
			currentFileStart = time.Now()
			stageStart = currentFileStart
			stageStartFilePercent = 0.0
			currentFilePercent = 0.0
			currentStageKey = "init"
		}
	}

	formatRemaining := func(start time.Time, doneRatio float64) string {
		if doneRatio <= 0 {
			return "Calculating..."
		}
		elapsed := time.Since(start)
		estimatedTotal := time.Duration(float64(elapsed) / doneRatio)
		remaining := estimatedTotal - elapsed
		if remaining < 0 {
			remaining = 0
		}
		return remaining.Round(time.Second).String()
	}

	stageKeyForStatus := func(msg string) string {
		m := strings.ToLower(strings.TrimSpace(msg))
		switch {
		case strings.Contains(m, "extract") || strings.Contains(m, "scan") || strings.Contains(m, "skip"):
			return "prepare"
		case strings.Contains(m, "upscal"):
			return "upscale"
		case strings.Contains(m, "assembl") || strings.Contains(m, "encod"):
			return "assemble"
		case strings.Contains(m, "complet"):
			return "done"
		default:
			return "misc"
		}
	}

	wrappedStatus := func(msg string) {
		// Reset stage ETA only when moving to a new processing stage.
		newStage := stageKeyForStatus(msg)
		if newStage != currentStageKey {
			currentStageKey = newStage
			stageStart = time.Now()
			stageStartFilePercent = currentFilePercent
		}
		if status != nil {
			status(msg)
		}
	}

	updateProgress := func(currentIdx int, filePercent float64) {
		if progress == nil {
			return
		}
		resetForFile(currentIdx)
		if filePercent < 0 {
			filePercent = 0
		}
		if filePercent > 1 {
			filePercent = 1
		}
		currentFilePercent = filePercent

		// 每个文件独立进度：0→100%，不受总文件数影响
		totalPercent := filePercent * 100.0
		fileETA := formatRemaining(currentFileStart, filePercent)
		// totalETA 与 fileETA 一致（当前文件剩余时间）
		totalETA := fileETA

		stageProgress := filePercent - stageStartFilePercent
		stageTotal := 1.0 - stageStartFilePercent
		stageRatio := 0.0
		if stageTotal > 0 {
			stageRatio = stageProgress / stageTotal
		}
		stageETA := formatRemaining(stageStart, stageRatio)

		progress(totalPercent, stageETA, fileETA, totalETA)
	}

	for idx, file := range files {
		select {
		case <-ctx.Done():
			return context.Canceled
		default:
		}

		updateProgress(idx, 0.0)
		ext := strings.ToLower(filepath.Ext(file))
		if _, ok := app.SupportedVideoExts[ext]; ok {
			wrappedStatus(fmt.Sprintf("%d/%d Upscaling video (single-file mode): %s", idx+1, total, filepath.Base(file)))
			out := prepareVideoOutput(file, opts)
			if err := ProcessVideo(ctx, ffmpegPath, file, out, opts, wrappedStatus, func(p float64) {
				updateProgress(idx, p/100.0)
			}); err != nil {
				return err
			}
			// Callback for completed file
			if onFileCompleted != nil {
				onFileCompleted(file)
			}
			continue
		}

		wrappedStatus(fmt.Sprintf("%d/%d Upscaling image: %s", idx+1, total, filepath.Base(file)))
		out := prepareImageOutput(file, opts)
		if err := ProcessImage(file, opts, out); err != nil {
			return err
		}
		updateProgress(idx, 1.0)
		// Callback for completed file
		if onFileCompleted != nil {
			onFileCompleted(file)
		}
	}

	// Flush any remaining batched log messages from AI/TensorRT inference.
	TRTLog.Flush()
	AILog.Flush()
	wrappedStatus("Completed")
	return nil
}

func prepareImageOutput(inputPath string, opts Options) string {
	outRoot := ""
	if opts.OutputPath != app.OutputPathCode {
		outRoot = opts.OutputPath
	}
	return PrepareOutputPath(inputPath, outRoot, opts.AIModel, opts.InputScalePercent, opts.OutputScalePercent, opts.BlendingFactor, opts.ImageExtension)
}

func prepareVideoOutput(inputPath string, opts Options) string {
	outRoot := ""
	if opts.OutputPath != app.OutputPathCode {
		outRoot = opts.OutputPath
	}
	// Use same logic as images but with video extension
	return PrepareOutputPath(inputPath, outRoot, opts.AIModel, opts.InputScalePercent, opts.OutputScalePercent, opts.BlendingFactor, opts.VideoExtension)
}
