package core

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"image"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

func ProcessVideo(ctx context.Context, ffmpegPath string, inputPath string, outputPath string, opts Options, status func(string), progress func(float64)) error {
	// 1. Create temp directories
	// The Python version keeps frames in a specific folder "extracted_frames".
	// Here we use a local folder "_temp" next to the output for processing.

	workDir := filepath.Join(filepath.Dir(outputPath), strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))+"_temp")
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return err
	}

	// Track success to perform cleanup only when finished successfully (if KeepFrames is OFF)
	// This allows resuming if the process was interrupted.
	success := false
	defer func() {
		if !opts.KeepFrames && success {
			os.RemoveAll(workDir)
		}
	}()

	framesDir := filepath.Join(workDir, "frames")
	upscaledDir := filepath.Join(workDir, "upscaled")
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		return err
	}
	if err := os.MkdirAll(upscaledDir, 0755); err != nil {
		return err
	}

	// 2. Get FPS
	fps, err := getVideoFPS(ctx, ffmpegPath, inputPath)
	if err != nil {
		return fmt.Errorf("failed to detect FPS: %w", err)
	}

	// 3. Extract Frames
	if status != nil {
		status(fmt.Sprintf("Extracting frames for: %s", filepath.Base(inputPath)))
	}

	// Check if extraction was already completed
	extractedMarker := filepath.Join(workDir, ".extracted")
	if _, err := os.Stat(extractedMarker); err == nil {
		// Extraction already done, verify we have frames
		if files, _ := ioutil.ReadDir(framesDir); len(files) > 0 {
			if status != nil {
				status("Skipping extraction (already done)")
			}
		} else {
			// Marker exists but no frames? Re-extract
			if err := extractFrames(ctx, ffmpegPath, inputPath, framesDir, progress); err != nil {
				return err
			}
			os.Create(extractedMarker)
		}
	} else {
		// Marker not found, extract
		if err := extractFrames(ctx, ffmpegPath, inputPath, framesDir, progress); err != nil {
			return err
		}
		os.Create(extractedMarker)
	}

	if progress != nil {
		progress(10.0) // Extraction done
	}

	// 4. Upscale Frames
	frameFiles, err := getFrameFiles(framesDir)
	if err != nil {
		return err
	}
	if len(frameFiles) == 0 {
		return fmt.Errorf("no frames extracted")
	}

	if err := upscaleFrames(ctx, frameFiles, upscaledDir, opts, progress, status); err != nil {
		return err
	}
	progress(90.0) // Upscaling done

	// 5. Assemble Video
	if status != nil {
		status(fmt.Sprintf("Assembling video: %s", filepath.Base(outputPath)))
	}
	if err := assembleVideo(ctx, ffmpegPath, upscaledDir, inputPath, outputPath, fps, opts, status, progress, len(frameFiles)); err != nil {
		return err
	}

	success = true
	return nil
}

func getVideoDurationAndFPS(ctx context.Context, ffmpegPath, inputPath string) (float64, string, error) {
	cmd := exec.CommandContext(ctx, ffmpegPath, "-i", inputPath)
	hideWindow(cmd)
	out, _ := cmd.CombinedOutput()
	output := string(out)

	fps := "30"
	reFPS := regexp.MustCompile(`, (\d+(?:\.\d+)?) fps`)
	if matches := reFPS.FindStringSubmatch(output); len(matches) > 1 {
		fps = matches[1]
	}

	var duration float64 = 0
	// Duration: 00:01:23.45, start: 0.000000, bitrate: ...
	reDur := regexp.MustCompile(`Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d{2})`)
	if matches := reDur.FindStringSubmatch(output); len(matches) == 4 {
		h, _ := strconv.ParseFloat(matches[1], 64)
		m, _ := strconv.ParseFloat(matches[2], 64)
		s, _ := strconv.ParseFloat(matches[3], 64)
		duration = h*3600 + m*60 + s
	}

	return duration, fps, nil
}

func getVideoFPS(ctx context.Context, ffmpegPath, inputPath string) (string, error) {
	_, fps, err := getVideoDurationAndFPS(ctx, ffmpegPath, inputPath)
	return fps, err
}

func extractFrames(ctx context.Context, ffmpegPath, inputPath, framesDir string, progress func(float64)) error {
	duration, fpsStr, _ := getVideoDurationAndFPS(ctx, ffmpegPath, inputPath)
	fps, _ := strconv.ParseFloat(fpsStr, 64)
	if fps <= 0 {
		fps = 30
	}
	totalFrames := duration * fps

	// Try hardware acceleration: use cuda if we have an NVIDIA GPU, otherwise auto.
	hwAccelArgs := []string{"-hwaccel", "auto"}
	if strings.Contains(strings.ToUpper(AIBackend()), "CUDA") {
		// Use CUDA for video decoding as well to show activity in the Video Decode engine
		hwAccelArgs = []string{"-hwaccel", "cuda"}
	}

	args := []string{
		"-threads", "0",
		"-i", inputPath,
		"-q:v", "2",
		filepath.Join(framesDir, "frame_%08d.jpg"),
	}

	fullArgs := append(hwAccelArgs, args...)

	runWithProgress := func(cmdArgs []string) error {
		cmd := exec.CommandContext(ctx, ffmpegPath, cmdArgs...)
		hideWindow(cmd)
		stderr, err := cmd.StderrPipe()
		if err != nil {
			return err
		}
		if err := cmd.Start(); err != nil {
			return err
		}

		reFrame := regexp.MustCompile(`frame=\s*(\d+)`)

		go func() {
			scanner := bufio.NewScanner(stderr)
			// FFmpeg uses \r to overwrite lines, so we split by \r
			split := func(data []byte, atEOF bool) (advance int, token []byte, err error) {
				if atEOF && len(data) == 0 {
					return 0, nil, nil
				}
				if i := bytes.IndexAny(data, "\r\n"); i >= 0 {
					return i + 1, data[0:i], nil
				}
				if atEOF {
					return len(data), data, nil
				}
				return 0, nil, nil
			}
			scanner.Split(split)

			for scanner.Scan() {
				line := scanner.Text()
				if progress != nil && totalFrames > 0 {
					matches := reFrame.FindStringSubmatch(line)
					if len(matches) > 1 {
						if currentFrame, err := strconv.ParseFloat(matches[1], 64); err == nil {
							p := (currentFrame / totalFrames) * 10.0
							if p > 10.0 {
								p = 10.0
							}
							progress(p)
						}
					}
				}
			}
		}()

		return cmd.Wait()
	}

	if err := runWithProgress(fullArgs); err != nil {
		// Fallback to software decoding
		softwareArgs := []string{
			"-threads", "0",
			"-i", inputPath,
			"-q:v", "2",
			filepath.Join(framesDir, "frame_%08d.jpg"),
		}
		if errFallback := runWithProgress(softwareArgs); errFallback != nil {
			return fmt.Errorf("extract frames failed (hardware & software)")
		}
	}
	return nil
}

func getFrameFiles(dir string) ([]string, error) {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, f := range files {
		if !f.IsDir() {
			paths = append(paths, filepath.Join(dir, f.Name()))
		}
	}
	return paths, nil
}

func upscaleFrames(ctx context.Context, files []string, outDir string, opts Options, progress func(float64), status func(string)) error {
	workers := opts.AIThreads
	if workers < 1 {
		workers = 1
	}

	inferWorkers := workers

	// Use shared session - ONNX Runtime handles concurrent calls safely
	// Creating multiple sessions would consume 4x GPU memory
	var sharedSession *ort.DynamicAdvancedSession
	var err error
	if !trtAvailable {
		sharedSession, err = GetAISession(opts.AIModel)
		if err != nil {
			return fmt.Errorf("failed to get AI session: %w", err)
		}
	}

	total := len(files)
	if total == 0 {
		return nil
	}

	outExt := opts.ImageExtension
	if outExt == "" {
		// For video frames, always use JPEG by default (much faster than PNG)
		// PNG is only useful for final output image, not temporary video frames
		outExt = ".jpg"
	}

	type frameJob struct {
		src     string
		outPath string
	}
	type frameResult struct {
		outPath string
		img     image.Image
	}

	// Phase 1: scan existing outputs once, then show a clear processing stage.
	pending := make([]frameJob, 0, total)
	existingCount := 0
	for i, src := range files {
		base := filepath.Base(src)
		baseNoExt := strings.TrimSuffix(base, filepath.Ext(base))
		outPath := filepath.Join(outDir, baseNoExt+outExt)

		if info, err := os.Stat(outPath); err == nil && info.Size() > 0 {
			existingCount++
		} else {
			pending = append(pending, frameJob{src: src, outPath: outPath})
		}

		if progress != nil && ((i+1)%32 == 0 || i+1 == total) {
			p := float64(i+1) / float64(total) * 80.0
			progress(10.0 + p)
		}
		if status != nil && ((i+1)%25 == 0 || i+1 == total) {
			status(fmt.Sprintf("Scanning existing frames: %d/%d", i+1, total))
		}
	}

	if len(pending) == 0 {
		if status != nil {
			status(fmt.Sprintf("All frames already upscaled: %d/%d", total, total))
		}
		if progress != nil {
			progress(90.0)
		}
		return nil
	}

	if status != nil {
		status(fmt.Sprintf("Upscaling pending frames: %d/%d (resume from frame %d)", len(pending), total, existingCount+1))
	}

	// Optimize channel sizes based on batch size
	batchSize := 8 // Process frames in batches for better cache locality
	jobs := make(chan frameJob, batchSize*inferWorkers)
	results := make(chan frameResult, batchSize*2)
	errCh := make(chan error, 1)
	setErr := func(err error) {
		if err == nil {
			return
		}
		select {
		case errCh <- err:
		default:
		}
	}

	// done is a broadcast stop signal: closing it notifies all goroutines to exit.
	done := make(chan struct{})
	var doneOnce sync.Once
	stopAll := func(e error) {
		setErr(e)
		doneOnce.Do(func() { close(done) })
	}

	var inferWG sync.WaitGroup
	var saveWG sync.WaitGroup
	var processedPending int32
	var lastProgressNS int64
	var lastStatusNS int64
	processingStartTime := time.Now()

	reportProgress := func(donePending int32, force bool) {
		if progress == nil {
			return
		}
		now := time.Now().UnixNano()
		if !force {
			last := atomic.LoadInt64(&lastProgressNS)
			if last != 0 && now-last < int64(150*time.Millisecond) {
				return
			}
		}
		atomic.StoreInt64(&lastProgressNS, now)
		doneTotal := existingCount + int(donePending)
		p := float64(doneTotal) / float64(total) * 80.0
		progress(10.0 + p)
	}

	reportStatus := func(msg string, force bool) {
		if status == nil {
			return
		}
		now := time.Now().UnixNano()
		if !force {
			last := atomic.LoadInt64(&lastStatusNS)
			if last != 0 && now-last < int64(250*time.Millisecond) {
				return
			}
		}
		atomic.StoreInt64(&lastStatusNS, now)

		// Add ETA to status message
		doneTotal := atomic.LoadInt32(&processedPending)
		if doneTotal > 0 && doneTotal < int32(total) {
			elapsed := time.Since(processingStartTime)
			rate := float64(doneTotal) / elapsed.Seconds()
			remaining := float64(total-int(doneTotal)) / rate
			if remaining > 0 {
				eta := time.Duration(remaining * float64(time.Second))
				if eta > time.Hour {
					msg += fmt.Sprintf(" ETA: %0.1fh", eta.Hours())
				} else if eta > time.Minute {
					msg += fmt.Sprintf(" ETA: %0.0fm", eta.Minutes())
				} else {
					msg += fmt.Sprintf(" ETA: %0.0fs", eta.Seconds())
				}
			}
		}

		status(msg)
	}

	// Each worker uses the shared session (ONNX Runtime handles concurrent calls)
	inferWorker := func(workerIndex int) {
		defer inferWG.Done()
		session := sharedSession
		for {
			select {
			case <-done:
				return
			case job, ok := <-jobs:
				if !ok {
					return
				}
				decodeStart := time.Now()
				img, err := decodeImageWithGOCV(job.src)
				if err != nil {
					img, err = decodeImage(job.src)
					if err != nil {
						stopAll(err)
						return
					}
				}
				decodeTime := time.Since(decodeStart)

				inferStart := time.Now()
				final, err := upscaleImageInMemoryWithSession(session, img, opts)
				if err != nil {
					stopAll(err)
					return
				}
				inferTime := time.Since(inferStart)

				select {
				case <-ctx.Done():
					return
				case <-done:
					return
				case results <- frameResult{outPath: job.outPath, img: final}:
				}

				_ = decodeTime
				_ = inferTime
			}
		}
	}

	// Increase save workers to match AI workers to avoid bottleneck
	// Encoding is CPU-bound, so we can have more save workers than AI workers
	saverCount := workers
	if saverCount < 2 {
		saverCount = 2
	}
	// Cap at reasonable limit to avoid too much memory pressure
	if saverCount > 8 {
		saverCount = 8
	}

	saveWorker := func(workerIndex int) {
		defer saveWG.Done()
		for result := range results {
			err := encodeImageWithGOCV(result.outPath, result.img, outExt, opts.PerformanceMode)
			if err != nil {
				// gocv disabled or unavailable - silently fall back to pure Go encoder
				err = encodeImage(result.outPath, result.img, outExt, opts.PerformanceMode)
				if err != nil {
					stopAll(err)
					return
				}
			}

			donePending := atomic.AddInt32(&processedPending, 1)
			reportProgress(donePending, donePending == int32(len(pending)))

			if status != nil {
				doneTotal := existingCount + int(donePending)
				reportStatus(fmt.Sprintf("Upscaling frame %d/%d", doneTotal, total), donePending == int32(len(pending)))
			}
		}
	}

	for i := 0; i < inferWorkers; i++ {
		inferWG.Add(1)
		go inferWorker(i)
	}
	for i := 0; i < saverCount; i++ {
		saveWG.Add(1)
		go saveWorker(i)
	}

	// Feed jobs to workers
feedLoop:
	for _, j := range pending {
		select {
		case <-ctx.Done():
			break feedLoop
		case <-done:
			break feedLoop
		case jobs <- j:
		}
	}
	close(jobs)

	inferWG.Wait()
	close(results)
	saveWG.Wait()

	if ctx.Err() != nil {
		return context.Canceled
	}

	select {
	case err := <-errCh:
		return err
	default:
	}

	reportProgress(int32(len(pending)), true)
	return nil
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func assembleVideo(ctx context.Context, ffmpegPath, frameDir, originalVideo, outputPath, fps string, opts Options, status func(string), progress func(float64), totalFrames int) error {
	// Detect what extension the frames have.
	// We look at the first file in frameDir.
	files, _ := ioutil.ReadDir(frameDir)
	if len(files) == 0 {
		return fmt.Errorf("no upscaled frames found")
	}
	ext := filepath.Ext(files[0].Name()) // e.g. ".png"

	// ffmpeg -r FPS -i frameDir/frame_%08d.png -i originalVideo -map 0:v -map 1:a?
	// We need to map audio from original.

	// Construct input pattern
	inputPattern := filepath.Join(frameDir, "frame_%08d"+ext)

	codec := opts.VideoCodec
	if codec == "x264" {
		codec = "libx264"
	}
	if codec == "x265" {
		codec = "libx265"
	}

	args := []string{
		"-y",
		"-framerate", fps,
		"-i", inputPattern,
		"-i", originalVideo,
		"-map", "0:v",
		"-map", "1:a?", // ? means ignore if no audio
		"-c:v", codec,
		"-pix_fmt", "yuv420p", // Important for compatibility
		"-c:a", "copy",
	}

	// Set high bitrate for upscaled videos (4x resolution needs much higher bitrate)
	// This matches Python version's -b:v 20000k
	if !strings.Contains(codec, "_nvenc") && !strings.Contains(codec, "_amf") && !strings.Contains(codec, "_qsv") {
		// For CPU encoders (x264/x265), use bitrate instead of CRF for consistent file size
		// Calculate bitrate based on resolution: base 20Mbps for 1080p, scale for other resolutions
		args = append(args, "-b:v", "20000k", "-maxrate", "25000k", "-bufsize", "50000k")
	} else if strings.Contains(codec, "_nvenc") {
		// For NVIDIA NVENC encoders, use high bitrate
		args = append(args, "-b:v", "20000k", "-maxrate", "25000k", "-bufsize", "50000k")
	} else if strings.Contains(codec, "_amf") || strings.Contains(codec, "_qsv") {
		// For AMD AMF and Intel QSV encoders
		args = append(args, "-b:v", "20000k")
	}

	// GPU encoding support?
	gpuIndex := parseGPUIndex(opts.GPU)
	if strings.Contains(codec, "_nvenc") && gpuIndex >= 0 {
		args = append(args, "-gpu", strconv.Itoa(gpuIndex))
		switch strings.ToLower(strings.TrimSpace(opts.PerformanceMode)) {
		case "extreme performance":
			args = append(args, "-preset", "p1")
		case "balanced":
			args = append(args, "-preset", "p4")
		}
	} else if strings.Contains(codec, "libx26") {
		// Don't use CRF, use bitrate instead for consistent file size
		// This is removed since we're using -b:v above
		switch strings.ToLower(strings.TrimSpace(opts.PerformanceMode)) {
		case "extreme performance":
			args = append(args, "-preset", "veryfast")
		case "balanced":
			args = append(args, "-preset", "fast")
		default:
			args = append(args, "-preset", "medium")
		}
	}

	args = append(args, outputPath)

	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	hideWindow(cmd)

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("assemble video: failed to open stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("assemble video: failed to start ffmpeg: %w", err)
	}

	reFrame := regexp.MustCompile(`frame=\s*(\d+)`)

	// Async goroutine: parse ffmpeg stderr and report 90→100% progress
	go func() {
		scanner := bufio.NewScanner(stderr)
		split := func(data []byte, atEOF bool) (advance int, token []byte, err error) {
			if atEOF && len(data) == 0 {
				return 0, nil, nil
			}
			if i := bytes.IndexAny(data, "\r\n"); i >= 0 {
				return i + 1, data[0:i], nil
			}
			if atEOF {
				return len(data), data, nil
			}
			return 0, nil, nil
		}
		scanner.Split(split)

		for scanner.Scan() {
			line := scanner.Text()
			if matches := reFrame.FindStringSubmatch(line); len(matches) > 1 {
				if currentFrame, err := strconv.ParseFloat(matches[1], 64); err == nil {
					if status != nil {
						status(fmt.Sprintf("Assembling video: frame %d/%d", int(currentFrame), totalFrames))
					}
					if progress != nil && totalFrames > 0 {
						p := 90.0 + (currentFrame/float64(totalFrames))*10.0
						if p > 100.0 {
							p = 100.0
						}
						progress(p)
					}
				}
			}
		}
	}()

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("assemble video failed (ffmpeg exited with error)")
	}

	// Report 100% on successful completion
	if progress != nil {
		progress(100.0)
	}
	return nil
}

func parseGPUIndex(gpu string) int {
	g := strings.TrimSpace(strings.ToLower(gpu))
	if g == "" || g == "auto" {
		return -1
	}
	parts := strings.Fields(g)
	if len(parts) != 2 || parts[0] != "gpu" {
		return -1
	}
	v, err := strconv.Atoi(parts[1])
	if err != nil || v < 1 {
		return -1
	}
	return v - 1
}

func exportFrames(ctx context.Context, ffmpegPath, outputVideoPath string) error {
	framesDir := strings.TrimSuffix(outputVideoPath, filepath.Ext(outputVideoPath)) + "_frames"
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		return err
	}

	args := []string{
		"-y",
		"-i", outputVideoPath,
		filepath.Join(framesDir, "frame_%06d.png"),
	}
	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	hideWindow(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg frame export failed: %w, output: %s", err, string(out))
	}
	return nil
}

func PrepareOutputPath(inputPath string, outputRoot string, aiModel string, inputScale int, outputScale int, blend float64, extension string) string {
	dir := filepath.Dir(inputPath)
	if outputRoot != "" {
		dir = outputRoot
	}

	base := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))

	// Format: {base}_{model}_InputR-{scale}_OutputR-{scale}[_Blending-{level}]{ext}
	name := fmt.Sprintf("%s_%s", base, aiModel)

	name += fmt.Sprintf("_InputR-%d", inputScale)
	name += fmt.Sprintf("_OutputR-%d", outputScale)

	// Blending level strings to match Python
	if blend > 0 {
		switch blend {
		case 0.3:
			name += "_Blending-Low"
		case 0.5:
			name += "_Blending-Medium"
		case 0.7:
			name += "_Blending-High"
		default:
			// Fallback if custom value
			name += fmt.Sprintf("_Blending-%.1f", blend)
		}
	}

	return filepath.Join(dir, name+extension)
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', 3, 64)
}
