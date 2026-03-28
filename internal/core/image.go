package core

import (
	"bufio"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	stddraw "image/draw"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/bmp"
	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/tiff"
	"golang.org/x/image/webp"
)

func IsVideo(path string, supported map[string]struct{}) bool {
	ext := strings.ToLower(filepath.Ext(path))
	_, ok := supported[ext]
	return ok
}

func ModelScale(model string) int {
	s := strings.ToLower(model)
	switch {
	case strings.Contains(s, "x4"):
		return 4
	case strings.Contains(s, "x3"):
		return 3
	case strings.Contains(s, "x2"):
		return 2
	default:
		return 1
	}
}

func ProcessImage(path string, opts Options, outputPath string) error {
	// Try gocv first for faster encoding/decoding
	if shouldUseGOCV(path, outputPath) {
		if err := ProcessImageWithGOCV(path, opts, outputPath); err == nil {
			return nil
		}
		// Fall back to pure Go on error
	}

	// Pure Go fallback
	img, err := decodeImage(path)
	if err != nil {
		return err
	}
	final, err := upscaleImageInMemory(img, opts)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	return encodeImage(outputPath, final, opts.ImageExtension, opts.PerformanceMode)
}

func decodeImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Fast path: check extension for optimized decoding
	ext := strings.ToLower(filepath.Ext(path))

	img, format, err := image.Decode(f)
	if err != nil {
		// Fallback for WebP which may fail with standard decoder
		if strings.EqualFold(ext, ".webp") {
			if _, errSeek := f.Seek(0, 0); errSeek == nil {
				if webpImg, errWebp := webp.Decode(f); errWebp == nil {
					img = webpImg
					err = nil
				}
			}
		}
		if err != nil {
			return nil, fmt.Errorf("decode failed (%s): %w", format, err)
		}
	}

	// Convert to RGBA for faster processing (avoids repeated At() calls)
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba, nil
	}

	// For large images, convert to RGBA to accelerate downstream operations
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	stddraw.Draw(rgba, bounds, img, bounds.Min, stddraw.Src)

	return rgba, nil
}

func upscaleImageInMemory(img image.Image, opts Options) (image.Image, error) {
	return upscaleImageInMemoryWithSession(nil, img, opts)
}

func upscaleImageInMemoryWithSession(session *ort.DynamicAdvancedSession, img image.Image, opts Options) (image.Image, error) {
	var inputScaled image.Image = img

	// Skip input scaling if it's 100% to avoid expensive Go resize operations
	if opts.InputScalePercent != 100 {
		inputScaled = resizeByPercentWithMode(img, opts.InputScalePercent, opts.PerformanceMode)
	}

	var aiScaled image.Image
	var aiErr error
	if session != nil {
		aiScaled, aiErr = RunAIInferenceWithSession(session, inputScaled, opts.AIModel)
	} else if trtAvailable {
		// If session is nil but TensorRT is available, try RunAIInference which will use TensorRT
		aiScaled, aiErr = RunAIInference(inputScaled, opts.AIModel)
	} else {
		aiScaled, aiErr = RunAIInference(inputScaled, opts.AIModel)
	}
	if aiErr != nil {
		aiScaled = resizeByFactorWithMode(inputScaled, ModelScale(opts.AIModel), opts.PerformanceMode)
	}

	final := aiScaled
	// Skip output scaling if it's 100% to avoid expensive Go resize operations
	if opts.OutputScalePercent != 100 {
		final = resizeByPercentWithMode(aiScaled, opts.OutputScalePercent, opts.PerformanceMode)
	}

	if opts.BlendingFactor > 0 {
		base := resizeToWithMode(img, final.Bounds().Dx(), final.Bounds().Dy(), opts.PerformanceMode)
		final = blend(base, final, opts.BlendingFactor)
	}

	return final, nil
}

func resizeByPercent(src image.Image, percent int) image.Image {
	return resizeByPercentWithMode(src, percent, "quality")
}

func resizeByPercentWithMode(src image.Image, percent int, mode string) image.Image {
	if percent <= 0 {
		percent = 100
	}
	if percent == 100 {
		return src
	}
	w := src.Bounds().Dx() * percent / 100
	h := src.Bounds().Dy() * percent / 100
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	return resizeToWithMode(src, w, h, mode)
}

func resizeByFactor(src image.Image, factor int) image.Image {
	return resizeByFactorWithMode(src, factor, "quality")
}

func resizeByFactorWithMode(src image.Image, factor int, mode string) image.Image {
	if factor <= 1 {
		return resizeToWithMode(src, src.Bounds().Dx(), src.Bounds().Dy(), mode)
	}
	return resizeToWithMode(src, src.Bounds().Dx()*factor, src.Bounds().Dy()*factor, mode)
}

func resizeTo(src image.Image, w, h int) image.Image {
	return resizeToWithMode(src, w, h, "quality")
}

func resizeToWithMode(src image.Image, w, h int, mode string) image.Image {
	if src.Bounds().Dx() == w && src.Bounds().Dy() == h {
		return src
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	var scaler xdraw.Scaler = xdraw.CatmullRom
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "extreme performance":
		scaler = xdraw.ApproxBiLinear
	case "balanced":
		scaler = xdraw.BiLinear
	}
	scaler.Scale(dst, dst.Bounds(), src, src.Bounds(), stddraw.Over, nil)
	return dst
}

func blend(base image.Image, upscaled image.Image, baseWeight float64) image.Image {
	if baseWeight <= 0 {
		return upscaled
	}
	if baseWeight > 0.95 {
		baseWeight = 0.95
	}
	upWeight := 1.0 - baseWeight

	b := upscaled.Bounds()
	dst := image.NewRGBA(b)

	// Convert to *image.RGBA for direct pixel access
	baseRGBA, ok := base.(*image.RGBA)
	if !ok {
		baseRGBA = image.NewRGBA(base.Bounds())
		stddraw.Draw(baseRGBA, base.Bounds(), base, base.Bounds().Min, stddraw.Src)
	}
	upRGBA, ok := upscaled.(*image.RGBA)
	if !ok {
		upRGBA = image.NewRGBA(upscaled.Bounds())
		stddraw.Draw(upRGBA, upRGBA.Bounds(), upscaled, upscaled.Bounds().Min, stddraw.Src)
	}

	width, height := b.Dx(), b.Dy()
	const parallelThreshold = 128 * 128 // 16k pixels

	if width*height >= parallelThreshold && height >= 2 {
		// Parallel blending for large images
		numWorkers := runtime.NumCPU()
		var wg sync.WaitGroup
		rowsPerWorker := height / numWorkers
		if rowsPerWorker < 1 {
			rowsPerWorker = 1
		}

		for worker := 0; worker < numWorkers; worker++ {
			startY := worker * rowsPerWorker
			endY := startY + rowsPerWorker
			if worker == numWorkers-1 || endY > height {
				endY = height
			}
			if startY >= height {
				break
			}

			wg.Add(1)
			go func(yStart, yEnd int) {
				defer wg.Done()
				for y := yStart; y < yEnd; y++ {
					baseRow := y * baseRGBA.Stride
					upRow := y * upRGBA.Stride
					dstRow := y * dst.Stride
					for x := 0; x < width; x++ {
						px := x * 4
						dstPix := dstRow + px

						dst.Pix[dstPix] = uint8((float64(baseRGBA.Pix[baseRow+px])*baseWeight + float64(upRGBA.Pix[upRow+px])*upWeight))
						dst.Pix[dstPix+1] = uint8((float64(baseRGBA.Pix[baseRow+px+1])*baseWeight + float64(upRGBA.Pix[upRow+px+1])*upWeight))
						dst.Pix[dstPix+2] = uint8((float64(baseRGBA.Pix[baseRow+px+2])*baseWeight + float64(upRGBA.Pix[upRow+px+2])*upWeight))
						dst.Pix[dstPix+3] = uint8((float64(baseRGBA.Pix[baseRow+px+3])*baseWeight + float64(upRGBA.Pix[upRow+px+3])*upWeight))
					}
				}
			}(startY, endY)
		}
		wg.Wait()
	} else {
		// Sequential blending for small images
		for y := 0; y < height; y++ {
			baseRow := y * baseRGBA.Stride
			upRow := y * upRGBA.Stride
			dstRow := y * dst.Stride
			for x := 0; x < width; x++ {
				px := x * 4
				dstPix := dstRow + px

				dst.Pix[dstPix] = uint8((float64(baseRGBA.Pix[baseRow+px])*baseWeight + float64(upRGBA.Pix[upRow+px])*upWeight))
				dst.Pix[dstPix+1] = uint8((float64(baseRGBA.Pix[baseRow+px+1])*baseWeight + float64(upRGBA.Pix[upRow+px+1])*upWeight))
				dst.Pix[dstPix+2] = uint8((float64(baseRGBA.Pix[baseRow+px+2])*baseWeight + float64(upRGBA.Pix[upRow+px+2])*upWeight))
				dst.Pix[dstPix+3] = uint8((float64(baseRGBA.Pix[baseRow+px+3])*baseWeight + float64(upRGBA.Pix[upRow+px+3])*upWeight))
			}
		}
	}

	return dst
}

var (
	// JPEG encoder buffer pool for better memory reuse
	jpegBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 0, 1024*1024) // 1MB initial buffer
		},
	}
)

func encodeImage(outputPath string, img image.Image, ext string, performanceMode string) error {
	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()

	// Use buffered writer for better I/O performance
	bufWriter := bufio.NewWriterSize(f, 512*1024) // 512KB buffer
	defer bufWriter.Flush()

	switch strings.ToLower(ext) {
	case ".jpg", ".jpeg":
		quality := 90 // Default to 90 (good quality, faster encoding)
		switch strings.ToLower(strings.TrimSpace(performanceMode)) {
		case "extreme performance":
			quality = 80 // Faster encoding, acceptable quality
		case "balanced":
			quality = 85
		}
		return jpeg.Encode(bufWriter, img, &jpeg.Options{Quality: quality})
	case ".bmp":
		return bmp.Encode(bufWriter, img)
	case ".tif", ".tiff":
		return tiff.Encode(bufWriter, img, &tiff.Options{Compression: tiff.Uncompressed})
	default:
		// For PNG, use encoder for better control over compression
		enc := png.Encoder{
			CompressionLevel: png.BestSpeed, // Faster compression
		}
		return enc.Encode(bufWriter, img)
	}
}


