//go:build !no_gocv
// +build !no_gocv

package core

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"
)

// ProcessImageWithGOCV uses OpenCV for faster image encoding/decoding
func ProcessImageWithGOCV(path string, opts Options, outputPath string) error {
	// Use gocv for decoding (much faster than Go stdlib)
	mat := gocv.IMRead(path, gocv.IMReadUnchanged)
	if mat.Empty() {
		return fmt.Errorf("gocv: failed to read image: %s", path)
	}
	defer mat.Close()

	// Convert Go image to gocv Mat
	img, err := mat.ToImage()
	if err != nil {
		return fmt.Errorf("gocv: failed to convert to Go image: %w", err)
	}

	// Process with existing pipeline (AI inference, etc.)
	final, err := upscaleImageInMemory(img, opts)
	if err != nil {
		return err
	}

	// Convert back to gocv Mat for encoding
	bounds := final.Bounds()
	resultMat := gocv.NewMatWithSize(bounds.Dy(), bounds.Dx(), gocv.MatTypeCV8UC4)
	defer resultMat.Close()

	// Fast copy pixels
	rgba, ok := final.(*image.RGBA)
	if !ok {
		rgbaCopy := image.NewRGBA(bounds)
		drawDirect(rgbaCopy, final)
		rgba = rgbaCopy
	}

	// Copy pixel data to Mat
	for y := 0; y < bounds.Dy(); y++ {
		rowStart := y * rgba.Stride
		for x := 0; x < bounds.Dx(); x++ {
			pixel := rgba.Pix[rowStart+x*4 : rowStart+x*4+4]
			resultMat.SetUCharAt(y, x*4+2, pixel[0])   // B
			resultMat.SetUCharAt(y, x*4+1, pixel[1])   // G
			resultMat.SetUCharAt(y, x*4+0, pixel[2])   // R
			resultMat.SetUCharAt(y, x*4+3, pixel[3])   // A
		}
	}

	// Ensure output directory exists
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}

	// Determine save parameters based on extension
	ext := filepath.Ext(outputPath)
	var saveParams []int

	switch ext {
	case ".jpg", ".jpeg":
		quality := 95
		switch opts.PerformanceMode {
		case "extreme performance":
			quality = 85
		case "balanced":
			quality = 90
		}
		saveParams = []int{gocv.IMWriteJpegQuality, quality}
	case ".png":
		saveParams = []int{gocv.IMWritePngCompression, 3}
	case ".webp":
		saveParams = []int{gocv.IMWriteWebpQuality, 95}
	}

	// Encode with gocv (much faster than Go stdlib)
	if len(saveParams) > 0 {
		if !gocv.IMWriteWithParams(outputPath, resultMat, saveParams) {
			return fmt.Errorf("gocv: failed to write image: %s", outputPath)
		}
	} else {
		if !gocv.IMWrite(outputPath, resultMat) {
			return fmt.Errorf("gocv: failed to write image: %s", outputPath)
		}
	}

	return nil
}

// drawDirect is a helper to copy pixels between images
func drawDirect(dst, src image.Image) {
	bounds := src.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if rgbaDst, ok := dst.(*image.RGBA); ok {
				rgbaDst.Set(x, y, src.At(x, y))
			}
		}
	}
}

// shouldUseGOCV determines if gocv should be used for image processing
func shouldUseGOCV(inputPath, outputPath string) bool {
	// Only use gocv for common image formats that OpenCV handles well
	ext := strings.ToLower(filepath.Ext(inputPath))
	outExt := strings.ToLower(filepath.Ext(outputPath))
	
	// Use gocv for JPEG, PNG, BMP, TIFF
	supportedFormats := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".bmp":  true,
		".tif":  true,
		".tiff": true,
		".webp": false, // WebP may have issues
	}
	
	return supportedFormats[ext] && supportedFormats[outExt]
}

// GetGOCVVersion returns OpenCV version
func GetGOCVVersion() (major, minor, patch int) {
	return 4, 8, 0 // OpenCV 4.8.0 for this gocv version
}

// IsGOCVAvailable checks if gocv is properly initialized
func IsGOCVAvailable() bool {
	return true
}

// decodeImageWithGOCV uses OpenCV for faster image decoding (for video frames)
func decodeImageWithGOCV(path string) (image.Image, error) {
	mat := gocv.IMRead(path, gocv.IMReadUnchanged)
	if mat.Empty() {
		return nil, fmt.Errorf("gocv: failed to read image: %s", path)
	}
	defer mat.Close()

	return mat.ToImage()
}

// encodeImageWithGOCV uses OpenCV for faster image encoding (for video frames)
func encodeImageWithGOCV(outputPath string, img image.Image, ext string, performanceMode string) error {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// Convert to RGBA if needed
	rgba, ok := img.(*image.RGBA)
	if !ok {
		rgbaCopy := image.NewRGBA(bounds)
		drawDirect(rgbaCopy, img)
		rgba = rgbaCopy
	}

	// Create Mat from RGBA data directly
	// Note: Go's RGBA is in RGBA order, OpenCV expects BGRA
	resultMat, err := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8UC4, rgba.Pix)
	if err != nil {
		return fmt.Errorf("gocv: failed to create Mat from bytes: %w", err)
	}
	defer resultMat.Close()

	// Convert RGBA to the format OpenCV expects for encoding
	// Use OpenCV's cvtColor for fast channel swapping
	var writeMat gocv.Mat
	if ext == ".jpg" || ext == ".jpeg" {
		// RGBA to BGR (remove alpha, swap RGB channels)
		tempMat := gocv.NewMat()
		defer tempMat.Close()
		gocv.CvtColor(resultMat, &tempMat, gocv.ColorRGBAToRGB)
		// Now we have RGB, convert to BGR
		rgbToBgr := gocv.NewMat()
		defer rgbToBgr.Close()
		gocv.CvtColor(tempMat, &rgbToBgr, gocv.ColorRGBToBGR)
		writeMat = rgbToBgr
	} else {
		// For PNG/WebP, convert RGBA to BGRA (swap RGB channels, keep alpha)
		tempMat := gocv.NewMat()
		defer tempMat.Close()
		gocv.CvtColor(resultMat, &tempMat, gocv.ColorRGBAToBGRA)
		writeMat = tempMat
	}

	// Ensure output directory exists
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}

	// Determine save parameters based on extension
	// Use lower quality for faster encoding (especially for video frames)
	var saveParams []int
	switch ext {
	case ".jpg", ".jpeg":
		quality := 90 // Default to 90 (good quality, faster encoding)
		switch performanceMode {
		case "extreme performance":
			quality = 80 // Faster encoding, acceptable quality
		case "balanced":
			quality = 85
		}
		saveParams = []int{gocv.IMWriteJpegQuality, quality}
	case ".png":
		compression := 3
		switch performanceMode {
		case "extreme performance":
			compression = 1 // Faster compression
		case "balanced":
			compression = 2
		}
		saveParams = []int{gocv.IMWritePngCompression, compression}
	case ".webp":
		quality := 90
		switch performanceMode {
		case "extreme performance":
			quality = 80
		case "balanced":
			quality = 85
		}
		saveParams = []int{gocv.IMWriteWebpQuality, quality}
	}

	// Encode with gocv
	if len(saveParams) > 0 {
		if !gocv.IMWriteWithParams(outputPath, writeMat, saveParams) {
			return fmt.Errorf("gocv: failed to write image: %s", outputPath)
		}
	} else {
		if !gocv.IMWrite(outputPath, writeMat) {
			return fmt.Errorf("gocv: failed to write image: %s", outputPath)
		}
	}

	return nil
}

