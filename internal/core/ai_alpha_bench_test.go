package core

import (
	"fmt"
	"image"
	"image/color"
	"testing"
	"time"
)

func BenchmarkRGBvsRGBWithAlpha(b *testing.B) {
	// Create a large image with alpha channel
	width, height := 1920, 1080
	imgWithAlpha := image.NewRGBA(image.Rect(0, 0, width, height))
	
	// Fill with gradient alpha
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			alpha := uint8((x*255 + y*255) / (width + height))
			imgWithAlpha.SetRGBA(x, y, color.RGBA{
				R: 255,
				G: 255,
				B: 255,
				A: alpha,
			})
		}
	}

	// Create same image without alpha (fully opaque)
	imgNoAlpha := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			imgNoAlpha.SetRGBA(x, y, color.RGBA{
				R: 255,
				G: 255,
				B: 255,
				A: 255,
			})
		}
	}

	b.Run("RGBOnly", func(b *testing.B) {
		// Simulate RGB-only processing (current path)
		for i := 0; i < b.N; i++ {
			// Simulate preprocessing overhead
			start := time.Now()
			// Just measure the check and processing overhead
			_ = checkAlphaChannel(imgNoAlpha)
			_ = time.Since(start)
		}
	})

	b.Run("RGBWithAlpha", func(b *testing.B) {
		// Simulate RGB+Alpha processing (optimized path)
		for i := 0; i < b.N; i++ {
			start := time.Now()
			// Check alpha channel
			hasAlpha := checkAlphaChannel(imgWithAlpha)
			if hasAlpha {
				// Simulate alpha upsampling (2x)
				_ = upsampleAlphaChannel(imgWithAlpha, width*2, height*2)
			}
			_ = time.Since(start)
		}
	})
}

// Manual benchmark for alpha upsampling
func TestAlphaUpscalingBenchmark(t *testing.T) {
	width, height := 1920, 1080
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	
	// Fill with gradient alpha
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			alpha := uint8((x + y) / 2)
			img.Pix[(y*width+x)*4+3] = alpha
		}
	}

	// Benchmark alpha upsampling (2x scale)
	iterations := 10
	start := time.Now()
	for i := 0; i < iterations; i++ {
		upsampleAlphaChannel(img, width*2, height*2)
	}
	duration := time.Since(start)

	avgTime := duration / time.Duration(iterations)
	fmt.Printf("Alpha upsampling (2x) average time: %v\n", avgTime)
	fmt.Printf("This is typically much faster than AI inference (~1s per frame)\n")
}
