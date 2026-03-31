package core

import (
	"image"
	"image/color"
	"testing"
)

func TestCheckAlphaChannel(t *testing.T) {
	tests := []struct {
		name     string
		img      *image.RGBA
		expected bool
	}{
		{
			name:     "Fully opaque image",
			img:      image.NewRGBA(image.Rect(0, 0, 100, 100)),
			expected: false,
		},
		{
			name:     "Image with some transparent pixels",
			img:      createImageWithAlpha(100, 100, 255),
			expected: true,
		},
		{
			name:     "Small image with alpha",
			img:      createImageWithAlpha(50, 50, 128),
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := checkAlphaChannel(tt.img)
			if result != tt.expected {
				t.Errorf("checkAlphaChannel() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestUpsampleAlphaChannel(t *testing.T) {
	// Create a simple gradient alpha channel
	srcImg := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			// Create a simple gradient from 0 to 255
			alpha := uint8((x + y) / 2)
			srcImg.Pix[(y*100+x)*4+3] = alpha
		}
	}

	// Upscale by 2x
	upscaled := upsampleAlphaChannel(srcImg, 200, 200)

	// Check dimensions
	if upscaled.Bounds().Dx() != 200 || upscaled.Bounds().Dy() != 200 {
		t.Errorf("Upscaled dimensions = %dx%d, want 200x200",
			upscaled.Bounds().Dx(), upscaled.Bounds().Dy())
	}

	// Check that alpha values are reasonable
	// Top-left should be near 0, bottom-right should be near 255
	tlAlpha := upscaled.Pix[3] // Top-left pixel alpha
	brAlpha := upscaled.Pix[(199*200+199)*4+3] // Bottom-right pixel alpha

	if tlAlpha > 50 {
		t.Errorf("Top-left alpha = %d, want <= 50", tlAlpha)
	}
	if brAlpha < 200 {
		t.Errorf("Bottom-right alpha = %d, want >= 200", brAlpha)
	}
}

func createImageWithAlpha(width, height int, alphaVal uint8) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	// Set first pixel to have a different alpha value
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 255, B: 255, A: alphaVal})
	return img
}
