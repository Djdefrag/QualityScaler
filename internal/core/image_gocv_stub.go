//go:build no_gocv
// +build no_gocv

package core

import (
	"fmt"
	"image"
)

// ProcessImageWithGOCV - stub when gocv is disabled
func ProcessImageWithGOCV(path string, opts Options, outputPath string) error {
	return fmt.Errorf("gocv disabled")
}

// decodeImageWithGOCV - stub when gocv is disabled
func decodeImageWithGOCV(path string) (image.Image, error) {
	return nil, fmt.Errorf("gocv disabled")
}

// encodeImageWithGOCV - stub when gocv is disabled
func encodeImageWithGOCV(path string, img image.Image, ext string, perfMode string) error {
	return fmt.Errorf("gocv disabled")
}

// shouldUseGOCV returns false when gocv is disabled
func shouldUseGOCV(inputPath, outputPath string) bool {
	return false
}
