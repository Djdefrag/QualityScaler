//go:build windows

package core

import (
	"fmt"
	"syscall"
	"unsafe"
)

// Use LoadLibraryW to load ONNX Runtime DLL on Windows
func loadDLL(path string) error {
	// Convert path to UTF-16
	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return fmt.Errorf("failed to convert path: %w", err)
	}

	// Try to load the DLL
	handle := syscall.NewLazyDLL("kernel32.dll")
	loadLibrary := handle.NewProc("LoadLibraryW")

	ret, _, err := loadLibrary.Call(uintptr(unsafe.Pointer(pathPtr)))
	if ret == 0 {
		return fmt.Errorf("LoadLibrary failed: %w (path: %s)", err, path)
	}

	return nil
}
