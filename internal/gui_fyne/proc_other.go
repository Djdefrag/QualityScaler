//go:build !windows

package gui_fyne

import "os/exec"

func hideWindow(cmd *exec.Cmd) {}
