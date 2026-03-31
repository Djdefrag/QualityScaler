//go:build !windows

package gui

import "os/exec"

func hideWindow(cmd *exec.Cmd) {}
