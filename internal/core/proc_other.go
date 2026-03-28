//go:build !windows

package core

import "os/exec"

func hideWindow(cmd *exec.Cmd) {}
