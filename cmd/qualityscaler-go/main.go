package main

import (
	"qualityscaler-go/internal/gui"
)


func main() {
	if err := gui.Run(); err != nil {
// log.Fatal(err)
	}
}
