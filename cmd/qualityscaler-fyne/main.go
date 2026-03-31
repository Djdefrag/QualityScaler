package main

import (
	"log"

	"qualityscaler-go/internal/gui_fyne"
)

func main() {
	if err := gui_fyne.Run(); err != nil {
		log.Fatal(err)
	}
}
