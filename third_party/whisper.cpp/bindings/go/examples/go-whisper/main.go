package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	// Packages
	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

func main() {
	flags, err := NewFlags(filepath.Base(os.Args[0]), os.Args[1:])
	if err == flag.ErrHelp {
		os.Exit(0)
	} else if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	} else if flags.GetModel() == "" {
		fmt.Fprintln(os.Stderr, "Use -model flag to specify which model file to use")
		os.Exit(1)
	} else if flags.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "No input files specified")
		os.Exit(1)
	}

	// Load model
	model, err := whisper.New(flags.GetModel())
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer model.Close()

	// Process files
	for _, filename := range flags.Args() {
		if err := Process(model, filename, flags); err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}
	}
}
