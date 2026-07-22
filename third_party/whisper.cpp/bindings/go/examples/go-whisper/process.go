package main

import (
	"fmt"
	"io"
	"os"
	"time"

	// Package imports
	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	wav "github.com/go-audio/wav"
)

func Process(model whisper.Model, path string, flags *Flags) error {
	var data []float32

	// Create processing context
	context, err := model.NewContext()
	if err != nil {
		return err
	}

	// Set the parameters
	if err := flags.SetParams(context); err != nil {
		return err
	}

	fmt.Printf("\n%s\n", context.SystemInfo())

	// Open the file
	fmt.Fprintf(flags.Output(), "Loading %q\n", path)
	fh, err := os.Open(path)
	if err != nil {
		return err
	}
	defer fh.Close()

	// Decode the WAV file - load the full buffer
	dec := wav.NewDecoder(fh)
	if buf, err := dec.FullPCMBuffer(); err != nil {
		return err
	} else if dec.SampleRate != whisper.SampleRate {
		return fmt.Errorf("unsupported sample rate: %d", dec.SampleRate)
	} else if dec.NumChans != 1 {
		return fmt.Errorf("unsupported number of channels: %d", dec.NumChans)
	} else {
		data = buf.AsFloat32Buffer().Data
	}

	// Segment callback when -tokens is specified
	var cb whisper.SegmentCallback
	if flags.IsTokens() {
		cb = func(segment whisper.Segment) {
			fmt.Fprintf(flags.Output(), "%02d [%6s->%6s] ", segment.Num, segment.Start.Truncate(time.Millisecond), segment.End.Truncate(time.Millisecond))
			for _, token := range segment.Tokens {
				if flags.IsColorize() && context.IsText(token) {
					fmt.Fprint(flags.Output(), Colorize(token.Text, int(token.P*24.0)), " ")
				} else {
					fmt.Fprint(flags.Output(), token.Text, " ")
				}
			}
			fmt.Fprintln(flags.Output(), "")
			fmt.Fprintln(flags.Output(), "")
		}
	}

	// Process the data
	fmt.Fprintf(flags.Output(), "  ...processing %q\n", path)
	context.ResetTimings()
	if err := context.Process(data, nil, cb, nil); err != nil {
		return err
	}

	context.PrintTimings()

	// Print out the results
	switch {
	case flags.GetOut() == "srt":
		return OutputSRT(os.Stdout, context)
	case flags.GetOut() == "none":
		return nil
	default:
		return Output(os.Stdout, context, flags.IsColorize())
	}
}

// Output text as SRT file
func OutputSRT(w io.Writer, context whisper.Context) error {
	n := 1
	for {
		segment, err := context.NextSegment()
		if err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
		fmt.Fprintln(w, n)
		fmt.Fprintln(w, srtTimestamp(segment.Start), " --> ", srtTimestamp(segment.End))
		fmt.Fprintln(w, segment.Text)
		fmt.Fprintln(w, "")
		n++
	}
}

// Output text to terminal
func Output(w io.Writer, context whisper.Context, colorize bool) error {
	for {
		segment, err := context.NextSegment()
		if err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
		fmt.Fprintf(w, "[%6s->%6s]", segment.Start.Truncate(time.Millisecond), segment.End.Truncate(time.Millisecond))
		if colorize {
			for _, token := range segment.Tokens {
				if !context.IsText(token) {
					continue
				}
				fmt.Fprint(w, " ", Colorize(token.Text, int(token.P*24.0)))
			}
			fmt.Fprint(w, "\n")
		} else {
			fmt.Fprintln(w, " ", segment.Text)
		}
	}
}

// Return srtTimestamp
func srtTimestamp(t time.Duration) string {
	return fmt.Sprintf("%02d:%02d:%02d,%03d", t/time.Hour, (t%time.Hour)/time.Minute, (t%time.Minute)/time.Second, (t%time.Second)/time.Millisecond)
}
