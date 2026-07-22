package main

import (
	"flag"
	"fmt"
	"strings"
	"time"

	// Packages
	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type Flags struct {
	*flag.FlagSet
}

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func NewFlags(name string, args []string) (*Flags, error) {
	flags := &Flags{
		FlagSet: flag.NewFlagSet(name, flag.ContinueOnError),
	}

	// Register the command line arguments
	registerFlags(flags)

	// Parse command line
	if err := flags.Parse(args); err != nil {
		return nil, err
	}

	// Return success
	return flags, nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

func (flags *Flags) GetModel() string {
	return flags.Lookup("model").Value.String()
}

func (flags *Flags) GetLanguage() string {
	return flags.Lookup("language").Value.String()
}

func (flags *Flags) IsTranslate() bool {
	return flags.Lookup("translate").Value.(flag.Getter).Get().(bool)
}

func (flags *Flags) GetOffset() time.Duration {
	return flags.Lookup("offset").Value.(flag.Getter).Get().(time.Duration)
}

func (flags *Flags) GetDuration() time.Duration {
	return flags.Lookup("duration").Value.(flag.Getter).Get().(time.Duration)
}

func (flags *Flags) GetThreads() uint {
	return flags.Lookup("threads").Value.(flag.Getter).Get().(uint)
}

func (flags *Flags) GetOut() string {
	return strings.ToLower(flags.Lookup("out").Value.String())
}

func (flags *Flags) IsTokens() bool {
	return flags.Lookup("tokens").Value.String() == "true"
}

func (flags *Flags) IsColorize() bool {
	return flags.Lookup("colorize").Value.String() == "true"
}

func (flags *Flags) GetMaxLen() uint {
	return flags.Lookup("max-len").Value.(flag.Getter).Get().(uint)
}

func (flags *Flags) GetMaxTokens() uint {
	return flags.Lookup("max-tokens").Value.(flag.Getter).Get().(uint)
}

func (flags *Flags) GetWordThreshold() float32 {
	return float32(flags.Lookup("word-thold").Value.(flag.Getter).Get().(float64))
}

func (flags *Flags) SetParams(context whisper.Context) error {
	if lang := flags.GetLanguage(); lang != "" && lang != "auto" {
		fmt.Fprintf(flags.Output(), "Setting language to %q\n", lang)
		if err := context.SetLanguage(lang); err != nil {
			return err
		}
	}
	if flags.IsTranslate() && context.IsMultilingual() {
		fmt.Fprintf(flags.Output(), "Setting translate to true\n")
		context.SetTranslate(true)
	}
	if offset := flags.GetOffset(); offset != 0 {
		fmt.Fprintf(flags.Output(), "Setting offset to %v\n", offset)
		context.SetOffset(offset)
	}
	if duration := flags.GetDuration(); duration != 0 {
		fmt.Fprintf(flags.Output(), "Setting duration to %v\n", duration)
		context.SetDuration(duration)
	}
	if threads := flags.GetThreads(); threads != 0 {
		fmt.Fprintf(flags.Output(), "Setting threads to %d\n", threads)
		context.SetThreads(threads)
	}
	if max_len := flags.GetMaxLen(); max_len != 0 {
		fmt.Fprintf(flags.Output(), "Setting max_segment_length to %d\n", max_len)
		context.SetMaxSegmentLength(max_len)
	}
	if max_tokens := flags.GetMaxTokens(); max_tokens != 0 {
		fmt.Fprintf(flags.Output(), "Setting max_tokens to %d\n", max_tokens)
		context.SetMaxTokensPerSegment(max_tokens)
	}
	if word_threshold := flags.GetWordThreshold(); word_threshold != 0 {
		fmt.Fprintf(flags.Output(), "Setting word_threshold to %f\n", word_threshold)
		context.SetTokenThreshold(word_threshold)
	}

	// Return success
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func registerFlags(flag *Flags) {
	flag.String("model", "", "Path to the model file")
	flag.String("language", "", "Spoken language")
	flag.Bool("translate", false, "Translate from source language to english")
	flag.Duration("offset", 0, "Time offset")
	flag.Duration("duration", 0, "Duration of audio to process")
	flag.Uint("threads", 0, "Number of threads to use")
	flag.Uint("max-len", 0, "Maximum segment length in characters")
	flag.Uint("max-tokens", 0, "Maximum tokens per segment")
	flag.Float64("word-thold", 0, "Maximum segment score")
	flag.Bool("tokens", false, "Display tokens")
	flag.Bool("colorize", false, "Colorize tokens")
	flag.String("out", "", "Output format (srt, none or leave as empty string)")
}
