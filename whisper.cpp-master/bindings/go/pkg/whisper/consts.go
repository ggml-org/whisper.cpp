package whisper

import (
	"errors"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

///////////////////////////////////////////////////////////////////////////////
// ERRORS

var (
	ErrUnableToLoadModel    = errors.New("unable to load model")
	ErrInternalAppError     = errors.New("internal application error")
	ErrProcessingFailed     = errors.New("processing failed")
	ErrUnsupportedLanguage  = errors.New("unsupported language")
	ErrModelNotMultilingual = errors.New("model is not multilingual")
)

///////////////////////////////////////////////////////////////////////////////
// CONSTANTS

// SampleRate is the sample rate of the audio data.
const SampleRate = whisper.SampleRate

// SampleBits is the number of bytes per sample.
const SampleBits = whisper.SampleBits
