package whisper

import (
	"fmt"
	"io"
	"time"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// SegmentCallback is the callback function for processing segments in real
// time. It is called during the Process function
type SegmentCallback func(Segment)

// ProgressCallback is the callback function for reporting progress during
// processing. It is called during the Process function
type ProgressCallback func(int)

// EncoderBeginCallback is the callback function for checking if we want to
// continue processing. It is called during the Process function
type EncoderBeginCallback func() bool

type ParamsConfigure func(*Parameters)

// Model is the interface to a whisper model. Create a new model with the
// function whisper.New(string)
// Deprecated: Use NewModel implementation struct instead of relying on this interface
type Model interface {
	io.Closer

	// Return a new speech-to-text context.
	// It may return an error is the model is not loaded or closed
	// Deprecated: Use NewContext implementation struct instead of relying on this interface
	NewContext() (Context, error)

	// Return true if the model is multilingual.
	// It returns false if the model is not loaded or closed
	IsMultilingual() bool

	// Return all languages supported.
	Languages() []string
}

// Context is the speech recognition context.
// Deprecated: Use NewContext implementation struct instead of relying on this interface
type Context interface {
	io.Closer

	// Deprecated: Use Params().SetLanguage() instead
	SetLanguage(string) error

	// Deprecated: Use Params().SetTranslate() instead
	SetTranslate(bool)

	// Deprecated: Use Params().SetSplitOnWord() instead
	SetSplitOnWord(bool)

	// Deprecated: Use Params().SetThreads() instead
	SetThreads(uint)

	// Deprecated: Use Params().SetOffset() instead
	SetOffset(time.Duration)

	// Deprecated: Use Params().SetDuration() instead
	SetDuration(time.Duration)

	// Deprecated: Use Params().SetTokenThreshold() instead
	SetTokenThreshold(float32)

	// Deprecated: Use Params().SetTokenSumThreshold() instead
	SetTokenSumThreshold(float32)
	// Deprecated: Use Params().SetMaxSegmentLength() instead

	SetMaxSegmentLength(uint)

	// Deprecated: Use Params().SetTokenTimestamps() instead
	SetTokenTimestamps(bool)

	// Deprecated: Use Params().SetMaxTokensPerSegment() instead
	SetMaxTokensPerSegment(uint)

	// Deprecated: Use Params().SetAudioCtx() instead
	SetAudioCtx(uint)

	// Deprecated: Use Params().SetMaxContext() instead
	SetMaxContext(int)

	// Deprecated: Use Params().SetBeamSize() instead
	SetBeamSize(int)

	// Deprecated: Use Params().SetEntropyThold() instead
	SetEntropyThold(float32)

	// Deprecated: Use Params().SetTemperature() instead
	SetTemperature(float32)

	// Deprecated: Use Params().SetTemperatureFallback() instead
	SetTemperatureFallback(float32)

	// Deprecated: Use Params().SetInitialPrompt() instead
	SetInitialPrompt(string)

	// Get language of the context parameters
	// Deprecated: Use Params().Language() instead
	Language() string

	// Deprecated: Use Model().IsMultilingual() instead
	IsMultilingual() bool

	// Get detected language
	DetectedLanguage() string

	// Voice Activity Detection (VAD) methods
	// Deprecated: Use Params().SetVAD() instead
	SetVAD(bool)
	// Deprecated: Use Params().SetVADModelPath() instead
	SetVADModelPath(string)
	// Deprecated: Use Params().SetVADThreshold() instead
	SetVADThreshold(float32)
	// Deprecated: Use Params().SetVADMinSpeechMs() instead
	SetVADMinSpeechMs(int)
	// Deprecated: Use Params().SetVADMinSilenceMs() instead
	SetVADMinSilenceMs(int)
	// Deprecated: Use Params().SetVADMaxSpeechSec() instead
	SetVADMaxSpeechSec(float32)
	// Deprecated: Use Params().SetVADSpeechPadMs() instead
	SetVADSpeechPadMs(int)
	// Deprecated: Use Params().SetVADSamplesOverlap() instead
	SetVADSamplesOverlap(float32)

	// Process mono audio data and return any errors.
	// If defined, newly generated segments are passed to the
	// callback function during processing.
	Process([]float32, EncoderBeginCallback, SegmentCallback, ProgressCallback) error

	// After process is called, return segments until the end of the stream
	// is reached, when io.EOF is returned.
	NextSegment() (Segment, error)

	// Deprecated: Use Model().TokenIdentifier().IsBEG() instead
	IsBEG(Token) bool

	// Deprecated: Use Model().TokenIdentifier().IsSOT() instead
	IsSOT(Token) bool

	// Deprecated: Use Model().TokenIdentifier().IsEOT() instead
	IsEOT(Token) bool

	// Deprecated: Use Model().TokenIdentifier().IsPREV() instead
	IsPREV(Token) bool

	// Deprecated: Use Model().TokenIdentifier().IsSOLM() instead
	IsSOLM(Token) bool

	// Deprecated: Use Model().TokenIdentifier().IsNOT() instead
	IsNOT(Token) bool

	// Deprecated: Use Model().TokenIdentifier().IsLANG() instead
	IsLANG(Token, string) bool

	// Deprecated: Use Model().TokenIdentifier().IsText() instead
	IsText(Token) bool

	// Deprecated: Use Model().PrintTimings() instead
	// these are model-level performance metrics
	PrintTimings()

	// Deprecated: Use Model().ResetTimings() instead
	// these are model-level performance metrics
	ResetTimings()

	// SystemInfo returns the system information
	SystemInfo() string
}

// Segment is the text result of a speech recognition.
type Segment struct {
	// Segment Number
	Num int

	// Time beginning and end timestamps for the segment.
	Start, End time.Duration

	// The text of the segment.
	Text string

	// The tokens of the segment.
	Tokens []Token

	// True if the next segment is predicted as a speaker turn (tinydiarize)
	// It works only with the diarization supporting models (like small.en-tdrz.bin) with the diarization enabled
	// using Parameters.SetDiarize(true)
	SpeakerTurnNext bool
}

func (s Segment) String() string {
	// foramt: [00:01:39.000 --> 00:01:50.000]   And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
	return fmt.Sprintf("[%s --> %s] %s", s.Start.Truncate(time.Millisecond), s.End.Truncate(time.Millisecond), s.Text)
}

// Token is a text or special token
type Token struct {
	// ID of the token
	Id int

	// Text of the token
	Text string

	// Probability of the token
	P float32

	// Timestamp of the token
	Start, End time.Duration
}
