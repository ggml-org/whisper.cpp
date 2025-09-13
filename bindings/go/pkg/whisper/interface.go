package whisper

import (
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

type TokenIdentifier interface {
	// Test for "begin" token
	IsBEG(Token) (bool, error)

	// Test for "start of transcription" token
	IsSOT(Token) (bool, error)

	// Test for "end of transcription" token
	IsEOT(Token) (bool, error)

	// Test for "start of prev" token
	IsPREV(Token) (bool, error)

	// Test for "start of lm" token
	IsSOLM(Token) (bool, error)

	// Test for "no timestamps" token
	IsNOT(Token) (bool, error)

	// Test for token associated with a specific language
	IsLANG(Token, string) (bool, error)

	// Test for text token
	IsText(Token) (bool, error)
}

type ParamsConfigure func(Parameters)

// Model is the interface to a whisper model. Create a new model with the
// function whisper.New(string)
type Model interface {
	io.Closer

	// Return a new speech-to-text context.
	// It may return an error is the model is not loaded or closed
	NewContext() (Context, error)

	// Return a new parameters wrapper
	// sampling is the sampling strategy to use
	// configure is the function to configure the parameters
	// It may return an error is the model is not loaded or closed
	NewParams(
		sampling SamplingStrategy,
		configure ParamsConfigure,
	) (Parameters, error)

	// Return a new speech-to-text context configured via the provided function
	// and sampling strategy. The context is backed by an isolated whisper_state.
	// It may return an error is the model is not loaded or closed
	NewContextWithParams(
		sampling SamplingStrategy,
		configure ParamsConfigure,
	) (Context, error)

	// Return true if the model is multilingual.
	// It returns false if the model is not loaded or closed
	IsMultilingual() bool

	// Return all languages supported.
	Languages() []string

	// Model performance timing methods
	// Print model performance timings to stdout
	PrintTimings()

	// Reset model performance timing counters
	ResetTimings()

	// WhisperContext returns the memory-safe whisper context wrapper of the raw whisper context
	// You may need to use this to get the raw whisper context
	// Ot check that the model's context is not closed
	WhisperContext() WhisperContext

	// Token identifier
	TokenIdentifier() TokenIdentifier
}

// Parameters configures decode / processing behavior
type Parameters interface {
	SetTranslate(bool)
	SetSplitOnWord(bool)
	SetThreads(uint)
	SetOffset(time.Duration)
	SetDuration(time.Duration)
	SetTokenThreshold(float32)
	SetTokenSumThreshold(float32)
	SetMaxSegmentLength(uint)
	SetTokenTimestamps(bool)
	SetMaxTokensPerSegment(uint)
	SetAudioCtx(uint)
	SetMaxContext(n int)
	SetBeamSize(n int)
	SetEntropyThold(t float32)
	SetInitialPrompt(prompt string)

	SetNoContext(bool)
	SetPrintSpecial(bool)
	SetPrintProgress(bool)
	SetPrintRealtime(bool)
	SetPrintTimestamps(bool)

	// Enable extra debug info (e.g., dump log_mel)
	SetDebugMode(bool)
	// Diarization (tinydiarize)
	SetDiarize(bool)

	// Voice Activity Detection (VAD)
	SetVAD(bool)
	SetVADModelPath(string)
	SetVADThreshold(float32)
	SetVADMinSpeechMs(int)
	SetVADMinSilenceMs(int)
	SetVADMaxSpeechSec(float32)
	SetVADSpeechPadMs(int)
	SetVADSamplesOverlap(float32)

	// Set the temperature
	SetTemperature(t float32)

	// Set the fallback temperature incrementation
	// Pass -1.0 to disable this feature
	SetTemperatureFallback(t float32)

	// Set the language
	// If the model is not multilingual, this will return an error
	SetLanguage(string) error

	// Set single segment mode
	SetSingleSegment(bool)

	// Getter methods
	Language() string
	Threads() int
}

// Context is the speech recognition context.
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

	// Return the model that the context is backed by
	Model() Model

	// Deprecated: Use Model().IsMultilingual() instead
	IsMultilingual() bool

	// Get detected language
	DetectedLanguage() string

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

	// Params returns a high-level parameters wrapper - preferred method
	Params() Parameters
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
