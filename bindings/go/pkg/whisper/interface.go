package whisper

import (
	"io"
	"time"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
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
	NewContext() (Context, error)

	NewParams(
		sampling SamplingStrategy,
		configure ParamsConfigure,
	) (Parameters, error)

	// Return true if the model is multilingual.
	IsMultilingual() bool

	// Return all languages supported.
	Languages() []string

	// Model performance timing methods
	// Print model performance timings to stdout
	PrintTimings()

	// Reset model performance timing counters
	ResetTimings()

	// WhisperContext returns the memory-safe whisper context wrapper of the raw whisper context
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
	SetLanguage(string) error

	// Set single segment mode
	SetSingleSegment(bool)

	// Getter methods
	Language() string
	Threads() int

	UnsafeParams() *whisper.Params
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

	// Return true if the model is multilingual.
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

	// Deprecated token methods - use Model.IsBEG(), Model.IsSOT(), etc. instead
	// Deprecated: Use Model.IsBEG() instead
	IsBEG(Token) bool

	// Deprecated: Use Model.IsSOT() instead
	IsSOT(Token) bool

	// Deprecated: Use Model.IsEOT() instead
	IsEOT(Token) bool

	// Deprecated: Use Model.IsPREV() instead
	IsPREV(Token) bool

	// Deprecated: Use Model.IsSOLM() instead
	IsSOLM(Token) bool

	// Deprecated: Use Model.IsNOT() instead
	IsNOT(Token) bool

	// Deprecated: Use Model.IsLANG() instead
	IsLANG(Token, string) bool

	// Deprecated: Use Model.IsText() instead
	IsText(Token) bool

	// Deprecated: Use Model.PrintTimings() instead - these are model-level performance metrics
	PrintTimings()

	// Deprecated: Use Model.ResetTimings() instead - these are model-level performance metrics
	ResetTimings()

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
	SpeakerTurnNext bool
}

// Token is a text or special token
type Token struct {
	Id         int
	Text       string
	P          float32
	Start, End time.Duration
}
