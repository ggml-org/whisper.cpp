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

// Model is the interface to a whisper model. Create a new model with the
// function whisper.New(string)
type Model interface {
	io.Closer

	// Return a new speech-to-text context.
	NewContext() (Context, error)

	// Return true if the model is multilingual.
	IsMultilingual() bool

	// Return all languages supported.
	Languages() []string
}

// Context is the speech recognition context.
type Context interface {
	SetLanguage(string) error // Set the language to use for speech recognition, use "auto" for auto detect language.
	SetTranslate(bool)        // Set translate flag
	IsMultilingual() bool     // Return true if the model is multilingual.
	Language() string         // Get language
	DetectedLanguage() string // Get detected language

	SetOffset(time.Duration)          // Set offset
	SetDuration(time.Duration)        // Set duration
	SetThreads(uint)                  // Set number of threads to use
	SetSplitOnWord(bool)              // Set split on word flag
	SetTokenThreshold(float32)        // Set timestamp token probability threshold
	SetTokenSumThreshold(float32)     // Set timestamp token sum probability threshold
	SetMaxSegmentLength(uint)         // Set max segment length in characters
	SetTokenTimestamps(bool)          // Set token timestamps flag
	SetMaxTokensPerSegment(uint)      // Set max tokens per segment (0 = no limit)
	SetAudioCtx(uint)                 // Set audio encoder context
	SetMaxContext(n int)              // Set maximum number of text context tokens to store
	SetBeamSize(n int)                // Set Beam Size
	SetEntropyThold(t float32)        // Set Entropy threshold
	SetInitialPrompt(prompt string)   // Set initial prompt
	SetTemperature(t float32)         // Set temperature
	SetTemperatureFallback(t float32) // Set temperature incrementation

	// Process mono audio data and return any errors.
	// If defined, newly generated segments are passed to the
	// callback function during processing.
	Process([]float32, EncoderBeginCallback, SegmentCallback, ProgressCallback) error

	// After process is called, return segments until the end of the stream
	// is reached, when io.EOF is returned.
	NextSegment() (Segment, error)

	IsBEG(Token) bool          // Test for "begin" token
	IsSOT(Token) bool          // Test for "start of transcription" token
	IsEOT(Token) bool          // Test for "end of transcription" token
	IsPREV(Token) bool         // Test for "start of prev" token
	IsSOLM(Token) bool         // Test for "start of lm" token
	IsNOT(Token) bool          // Test for "No timestamps" token
	IsLANG(Token, string) bool // Test for token associated with a specific language
	IsText(Token) bool         // Test for text token

	// Timings
	PrintTimings()
	ResetTimings()

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
}

// Token is a text or special token
type Token struct {
	Id         int
	Text       string
	P          float32
	Start, End time.Duration
}
