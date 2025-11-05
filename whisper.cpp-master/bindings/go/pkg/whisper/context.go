package whisper

import (
	"fmt"
	"io"
	"runtime"
	"strings"
	"time"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type context struct {
	n      int
	model  *model
	params whisper.Params
}

// Make sure context adheres to the interface
var _ Context = (*context)(nil)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func newContext(model *model, params whisper.Params) (Context, error) {
	context := new(context)
	context.model = model
	context.params = params

	// Return success
	return context, nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Set the language to use for speech recognition.
func (context *context) SetLanguage(lang string) error {
	if context.model.ctx == nil {
		return ErrInternalAppError
	}
	if !context.model.IsMultilingual() {
		return ErrModelNotMultilingual
	}

	if lang == "auto" {
		context.params.SetLanguage(-1)
	} else if id := context.model.ctx.Whisper_lang_id(lang); id < 0 {
		return ErrUnsupportedLanguage
	} else if err := context.params.SetLanguage(id); err != nil {
		return err
	}
	// Return success
	return nil
}

func (context *context) IsMultilingual() bool {
	return context.model.IsMultilingual()
}

// Get language
func (context *context) Language() string {
	id := context.params.Language()
	if id == -1 {
		return "auto"
	}
	return whisper.Whisper_lang_str(context.params.Language())
}

func (context *context) DetectedLanguage() string {
	return whisper.Whisper_lang_str(context.model.ctx.Whisper_full_lang_id())
}

// Set translate flag
func (context *context) SetTranslate(v bool) {
	context.params.SetTranslate(v)
}

func (context *context) SetSplitOnWord(v bool) {
	context.params.SetSplitOnWord(v)
}

// Set number of threads to use
func (context *context) SetThreads(v uint) {
	context.params.SetThreads(int(v))
}

// Set time offset
func (context *context) SetOffset(v time.Duration) {
	context.params.SetOffset(int(v.Milliseconds()))
}

// Set duration of audio to process
func (context *context) SetDuration(v time.Duration) {
	context.params.SetDuration(int(v.Milliseconds()))
}

// Set timestamp token probability threshold (~0.01)
func (context *context) SetTokenThreshold(t float32) {
	context.params.SetTokenThreshold(t)
}

// Set timestamp token sum probability threshold (~0.01)
func (context *context) SetTokenSumThreshold(t float32) {
	context.params.SetTokenSumThreshold(t)
}

// Set max segment length in characters
func (context *context) SetMaxSegmentLength(n uint) {
	context.params.SetMaxSegmentLength(int(n))
}

// Set token timestamps flag
func (context *context) SetTokenTimestamps(b bool) {
	context.params.SetTokenTimestamps(b)
}

// Set max tokens per segment (0 = no limit)
func (context *context) SetMaxTokensPerSegment(n uint) {
	context.params.SetMaxTokensPerSegment(int(n))
}

// Set audio encoder context
func (context *context) SetAudioCtx(n uint) {
	context.params.SetAudioCtx(int(n))
}

// Set maximum number of text context tokens to store
func (context *context) SetMaxContext(n int) {
	context.params.SetMaxContext(n)
}

// Set Beam Size
func (context *context) SetBeamSize(n int) {
	context.params.SetBeamSize(n)
}

// Set Entropy threshold
func (context *context) SetEntropyThold(t float32) {
	context.params.SetEntropyThold(t)
}

// Set Temperature
func (context *context) SetTemperature(t float32) {
	context.params.SetTemperature(t)
}

// Set the fallback temperature incrementation
// Pass -1.0 to disable this feature
func (context *context) SetTemperatureFallback(t float32) {
	context.params.SetTemperatureFallback(t)
}

// Set initial prompt
func (context *context) SetInitialPrompt(prompt string) {
	context.params.SetInitialPrompt(prompt)
}

// ResetTimings resets the mode timings. Should be called before processing
func (context *context) ResetTimings() {
	context.model.ctx.Whisper_reset_timings()
}

// PrintTimings prints the model timings to stdout.
func (context *context) PrintTimings() {
	context.model.ctx.Whisper_print_timings()
}

// SystemInfo returns the system information
func (context *context) SystemInfo() string {
	return fmt.Sprintf("system_info: n_threads = %d / %d | %s\n",
		context.params.Threads(),
		runtime.NumCPU(),
		whisper.Whisper_print_system_info(),
	)
}

// Use mel data at offset_ms to try and auto-detect the spoken language
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// Returns the probabilities of all languages.
func (context *context) WhisperLangAutoDetect(offset_ms int, n_threads int) ([]float32, error) {
	langProbs, err := context.model.ctx.Whisper_lang_auto_detect(offset_ms, n_threads)
	if err != nil {
		return nil, err
	}
	return langProbs, nil
}

// Process new sample data and return any errors
func (context *context) Process(
	data []float32,
	callEncoderBegin EncoderBeginCallback,
	callNewSegment SegmentCallback,
	callProgress ProgressCallback,
) error {
	if context.model.ctx == nil {
		return ErrInternalAppError
	}
	// If the callback is defined then we force on single_segment mode
	if callNewSegment != nil {
		context.params.SetSingleSegment(true)
	}

	// We don't do parallel processing at the moment
	processors := 0
	if processors > 1 {
		if err := context.model.ctx.Whisper_full_parallel(context.params, data, processors, callEncoderBegin,
			func(new int) {
				if callNewSegment != nil {
					num_segments := context.model.ctx.Whisper_full_n_segments()
					s0 := num_segments - new
					for i := s0; i < num_segments; i++ {
						callNewSegment(toSegment(context.model.ctx, i))
					}
				}
			}); err != nil {
			return err
		}
	} else if err := context.model.ctx.Whisper_full(context.params, data, callEncoderBegin,
		func(new int) {
			if callNewSegment != nil {
				num_segments := context.model.ctx.Whisper_full_n_segments()
				s0 := num_segments - new
				for i := s0; i < num_segments; i++ {
					callNewSegment(toSegment(context.model.ctx, i))
				}
			}
		}, func(progress int) {
			if callProgress != nil {
				callProgress(progress)
			}
		}); err != nil {
		return err
	}

	// Return success
	return nil
}

// Return the next segment of tokens
func (context *context) NextSegment() (Segment, error) {
	if context.model.ctx == nil {
		return Segment{}, ErrInternalAppError
	}
	if context.n >= context.model.ctx.Whisper_full_n_segments() {
		return Segment{}, io.EOF
	}

	// Populate result
	result := toSegment(context.model.ctx, context.n)

	// Increment the cursor
	context.n++

	// Return success
	return result, nil
}

// Test for text tokens
func (context *context) IsText(t Token) bool {
	switch {
	case context.IsBEG(t):
		return false
	case context.IsSOT(t):
		return false
	case whisper.Token(t.Id) >= context.model.ctx.Whisper_token_eot():
		return false
	case context.IsPREV(t):
		return false
	case context.IsSOLM(t):
		return false
	case context.IsNOT(t):
		return false
	default:
		return true
	}
}

// Test for "begin" token
func (context *context) IsBEG(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_beg()
}

// Test for "start of transcription" token
func (context *context) IsSOT(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_sot()
}

// Test for "end of transcription" token
func (context *context) IsEOT(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_eot()
}

// Test for "start of prev" token
func (context *context) IsPREV(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_prev()
}

// Test for "start of lm" token
func (context *context) IsSOLM(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_solm()
}

// Test for "No timestamps" token
func (context *context) IsNOT(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_not()
}

// Test for token associated with a specific language
func (context *context) IsLANG(t Token, lang string) bool {
	if id := context.model.ctx.Whisper_lang_id(lang); id >= 0 {
		return whisper.Token(t.Id) == context.model.ctx.Whisper_token_lang(id)
	} else {
		return false
	}
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func toSegment(ctx *whisper.Context, n int) Segment {
	return Segment{
		Num:    n,
		Text:   strings.TrimSpace(ctx.Whisper_full_get_segment_text(n)),
		Start:  time.Duration(ctx.Whisper_full_get_segment_t0(n)) * time.Millisecond * 10,
		End:    time.Duration(ctx.Whisper_full_get_segment_t1(n)) * time.Millisecond * 10,
		Tokens: toTokens(ctx, n),
	}
}

func toTokens(ctx *whisper.Context, n int) []Token {
	result := make([]Token, ctx.Whisper_full_n_tokens(n))
	for i := 0; i < len(result); i++ {
		data := ctx.Whisper_full_get_token_data(n, i)

		result[i] = Token{
			Id:    int(ctx.Whisper_full_get_token_id(n, i)),
			Text:  ctx.Whisper_full_get_token_text(n, i),
			P:     ctx.Whisper_full_get_token_p(n, i),
			Start: time.Duration(data.T0()) * time.Millisecond * 10,
			End:   time.Duration(data.T1()) * time.Millisecond * 10,
		}
	}
	return result
}
